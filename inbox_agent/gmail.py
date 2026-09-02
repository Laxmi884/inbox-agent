"""Gmail adapters.

Stage A runs entirely against a frozen snapshot so every architecture is compared
on identical input (spec section 6) and no iteration can touch the real mailbox.
A LiveGmailClient implementing the same protocol lands when live writes are in
scope; nothing above this module changes when it does.
"""
from __future__ import annotations

import base64
import html as html_mod
import json
import re
import threading
from pathlib import Path
from typing import Any, Protocol

from .models import Thread


# The subset of Gmail search syntax the snapshot client can honour. Kept
# deliberately tiny: the live client hands `query` to the Gmail API verbatim and
# never calls this function at all. It exists so that both clients answer the
# SAME query string, which is the only thing that makes a snapshot test evidence
# about live behaviour.
def matches_query(thread: Thread, query: str) -> bool:
    """True if `thread` satisfies every term in `query`.

    Raises on a term this parser does not implement, rather than ignoring it. An
    ignored term would pass live (Gmail understands it) and silently do nothing
    here, turning every snapshot test for that term into a false negative.
    """
    labels = {label.upper() for label in thread.label_ids}
    for term in query.split():
        negate = term.startswith("-")
        bare = term[1:] if negate else term
        if bare == "is:unread":
            present = "UNREAD" in labels
        elif bare == "in:inbox":
            present = "INBOX" in labels
        elif bare.startswith("label:"):
            present = bare[len("label:"):].upper() in labels
        else:
            raise ValueError(
                f"{bare!r} is not a query term the snapshot client implements")
        if present == negate:
            return False
    return True


class GmailClient(Protocol):
    def list_threads(self, limit: int = 50, query: str = "") -> list[Thread]: ...
    def get_thread(self, thread_id: str) -> Thread: ...
    def apply_label(self, thread_id: str, label: str) -> dict[str, Any]: ...
    def remove_label(self, thread_id: str, label: str) -> dict[str, Any]: ...
    def archive(self, thread_id: str) -> dict[str, Any]: ...
    def trash(self, thread_id: str) -> dict[str, Any]: ...
    def create_draft(self, thread_id: str, body: str) -> dict[str, Any]: ...


def load_snapshot(path: Path | str) -> list[Thread]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No snapshot at {path}. Ask Claude to pull one with the Gmail MCP tools: "
            f"'Pull 50 recent inbox threads and write them to {path}'."
        )
    return [Thread.model_validate(d) for d in json.loads(path.read_text())]


class SnapshotGmailClient:
    """Reads a frozen snapshot; mutations change in-memory state only.

    Every mutating method returns {"simulated": True} so callers can never mistake
    a snapshot write for a real one.
    """

    def __init__(self, path: Path | str):
        self._threads: dict[str, Thread] = {t.id: t for t in load_snapshot(path)}
        self._order: list[str] = list(self._threads)

    def list_threads(self, limit: int = 50, query: str = "") -> list[Thread]:
        # Filter BEFORE limiting. Limiting first would mean "the 50 newest, of
        # which some are unread", so an inbox with a long read run reads as empty.
        threads = (self._threads[i] for i in self._order)
        if query:
            threads = (t for t in threads if matches_query(t, query))
        out = []
        for thread in threads:
            out.append(thread)
            if len(out) >= limit:
                break
        return out

    def get_thread(self, thread_id: str) -> Thread:
        if thread_id not in self._threads:
            raise KeyError(f"thread {thread_id!r} not in snapshot")
        return self._threads[thread_id]

    def _labels(self, thread_id: str) -> list[str]:
        return self.get_thread(thread_id).label_ids

    def apply_label(self, thread_id: str, label: str) -> dict[str, Any]:
        labels = self._labels(thread_id)
        if label not in labels:
            labels.append(label)
        return {"simulated": True, "thread_id": thread_id, "label": label}

    def remove_label(self, thread_id: str, label: str) -> dict[str, Any]:
        labels = self._labels(thread_id)
        if label in labels:
            labels.remove(label)
        return {"simulated": True, "thread_id": thread_id, "label": label}

    def archive(self, thread_id: str) -> dict[str, Any]:
        return self.remove_label(thread_id, "INBOX") | {"action": "archive"}

    def trash(self, thread_id: str) -> dict[str, Any]:
        self.remove_label(thread_id, "INBOX")
        return self.apply_label(thread_id, "TRASH") | {"action": "trash"}

    def create_draft(self, thread_id: str, body: str) -> dict[str, Any]:
        return {"simulated": True, "thread_id": thread_id, "draft_chars": len(body)}


class _LabelMap:
    """Two-way map between Gmail label ids and display names.

    `Thread.label_ids` holds NAMES in both client implementations. That is the
    contract the comment above `matches_query` exists to protect: both clients
    answering the same query string is the only thing that makes a snapshot
    test evidence about live behaviour. Storing raw ids live would mean a
    snapshot test asserting on 'agent/triaged' and a live run asserting on
    'Label_12' are no longer the same assertion.

    Ids cannot be derived from names and have no reliable shape - `Label_1` and
    `Label_6111317184412779502` are both in use in one real account - so the map
    must come from labels.list, never a heuristic. Built once per client,
    refreshed on a miss.

    Knowingly accepted: a label renamed in Gmail changes identity from this
    project's point of view. That is correct - the name is what the owner sees
    and what the digest reports.

    Thread safety. The client hydrates threads through a ThreadPoolExecutor, and
    every one of those workers calls `to_name`. Two rules keep that sound:

    - Writers hold `_lock`; readers never do. A read is a single dict lookup on
      an attribute that is only ever REPLACED, never mutated in place, so a
      reader either sees the whole old map or the whole new one.
    - `refresh` therefore builds new dicts and rebinds. Clearing and
      repopulating in place would let a concurrent reader observe a half-built
      map and surface a raw id where a name was available - an intermittent
      wrong label in the digest, and close to undiagnosable afterwards.
    """

    def __init__(self, service):
        self._service = service
        self._by_id: dict[str, str] = {}
        self._by_name: dict[str, str] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def _fetch(self) -> None:
        """Rebuild from labels.list. Caller holds the lock."""
        result = self._service.users().labels().list(userId="me").execute()
        by_id, by_name = {}, {}
        for label in result.get("labels", []):
            by_id[label["id"]] = label["name"]
            by_name[label["name"]] = label["id"]
        # Rebind, never mutate: see the thread-safety note above.
        self._by_id, self._by_name = by_id, by_name
        self._loaded = True

    def refresh(self) -> None:
        with self._lock:
            self._fetch()

    def _ensure(self) -> None:
        """Build the map if it has not been built.

        Called explicitly before fanning out across threads, so the initial
        build happens once rather than being raced for by five workers.
        """
        if not self._loaded:
            with self._lock:
                if not self._loaded:      # another thread may have won the race
                    self._fetch()

    def to_name(self, label_id: str) -> str:
        """Id -> display name.

        A miss means a label created in Gmail since the map was built, so
        refetch once. If it is still unknown, surface the raw id rather than
        dropping it: a dropped label is silent data loss into the classifier's
        `Current labels:` line, and a visibly odd id is far easier to diagnose
        than a label that quietly vanished.
        """
        self._ensure()
        name = self._by_id.get(label_id)
        if name is not None:
            return name
        self.refresh()
        return self._by_id.get(label_id, label_id)

    def to_id(self, name: str) -> str:
        """Display name -> id, creating the label if it does not exist.

        Creation is what makes the live path survivable: `agent/triaged` does
        not exist in the mailbox - confirmed against it - and `mark_triaged`
        needs it on the first run, for every thread processed.

        The refetch before creating is not belt-and-braces. Without it, a label
        created in Gmail (or by another process) after this map was built would
        be created a second time, leaving two labels sharing one name and a
        `to_name` lookup that depends on dict ordering.
        """
        self._ensure()
        label_id = self._by_name.get(name)
        if label_id is not None:
            return label_id

        # The whole miss path is serialised. Two workers missing the same name
        # would otherwise both refetch, both still miss, and both create it.
        with self._lock:
            self._fetch()
            label_id = self._by_name.get(name)
            if label_id is not None:
                return label_id
            created = self._service.users().labels().create(
                userId="me",
                body={"name": name,
                      "labelListVisibility": "labelShow",
                      "messageListVisibility": "show"}).execute()
            self._by_id = self._by_id | {created["id"]: created["name"]}
            self._by_name = self._by_name | {created["name"]: created["id"]}
            return created["id"]


# Everything between these tags is markup or code, never prose. Removed
# wholesale rather than untagged, because stripping only the tags leaves CSS
# and JS in the prompt - pure token cost, and confusing input for a classifier.
_DROP_ELEMENTS = re.compile(r"<(script|style|head)\b[^>]*>.*?</\1>", re.I | re.S)
# Block-level boundaries become newlines BEFORE the general tag strip. Without
# this the whole mail collapses to a single line, so _fence's 4000-char cut
# lands mid-sentence with no structure to orient on, and _BLANK_LINES below
# never has anything to normalise.
_BLOCK_BREAK = re.compile(
    r"</?(p|div|br|tr|li|h[1-6]|table|blockquote)\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
# Zero-width filler that bulk senders inject in bulk to pad the inbox preview
# line. It carries no meaning and cannot be seen, but it is charged for: on a
# real Strava mail it was 22.4% of the first 4000 characters, which is the slice
# _fence actually hands the model. Deleted outright rather than collapsed - a
# space would be just as wrong, only shorter.
# U+034F COMBINING GRAPHEME JOINER is in here because a real Strava mail used
# 448 of them - found by counting what survived a first pass, not by guessing
# at the set. Bulk senders reach for whatever their ESP offers, so this list is
# empirical and will grow.
_INVISIBLE = re.compile(
    "[\u200b\u200c\u200d\ufeff\u00ad\u034f\u115f\u1160\u3164\u2800]+")
# Exotic spaces normalised into ordinary ones so the collapse below catches
# them. NBSP and figure-space ARE spaces - unlike the class above they mean
# something - so they are converted, never dropped.
_WS = re.compile(r"[ \t\r\f\v\u00a0\u2007\u2009\u200a\u202f\u3000]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _b64url(data: str) -> str:
    """Decode Gmail's base64url, restoring the padding it strips.

    Two traps, both routine rather than exotic: Gmail drops the '=' padding, so
    a plain b64decode raises binascii.Error on roughly three quarters of all
    messages; and the URL-safe alphabet uses '-' and '_' where standard base64
    uses '+' and '/'. errors="replace" on the final decode because a mislabelled
    charset is common and must never take down a run.
    """
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
    except Exception:
        return ""
    return raw.decode("utf-8", errors="replace")


def _strip_invisible(text: str) -> str:
    """Drop zero-width filler and normalise exotic spaces.

    Applied to plain-text parts too, not only HTML: the padding arrives in
    whichever alternative the sender wrote it into, and on real mail the
    text/plain part is where it showed up.
    """
    return _INVISIBLE.sub("", text)


def _normalise(text: str) -> str:
    """Drop invisible filler, collapse runs of space, and delete lines that are
    now empty.

    That last step is not cosmetic. Stripping the filler out of a padded mail
    leaves the rows it was padding as whitespace-only lines, and a run of
    "\n \n \n" is charged for exactly like content. Applied to plain-text and
    HTML alike so the two branches cannot drift.
    """
    text = _strip_invisible(text)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def _html_to_text(html: str) -> str:
    text = _DROP_ELEMENTS.sub(" ", html)
    text = _BLOCK_BREAK.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = html_mod.unescape(text)
    return _normalise(text)


def _walk_parts(payload: dict):
    """Depth-first over the MIME tree, skipping attachments.

    A part with a filename is an attachment even when its mimeType is
    text/plain, so a .txt attachment never becomes the body.
    """
    if payload.get("filename"):
        return
    parts = payload.get("parts")
    if parts:
        for part in parts:
            yield from _walk_parts(part)
    else:
        yield payload


def _extract_body(payload: dict) -> str:
    """Best text for one message payload: text/plain if there is any, else
    stripped text/html, else empty.

    Empty is a legitimate answer, not an error: a calendar invite or a bare
    attachment has no text part, and such a thread must still classify on its
    subject and sender.

    The fallback turns on whether a part yielded TEXT, not on whether a part
    was present. A multipart/alternative whose plain part is whitespace - or
    whose data is corrupt, which decodes to "" - would otherwise return "" and
    never look at the html part carrying the entire message, and the result
    would be indistinguishable from mail that genuinely had no body.
    """
    if not payload:
        return ""
    plain, html = [], []
    for part in _walk_parts(payload):
        # Gmail reports 'text/plain; charset="UTF-8"', never a bare mime type.
        mime = (part.get("mimeType") or "").lower()
        data = (part.get("body") or {}).get("data") or ""
        if not data:
            continue
        if mime.startswith("text/plain"):
            plain.append(_b64url(data))
        elif mime.startswith("text/html"):
            html.append(_b64url(data))

    text = "\n".join(p for p in plain if p.strip())
    if text.strip():
        return _normalise(text)
    text = _html_to_text("\n".join(h for h in html if h.strip()))
    return text or ""

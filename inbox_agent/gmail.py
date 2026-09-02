"""Gmail adapters.

Stage A runs entirely against a frozen snapshot so every architecture is compared
on identical input (spec section 6) and no iteration can touch the real mailbox.
A LiveGmailClient implementing the same protocol lands when live writes are in
scope; nothing above this module changes when it does.
"""
from __future__ import annotations

import json
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

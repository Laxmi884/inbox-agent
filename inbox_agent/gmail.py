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
import http.client
import logging
import random
import re
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional, Protocol

from .models import Thread

log = logging.getLogger(__name__)


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
    def list_thread_ids(self, query: str = "", max_ids: int = 500) -> list[str]: ...
    def get_thread(self, thread_id: str) -> Thread: ...
    def get_threads(self, thread_ids: list[str],
                    *, workers: int | None = None) -> list[Thread]: ...
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

    def list_thread_ids(self, query: str = "", max_ids: int = 500) -> list[str]:
        # Deliberately the same filter as list_threads. Both clients answering
        # one query string identically is the contract matches_query exists to
        # protect, and an ids-only path that drifted from it would be a
        # sampler that sees different mail depending on the backend.
        return [t.id for t in self.list_threads(limit=max_ids, query=query)]

    def get_thread(self, thread_id: str) -> Thread:
        if thread_id not in self._threads:
            raise KeyError(f"thread {thread_id!r} not in snapshot")
        return self._threads[thread_id]

    def get_threads(self, thread_ids: list[str],
                    *, workers: int | None = None) -> list[Thread]:
        return [self.get_thread(i) for i in thread_ids]

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

    def __init__(self, service, http=None):
        self._service = service
        # `http` is a zero-arg callable returning this thread's transport, or
        # None. See LiveGmailClient._http: a refresh triggered by a miss inside
        # a hydration worker would otherwise share the service's single
        # httplib2.Http with whatever else that pool is doing.
        self._http = http or (lambda: None)
        self._by_id: dict[str, str] = {}
        self._by_name: dict[str, str] = {}
        # Gmail enforces name uniqueness case-INSENSITIVELY. Without this index
        # a lookup for 'learning' misses the mailbox's 'Learning', creation is
        # attempted, and Gmail answers 409. See to_id.
        self._by_lower: dict[str, str] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def _fetch(self) -> None:
        """Rebuild from labels.list. Caller holds the lock."""
        result = self._service.users().labels().list(
            userId="me").execute(http=self._http())
        by_id, by_name, by_lower = {}, {}, {}
        for label in result.get("labels", []):
            by_id[label["id"]] = label["name"]
            by_name[label["name"]] = label["id"]
            # setdefault, so that when two labels differ only by case the
            # fallback is stable rather than dependent on labels.list ordering.
            # Which one it picks only matters when neither is an exact match.
            by_lower.setdefault(label["name"].lower(), label["id"])
        # Rebind, never mutate: see the thread-safety note above.
        self._by_id, self._by_name, self._by_lower = by_id, by_name, by_lower
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

        Case. Gmail's uniqueness rule for label names is case-INSENSITIVE, so
        `create("learning")` against a mailbox holding 'Learning' does not
        return the existing label - it fails with 409 "Label name exists or
        conflicts". An exact-case dict cannot see that collision coming, which
        is why the first live run died mid-batch with ten actions already
        written to the mailbox. So a miss falls back to a case-folded lookup
        before concluding the label is new.

        Exact match still wins. Resolving 'learning' to an existing 'Learning'
        is adopting the label the owner already keeps, which is the intent; but
        if both spellings somehow exist, the one asked for is the one meant.
        """
        self._ensure()
        label_id = self._resolve(name)
        if label_id is not None:
            return label_id

        # The whole miss path is serialised. Two workers missing the same name
        # would otherwise both refetch, both still miss, and both create it.
        with self._lock:
            self._fetch()
            label_id = self._resolve(name)
            if label_id is not None:
                return label_id
            created = self._service.users().labels().create(
                userId="me",
                body={"name": name,
                      "labelListVisibility": "labelShow",
                      "messageListVisibility": "show"}).execute(
                          http=self._http())
            self._by_id = self._by_id | {created["id"]: created["name"]}
            self._by_name = self._by_name | {created["name"]: created["id"]}
            self._by_lower = ({created["name"].lower(): created["id"]}
                              | self._by_lower)
            return created["id"]

    def _resolve(self, name: str) -> Optional[str]:
        """Exact name first, then case-folded. None if the label is genuinely new.

        Readers never hold the lock, so this reads each dict exactly once and
        relies on both only ever being REPLACED - the same rule as to_name.
        """
        label_id = self._by_name.get(name)
        if label_id is not None:
            return label_id
        return self._by_lower.get(name.lower())


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


# One list call plus one get per thread. 50 threads sequentially is ~10s of
# almost pure round-trip latency. Five at a time is well inside quota (a
# threads.get is 10 units against 250 units/sec/user, so 50 gets is 500 units)
# and is simpler than BatchHttpRequest, which needs its own callback plumbing
# and error handling for a saving we do not need at this size.
_HYDRATE_WORKERS = 5

# Backoff applies to 429 and 5xx. A 4xx is otherwise a bug in our request - a
# bad label id, a malformed query - and retrying it just makes the same mistake
# more slowly while hiding it from the caller.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# 403 is the exception, and it cost a live run. Gmail signals "you are going
# too fast" with 403, not 429, so the status alone carries two opposite
# meanings: a permission denial that will never succeed, and a rate limit that
# will succeed shortly. The per-error `reason` is what separates them. On
# 2026-09-03 a live /triage 20 died on "Quota exceeded ... Units per minute
# per user" after applying 32 actions to the real mailbox, because the policy
# read the status, saw a 4xx, and gave up.
_RATE_LIMIT_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded"})

# And below the statuses there is a third failure: no status at all. A keepalive
# socket that has gone stale dies on the next write, and _status_of returns None
# for that, which is in no status set - so the policy re-raised instantly. Seen
# live on 2026-09-03: idle from 14:02, a /triage at 15:18:48.429 failed at
# 15:18:48.664. 235ms, in `fetch`, before a single thread was read.
#
# One dead socket surfaces as several different exceptions depending on where it
# dies, so catching only the SSLEOFError we happened to see would leave the rest.
_TRANSPORT_ERRORS = (ssl.SSLError, http.client.HTTPException, OSError)

# Two schedules, because the two failures clear on different timescales. A 5xx
# is a blip: 1+2+4 = 7s. A quota is measured per MINUTE, so the short schedule
# would give up long before it could possibly clear - 2+4+8+16+32 = 62s.
# A dead socket takes a handshake, not a minute, so it keeps the short one.
_MAX_ATTEMPTS = 4
_QUOTA_ATTEMPTS = 6
_QUOTA_BASE_DELAY = 2.0


def _status_of(error) -> Optional[int]:
    resp = getattr(error, "resp", None)
    status = getattr(resp, "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _reasons_of(error) -> frozenset:
    """The per-error `reason` codes Google attaches, or empty if there are none.

    googleapiclient parses these off the response body into `error_details`.
    Absent or malformed, the answer is the empty set, which reads as "not a
    rate limit" - the safe way round for an ambiguous 403, because surfacing a
    real permission problem beats silently waiting a minute on one.
    """
    details = getattr(error, "error_details", None)
    if not isinstance(details, list):
        return frozenset()
    return frozenset(d.get("reason") for d in details
                     if isinstance(d, dict) and d.get("reason"))


def _is_rate_limit(error, status) -> bool:
    """Is this Gmail asking us to slow down, whatever status it used to say so?"""
    if status == 429:
        return True
    return status == 403 and bool(_reasons_of(error) & _RATE_LIMIT_REASONS)


def _is_transport_error(error) -> bool:
    """Did the connection die, rather than the server answer?

    No HttpError guard is needed, and one was written here before being
    checked: googleapiclient's HttpError derives from Exception directly, not
    from OSError or ssl.SSLError, so it cannot reach this test. The status
    branches above would take it first in any case.
    """
    return isinstance(error, _TRANSPORT_ERRORS)


def _with_backoff(call, *, what: str, reset=None, idempotent: bool = True):
    """Execute a Gmail request, retrying only what is worth retrying.

    Everything else propagates. A silently swallowed HttpError against a real
    mailbox is the worst outcome available here: the run reports success and the
    mail was never touched.

    Waits are jittered. Threads are hydrated _HYDRATE_WORKERS at a time against
    one shared quota, so an un-jittered schedule has all of them fail together,
    sleep the identical interval and collide again on every retry - which is
    how 32 actions went out in 12 seconds and exhausted a per-minute limit.

    The jitter only ever lengthens a wait. Textbook full jitter spreads either
    side of the nominal delay, but half of that range is spent shortening the
    schedule, and shortening is the wrong direction to err when the server has
    just said there were too many requests: it would let the quota schedule
    finish in 31s, well short of the minute the limit is measured over.

    `reset` discards the caller's cached transport, and a retry after a dead
    socket is worthless without it: httplib2 keeps the connection in its own
    pool and writes to the same corpse again. Called only for transport
    errors - a 403 arrived over a healthy connection, and throwing that away
    would buy a fresh TLS handshake on every quota retry for nothing.

    `idempotent=False` refuses to repeat a call after a TRANSPORT error only.
    Such an error is ambiguous: the request may have died on the way out, or
    after Gmail had already acted on it. Every other wrapped call is
    idempotent - labelling or trashing a thread twice lands in the same state
    - so repeating them costs nothing, while repeating drafts.create puts a
    second draft in the owner's mailbox. A rate limit stays retryable even
    then, because a refusal is not ambiguous: nothing was created.
    """
    attempt = 0
    delay = 1.0
    while True:
        attempt += 1
        try:
            return call()
        except Exception as exc:
            status = _status_of(exc)
            if _is_rate_limit(exc, status):
                limit, delay = _QUOTA_ATTEMPTS, max(delay, _QUOTA_BASE_DELAY)
            elif status in _RETRY_STATUSES:
                limit = _MAX_ATTEMPTS
            elif _is_transport_error(exc) and idempotent:
                limit = _MAX_ATTEMPTS
                if reset is not None:
                    reset()
            else:
                raise
            if attempt >= limit:
                raise
            pause = delay * random.uniform(1.0, 1.5)
            # "returned None" would be a lie for a dropped connection: nothing
            # was returned, the socket died. Name the exception instead - it is
            # the only thing that tells the reader which failure this was.
            cause = (f"returned {status} "
                     f"({','.join(sorted(_reasons_of(exc))) or '-'})"
                     if status is not None else
                     f"connection died ({type(exc).__name__})")
            log.warning("gmail %s %s, retry %d/%d in %.1fs",
                        what, cause, attempt, limit - 1, pause)
            time.sleep(pause)
            delay *= 2


def _header(headers: list[dict], name: str) -> str:
    lowered = name.lower()
    for h in headers:
        if (h.get("name") or "").lower() == lowered:
            return h.get("value") or ""
    return ""


def _iso_date(raw: str) -> str:
    """RFC 2822 -> ISO 8601.

    The snapshot stores ISO and recency.py does date arithmetic on this field,
    so both clients must produce the same shape. An unparseable date returns the
    raw string rather than raising: a malformed Date header is the sender's
    fault and must not cost the owner a whole run.
    """
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).isoformat()
    except (TypeError, ValueError):
        return raw


class LiveGmailClient:
    """The real mailbox, behind the same seven-method protocol as the snapshot.

    Takes a BUILT service object rather than credentials, which is what makes it
    testable: every test drives it through a fake of the googleapiclient
    resource chain, with no network and no credentials anywhere.

    Two invariants it shares with SnapshotGmailClient and must never break:
    `query` is answered the same way by both - here, by handing it to Gmail
    verbatim - and `Thread.label_ids` holds display NAMES.
    """

    def __init__(self, service, http_factory=None):
        self._service = service
        self._http_factory = http_factory
        self._local = threading.local()
        self._labels = _LabelMap(service, http=self._http)

    def _http(self):
        """This thread's transport, or None to use the service's own.

        httplib2.Http is NOT thread-safe, and googleapiclient's service object
        holds exactly one. Sharing it across the hydration pool corrupts the
        SSL socket state, and it surfaces as

            ssl.SSLError: [SSL: WRONG_VERSION_NUMBER] wrong version number

        which names nothing whatsoever about threads. Reproduced against the
        real mailbox before this existed: one worker succeeded every time, five
        workers failed every time.

        None is a valid return and is what the tests use - HttpRequest.execute
        falls back to its own http when passed None, so the fakes never need a
        transport at all.
        """
        if self._http_factory is None:
            return None
        http = getattr(self._local, "http", None)
        if http is None:
            http = self._local.http = self._http_factory()
        return http

    def _reset_http(self) -> None:
        """Throw this thread's transport away, so the next call reconnects.

        Cached above for the lifetime of the thread, which is right until the
        socket underneath it dies. Gmail closes an idle keepalive, httplib2
        keeps the dead connection in its own pool, and the next write raises
        SSLEOFError - then the retry writes to the same corpse and raises
        again. Retrying without this is not a fix, it is the same failure
        three more times.
        """
        self._local.http = None

    # --- reads --------------------------------------------------------------

    def _threads_resource(self):
        return self._service.users().threads()

    def list_threads(self, limit: int = 50, query: str = "") -> list[Thread]:
        """Ids from Gmail, then one get per id to hydrate.

        `query` goes to the API verbatim; `matches_query` is never called here.
        Gmail's `q` accepts display names for `label:` - verified against the
        real mailbox with `-label:Education/AI` - so `settings.inbox_query`
        needs no translation. Only `label_ids` on the way back does.
        """
        result = _with_backoff(
            lambda: self._threads_resource().list(
                userId="me", q=query,
                maxResults=limit).execute(http=self._http()),
            what="threads.list", reset=self._reset_http)
        return self.get_threads([t["id"] for t in (result.get("threads") or [])])

    def list_thread_ids(self, query: str = "", max_ids: int = 500) -> list[str]:
        """Ids only, paged, with nothing hydrated.

        list_threads costs one threads.get per id, so it cannot be pointed at
        more mail than you intend to read. Sampling wants the opposite shape:
        see thousands of ids cheaply, choose a few, hydrate only those. A page
        of 500 ids costs 5 quota units; hydrating one thread costs 10. So the
        whole 20 000-thread mailbox is enumerable for less than the cost of
        reading twenty of it.

        max_ids bounds a mailbox that has no natural end. It is a per-stratum
        ceiling for the sampler, not a limit on the answer: a stratum with more
        threads than this is sampled from its newest max_ids, which is a bias
        worth knowing about and the reason the strata are cut by year.
        """
        ids: list[str] = []
        token = None
        while len(ids) < max_ids:
            page = _with_backoff(
                lambda: self._threads_resource().list(
                    userId="me", q=query, pageToken=token,
                    maxResults=min(500, max_ids - len(ids))).execute(
                        http=self._http()),
                what="threads.list", reset=self._reset_http)
            ids.extend(t["id"] for t in (page.get("threads") or []))
            token = page.get("nextPageToken")
            if not token:
                break
        return ids[:max_ids]

    def get_threads(self, thread_ids: list[str],
                    *, workers: int | None = None) -> list[Thread]:
        """Hydrate an explicit list of ids, `workers` at a time.

        _HYDRATE_WORKERS is tuned for a triage run - twenty or so threads,
        fetched once, where five in flight is the difference between a
        responsive digest and a slow one. A caller pulling hundreds in one
        burst is a different problem: five workers there re-consume Gmail's
        per-minute quota the moment it refills, so every retry in _with_backoff
        collides with the four siblings that were also waiting, and six
        attempts run out while the limit is still exhausted. Seen on
        2026-09-04 hydrating a 200-thread sample. Such a caller should pace
        itself and say so here.
        """
        if not thread_ids:
            return []
        # Built once here rather than lazily inside each worker: five threads
        # racing to build it would issue five labels.list calls.
        self._labels._ensure()

        with ThreadPoolExecutor(
                max_workers=workers or _HYDRATE_WORKERS) as pool:
            # pool.map preserves input order. Gmail returns newest first and the
            # digest renders in that order, so ordering is contract rather than
            # an accident of scheduling.
            return list(pool.map(self.get_thread, thread_ids))

    def get_thread(self, thread_id: str) -> Thread:
        raw = _with_backoff(
            lambda: self._threads_resource().get(
                userId="me", id=thread_id,
                format="full").execute(http=self._http()),
            what="threads.get", reset=self._reset_http)
        return self._to_thread(raw)

    def _to_thread(self, raw: dict) -> Thread:
        messages = raw.get("messages") or []
        if not messages:
            return Thread(id=raw.get("id", ""), subject="", sender="", to=[],
                          date="", snippet="", body="", label_ids=[])

        first = messages[0]
        headers = (first.get("payload") or {}).get("headers") or []

        # Labels are per-message in Gmail but per-thread everywhere above this
        # line, so union them: a thread is UNREAD if any message in it is.
        label_ids: list[str] = []
        for msg in messages:
            for lid in msg.get("labelIds") or []:
                name = self._labels.to_name(lid)
                if name not in label_ids:
                    label_ids.append(name)

        to_raw = _header(headers, "To")
        return Thread(
            id=raw.get("id", ""),
            subject=_header(headers, "Subject"),
            sender=_header(headers, "From"),
            to=[a.strip() for a in to_raw.split(",") if a.strip()],
            date=_iso_date(_header(headers, "Date")),
            snippet=first.get("snippet", "") or "",
            body=_extract_body(first.get("payload") or {}),
            label_ids=label_ids,
        )

    # --- writes -------------------------------------------------------------

    def _modify(self, thread_id: str, body: dict) -> dict[str, Any]:
        _with_backoff(
            lambda: self._threads_resource().modify(
                userId="me", id=thread_id,
                body=body).execute(http=self._http()),
            what="threads.modify", reset=self._reset_http)
        return {"thread_id": thread_id, **body}

    def apply_label(self, thread_id: str, label: str) -> dict[str, Any]:
        return self._modify(
            thread_id, {"addLabelIds": [self._labels.to_id(label)]}
        ) | {"label": label}

    def remove_label(self, thread_id: str, label: str) -> dict[str, Any]:
        return self._modify(
            thread_id, {"removeLabelIds": [self._labels.to_id(label)]}
        ) | {"label": label}

    def archive(self, thread_id: str) -> dict[str, Any]:
        # INBOX is a system label whose id IS "INBOX", so this needs no lookup
        # and cannot create anything.
        return self._modify(
            thread_id, {"removeLabelIds": ["INBOX"]}) | {"action": "archive"}

    def trash(self, thread_id: str) -> dict[str, Any]:
        """threads.trash(), not a TRASH label.

        The real endpoint is what untrash() reverses, and what puts the thread
        in Trash with the 30-day recovery window the owner expects. Adding a
        TRASH label by hand is a different operation that does not reverse the
        same way.
        """
        _with_backoff(
            lambda: self._threads_resource().trash(
                userId="me", id=thread_id).execute(http=self._http()),
            what="threads.trash", reset=self._reset_http)
        return {"thread_id": thread_id, "action": "trash"}

    def create_draft(self, thread_id: str, body: str) -> dict[str, Any]:
        """A reply draft attached to the thread.

        threadId alone is what makes Gmail file the draft in the right
        conversation; In-Reply-To and References are set from the LAST message
        so other mail clients thread it too.
        """
        thread = _with_backoff(
            lambda: self._threads_resource().get(
                userId="me", id=thread_id,
                format="full").execute(http=self._http()),
            what="threads.get", reset=self._reset_http)
        messages = thread.get("messages") or []
        headers = ((messages[-1].get("payload") or {}).get("headers") or []
                   if messages else [])
        message_id = _header(headers, "Message-ID")
        to = _header(headers, "From")
        subject = _header(headers, "Subject")
        if subject and not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        lines = [f"To: {to}", f"Subject: {subject}"]
        if message_id:
            lines.append(f"In-Reply-To: {message_id}")
            lines.append(f"References: {message_id}")
        raw = "\r\n".join(lines) + "\r\n\r\n" + body
        encoded = base64.urlsafe_b64encode(raw.encode()).decode()

        created = _with_backoff(
            lambda: self._service.users().drafts().create(
                userId="me",
                body={"message": {"threadId": thread_id, "raw": encoded}}
            ).execute(http=self._http()),
            # Not idempotent: a transport error is ambiguous, and a duplicate
            # draft in the owner's mailbox is worse than one they can ask for
            # again. A rate limit is still retried - a refusal creates nothing.
            what="drafts.create", reset=self._reset_http, idempotent=False)
        return {"thread_id": thread_id, "draft_id": created.get("id"),
                "draft_chars": len(body)}

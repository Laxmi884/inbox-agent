"""Gmail adapters.

Stage A runs entirely against a frozen snapshot so every architecture is compared
on identical input (spec section 6) and no iteration can touch the real mailbox.
A LiveGmailClient implementing the same protocol lands when live writes are in
scope; nothing above this module changes when it does.
"""
from __future__ import annotations

import json
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

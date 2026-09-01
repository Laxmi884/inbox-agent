"""Callback encoding, and the index -> thread_id boundary (spec section 4.3).

A naive callback encodes the thread id: `approve:1a040d7d5d69e611`. It fits
inside Telegram's 64-byte cap and it is still wrong.

Callbacks here carry a POSITION within the review request. The mapping back to a
thread id is read from the review payload persisted in the checkpoint - the
exact thing the human was shown - never from the callback itself. So there is no
callback string that can name a thread outside the reviewed batch, because
thread ids never travel in callback data at all.

That makes the trust boundary structural rather than validated. The graph's
existing "not part of the reviewed batch" guard stays as the second mechanism,
the same belt-and-braces the deny-list uses (config unions ALWAYS_FORBIDDEN in,
and the chokepoint re-ORs it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence

from ..models import Action, ReviewRequest, ReviewResponse

# Telegram hard-caps callback_data at 64 bytes. This is protocol, not policy.
CB_MAX_BYTES = 64

Kind = Literal["approve", "reject", "label", "prev", "next", "approve_all",
               "open", "list", "noop"]

_CODE_TO_KIND: dict[str, Kind] = {
    "a": "approve", "r": "reject", "l": "label",
    "p": "prev", "n": "next", "A": "approve_all",
    # Opening one item from the digest. Navigation, not a verdict - the digest
    # shipped with only "approve all", which made it a read-only screen.
    "o": "open",
    # Back to the digest from a single item.
    "L": "list",
}
_KIND_TO_CODE = {v: k for k, v in _CODE_TO_KIND.items()}

# Bound on what we will parse at all. An index cannot plausibly exceed a
# mailbox page, and refusing absurd input early keeps int() away from a
# 400-digit string.
_MAX_PARSED_INDEX = 10_000


@dataclass(frozen=True)
class Intent:
    kind: Kind
    index: Optional[int] = None
    label_index: Optional[int] = None


def encode(kind: Kind, index: Optional[int] = None,
           label_index: Optional[int] = None) -> str:
    code = _KIND_TO_CODE[kind]
    parts = [code]
    if index is not None:
        parts.append(str(index))
    if label_index is not None:
        parts.append(str(label_index))
    return ":".join(parts)


def decode(data: str) -> Intent:
    """Parse callback data. NEVER raises - hostile input becomes a no-op.

    This runs on bytes that came back from outside the process. Anything it
    cannot make sense of is `noop`, which every caller treats as "do nothing".
    """
    if not isinstance(data, str) or not data:
        return Intent("noop")
    parts = data.split(":")
    kind = _CODE_TO_KIND.get(parts[0])
    if kind is None:
        return Intent("noop")

    if kind in ("prev", "next", "approve_all", "list"):
        return Intent(kind)

    def parse(raw: str) -> Optional[int]:
        # str.isdigit() rejects "-1", "1e5", "", and anything non-ASCII-numeric,
        # so a negative index cannot be constructed here at all.
        if not raw.isdigit() or len(raw) > 5:
            return None
        value = int(raw)
        return value if value <= _MAX_PARSED_INDEX else None

    if kind in ("approve", "reject", "open"):
        if len(parts) != 2:
            return Intent("noop")
        index = parse(parts[1])
        return Intent(kind, index) if index is not None else Intent("noop")

    # label needs both an item index and a category index
    if len(parts) != 3:
        return Intent("noop")
    index, label_index = parse(parts[1]), parse(parts[2])
    if index is None or label_index is None:
        return Intent("noop")
    return Intent("label", index, label_index)


def resolve_thread_id(intent: Intent, request: ReviewRequest) -> Optional[str]:
    """Position -> thread id, against what the human was actually shown.

    Returns None for anything out of range. Note the explicit `< 0` check:
    Python would happily let index -1 select the last item, which would turn a
    malformed callback into an action on a real thread.
    """
    i = intent.index
    if i is None or i < 0 or i >= len(request.items):
        return None
    return request.items[i].thread_id


def to_response(
    request: ReviewRequest,
    intents: Mapping[int, Intent],
    categories: Sequence[str] = (),
) -> ReviewResponse:
    """Build the resume payload. Everything defaults to approve.

    Matching render.respond(): reviewing 50 proposals should not require 50
    decisions, it should require noticing the two that are wrong. If approval is
    expensive people stop reviewing and start rubber-stamping, and the
    human-in-the-loop becomes theatre.
    """
    decisions: dict[str, str] = {i.thread_id: "approve" for i in request.items}
    edits: dict[str, list[Action]] = {}

    for index, intent in intents.items():
        thread_id = resolve_thread_id(Intent(intent.kind, index, intent.label_index),
                                      request)
        if thread_id is None:
            continue  # forged, stale or replayed - contributes nothing

        if intent.kind == "open":
            continue  # navigation carries no verdict
        if intent.kind == "reject":
            decisions[thread_id] = "reject"
        elif intent.kind == "label":
            li = intent.label_index
            if li is None or li < 0 or li >= len(categories):
                continue  # a label we cannot name is not a label we will apply
            decisions[thread_id] = "edit"
            edits[thread_id] = [Action(kind="label", thread_id=thread_id,
                                       params={"label": categories[li]})]
        # "approve" needs no branch: it is already the default.

    return ReviewResponse(decisions=decisions, edits=edits, instructions=[])

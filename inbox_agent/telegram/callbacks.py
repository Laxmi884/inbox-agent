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

# Four hex characters, minted per digest. Positions used to resolve against a
# parked checkpoint, so an index always named the list the human was shown. With
# a queue that outlives runs, positions shift between digests and a tap on
# yesterday's message would land on today's item 3. The id makes the message
# itself identify which list it belongs to.
#
# Ambiguity: a 4-digit decimal index like "1234" is also valid hex, so
# decode("a:1234") reads the index as a digest id and returns noop. This is the
# safe direction — a callback with no digest id is exactly the stale shape we
# want rejected — and every call site passes a digest_id, so real encodes never
# hit this. It bites only above 999 held items.
DIGEST_ID_LEN = 4
_HEX = set("0123456789abcdef")

Kind = Literal["approve", "reject", "label", "prev", "next", "approve_all",
               "open", "list", "done", "approve_attention",
               "trash_all", "trash_all_go",
               # Corrections on work already done. Each maps to exactly one
               # action sequence, so what gets taught is what the button said.
               "keep", "relabel", "teach_trash",
               # How wide to teach it: this sender, or every thread of this
               # category. Asked rather than assumed, because the two are
               # different intentions behind the same tap.
               "scope_narrow", "scope_wide",
               "noop"]

_CODE_TO_KIND: dict[str, Kind] = {
    "a": "approve", "r": "reject", "l": "label",
    "p": "prev", "n": "next", "A": "approve_all",
    # Opening one item from the digest. Navigation, not a verdict - the digest
    # shipped with only "approve all", which made it a read-only screen.
    "o": "open",
    # Back to the digest from a single item.
    "L": "list",
    # Opening the run's audit records, and the attention-tier one-tap approve.
    "D": "done", "T": "approve_attention",
    # Bulk trash, and its confirmation. TWO codes rather than one carrying an
    # "are you sure" flag: a single kind would let a replayed callback skip the
    # confirmation, and the confirmation is the whole safeguard.
    "B": "trash_all", "G": "trash_all_go",
    # Verdicts and their blast radius. Single characters because callback_data
    # is capped at 64 bytes and an index can reach three digits on a backlog.
    "k": "keep", "R": "relabel", "X": "teach_trash",
    "s": "scope_narrow", "S": "scope_wide",
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
    digest_id: str = ""


def encode(kind: Kind, index: Optional[int] = None,
           label_index: Optional[int] = None, *, digest_id: str = "") -> str:
    parts = [_KIND_TO_CODE[kind]]
    if index is not None:
        parts.append(str(index))
    if label_index is not None:
        parts.append(str(label_index))
    if digest_id:
        parts.append(digest_id)
    return ":".join(parts)


def _is_digest_id(raw: str) -> bool:
    """A digest id is exactly DIGEST_ID_LEN lowercase hex characters.

    Length plus alphabet is what keeps it distinguishable from an index: an
    index is short and decimal, so "7f2a" can never be one. The length check
    short-circuits before scanning an attacker-supplied unbounded string
    character-by-character, the same safety as _MAX_PARSED_INDEX.
    """
    return len(raw) == DIGEST_ID_LEN and all(c in _HEX for c in raw)


def _parse(raw: str) -> Optional[int]:
    # str.isdigit() alone is unsafe: Unicode digit characters like ² and ٣ return
    # True but int() raises ValueError on them. isascii() is required to exclude
    # non-ASCII digits. This also guards against negative indices: "-1" and "1e5"
    # both fail both checks.
    if not (raw.isascii() and raw.isdigit()) or len(raw) > 5:
        return None
    value = int(raw)
    return value if value <= _MAX_PARSED_INDEX else None


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
    rest = parts[1:]

    digest_id = ""
    if rest and _is_digest_id(rest[-1]):
        digest_id = rest[-1]
        rest = rest[:-1]

    if kind in ("prev", "next", "approve_all", "list", "done",
                "approve_attention", "trash_all", "trash_all_go"):
        return Intent(kind, digest_id=digest_id) if not rest else Intent("noop")

    if kind in ("approve", "reject", "open", "keep", "teach_trash",
                "scope_narrow", "scope_wide"):
        if len(rest) != 1:
            return Intent("noop")
        index = _parse(rest[0])
        return (Intent(kind, index, digest_id=digest_id)
                if index is not None else Intent("noop"))

    # `label` and `relabel` are the two-index kinds: which item, and which
    # category. relabel also arrives with one index - the first tap, before a
    # category has been chosen - so it is allowed both shapes.
    if kind == "relabel" and len(rest) == 1:
        index = _parse(rest[0])
        return (Intent(kind, index, digest_id=digest_id)
                if index is not None else Intent("noop"))
    if len(rest) != 2:
        return Intent("noop")
    index, label_index = _parse(rest[0]), _parse(rest[1])
    if index is None or label_index is None:
        return Intent("noop")
    return Intent(kind, index, label_index, digest_id=digest_id)


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

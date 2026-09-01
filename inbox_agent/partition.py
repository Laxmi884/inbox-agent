"""The autonomy ladder as code (spec section 2.2).

The Stage A spec grants `label`, `archive` and `draft` "always" authority and
gates only `trash`. The implementation gated everything, which is a deviation
from our own design rather than a conservative reading of it: a gate that fires
on every item is rubber-stamped, and a rubber-stamped gate reviews nothing.

Pure by intent. Whether the agent may act alone is the highest-consequence
judgment in the system, so it is a function over a ReviewItem with no graph, no
store and no network anywhere near it.
"""
from __future__ import annotations

from typing import Literal, Optional

from .models import ReviewItem
from .render import LOW_CONFIDENCE

HoldReason = Literal["trash", "low_confidence", "needs_reply", "security_alert"]

# Held because the thread wants a person, not because the action is risky. The
# proposed action for these is typically a draft or a label - harmless - so they
# are the tier the one-tap approve button covers.
ATTENTION_CATEGORIES = ("needs_reply", "security_alert")


def hold_reason(item: ReviewItem) -> Optional[HoldReason]:
    """Why this item must wait for the owner, or None if the agent may act.

    Order is precedence, first match wins: trash, low_confidence, needs_reply,
    security_alert. Authorisation always outranks attention, so a low-confidence
    needs_reply is held as low_confidence and never becomes one-tap approvable.
    """
    kinds = {action.kind for action in item.proposed}
    if "trash" in kinds:
        # "trash: learned rule, else human approval at interrupt". A rule the
        # owner taught IS the authorisation; without this clause trash can never
        # graduate and that row of the ladder never does anything.
        if not (item.source == "rule" and item.rule_id):
            return "trash"
    if item.confidence < LOW_CONFIDENCE:
        return "low_confidence"
    if item.category in ATTENTION_CATEGORIES:
        return item.category  # type: ignore[return-value]
    return None


def partition(
    items: list[ReviewItem],
) -> tuple[list[ReviewItem], list[tuple[ReviewItem, HoldReason]]]:
    """Split into (act now, wait for the owner), preserving input order."""
    auto: list[ReviewItem] = []
    held: list[tuple[ReviewItem, HoldReason]] = []
    for item in items:
        reason = hold_reason(item)
        if reason is None:
            auto.append(item)
        else:
            held.append((item, reason))
    return auto, held

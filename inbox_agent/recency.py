"""Age-based demotion: a two-year-old `needs_reply` is not a needs-reply.

`Thread.date` has existed since Stage A and took part in no decision anywhere -
it was printed into the prompt and nothing more. That is a real gap when
clearing a backlog: whatever was waiting on the owner two years ago happened
without them, and holding it in the inbox as urgent is actively wrong.

Deterministic, not model-driven. Asking a 12B model to reason about dates is
precisely the sort of thing it does unreliably, and the project's stance
throughout is to keep decidable things out of the model - the same reasoning
that puts the deny-list in code and the prefilter ahead of the classifier.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .models import Action, Decision, Thread

# Categories whose whole point is that the thread STAYS in the inbox. Demotion
# means "stop holding this"; a category that was already leaving has nothing to
# demote.
KEEP_IN_INBOX = frozenset({"needs_reply", "important_fyi"})


def age_days(thread: Thread, now: Optional[datetime] = None) -> Optional[float]:
    """Age in days, or None if the date cannot be read.

    None means "unknown", never "ancient". Silently archiving mail whose date
    header failed to parse would be exactly the kind of invisible failure the
    audit design exists to prevent.
    """
    raw = (thread.date or "").strip()
    if not raw:
        return None
    try:
        # Gmail hands us RFC-3339 with a Z; fromisoformat wants +00:00 before
        # 3.11 and tolerates Z after, so normalise rather than depend on it.
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    now = now or datetime.now(timezone.utc)
    delta = (now - parsed).total_seconds() / 86400.0
    # Clock skew and badly-behaved senders produce future dates. Those are not
    # fresh by a negative amount; they are simply not stale.
    return max(0.0, delta)


def demote_stale(decision: Decision, thread: Thread, stale_after_days: int,
                 now: Optional[datetime] = None) -> Decision:
    """Stop holding a stale thread in the inbox. Returns the decision unchanged
    when it does not apply.

    Three things it deliberately does NOT do:

    - It does not rewrite `category`. The model said `needs_reply` and that was
      a true statement about the mail; changing it would leave the audit trail
      claiming a judgement that was never made. The category stays, the action
      changes, and `reason` records why.
    - It does not touch rule-decided threads. A rule is the owner's explicit
      instruction, which outranks an inference about age - the authority
      ordering from the spec's "how it learns".
    - It never escalates to `trash`. Stale is a reason to stop holding
      something, never a reason to destroy it, and trash is the one reversible
      action whose reversal has a deadline.
    """
    if decision.source == "rule":
        return decision
    if decision.category not in KEEP_IN_INBOX:
        return decision

    age = age_days(thread, now)
    if age is None or age <= stale_after_days:
        return decision

    kinds = [a.kind for a in decision.actions]
    if "archive" in kinds or "trash" in kinds:
        return decision

    # Keep whatever was already proposed except the no-op, and add the archive.
    # A stale thread that was going to be labelled still gets its label; it just
    # also leaves the inbox.
    actions = [a for a in decision.actions if a.kind != "none"]
    actions.append(Action(kind="archive", thread_id=decision.thread_id))

    note = (f"{decision.reason} [archived: {int(age)} days old, "
            f"stale after {stale_after_days}]").strip()
    return decision.model_copy(update={"actions": actions, "reason": note})

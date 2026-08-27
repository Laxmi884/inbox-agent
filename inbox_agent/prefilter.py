"""Deterministic rule application - zero LLM calls (spec section 4.2).

This is what stops a 200-thread inbox from becoming 200 Gemma calls. Anything a
learned rule already covers is decided here, cheaply and with a citable rule id;
only genuinely novel mail reaches the model.
"""
from __future__ import annotations

from .models import Action, Decision, Thread
from .store import PreferenceStore


def prefilter(
    threads: list[Thread], prefs: PreferenceStore
) -> tuple[list[Decision], list[Thread]]:
    """Split a batch into (decided by rule, still needing the model)."""
    decided: list[Decision] = []
    undecided: list[Thread] = []

    for thread in threads:
        matches = prefs.matching(thread)
        if not matches:
            undecided.append(thread)
            continue

        # Most recently created rule wins: the owner's latest word is the current one.
        rule = max(matches, key=lambda r: r.created_at)
        prefs.record_hit(rule.id)
        decided.append(Decision(
            thread_id=thread.id,
            category="rule_match",
            actions=[Action(kind=rule.action, thread_id=thread.id)],
            reason=f"matched {rule.scope} rule {rule.pattern!r} -> {rule.action}",
            confidence=1.0,
            source="rule",
            rule_id=rule.id,
        ))

    return decided, undecided

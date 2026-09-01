"""Deterministic rule application - zero LLM calls (spec section 4.2).

This is what stops a 200-thread inbox from becoming 200 Gemma calls. Anything a
learned rule already covers is decided here, cheaply and with a citable rule id;
only genuinely novel mail reaches the model.
"""
from __future__ import annotations

from .models import Action, Decision, Rule, Thread
from .store import PreferenceStore



class UnbindableRuleError(Exception):
    """A stored rule cannot be turned into an action against a thread.

    Raised here, at bind time, rather than at execution time. The only rules
    that reach it are legacy `label` rules converted by Rule's validator, which
    have no label to apply: `_dispatch` would raise KeyError mid-run, after
    earlier actions in the same run had already gone through the chokepoint.
    Naming the rule tells the owner exactly which one to teach again.
    """


def _bind(rule: Rule, thread: Thread) -> list[Action]:
    """Join a rule's templates to one thread.

    A rule stores templates because an Action requires a thread_id and a rule
    has not met a thread yet. This is the only place the two are joined.
    """
    bound = []
    for template in rule.actions:
        if template.kind in ("label", "unlabel") and not template.params.get("label"):
            raise UnbindableRuleError(
                f"rule {rule.id} ({rule.scope} {rule.pattern!r}) says "
                f"{template.kind!r} but names no label. It predates labelled "
                f"rules; correct one of these threads again to re-teach it.")
        bound.append(Action(kind=template.kind, thread_id=thread.id,
                            params=dict(template.params)))
    return bound


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
            actions=_bind(rule, thread),
            reason=f"matched {rule.scope} rule {rule.pattern!r} -> {rule.summary}",
            confidence=1.0,
            source="rule",
            rule_id=rule.id,
        ))

    return decided, undecided


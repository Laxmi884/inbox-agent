from datetime import datetime, timezone

import pytest

from inbox_agent.models import ActionTemplate, Rule, Thread
from inbox_agent.prefilter import UnbindableRuleError, prefilter
from inbox_agent.store import PreferenceStore, build_store, rule_from_correction


def thread(**kw) -> Thread:
    base = dict(id="t1", subject="Sale 50%", sender="deals@shop.com", to=[],
                date="2026-08-26T10:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    return Thread(**(base | kw))


def test_thread_with_no_rule_is_left_undecided():
    decided, undecided = prefilter([thread()], PreferenceStore(build_store()))
    assert decided == []
    assert len(undecided) == 1


def test_thread_matching_a_rule_is_decided_without_the_model():
    prefs = PreferenceStore(build_store())
    prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "user archived it"))
    decided, undecided = prefilter([thread()], prefs)
    assert len(decided) == 1
    assert undecided == []
    assert decided[0].source == "rule"
    assert decided[0].actions[0].kind == "archive"
    assert decided[0].category == "rule_match"
    assert decided[0].confidence == 1.0


def test_decided_thread_cites_the_rule_that_decided_it():
    prefs = PreferenceStore(build_store())
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "user archived it")
    prefs.add_rule(r)
    decided, _ = prefilter([thread()], prefs)
    assert decided[0].rule_id == r.id


def test_matching_a_rule_increments_its_hit_count():
    prefs = PreferenceStore(build_store())
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    prefs.add_rule(r)
    prefilter([thread()], prefs)
    assert prefs.rules()[0].hit_count == 1


def test_prefilter_splits_a_mixed_batch():
    """This split is what keeps a 200-thread inbox affordable on a local model."""
    prefs = PreferenceStore(build_store())
    prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note"))
    batch = [thread(), thread(id="t2", sender="boss@work.com", subject="Re: budget")]
    decided, undecided = prefilter(batch, prefs)
    assert [d.thread_id for d in decided] == ["t1"]
    assert [t.id for t in undecided] == ["t2"]


def test_trash_rule_produces_a_trash_action():
    prefs = PreferenceStore(build_store())
    prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="trash")], "owner told it to delete these"))
    decided, _ = prefilter([thread()], prefs)
    assert decided[0].actions[0].kind == "trash"


def test_most_recently_created_rule_wins_when_several_match():
    """When multiple rules match one thread, the owner's most recent instruction wins.
    This allows someone to change their mind about a sender."""
    prefs = PreferenceStore(build_store())
    t = thread()

    # Create two rules that both match the same thread but with different created_at
    # Older rule: archive
    older_time = datetime(2026, 8, 24, 10, 0, 0, tzinfo=timezone.utc)
    older_rule = Rule(
        id="rule_old",
        scope="sender",
        pattern="deals@shop.com",
        actions=[ActionTemplate(kind="archive")],
        provenance="user",
        created_at=older_time,
    )

    # Newer rule: label (the owner changed their mind)
    newer_time = datetime(2026, 8, 26, 10, 0, 0, tzinfo=timezone.utc)
    newer_rule = Rule(
        id="rule_new",
        scope="sender",
        pattern="deals@shop.com",
        # Names its label, which is the whole change: a rule that said only
        # "label" bound to params={} and raised KeyError at the chokepoint.
        actions=[ActionTemplate(kind="label", params={"label": "promotion"})],
        provenance="user",
        created_at=newer_time,
    )

    prefs.add_rule(older_rule)
    prefs.add_rule(newer_rule)

    decided, _ = prefilter([t], prefs)
    # Should match the newer rule with action "label"
    assert len(decided) == 1
    assert decided[0].actions[0].kind == "label"
    assert decided[0].actions[0].params == {"label": "promotion"}
    assert decided[0].rule_id == newer_rule.id
    assert decided[0].source == "rule"
    assert "deals@shop.com" in decided[0].reason


# --- binding a rule to a thread ---------------------------------------------
# A rule is a template. Action requires a thread_id, so the two are joined here,
# and this is where a rule that cannot be joined has to be refused.

def _rule(actions, rid="r-1", scope="sender", pattern="deals@shop.com"):
    return Rule(id=rid, scope=scope, pattern=pattern, actions=actions,
                provenance="p", created_at=datetime(2026, 9, 1, tzinfo=timezone.utc))


def test_a_label_rule_applies_the_label_it_names():
    """The regression. Before this, a label rule produced params={} and raised
    KeyError inside _dispatch - reported as "simulated" under dry-run, so a live
    mailbox was the only place it could ever be noticed."""
    s = PreferenceStore(build_store())
    s.add_rule(_rule([ActionTemplate(kind="label", params={"label": "promotion"})]))
    decided, _ = prefilter([thread()], s)
    action = decided[0].actions[0]
    assert (action.kind, action.thread_id) == ("label", "t1")
    assert action.params == {"label": "promotion"}


def test_a_rule_binds_every_action_in_its_sequence():
    s = PreferenceStore(build_store())
    s.add_rule(_rule([ActionTemplate(kind="label", params={"label": "recruiter"}),
                      ActionTemplate(kind="archive")]))
    decided, _ = prefilter([thread()], s)
    assert [(a.kind, a.thread_id) for a in decided[0].actions] == [
        ("label", "t1"), ("archive", "t1")]


def test_a_legacy_label_rule_is_refused_by_name_not_by_keyerror():
    """A converted legacy rule cannot say which label. Executing it raises
    KeyError partway through a run - which the bot reports as "the run did not
    finish" - so it is refused here, naming the rule to re-teach."""
    s = PreferenceStore(build_store())
    s.add_rule(Rule.model_validate(
        {"id": "r-old", "scope": "sender", "pattern": "deals@shop.com",
         "action": "label", "provenance": "p",
         "created_at": datetime(2026, 9, 1, tzinfo=timezone.utc)}))
    with pytest.raises(UnbindableRuleError) as exc:
        prefilter([thread()], s)
    assert "r-old" in str(exc.value)


def test_the_reason_names_what_the_rule_does():
    s = PreferenceStore(build_store())
    s.add_rule(_rule([ActionTemplate(kind="label", params={"label": "recruiter"}),
                      ActionTemplate(kind="archive")]))
    decided, _ = prefilter([thread()], s)
    assert "label(recruiter), archive" in decided[0].reason

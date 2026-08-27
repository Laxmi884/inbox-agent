from inbox_agent.models import Thread
from inbox_agent.prefilter import prefilter
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
    prefs.add_rule(rule_from_correction(thread(), "archive", "user archived it"))
    decided, undecided = prefilter([thread()], prefs)
    assert len(decided) == 1
    assert undecided == []
    assert decided[0].source == "rule"
    assert decided[0].actions[0].kind == "archive"


def test_decided_thread_cites_the_rule_that_decided_it():
    prefs = PreferenceStore(build_store())
    r = rule_from_correction(thread(), "archive", "user archived it")
    prefs.add_rule(r)
    decided, _ = prefilter([thread()], prefs)
    assert decided[0].rule_id == r.id


def test_matching_a_rule_increments_its_hit_count():
    prefs = PreferenceStore(build_store())
    r = rule_from_correction(thread(), "archive", "note")
    prefs.add_rule(r)
    prefilter([thread()], prefs)
    assert prefs.rules()[0].hit_count == 1


def test_prefilter_splits_a_mixed_batch():
    """This split is what keeps a 200-thread inbox affordable on a local model."""
    prefs = PreferenceStore(build_store())
    prefs.add_rule(rule_from_correction(thread(), "archive", "note"))
    batch = [thread(), thread(id="t2", sender="boss@work.com", subject="Re: budget")]
    decided, undecided = prefilter(batch, prefs)
    assert [d.thread_id for d in decided] == ["t1"]
    assert [t.id for t in undecided] == ["t2"]


def test_trash_rule_produces_a_trash_action():
    prefs = PreferenceStore(build_store())
    prefs.add_rule(rule_from_correction(thread(), "trash", "owner told it to delete these"))
    decided, _ = prefilter([thread()], prefs)
    assert decided[0].actions[0].kind == "trash"

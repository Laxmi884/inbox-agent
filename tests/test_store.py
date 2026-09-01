# tests/test_store.py
from datetime import datetime, timezone

from inbox_agent.models import ActionTemplate, Thread
from inbox_agent.store import PreferenceStore, build_store, rule_from_correction


def thread(**kw) -> Thread:
    base = dict(id="t1", subject="Sale 50%", sender="deals@shop.com", to=[],
                date="2026-08-26T10:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    return Thread(**(base | kw))


def store() -> PreferenceStore:
    return PreferenceStore(build_store())  # no embeddings: exact matching only


def test_added_rule_is_retrievable():
    s = store()
    s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "user archived it"))
    assert len(s.rules()) == 1


def test_rule_matches_thread_by_sender():
    s = store()
    s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "user archived it"))
    assert len(s.matching(thread())) == 1


def test_rule_does_not_match_a_different_sender():
    s = store()
    s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note"))
    assert s.matching(thread(sender="boss@work.com")) == []


def test_provenance_survives_the_round_trip():
    s = store()
    s.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "rejected proposal on t1"))
    assert "rejected proposal on t1" in s.rules()[0].provenance


def test_record_hit_increments_the_counter():
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.record_hit(r.id)
    s.record_hit(r.id)
    assert s.rules()[0].hit_count == 2


def test_mark_overridden_flags_the_rule_without_deleting_it():
    """An overridden rule is evidence. It stays visible in the audit trail."""
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.mark_overridden(r.id)
    assert s.rules()[0].overridden is True
    assert len(s.rules()) == 1


def test_overridden_rules_stop_matching():
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.mark_overridden(r.id)
    assert s.matching(thread()) == []


def test_delete_rule_removes_it():
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.delete_rule(r.id)
    assert s.rules() == []


def test_as_table_is_human_readable():
    """The owner must always be able to read what the agent thinks it knows."""
    s = store()
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "note")
    s.add_rule(r)
    s.record_hit(r.id)
    row = s.as_table()[0]
    for col in ("id", "scope", "pattern", "action", "hit_count", "provenance"):
        assert col in row
    assert row["id"] == r.id
    assert row["scope"] == "sender"
    assert row["pattern"] == "deals@shop.com"
    assert row["action"] == "archive"
    assert row["hit_count"] == 1
    assert row["provenance"] == "note"
    assert row["overridden"] is False
    assert row.get("created_at")


def test_rules_returns_all_rules_beyond_the_default_search_limit():
    """BaseStore.search() defaults to limit=10. rules() must paginate past that
    default or the rule set silently truncates once it grows past 10 - exactly
    the silent-wrong-answer failure the audit design exists to prevent."""
    s = store()
    for i in range(25):
        s.add_rule(rule_from_correction(thread(sender=f"sender{i}@shop.com"), [ActionTemplate(kind="archive")], f"note {i}"))
    assert len(s.rules()) == 25

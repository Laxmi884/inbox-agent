"""The record of what a run did, kept past the run after it."""
from datetime import datetime, timedelta, timezone

from inbox_agent.models import Action, DoneRecord, ReviewItem, RunReport
from inbox_agent.store import MAX_REPORTS, DoneStore, build_store

T0 = datetime(2026, 9, 8, 8, 0, tzinfo=timezone.utc)


def item(tid="t1"):
    return ReviewItem(
        thread_id=tid, category="promotion", subject=f"Subject {tid}",
        sender="deals@shop.com", snippet="s",
        proposed=[Action(kind="archive", thread_id=tid)],
        reason="a sale", confidence=0.9, source="model")


def report(run_id="r1", at=T0, threads=("t1",)):
    return RunReport(
        run_id=run_id, ran_at=at, total=len(threads), remaining=3,
        done=[DoneRecord(thread_id=t, item=item(t),
                         actions=[("archive", None), ("label", "promo")],
                         rule_id="r-abc")
              for t in threads])


def store():
    return DoneStore(build_store())


def test_a_report_round_trips_with_every_field():
    s = store()
    s.record(report())
    got = s.get("r1")
    assert got is not None
    assert got.run_id == "r1"
    assert got.ran_at == T0
    assert got.total == 1 and got.remaining == 3
    row = got.done[0]
    assert row.thread_id == "t1"
    # The whole ReviewItem, not a flattened subject: category and reason are
    # what the correction path reads off it.
    assert row.item.subject == "Subject t1"
    assert row.item.category == "promotion"
    assert row.item.reason == "a sale"
    assert row.actions == [("archive", None), ("label", "promo")]
    assert row.rule_id == "r-abc"


def test_get_returns_none_for_an_unknown_run():
    assert store().get("nope") is None


def test_recent_is_newest_first():
    s = store()
    s.record(report("old", T0))
    s.record(report("new", T0 + timedelta(hours=4)))
    assert [r.run_id for r in s.recent()] == ["new", "old"]


def test_an_eleventh_report_prunes_the_oldest():
    s = store()
    for i in range(MAX_REPORTS + 1):
        s.record(report(f"r{i}", T0 + timedelta(minutes=i)))
    kept = [r.run_id for r in s.recent()]
    assert len(kept) == MAX_REPORTS
    assert "r0" not in kept, "the oldest report survived the prune"
    assert s.get("r0") is None
    assert kept[0] == f"r{MAX_REPORTS}"


def test_reading_never_prunes():
    """Pruning on read would make /done mutate the record it is showing."""
    s = store()
    for i in range(MAX_REPORTS + 1):
        s.record(report(f"r{i}", T0 + timedelta(minutes=i)))
    s._store.put(("done", "reports"), "extra",
                 {"report": report("extra", T0).model_dump(mode="json")})
    assert len(s.recent(limit=99)) == MAX_REPORTS + 1
    assert s.get("extra") is not None


def test_a_run_that_did_nothing_still_records_a_report():
    """"It ran and did nothing" must be distinguishable from "it never ran"."""
    s = store()
    s.record(RunReport(run_id="quiet", ran_at=T0, total=0, remaining=0, done=[]))
    got = s.get("quiet")
    assert got is not None and got.done == []


def test_a_naive_timestamp_is_read_back_as_utc():
    """Reports are sorted against each other; naive vs aware raises."""
    s = store()
    s.record(RunReport(run_id="naive", ran_at=datetime(2026, 9, 8, 8, 0)))
    s.record(report("aware", T0 + timedelta(hours=1)))
    assert [r.run_id for r in s.recent()] == ["aware", "naive"]

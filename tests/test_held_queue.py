"""The queue that makes held items outlive the run that produced them."""
from datetime import datetime, timedelta, timezone

from inbox_agent.models import Action, HeldItem, ReviewItem
from inbox_agent.store import HeldQueue, build_store

T0 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def item(tid="t1", conf=0.9):
    return ReviewItem(
        thread_id=tid, category="promotion", subject=f"Subject {tid}",
        sender="a@b.com", snippet="s",
        proposed=[Action(kind="trash", thread_id=tid)],
        reason="because", confidence=conf, source="model")


def queue():
    return HeldQueue(build_store())


def test_add_then_get_round_trips():
    q = queue()
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    got = q.get("t1")
    assert got is not None
    assert got.thread_id == "t1"
    assert got.hold_reason == "trash"
    assert got.run_id == "r1"
    assert got.first_held_at == T0
    assert got.item.subject == "Subject t1"


def test_get_returns_none_for_an_unknown_thread():
    assert queue().get("nope") is None


def test_all_is_ordered_oldest_first():
    q = queue()
    q.add(item("t2"), run_id="r2", reason="trash", now=T0 + timedelta(hours=10))
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    assert [h.thread_id for h in q.all()] == ["t1", "t2"]


def test_re_adding_a_held_thread_keeps_the_ORIGINAL_first_held_at():
    """This is what makes 'waiting since Tue 8:00' true.

    An item held at 8am and still held at 6pm is one item that has been waiting
    ten hours, not a fresh one. Overwriting first_held_at would reset the age on
    every run and the queue would never look old, which is the entire signal.
    """
    q = queue()
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    q.add(item("t1"), run_id="r2", reason="trash", now=T0 + timedelta(hours=10))
    assert q.get("t1").first_held_at == T0
    assert len(q.all()) == 1


def test_re_adding_refreshes_the_item_and_the_reason():
    """The age is sticky; the content is not. A re-classified thread should
    show its current proposal and current reason for being held."""
    q = queue()
    q.add(item("t1", conf=0.9), run_id="r1", reason="trash", now=T0)
    q.add(item("t1", conf=0.2), run_id="r2", reason="low_confidence",
          now=T0 + timedelta(hours=10))
    got = q.get("t1")
    assert got.hold_reason == "low_confidence"
    assert got.item.confidence == 0.2
    assert got.run_id == "r2"


def test_remove_takes_it_out_of_the_queue():
    q = queue()
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    q.remove("t1")
    assert q.get("t1") is None
    assert q.all() == []


def test_removing_an_absent_thread_is_a_no_op():
    queue().remove("never-held")


def test_all_reads_past_the_default_search_page():
    """BaseStore.search() defaults to limit=10. The same truncation trap that
    rules() had to paginate around applies here."""
    q = queue()
    for i in range(25):
        q.add(item(f"t{i:02d}"), run_id="r1", reason="trash",
              now=T0 + timedelta(minutes=i))
    assert len(q.all()) == 25


def test_held_item_is_json_serialisable():
    """It crosses the same boundary ReviewRequest does."""
    q = queue()
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    dumped = q.get("t1").model_dump(mode="json")
    assert HeldItem.model_validate(dumped).thread_id == "t1"

# tests/test_persistence.py
"""What has to survive the process, and why it now has to.

The queue was in-memory when it was only ever read by the run that filled it.
`mark_triaged` changed that: every processed thread leaves the fetch query, held
ones included, and `fetch` uses `settings.inbox_query` in both modes - so a
thread lost from the queue is not re-fetched by `/triage`, and not by `/backlog`
either. Losing the queue therefore stopped being "the digest is short this
morning" and became "fifteen threads are unreachable except by hand in Gmail".

These tests are about the restart, so every one of them closes the store and
opens a second one over the same file rather than reusing the object.
"""
from datetime import datetime, timezone

import pytest

from inbox_agent.models import Action, ReviewItem, Thread
from inbox_agent.store import (
    HeldQueue, PreferenceStore, open_store, rule_from_correction,
)


def thread(**kw) -> Thread:
    base = dict(id="t1", subject="Sale 50%", sender="deals@shop.com", to=[],
                date="2026-08-26T10:00:00Z", snippet="s", body="b",
                label_ids=["INBOX"])
    return Thread(**(base | kw))


def review_item(thread_id="t1") -> ReviewItem:
    return ReviewItem(
        thread_id=thread_id, subject="Sale 50%", sender="deals@shop.com",
        snippet="s", proposed=[Action(kind="trash", thread_id=thread_id)],
        reason="Marketing mail, no order reference.", confidence=0.92,
        source="model")


@pytest.fixture
def db(tmp_path):
    """The path both halves of each test open. No embeddings: the index is
    orthogonal to persistence and would put Ollama on the critical path of the
    suite."""
    return tmp_path / "store" / "s.sqlite"


def test_a_held_item_is_still_there_after_a_restart(db):
    store = open_store(db)
    HeldQueue(store).add(review_item(), run_id="r1", reason="trash")
    store.conn.close()

    survived = HeldQueue(open_store(db)).all()
    assert [h.thread_id for h in survived] == ["t1"]
    assert survived[0].hold_reason == "trash"
    assert survived[0].item.subject == "Sale 50%"


def test_the_wait_is_measured_from_the_first_hold_not_from_the_restart(db):
    """`first_held_at` is what makes an ignored queue read as ignored. Restarting
    the bot must not reset the ageing and quietly make everything look fresh."""
    store = open_store(db)
    held = HeldQueue(store).add(review_item(), run_id="r1", reason="trash")
    store.conn.close()

    assert HeldQueue(open_store(db)).all()[0].first_held_at == held.first_held_at


def test_removing_an_item_is_what_persists_not_just_adding(db):
    """The queue drains as well as fills. A remove that only reached memory
    would resurrect the item on the next start, and the owner would be asked to
    authorise the same trash twice."""
    store = open_store(db)
    queue = HeldQueue(store)
    queue.add(review_item("t1"), run_id="r1", reason="trash")
    queue.add(review_item("t2"), run_id="r1", reason="trash")
    queue.remove("t1")
    store.conn.close()

    assert [h.thread_id for h in HeldQueue(open_store(db)).all()] == ["t2"]


def test_learned_rules_and_instructions_survive_a_restart(db):
    """Rules are the whole point of the learning loop: every correction the
    owner makes is spent teaching a store that used to be thrown away when the
    process ended."""
    store = open_store(db)
    prefs = PreferenceStore(store)
    prefs.add_rule(rule_from_correction(thread(), "archive", "user archived it"))
    prefs.add_instruction("never trash mail from my accountant")
    store.conn.close()

    reopened = PreferenceStore(open_store(db))
    assert len(reopened.rules()) == 1
    assert reopened.instructions() == ["never trash mail from my accountant"]
    assert [r.pattern for r in reopened.matching(thread())] == ["deals@shop.com"]


def test_a_rules_hit_count_survives_a_restart(db):
    """Precision is hits against overrides. If either counter resets on restart,
    the auto-demotion below 0.5 can never fire on a bot that is restarted more
    often than a rule fires four times."""
    store = open_store(db)
    prefs = PreferenceStore(store)
    rule = prefs.add_rule(rule_from_correction(thread(), "archive", "because"))
    for _ in range(3):
        prefs.record_hit(rule.id)
    prefs.record_override(rule.id)
    store.conn.close()

    survived = PreferenceStore(open_store(db)).rules()[0]
    assert survived.hit_count == 3
    assert survived.override_count == 1


def test_the_directory_is_created_rather_than_demanded(db):
    """First run on a new machine. Refusing to start because a directory the
    agent owns does not exist yet is a failure mode with no upside."""
    assert not db.parent.exists()
    open_store(db).conn.close()
    assert db.exists()

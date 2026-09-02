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

from inbox_agent.models import ActionTemplate, Action, ReviewItem, Thread
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
    prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "user archived it"))
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
    rule = prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "because"))
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


# --- a run's verdict supersedes what it held before ---------------------------
# Found on a live phone. A LangChain workshop was held as promotion -> trash.
# The policy was corrected, the SAME thread was re-triaged, the model returned
# learning -> label, and it was auto-executed. The stale trash entry stayed in
# the queue: the digest then showed one thread twice with contradictory
# verdicts, and approving the stale one would have trashed a thread the agent
# had just filed as learning material - executing a proposal the agent itself
# had superseded.
#
# The queue outliving runs is the design (Plan 1). Nothing reconciling it
# against a newer verdict was the bug.

def test_a_rerun_that_no_longer_holds_a_thread_drops_the_stale_entry(tmp_path):
    from inbox_agent.models import Action, ReviewItem
    from inbox_agent.store import HeldQueue, open_store

    q = HeldQueue(open_store(tmp_path / "held.sqlite"))
    stale = ReviewItem(thread_id="t1", category="promotion", subject="Workshop",
                       sender="hello@mail.langchain.com", snippet="s",
                       proposed=[Action(kind="trash", thread_id="t1")],
                       reason="promotional", confidence=1.0, source="model")
    q.add(stale, run_id="r1", reason="trash")
    assert len(q.all()) == 1

    # Run 2 decides the same thread needs no holding at all.
    q.remove("t1")
    assert q.all() == []


def test_removing_a_thread_that_was_never_held_is_not_an_error(tmp_path):
    """The reconciliation runs over every auto-executed thread, and most of
    them were never in the queue."""
    from inbox_agent.store import HeldQueue, open_store

    q = HeldQueue(open_store(tmp_path / "held.sqlite"))
    q.remove("never-seen")          # must not raise
    assert q.all() == []


def test_a_thread_this_run_did_not_touch_stays_held(tmp_path):
    """The carry-over the persistent queue exists for. Only threads the run
    actually processed may have their held entry superseded."""
    from inbox_agent.models import Action, ReviewItem
    from inbox_agent.store import HeldQueue, open_store

    q = HeldQueue(open_store(tmp_path / "held.sqlite"))
    for tid in ("t1", "t2"):
        q.add(ReviewItem(thread_id=tid, category="promotion", subject=tid,
                         sender=f"{tid}@x.com", snippet="s",
                         proposed=[Action(kind="trash", thread_id=tid)],
                         reason="r", confidence=1.0, source="model"),
              run_id="r1", reason="trash")
    q.remove("t1")                  # only t1 was re-processed
    assert [h.thread_id for h in q.all()] == ["t2"]

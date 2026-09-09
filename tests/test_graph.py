import json
from datetime import datetime, timezone

import pytest

from langgraph.types import Command

from inbox_agent.audit import AuditLog
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.classify import ThreadJudgment
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.graph import build_graph, learn_from_response
from inbox_agent.models import (
    Action, ActionTemplate, Decision, ReviewResponse, Rule, Thread,
)
from inbox_agent.policy import Policy
from inbox_agent.store import DoneStore, HeldQueue, PreferenceStore, build_store


class FakeLLM:
    def __init__(self, judgment=None):
        self._j = judgment or ThreadJudgment(category="promotion", action="archive",
                                             label=None, reason="a sale", confidence=0.9)
    def with_structured_output(self, schema): return self
    def invoke(self, messages): return self._j


@pytest.fixture
def snapshot_file(tmp_path):
    data = [{"id": "t1", "subject": "Sale", "sender": "deals@shop.com", "to": [],
             "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "b",
             "label_ids": ["INBOX", "UNREAD"]}]
    p = tmp_path / "threads.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def wiring(tmp_path, snapshot_file):
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    return dict(
        client=SnapshotGmailClient(snapshot_file),
        prefs=PreferenceStore(build_store()),
        policy=Policy(text="TEST", version="local:test", source="local"),
        llm=FakeLLM(), settings=settings, log=AuditLog(settings.audit_log),
        held=HeldQueue(build_store()), done=DoneStore(build_store()),
    )


def test_graph_suspends_at_the_review_interrupt(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-1"}}
    result = graph.invoke({"limit": 10, "mode": "backlog"}, config)
    assert "__interrupt__" in result


@pytest.fixture
def three_threads(tmp_path):
    data = [{"id": f"r{i}", "subject": f"Sale {i}", "sender": "deals@shop.com",
             "to": [], "date": "2026-08-26T10:00:00Z", "snippet": "s",
             "body": "b", "label_ids": ["INBOX", "UNREAD"]} for i in range(3)]
    p = tmp_path / "three.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def wiring_of_three(wiring, three_threads):
    return {**wiring, "client": SnapshotGmailClient(three_threads)}


def test_fetch_reports_how_many_the_limit_left_behind(wiring_of_three):
    """The probe is ids-only: it answers 'was the cap the reason this run
    stopped?' without paying to read the answer."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring_of_three, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 2, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-remaining"}})
    assert len(result["thread_ids"]) == 2
    assert result["remaining"] == 1


def test_fetch_reports_no_remainder_when_the_limit_was_not_reached(
        wiring_of_three):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring_of_three, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 50, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-no-remaining"}})
    assert result["remaining"] == 0


@pytest.fixture
def five_threads(tmp_path):
    data = [{"id": f"r{i}", "subject": f"Sale {i}", "sender": "deals@shop.com",
             "to": [], "date": "2026-08-26T10:00:00Z", "snippet": "s",
             "body": "b", "label_ids": ["INBOX", "UNREAD"]} for i in range(5)]
    p = tmp_path / "five.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def wiring_of_five(wiring, five_threads):
    return {**wiring, "client": SnapshotGmailClient(five_threads)}


def test_fetch_reports_the_real_remainder_not_just_whether_the_cap_bound(
        wiring_of_five):
    """5 matching threads, limit=2: under the old `max_ids=limit + 1` probe
    this would have asked for 3 ids and reported remaining=1 no matter how
    far behind the owner actually was - a boolean typed as an int. Probing a
    full page (max_ids=500) instead sees all 5 and reports the true
    remainder, 3. 5-and-2 rather than 3-and-2 on purpose: with only 3 matching
    threads the old bug and the fix agree (both say 1), so this needs a
    corpus wider than limit + 1 to tell them apart."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring_of_five, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 2, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-real-remaining"}})
    assert len(result["thread_ids"]) == 2
    assert result["remaining"] == 3


def test_interrupt_payload_is_json_serialisable(wiring):
    """It must survive the trip to a Telegram renderer unchanged."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-2"}}
    result = graph.invoke({"limit": 10, "mode": "backlog"}, config)
    payload = result["__interrupt__"][0].value
    json.dumps(payload)  # must not raise
    assert payload["items"][0]["thread_id"] == "t1"
    assert payload["policy_version"] == "local:test"


def test_approving_executes_the_action(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-3"}}
    graph.invoke({"limit": 10, "mode": "backlog"}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"t1": "approve"}, "edits": {}, "instructions": []}),
        config)
    # FakeLLM judges promotion/archive, and a filed promotion also gets the
    # UNREAD marker (MARK_READ_ON_ARCHIVE), so assert on the action this test
    # is about rather than on how many rode along with it.
    executed = [r["action"] for r in final["executed"]]
    assert "archive" in executed, executed
    assert wiring["log"].records()[0].action == "archive"


def test_executed_action_carries_a_traceable_checkpoint_identifier(wiring):
    """An AuditRecord must be joinable back to the run that produced it
    (spec's audit-worthiness pillar). langgraph 1.2.11 does not populate
    config["configurable"]["checkpoint_id"] inside a node on either the
    initial or the resumed invocation (verified at runtime), so thread_id -
    which IS always present - is what gets recorded in that field."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-checkpoint"}}
    graph.invoke({"limit": 10, "mode": "backlog"}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"t1": "approve"}, "edits": {}, "instructions": []}),
        config)
    # FakeLLM judges promotion/archive, and a filed promotion also gets the
    # UNREAD marker (MARK_READ_ON_ARCHIVE), so assert on the action this test
    # is about rather than on how many rode along with it.
    executed = [r["action"] for r in final["executed"]]
    assert "archive" in executed, executed
    record = wiring["log"].records()[0]
    assert record.checkpoint_id is not None
    assert record.checkpoint_id == "run-checkpoint"


def test_rejecting_executes_nothing(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-4"}}
    graph.invoke({"limit": 10, "mode": "backlog"}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"t1": "reject"}, "edits": {}, "instructions": []}),
        config)
    assert final["executed"] == []
    # mark_triaged still runs on every processed thread (including a rejected
    # one) and writes its own "simulated" label record, so the log is no
    # longer empty - but nothing beyond that bookkeeping may have happened,
    # which `records() == []` used to guarantee on its own.
    assert all(r.action == "label" for r in wiring["log"].records())


def test_rejection_becomes_a_learned_rule(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-5"}}
    graph.invoke({"limit": 10, "mode": "backlog"}, config)
    graph.invoke(
        Command(resume={"decisions": {"t1": "reject"},
                        "edits": {"t1": [{"kind": "label", "thread_id": "t1",
                                          "params": {"label": "Keep"}}]},
                        "instructions": []}),
        config)
    rules = wiring["prefs"].rules()
    assert len(rules) == 1
    assert [a.kind for a in rules[0].actions] == ["label"]
    assert "t1" in rules[0].provenance


def test_learn_from_response_records_provenance():
    prefs = PreferenceStore(build_store())
    threads = [Thread(id="t1", subject="Sale", sender="deals@shop.com", to=[],
                      date="2026-08-26T10:00:00Z", snippet="s", body="b",
                      label_ids=["INBOX"])]
    learn_from_response(
        ReviewResponse(decisions={"t1": "edit"},
                       edits={"t1": [Action(kind="archive", thread_id="t1")]},
                       instructions=[]),
        threads, prefs)
    assert [a.kind for a in prefs.rules()[0].actions] == ["archive"]
    assert "corrected" in prefs.rules()[0].provenance


def test_rule_matched_threads_skip_the_model(wiring):
    """A thread the prefilter decides must never be sent to the LLM."""
    from langgraph.checkpoint.memory import InMemorySaver
    from inbox_agent.store import rule_from_correction

    thread = wiring["client"].get_thread("t1")
    wiring["prefs"].add_rule(rule_from_correction(thread, [ActionTemplate(kind="trash")], "owner said delete"))

    class ExplodingLLM:
        def with_structured_output(self, schema): return self
        def invoke(self, messages): raise AssertionError("model must not be called")

    graph = build_graph(**{**wiring, "llm": ExplodingLLM()}, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10, "mode": "backlog"},
                          {"configurable": {"thread_id": "run-6"}})
    assert result["__interrupt__"][0].value["items"][0]["source"] == "rule"


def test_approve_unknown_thread_id_does_not_crash(wiring):
    """A ghost thread id on the approve path must be skipped, not KeyError'd."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-7"}}
    graph.invoke({"limit": 10, "mode": "backlog"}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"ghost-thread": "approve"},
                        "edits": {}, "instructions": []}),
        config)
    assert final["executed"] == []
    assert any(s["thread_id"] == "ghost-thread" for s in final["skipped"])
    # ghost-thread was never in the fetched batch, so mark_triaged - which only
    # walks state["thread_ids"] - never touches it either, unlike t1, which
    # legitimately picks up a triaged-label record.
    assert not any(r.thread_id == "ghost-thread" for r in wiring["log"].records())


def test_edit_unknown_thread_id_executes_nothing(wiring):
    """A forged/never-proposed thread id on the edit path must not fail open."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-8"}}
    graph.invoke({"limit": 10, "mode": "backlog"}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"ghost-thread-2": "edit"},
                        "edits": {"ghost-thread-2": [
                            {"kind": "trash", "thread_id": "ghost-thread-2"}]},
                        "instructions": []}),
        config)
    assert final["executed"] == []
    assert any(s["thread_id"] == "ghost-thread-2" for s in final["skipped"])
    # Same reasoning as the ghost-thread case above: mark_triaged only walks
    # the real fetched batch, so a forged id gets no record of any kind.
    assert not any(r.thread_id == "ghost-thread-2" for r in wiring["log"].records())


def test_forbidden_action_is_visible_in_state(wiring):
    """A refusal must surface in TriageState, not just in print()/the JSONL file.

    Uses a caller-configured forbidden kind ("archive") rather than an
    ALWAYS_FORBIDDEN one ("send_message"/"delete_forever"): those aren't valid
    Rule action literals, and routing one through the edit path would also hit
    learn_from_response's rule-creation step - a separate, pre-existing gap
    (rule_from_correction assumes edits[0].kind is a valid ActionKind) that is
    out of scope for this fix round and is called out in the report instead.
    """
    from dataclasses import replace
    from langgraph.checkpoint.memory import InMemorySaver

    settings = replace(wiring["settings"],
                        forbidden_actions=wiring["settings"].forbidden_actions
                        | frozenset({"archive"}))
    graph = build_graph(**{**wiring, "settings": settings}, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-9"}}
    graph.invoke({"limit": 10, "mode": "backlog"}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"t1": "edit"},
                        "edits": {"t1": [{"kind": "archive", "thread_id": "t1"}]},
                        "instructions": []}),
        config)
    assert final["executed"] == []
    assert len(final["refused"]) == 1
    assert final["refused"][0]["thread_id"] == "t1"
    assert final["refused"][0]["kind"] == "archive"


def test_edit_naming_forbidden_kind_is_refused_and_not_learned(wiring):
    """A human edit naming a forbidden kind (send_message) is reachable because
    Action.kind is str, not the Literal. It must be refused by execute (not
    performed), must not crash learn (the graph's final node), and must not
    become a rule - there is nothing to learn from an action the system will
    never perform under any circumstances."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-10"}}
    graph.invoke({"limit": 10, "mode": "backlog"}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"t1": "edit"},
                        "edits": {"t1": [{"kind": "send_message", "thread_id": "t1"}]},
                        "instructions": []}),
        config)
    assert final["executed"] == []
    assert any(r["thread_id"] == "t1" for r in final["refused"])
    assert wiring["prefs"].rules() == []
    assert any(s["thread_id"] == "t1" and s.get("stage") == "learn"
               for s in final["skipped"])


# --- bounded state ----------------------------------------------------------
# LangGraph persists the FULL state after every node, so anything held in state
# is duplicated once per checkpoint - measured at 11-16 checkpoints for a single
# run. The snapshot's bodies are empty (max 0 chars), which hid this: with real
# Gmail bodies (live threads measured up to 204 KB) a 50-thread run would write
# the same bodies ~11 times over. State holds identifiers; the client is the
# source of truth and is already injected.


def test_state_does_not_carry_email_bodies(wiring):
    """Bodies must not be duplicated into every checkpoint."""
    graph = build_graph(**wiring)
    cfg = {"configurable": {"thread_id": "bounded-1"}}
    result = graph.invoke({"limit": 5, "mode": "backlog"}, cfg)

    blob = json.dumps({k: v for k, v in result.items() if k != "__interrupt__"})
    assert '"body"' not in blob, "state is carrying email bodies"
    assert "thread_ids" in result, "state should carry ids"
    assert result["thread_ids"] == ["t1"]


def test_state_does_not_carry_email_bodies_on_the_incremental_path(wiring):
    """Same bound, the other branch out of partition(): auto_execute's and
    enqueue_held's state (executed/refused/auto/held) must not carry bodies
    either - this is a property of TriageState, not just of the review node."""
    graph = build_graph(**wiring)
    cfg = {"configurable": {"thread_id": "bounded-1b"}}
    result = graph.invoke({"limit": 5}, cfg)

    blob = json.dumps(result)
    assert '"body"' not in blob, "state is carrying email bodies"
    assert "thread_ids" in result, "state should carry ids"
    assert result["thread_ids"] == ["t1"]


def test_skipped_accumulates_across_nodes_via_reducer(wiring):
    """execute() and learn() both write `skipped`. Without a reducer the second
    silently discards the first - and `skipped` is where refusals are recorded."""
    from inbox_agent.graph import TriageState
    import typing

    hints = typing.get_type_hints(TriageState, include_extras=True)
    meta = getattr(hints["skipped"], "__metadata__", ())
    assert meta, "skipped needs a reducer so upstream skips cannot be overwritten"


def test_execute_skip_survives_the_learn_node(wiring):
    """End-to-end: a ghost id skipped in execute must still be in final state
    after learn has also written to `skipped`."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "bounded-2"}}
    graph.invoke({"limit": 5, "mode": "backlog"}, cfg)
    final = graph.invoke(Command(resume={
        "decisions": {"t1": "approve", "ghost": "approve"},
        "edits": {}, "instructions": [],
    }), cfg)
    assert any(s["thread_id"] == "ghost" for s in final["skipped"])


def test_stale_threads_are_demoted_inside_the_pipeline(tmp_path):
    """End-to-end: a two-year-old needs_reply leaves the inbox."""
    import json as _json
    from datetime import datetime, timezone
    from langgraph.checkpoint.memory import InMemorySaver

    old = (datetime.now(timezone.utc).replace(year=datetime.now().year - 2)
           .isoformat().replace("+00:00", "Z"))
    data = [{"id": "old1", "subject": "Are you still interested?",
             "sender": "someone@example.com", "to": [], "date": old,
             "snippet": "waiting on you", "body": "", "label_ids": ["INBOX", "UNREAD"]}]
    snap = tmp_path / "threads.json"
    snap.write_text(_json.dumps(data))

    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "a.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev",
                        stale_after_days=90)

    class NeedsReply:
        def with_structured_output(self, schema): return self
        def invoke(self, m):
            return ThreadJudgment(category="needs_reply", action="none",
                                  reason="a person is waiting", confidence=0.9)

    graph = build_graph(client=SnapshotGmailClient(snap),
                        prefs=PreferenceStore(build_store()),
                        policy=Policy(text="P", version="v", source="local"),
                        llm=NeedsReply(), settings=settings,
                        log=AuditLog(settings.audit_log),
                        held=HeldQueue(build_store()),
                        done=DoneStore(build_store()),
                        checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 5, "mode": "backlog"},
                          {"configurable": {"thread_id": "stale-1"}})
    item = result["__interrupt__"][0].value["items"][0]

    # NOTE: ReviewItem carries no `category` - it is dropped when Decision is
    # turned into a ReviewItem, so the human never sees the classification, only
    # the action. Category preservation is asserted at the unit level in
    # tests/test_recency.py; here we can only observe the action and the reason.
    assert [a["kind"] for a in item["proposed"]] == ["archive"]
    assert "stale" in item["reason"].lower() or "days old" in item["reason"]


def test_a_recent_needs_reply_still_stays_in_the_inbox(tmp_path):
    """The control: demotion must not fire on fresh mail."""
    import json as _json
    from datetime import datetime, timezone
    from langgraph.checkpoint.memory import InMemorySaver

    data = [{"id": "new1", "subject": "Are you free tomorrow?",
             "sender": "someone@example.com", "to": [],
             "date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
             "snippet": "waiting", "body": "", "label_ids": ["INBOX", "UNREAD"]}]
    snap = tmp_path / "threads.json"
    snap.write_text(_json.dumps(data))

    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "a.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev",
                        stale_after_days=90)

    class NeedsReply:
        def with_structured_output(self, schema): return self
        def invoke(self, m):
            return ThreadJudgment(category="needs_reply", action="none",
                                  reason="a person is waiting", confidence=0.9)

    graph = build_graph(client=SnapshotGmailClient(snap),
                        prefs=PreferenceStore(build_store()),
                        policy=Policy(text="P", version="v", source="local"),
                        llm=NeedsReply(), settings=settings,
                        log=AuditLog(settings.audit_log),
                        held=HeldQueue(build_store()),
                        done=DoneStore(build_store()),
                        checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 5, "mode": "backlog"},
                          {"configurable": {"thread_id": "fresh-1"}})
    item = result["__interrupt__"][0].value["items"][0]
    assert [a["kind"] for a in item["proposed"]] == ["none"]


def test_a_bare_reject_now_teaches_a_rule(wiring):
    """Skip used to teach nothing: learn_from_response required an edit, so
    correcting the agent by skipping never made it better."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "reject-learns"}}
    graph.invoke({"limit": 5, "mode": "backlog"}, cfg)
    final = graph.invoke(Command(resume={
        "decisions": {"t1": "reject"}, "edits": {}, "instructions": []}), cfg)

    assert final["learned"], "a bare reject taught nothing"
    rule = wiring["prefs"].rules()[0]
    assert rule.rejected_action == "archive", "did not record WHAT was rejected"
    assert [a.kind for a in rule.actions] == ["none"]


# --- the autonomy split ------------------------------------------------------
# partition() sends confident, reversible actions straight to Gmail and holds
# the rest. interrupt() survives, but only mode="backlog" reaches it - the one
# job that genuinely waits is a future bulk sweep of historical threads, which
# must be previewed before it commits.


def _snapshot(tmp_path, threads):
    p = tmp_path / "threads.json"
    p.write_text(json.dumps(threads))
    return p


def _row(tid, subject="Sale", sender="deals@shop.com"):
    return {"id": tid, "subject": subject, "sender": sender, "to": [],
            "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "",
            "label_ids": ["INBOX", "UNREAD"]}


def test_incremental_run_completes_without_an_interrupt(wiring):
    """Act-then-report: the run does not wait. It acts and finishes."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-a"}})
    assert "__interrupt__" not in result


def test_incremental_run_executes_the_auto_tier(wiring):
    """FakeLLM proposes archive at 0.9, which is 'always' authority."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-b"}})
    executed = [r["action"] for r in result["executed"]]
    assert "archive" in executed, executed
    assert result["executed"][0]["action"] == "archive"


def test_incremental_run_holds_trash_instead_of_executing_it(tmp_path, wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    wiring = dict(wiring)
    wiring["llm"] = FakeLLM(ThreadJudgment(
        category="newsletter_noise", action="trash", label=None,
        reason="junk", confidence=0.9))
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-c"}})
    assert result["executed"] == []
    assert [h.hold_reason for h in wiring["held"].all()] == ["trash"]


def test_held_items_reach_the_queue_with_the_run_id(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    wiring = dict(wiring)
    wiring["llm"] = FakeLLM(ThreadJudgment(
        category="other", action="archive", label=None,
        reason="not sure", confidence=0.2))
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-d"}})
    queued = wiring["held"].all()
    assert len(queued) == 1
    assert queued[0].hold_reason == "low_confidence"
    assert queued[0].run_id


def test_two_runs_accumulate_held_items_rather_than_replacing_them(tmp_path):
    """Carryover. The evening digest must still show the morning's held items."""
    from langgraph.checkpoint.memory import InMemorySaver
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    snap = _snapshot(tmp_path, [_row("t1"), _row("t2")])
    held = HeldQueue(build_store())
    graph = build_graph(
        client=SnapshotGmailClient(snap), prefs=PreferenceStore(build_store()),
        policy=Policy(text="T", version="local:test", source="local"),
        llm=FakeLLM(ThreadJudgment(category="other", action="archive", label=None,
                                   reason="unsure", confidence=0.2)),
        settings=settings, log=AuditLog(settings.audit_log), held=held,
        done=DoneStore(build_store()), checkpointer=InMemorySaver())

    graph.invoke({"limit": 1}, {"configurable": {"thread_id": "run-1"}})
    assert len(held.all()) == 1
    first_held_at = held.get("t1").first_held_at
    run_id_1 = held.get("t1").run_id

    graph.invoke({"limit": 2}, {"configurable": {"thread_id": "run-2"}})
    assert len(held.all()) == 2

    # The property under test is carryover, not the count: under replacement
    # semantics (queue cleared and rebuilt each run) run 2 would ALSO end with
    # two items, so len(held.all()) == 2 alone cannot distinguish the two
    # designs. t1's first_held_at must be the same instant across both runs -
    # it has been waiting since run 1, not since run 2 - even though it is
    # legitimately re-held (and its run_id refreshed, per HeldQueue.add's
    # docstring) each time a later run still proposes the same action for it.
    t1 = held.get("t1")
    assert t1 is not None, "t1 must still be in the queue after run 2"
    assert t1.first_held_at == first_held_at, \
        "carryover broke: t1 looks freshly held instead of still waiting"
    assert t1.run_id != run_id_1, "run_id should refresh to the later run"


def test_backlog_mode_still_suspends_at_the_interrupt(wiring):
    """interrupt() survives, scoped to the one job that genuinely waits."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10, "mode": "backlog"},
                          {"configurable": {"thread_id": "run-e"}})
    assert "__interrupt__" in result


def test_backlog_mode_executes_nothing_before_approval(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    graph.invoke({"limit": 10, "mode": "backlog"},
                 {"configurable": {"thread_id": "run-f"}})
    assert wiring["log"].records() == []
    assert wiring["held"].all() == []


def test_an_unrecognised_mode_raises_rather_than_acting(wiring):
    """route_after_partition fails closed. A near-miss like "Backlog" or a
    future typo must stop the run, not silently fall through to the branch
    that acts on a real mailbox."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    with pytest.raises(ValueError, match="unknown run mode"):
        graph.invoke({"limit": 10, "mode": "Backlog"},
                     {"configurable": {"thread_id": "run-g"}})
    assert wiring["log"].records() == [], "must not have acted before raising"


# --- fetch only unread untriaged mail, and mark what was processed ----------


def test_fetch_only_picks_unread_untriaged_inbox_mail(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    rows = [
        _row("unread") | {"label_ids": ["INBOX", "UNREAD"]},
        _row("read") | {"label_ids": ["INBOX"]},
        _row("done") | {"label_ids": ["INBOX", "UNREAD", "agent/triaged"]},
        _row("archived") | {"label_ids": ["UNREAD"]},
    ]
    snap = _snapshot(tmp_path, rows)
    graph = build_graph(
        client=SnapshotGmailClient(snap), prefs=PreferenceStore(build_store()),
        policy=Policy(text="T", version="local:test", source="local"),
        llm=FakeLLM(), settings=settings, log=AuditLog(settings.audit_log),
        held=HeldQueue(build_store()), done=DoneStore(build_store()),
        checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-q"}})
    assert result["thread_ids"] == ["unread"]


def test_every_processed_thread_gets_the_triaged_label(tmp_path):
    """Both tiers. A held item is processed too - it is in the queue, and
    leaving it unlabelled would re-triage it on the next run."""
    from langgraph.checkpoint.memory import InMemorySaver
    # dry_run=False: mark_triaged goes through execute_action like any other
    # action (that's the whole point), and execute_action's dry-run branch
    # never dispatches to the client (tests/test_audit.py::
    # test_dry_run_does_not_touch_the_client pins this down) - so observing
    # the label actually land on the thread requires a live run, the same way
    # tests/test_audit.py::test_live_run_reaches_the_client does for archive.
    settings = Settings(backend="offline", dry_run=False, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    snap = _snapshot(tmp_path, [_row("t1"), _row("t2")])
    client = SnapshotGmailClient(snap)
    log = AuditLog(settings.audit_log)
    graph = build_graph(
        client=client, prefs=PreferenceStore(build_store()),
        policy=Policy(text="T", version="local:test", source="local"),
        llm=FakeLLM(ThreadJudgment(category="other", action="archive", label=None,
                                   reason="unsure", confidence=0.2)),
        settings=settings, log=log,
        held=HeldQueue(build_store()), done=DoneStore(build_store()),
        checkpointer=InMemorySaver())
    graph.invoke({"limit": 2}, {"configurable": {"thread_id": "run-m"}})
    for tid in ("t1", "t2"):
        assert settings.triaged_label in client.get_thread(tid).label_ids
    # checkpoint_id must carry the run's thread_id, matching every other
    # action record (test_executed_action_carries_a_traceable_checkpoint_identifier) -
    # mark_triaged used to call _context(None), leaving this field None.
    label_records = [r for r in log.records() if r.action == "label"]
    assert label_records and all(r.checkpoint_id == "run-m" for r in label_records)


def test_auto_executed_thread_gets_both_the_action_and_the_triaged_label(tmp_path):
    """The auto tier's counterpart to the held-tier test above: a confident,
    reversible action executes immediately through auto_execute, AND the same
    thread still comes out of mark_triaged with the label - both tiers reach
    mark_triaged, not just the held one."""
    from langgraph.checkpoint.memory import InMemorySaver
    # dry_run=False for the same reason as the held-tier test above: neither
    # the archive nor the label lands on the client under dry-run.
    settings = Settings(backend="offline", dry_run=False, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    snap = _snapshot(tmp_path, [_row("t1")])
    client = SnapshotGmailClient(snap)
    graph = build_graph(
        # FakeLLM()'s default judgment is archive at confidence 0.9 - "always"
        # authority, the same one test_incremental_run_executes_the_auto_tier
        # relies on to land in the auto tier rather than the held queue.
        client=client, prefs=PreferenceStore(build_store()),
        policy=Policy(text="T", version="local:test", source="local"),
        llm=FakeLLM(), settings=settings, log=AuditLog(settings.audit_log),
        held=HeldQueue(build_store()), done=DoneStore(build_store()),
        checkpointer=InMemorySaver())
    graph.invoke({"limit": 1}, {"configurable": {"thread_id": "run-auto"}})
    labels = client.get_thread("t1").label_ids
    assert "INBOX" not in labels, "the auto-executed archive did not reach the client"
    assert settings.triaged_label in labels, "mark_triaged did not reach the auto tier"


def test_deny_listed_label_kind_is_visible_in_skipped_and_does_not_crash(wiring):
    """A configured-out `label` kind must not make mark_triaged silently vanish
    the refusal (the exact failure this task exists to prevent, made silent:
    with INBOX_FORBIDDEN_ACTIONS=label, every run would re-triage the same
    threads forever while looking like a normal successful run). The run must
    complete and the refusal must land in `skipped`, not `refused` - `refused`
    has no reducer and returning it here would clobber whatever execute()/
    auto_execute() already wrote there (see TriageState)."""
    from dataclasses import replace
    from langgraph.checkpoint.memory import InMemorySaver

    settings = replace(wiring["settings"],
                        forbidden_actions=wiring["settings"].forbidden_actions
                        | frozenset({"label"}))
    graph = build_graph(**{**wiring, "settings": settings}, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-deny-label"}})
    assert "__interrupt__" not in result, "the run must complete, not hang"
    assert any(s["thread_id"] == "t1" and s.get("stage") == "mark_triaged"
               for s in result["skipped"])


# --- category rules, applied after the model --------------------------------
# The case that started this: the model is right that a newsletter is valuable
# and wrong to archive it. A category rule rewrites the actions; the category,
# which the model got right, stands.

def _category_rule(category="promotion", rid="r-cat", kinds=(("label", "promotion"),)):
    return Rule(id=rid, scope="category", pattern=category,
                actions=[ActionTemplate(kind=k, params={"label": v} if v else {})
                         for k, v in kinds],
                provenance="owner keeps these in the inbox",
                created_at=datetime(2026, 9, 1, tzinfo=timezone.utc))


def test_a_category_rule_rewrites_what_the_model_proposed(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    wiring["prefs"].add_rule(_category_rule())
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    out = graph.invoke({"limit": 10, "mode": "incremental"},
                       {"configurable": {"thread_id": "cat-1"}})
    decisions = [Decision.model_validate(d) for d in out["decisions"]]
    kinds = {k for d in decisions for k in [a.kind for a in d.actions]}
    assert "archive" not in kinds, "the category rule did not remove the archive"
    assert "label" in kinds


def test_a_rewritten_decision_is_credited_to_the_rule(wiring):
    """The digest counts "came from rules you taught me". A rewrite IS the rule
    deciding, so it has to be attributed or the learning stays invisible - and
    precision cannot move on a rule that never records a hit."""
    from langgraph.checkpoint.memory import InMemorySaver
    wiring["prefs"].add_rule(_category_rule())
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    out = graph.invoke({"limit": 10, "mode": "incremental"},
                       {"configurable": {"thread_id": "cat-2"}})
    decisions = [Decision.model_validate(d) for d in out["decisions"]]
    assert all(d.source == "rule" and d.rule_id == "r-cat" for d in decisions)
    assert wiring["prefs"].rules()[0].hit_count == len(decisions)


def test_a_thread_with_no_category_rule_is_left_alone(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    out = graph.invoke({"limit": 10, "mode": "incremental"},
                       {"configurable": {"thread_id": "cat-3"}})
    decisions = [Decision.model_validate(d) for d in out["decisions"]]
    assert any("archive" in [a.kind for a in d.actions] for d in decisions)


def test_a_sender_rule_still_short_circuits_the_model(wiring):
    """Precedence, asserted rather than described: a pre-model rule means the
    model never runs, so apply_rules never sees the thread."""
    from langgraph.checkpoint.memory import InMemorySaver
    wiring["prefs"].add_rule(Rule(
        id="r-send", scope="sender", pattern="deals@shop.com",
        actions=[ActionTemplate(kind="trash")], provenance="p",
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc)))
    wiring["prefs"].add_rule(_category_rule())
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    out = graph.invoke({"limit": 10, "mode": "incremental"},
                       {"configurable": {"thread_id": "cat-4"}})
    by_id = {d["thread_id"]: Decision.model_validate(d) for d in out["decisions"]}
    hit = next(d for d in by_id.values() if d.rule_id == "r-send")
    assert hit.category == "rule_match", "the model ran on a thread a rule covered"
    assert [a.kind for a in hit.actions] == ["trash"]


def test_the_rewritten_reason_says_it_was_the_owners_rule(wiring):
    """The reason is the only record of a judgement the owner ever sees, and
    after a rewrite the model's sentence alone would be a lie by omission."""
    from langgraph.checkpoint.memory import InMemorySaver
    wiring["prefs"].add_rule(_category_rule())
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    out = graph.invoke({"limit": 10, "mode": "incremental"},
                       {"configurable": {"thread_id": "cat-5"}})
    reason = Decision.model_validate(out["decisions"][0]).reason
    assert "your rule" in reason.lower()
    assert "promotion" in reason


# --- a run's verdict supersedes what an earlier run held ---------------------

def test_a_rerun_that_stops_holding_a_thread_clears_the_stale_held_entry(wiring):
    """Found on a live phone, and dangerous.

    A LangChain workshop was held as `promotion` -> trash. The policy was
    corrected, the SAME thread re-triaged, the model returned `learning` ->
    label, and it was auto-executed. The stale trash entry stayed in the queue,
    so the digest showed one thread twice with contradictory verdicts - and
    approving the stale one would have trashed a thread the agent had just
    filed as learning material, executing a proposal the agent itself had
    superseded.

    The queue outliving runs is the design. Nothing reconciling it against a
    newer verdict was the bug.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    # Run 1: trash, which is held for authorisation.
    wiring["llm"] = FakeLLM(ThreadJudgment(
        category="promotion", action="trash", label=None,
        reason="worthless", confidence=1.0))
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-1"}})
    assert [h.thread_id for h in wiring["held"].all()] == ["t1"]
    assert wiring["held"].all()[0].item.category == "promotion"

    # Run 2, same thread: now a confident label, which auto-executes.
    wiring["llm"] = FakeLLM(ThreadJudgment(
        category="learning", action="label", label="learning",
        reason="a technical workshop", confidence=1.0))
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-2"}})
    assert wiring["held"].all() == [], (
        "the superseded trash proposal is still in the queue; the digest would "
        "show this thread twice and approving it would trash it")


def test_a_thread_the_rerun_never_saw_stays_held(wiring):
    """The carry-over the persistent queue exists for. Only threads a run
    actually processed may have their held entry superseded - otherwise a
    /triage of 10 would silently empty a queue holding 40."""
    from langgraph.checkpoint.memory import InMemorySaver
    from inbox_agent.models import Action, ReviewItem

    wiring["held"].add(
        ReviewItem(thread_id="not-in-this-run", category="promotion",
                   subject="Older", sender="old@x.com", snippet="s",
                   proposed=[Action(kind="trash", thread_id="not-in-this-run")],
                   reason="r", confidence=1.0, source="model"),
        run_id="r0", reason="trash")

    wiring["llm"] = FakeLLM(ThreadJudgment(
        category="learning", action="label", label="learning",
        reason="a workshop", confidence=1.0))
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-1"}})
    assert [h.thread_id for h in wiring["held"].all()] == ["not-in-this-run"]


# --- the execute phases report progress too ---------------------------------
#
# Slice 1 instrumented classify and moved the blind spot downstream rather than
# removing it: on 2026-09-04 classify finished at 12:04:32 and the digest
# arrived at 12:06:19, 107 seconds later, with not one log line in between. Two
# Gmail phases run in that gap - auto_execute and mark_triaged - and a run
# stalled in either looked exactly like one about to finish.

import logging as _logging


def _run_incremental(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    return graph.invoke({"limit": 10, "mode": "incremental"},
                        {"configurable": {"thread_id": "progress-1"}})


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_auto_execute_reports_which_item_it_is_on(wiring, caplog):
    with caplog.at_level(_logging.INFO, logger="inbox_agent.graph"):
        _run_incremental(wiring)
    lines = [m for m in _messages(caplog) if m.startswith("execute ")]
    assert lines, "auto_execute ran silently"
    assert "/" in lines[0], "no N/M progress in the execute line"


def test_mark_triaged_announces_the_work_before_doing_it(wiring, caplog):
    """Announced up front: on a twenty-thread run this is twenty more round
    trips after the digest already looks ready."""
    with caplog.at_level(_logging.INFO, logger="inbox_agent.graph"):
        _run_incremental(wiring)
    assert any(m.startswith("mark_triaged: labelling") for m in _messages(caplog))


def test_mark_triaged_reports_each_thread(wiring, caplog):
    with caplog.at_level(_logging.INFO, logger="inbox_agent.graph"):
        _run_incremental(wiring)
    assert any(m.startswith("mark_triaged ") and "/" in m
               for m in _messages(caplog))


def test_learning_uses_the_sender_history_the_run_recorded():
    """The teach path reads the history, so a bulk address widens to itself.

    End-to-end over the wiring rather than over choose_scope: the sighting that
    makes a display name look like payload is recorded by the triage node, and
    the rule is written by learn_from_response, and before this the two were
    not connected at all - every rule the owner taught carried the full From
    header and could never fire twice.
    """
    prefs = PreferenceStore(build_store())
    for name in ("Charmain Guia", "Daphna Cibulski-Cohen"):
        prefs.note_sender(f"{name} <invitations@linkedin.com>")

    threads = [Thread(id="t1", subject="I want to connect",
                      sender="Daphna Cibulski-Cohen <invitations@linkedin.com>",
                      to=[], date="2026-09-08T10:00:00Z", snippet="s", body="b",
                      label_ids=["INBOX"])]
    learn_from_response(
        ReviewResponse(decisions={"t1": "edit"},
                       edits={"t1": [Action(kind="trash", thread_id="t1")]},
                       instructions=[]),
        threads, prefs)

    rule = prefs.rules()[0]
    assert rule.pattern == "invitations@linkedin.com"

    # The next invitation is a different human at the same address.
    later = Thread(id="t2", subject="I want to connect",
                   sender="Rajesh Kumar <invitations@linkedin.com>", to=[],
                   date="2026-09-08T19:00:00Z", snippet="s", body="b",
                   label_ids=["INBOX"])
    assert [r.id for r in prefs.matching(later)] == [rule.id]

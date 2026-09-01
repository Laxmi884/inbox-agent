import json
import pytest

from langgraph.types import Command

from inbox_agent.audit import AuditLog
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.classify import ThreadJudgment
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.graph import build_graph, learn_from_response
from inbox_agent.models import Action, ReviewResponse, Thread
from inbox_agent.policy import Policy
from inbox_agent.store import HeldQueue, PreferenceStore, build_store


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
        held=HeldQueue(build_store()),
    )


def test_graph_suspends_at_the_review_interrupt(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-1"}}
    result = graph.invoke({"limit": 10, "mode": "backlog"}, config)
    assert "__interrupt__" in result


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
    assert len(final["executed"]) == 1
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
    assert len(final["executed"]) == 1
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
    assert rules[0].action == "label"
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
    assert prefs.rules()[0].action == "archive"
    assert "corrected" in prefs.rules()[0].provenance


def test_rule_matched_threads_skip_the_model(wiring):
    """A thread the prefilter decides must never be sent to the LLM."""
    from langgraph.checkpoint.memory import InMemorySaver
    from inbox_agent.store import rule_from_correction

    thread = wiring["client"].get_thread("t1")
    wiring["prefs"].add_rule(rule_from_correction(thread, "trash", "owner said delete"))

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
    Rule.action literals, and routing one through the edit path would also hit
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
    assert rule.action == "none"


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
    assert len(result["executed"]) == 1
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
        checkpointer=InMemorySaver())

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
        held=HeldQueue(build_store()), checkpointer=InMemorySaver())
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
        held=HeldQueue(build_store()), checkpointer=InMemorySaver())
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
        held=HeldQueue(build_store()), checkpointer=InMemorySaver())
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

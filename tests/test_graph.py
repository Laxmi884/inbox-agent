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
from inbox_agent.store import PreferenceStore, build_store


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
             "label_ids": ["INBOX"]}]
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
    )


def test_graph_suspends_at_the_review_interrupt(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-1"}}
    result = graph.invoke({"limit": 10}, config)
    assert "__interrupt__" in result


def test_interrupt_payload_is_json_serialisable(wiring):
    """It must survive the trip to a Telegram renderer unchanged."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-2"}}
    result = graph.invoke({"limit": 10}, config)
    payload = result["__interrupt__"][0].value
    json.dumps(payload)  # must not raise
    assert payload["items"][0]["thread_id"] == "t1"
    assert payload["policy_version"] == "local:test"


def test_approving_executes_the_action(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-3"}}
    graph.invoke({"limit": 10}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"t1": "approve"}, "edits": {}, "instructions": []}),
        config)
    assert len(final["executed"]) == 1
    assert wiring["log"].records()[0].action == "archive"


def test_rejecting_executes_nothing(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-4"}}
    graph.invoke({"limit": 10}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"t1": "reject"}, "edits": {}, "instructions": []}),
        config)
    assert final["executed"] == []
    assert wiring["log"].records() == []


def test_rejection_becomes_a_learned_rule(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-5"}}
    graph.invoke({"limit": 10}, config)
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
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-6"}})
    assert result["__interrupt__"][0].value["items"][0]["source"] == "rule"


def test_approve_unknown_thread_id_does_not_crash(wiring):
    """A ghost thread id on the approve path must be skipped, not KeyError'd."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-7"}}
    graph.invoke({"limit": 10}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"ghost-thread": "approve"},
                        "edits": {}, "instructions": []}),
        config)
    assert final["executed"] == []
    assert any(s["thread_id"] == "ghost-thread" for s in final["skipped"])
    assert wiring["log"].records() == []


def test_edit_unknown_thread_id_executes_nothing(wiring):
    """A forged/never-proposed thread id on the edit path must not fail open."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-8"}}
    graph.invoke({"limit": 10}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"ghost-thread-2": "edit"},
                        "edits": {"ghost-thread-2": [
                            {"kind": "trash", "thread_id": "ghost-thread-2"}]},
                        "instructions": []}),
        config)
    assert final["executed"] == []
    assert any(s["thread_id"] == "ghost-thread-2" for s in final["skipped"])
    assert wiring["log"].records() == []


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
    graph.invoke({"limit": 10}, config)
    final = graph.invoke(
        Command(resume={"decisions": {"t1": "edit"},
                        "edits": {"t1": [{"kind": "archive", "thread_id": "t1"}]},
                        "instructions": []}),
        config)
    assert final["executed"] == []
    assert len(final["refused"]) == 1
    assert final["refused"][0]["thread_id"] == "t1"
    assert final["refused"][0]["kind"] == "archive"

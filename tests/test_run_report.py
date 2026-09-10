"""Every run leaves a record of what it did, written by the graph."""
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from inbox_agent.audit import AuditLog
from inbox_agent.classify import ThreadJudgment
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.graph import build_graph, run_report_from_state
from inbox_agent.models import Action, ReviewItem, ReviewRequest
from inbox_agent.policy import Policy
from inbox_agent.store import DoneStore, HeldQueue, PreferenceStore, build_store


class FakeLLM:
    def with_structured_output(self, schema): return self
    def invoke(self, messages):
        return ThreadJudgment(category="promotion", action="archive",
                              reason="a sale", confidence=0.9)


@pytest.fixture
def wiring(tmp_path):
    data = [{"id": f"t{i}", "subject": f"Sale {i}", "sender": "deals@shop.com",
             "to": [], "date": "2026-09-08T10:00:00Z", "snippet": "s", "body": "",
             "label_ids": ["INBOX", "UNREAD"]} for i in range(2)]
    snap = tmp_path / "threads.json"
    snap.write_text(json.dumps(data))
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    return dict(client=SnapshotGmailClient(snap),
                prefs=PreferenceStore(build_store()),
                policy=Policy(text="P", version="local:t", source="local"),
                llm=FakeLLM(), settings=settings,
                log=AuditLog(settings.audit_log),
                held=HeldQueue(build_store()), done=DoneStore(build_store()))


def _state(executed, *, items, triaged="agent/triaged", thread_ids=("t1",),
           remaining=4):
    request = ReviewRequest(run_id="abc12345", policy_version="local:t",
                            items=items)
    return {"review": request.model_dump(mode="json"),
            "executed": executed, "thread_ids": list(thread_ids),
            "remaining": remaining}


def _item(tid="t1"):
    return ReviewItem(thread_id=tid, category="promotion", subject=f"Sale {tid}",
                      sender="deals@shop.com", snippet="s",
                      proposed=[Action(kind="archive", thread_id=tid)],
                      reason="a sale", confidence=0.9, source="model")


# --- the join ---------------------------------------------------------------

def test_the_report_joins_audit_records_to_their_proposals():
    state = _state([{"action": "archive", "thread_id": "t1", "actor": "model"},
                    {"action": "label", "thread_id": "t1",
                     "params": {"label": "promo"}, "actor": "rule:r-7"}],
                   items=[_item("t1")])
    report = run_report_from_state(state, triaged_label="agent/triaged")
    assert report.run_id == "abc12345"
    assert report.total == 1 and report.remaining == 4
    assert len(report.done) == 1
    row = report.done[0]
    assert row.item.subject == "Sale t1"
    assert row.item.reason == "a sale"
    assert row.actions == [("archive", None), ("label", "promo")]
    assert row.rule_id == "r-7", "the rule that decided it was not recorded"


def test_the_bookkeeping_label_is_not_reported_as_work():
    state = _state([{"action": "label", "thread_id": "t1",
                     "params": {"label": "agent/triaged"}, "actor": "model"},
                    {"action": "archive", "thread_id": "t1", "actor": "model"}],
                   items=[_item("t1")])
    report = run_report_from_state(state, triaged_label="agent/triaged")
    assert report.done[0].actions == [("archive", None)]


def test_a_record_with_no_proposal_is_still_reported_by_id():
    """An action with no visible subject is strange; hiding it would be worse."""
    state = _state([{"action": "archive", "thread_id": "ghost", "actor": "model"}],
                   items=[])
    report = run_report_from_state(state, triaged_label="agent/triaged")
    assert report.done[0].thread_id == "ghost"
    assert report.done[0].item.subject == "ghost"


def test_a_malformed_record_does_not_raise():
    """This runs after the graph acted: an exception costs the owner the report
    for work that already reached Gmail."""
    state = _state([{"actor": "model"}, {"action": "archive"}],
                   items=[_item("t1")])
    assert run_report_from_state(state, triaged_label="agent/triaged").done == []


# --- wired into the graph ---------------------------------------------------

def test_a_run_records_its_report_under_its_own_run_id(wiring):
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-1"}})
    run_id = result["review"]["run_id"]
    report = wiring["done"].get(run_id)
    assert report is not None, "the run left no record of what it did"
    assert report.total == 2
    assert {r.thread_id for r in report.done} == {"t0", "t1"}
    assert all(("archive", None) in r.actions for r in report.done)


def test_a_run_that_executed_nothing_still_writes_a_report(wiring, tmp_path):
    """"It ran and did nothing" must be distinguishable from "it never ran"."""
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    wiring["client"] = SnapshotGmailClient(empty)
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-1"}})
    report = wiring["done"].get(result["review"]["run_id"])
    assert report is not None and report.done == []


def test_two_runs_leave_two_reports(wiring):
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    first = graph.invoke({"limit": 10, "mode": "incremental"},
                         {"configurable": {"thread_id": "run-1"}})
    second = graph.invoke({"limit": 10, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-2"}})
    ids = [r.run_id for r in wiring["done"].recent()]
    assert set(ids) == {first["review"]["run_id"], second["review"]["run_id"]}


# --- a report failure must not cost the run ---------------------------------

class ExplodingDoneStore(DoneStore):
    """A full disk, a row pydantic will not validate - the cause does not
    matter, only that it raises where the report is written."""
    def record(self, report):
        raise RuntimeError("disk full")


def test_a_failed_report_does_not_cost_the_triaged_label(wiring, tmp_path):
    """The report is written between auto_execute and mark_triaged, so an
    exception escaping it aborted graph.invoke AFTER the actions reached Gmail
    and BEFORE the label landed - and the next run refetched those threads and
    executed them a second time against the real mailbox. Losing the report is
    the acceptable cost; acting twice on the owner's mail is not.

    dry_run=False for the same reason as
    test_graph.py::test_every_processed_thread_gets_the_triaged_label: the
    dry-run branch never dispatches to the client, so watching the label
    actually land takes a live run.
    """
    settings = replace(wiring["settings"], dry_run=False)
    wired = wiring | {"settings": settings,
                      "done": ExplodingDoneStore(build_store())}
    graph = build_graph(**wired, checkpointer=InMemorySaver())

    graph.invoke({"limit": 10, "mode": "incremental"},
                 {"configurable": {"thread_id": "run-1"}})

    for tid in ("t0", "t1"):
        assert settings.triaged_label in wiring["client"].get_thread(tid).label_ids, (
            "the run aborted before mark_triaged; the next run will re-execute it")

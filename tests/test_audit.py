import json
import pytest

from inbox_agent.audit import (
    AuditLog, ExecutionContext, ForbiddenActionError, execute_action,
)
from inbox_agent.config import Settings
from inbox_agent.models import Action
from inbox_agent.gmail import SnapshotGmailClient


@pytest.fixture
def snapshot_file(tmp_path):
    data = [{"id": "t1", "subject": "Sale", "sender": "deals@shop.com", "to": [],
             "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "b",
             "label_ids": ["INBOX", "UNREAD"]}]
    p = tmp_path / "threads.json"
    p.write_text(json.dumps(data))
    return p


def make_settings(tmp_path, dry_run: bool) -> Settings:
    from inbox_agent.config import ALWAYS_FORBIDDEN
    return Settings(
        backend="offline", dry_run=dry_run, snapshot_dir=tmp_path,
        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
        forbidden_actions=ALWAYS_FORBIDDEN,
        context_hub_skill="inbox-triage", context_hub_tag="dev",
    )


@pytest.fixture
def ctx():
    return ExecutionContext(model="gemma4:12b-mlx", backend="ollama",
                            policy_version="local:abc", checkpoint_id="ckpt-1")


def test_send_message_is_refused_even_when_dry_run_is_off(tmp_path, snapshot_file, ctx):
    """The deny-list is the last line of defence and does not depend on dry-run."""
    settings = make_settings(tmp_path, dry_run=False)
    log = AuditLog(settings.audit_log)
    with pytest.raises(ForbiddenActionError, match="send_message"):
        execute_action(Action(kind="send_message", thread_id="t1"),
                       client=SnapshotGmailClient(snapshot_file),
                       settings=settings, log=log, actor="agent", context=ctx)


def test_refused_action_is_still_written_to_the_audit_log(tmp_path, snapshot_file, ctx):
    """An attempted forbidden action is exactly what an auditor needs to see."""
    settings = make_settings(tmp_path, dry_run=False)
    log = AuditLog(settings.audit_log)
    with pytest.raises(ForbiddenActionError):
        execute_action(Action(kind="send_message", thread_id="t1"),
                       client=SnapshotGmailClient(snapshot_file),
                       settings=settings, log=log, actor="agent", context=ctx)
    records = log.records()
    assert len(records) == 1
    assert records[0].result.startswith("refused")


def test_dry_run_does_not_touch_the_client(tmp_path, snapshot_file, ctx):
    settings = make_settings(tmp_path, dry_run=True)
    client = SnapshotGmailClient(snapshot_file)
    rec = execute_action(Action(kind="archive", thread_id="t1"), client=client,
                         settings=settings, log=AuditLog(settings.audit_log),
                         actor="agent", context=ctx)
    assert rec.dry_run is True
    assert rec.result == "simulated"
    assert "INBOX" in client.get_thread("t1").label_ids  # untouched


def test_live_run_reaches_the_client(tmp_path, snapshot_file, ctx):
    settings = make_settings(tmp_path, dry_run=False)
    client = SnapshotGmailClient(snapshot_file)
    execute_action(Action(kind="archive", thread_id="t1"), client=client,
                   settings=settings, log=AuditLog(settings.audit_log),
                   actor="agent", context=ctx)
    assert "INBOX" not in client.get_thread("t1").label_ids


def test_undo_token_captures_prior_labels_for_archive(tmp_path, snapshot_file, ctx):
    settings = make_settings(tmp_path, dry_run=False)
    rec = execute_action(Action(kind="archive", thread_id="t1"),
                         client=SnapshotGmailClient(snapshot_file), settings=settings,
                         log=AuditLog(settings.audit_log), actor="agent", context=ctx)
    assert rec.undo_token["restore_labels"] == ["INBOX", "UNREAD"]
    assert rec.reversible is True


def test_audit_log_is_append_only_jsonl(tmp_path, snapshot_file, ctx):
    settings = make_settings(tmp_path, dry_run=True)
    log = AuditLog(settings.audit_log)
    client = SnapshotGmailClient(snapshot_file)
    for kind in ("label", "archive"):
        execute_action(Action(kind=kind, thread_id="t1", params={"label": "X"}),
                       client=client, settings=settings, log=log,
                       actor="agent", context=ctx)
    lines = settings.audit_log.read_text().strip().split("\n")
    assert len(lines) == 2
    assert all(json.loads(line)["thread_id"] == "t1" for line in lines)


def test_record_carries_provenance_and_policy_version(tmp_path, snapshot_file, ctx):
    settings = make_settings(tmp_path, dry_run=True)
    rec = execute_action(Action(kind="archive", thread_id="t1"),
                         client=SnapshotGmailClient(snapshot_file), settings=settings,
                         log=AuditLog(settings.audit_log), actor="rule:r1",
                         context=ctx, rule_provenance="learned from correction on t9")
    assert rec.actor == "rule:r1"
    assert rec.rule_provenance == "learned from correction on t9"
    assert rec.policy_version == "local:abc"
    assert rec.model == "gemma4:12b-mlx"

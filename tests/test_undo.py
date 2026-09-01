"""Undoing an executed action.

`undo_token` has been written on every audit record since Stage A and
`undo_candidates()` has existed to find them - but nothing ever consumed either.
The machinery was built and the door was never fitted.

This matters more now than it did. Reporting reversible actions instead of
gating them is only safe if the tenth one, the one that was wrong, is cheap to
reverse. Without undo, "1 in 10 is fine" is not true.
"""
import json
from datetime import datetime, timezone

import pytest

from inbox_agent.audit import (
    AuditLog, ExecutionContext, ForbiddenActionError, execute_action, undo_action,
)
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.models import Action


@pytest.fixture
def wired(tmp_path):
    data = [{"id": "t1", "subject": "S", "sender": "a@b.com", "to": [],
             "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "",
             "label_ids": ["INBOX", "Finance"]}]
    snap = tmp_path / "threads.json"
    snap.write_text(json.dumps(data))
    settings = Settings(backend="offline", dry_run=False, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    return (SnapshotGmailClient(snap), settings, AuditLog(settings.audit_log),
            ExecutionContext(model="m", backend="none", policy_version="v"))


def act(kind, wired, **params):
    client, settings, log, ctx = wired
    return execute_action(Action(kind=kind, thread_id="t1", params=params),
                          client=client, settings=settings, log=log,
                          actor="agent", context=ctx)


# --- identity ---------------------------------------------------------------

def test_every_executed_action_gets_an_id_so_it_can_be_referenced(wired):
    rec = act("archive", wired)
    assert rec.id, "an action with no id cannot be undone"


def test_ids_are_unique_across_actions(wired):
    ids = {act("label", wired, label=f"L{i}").id for i in range(5)}
    assert len(ids) == 5


# --- reversing --------------------------------------------------------------

def test_undoing_an_archive_puts_the_thread_back_in_the_inbox(wired):
    client, settings, log, ctx = wired
    rec = act("archive", wired)
    assert "INBOX" not in client.get_thread("t1").label_ids

    undo_action(rec, client=client, settings=settings, log=log,
                actor="human", context=ctx)
    assert "INBOX" in client.get_thread("t1").label_ids


def test_undoing_an_archive_restores_the_other_labels_too(wired):
    """undo_token captures ALL prior labels, not just INBOX."""
    client, settings, log, ctx = wired
    rec = act("archive", wired)
    undo_action(rec, client=client, settings=settings, log=log,
                actor="human", context=ctx)
    assert "Finance" in client.get_thread("t1").label_ids


def test_undoing_a_label_removes_it(wired):
    client, settings, log, ctx = wired
    rec = act("label", wired, label="Recruiter")
    assert "Recruiter" in client.get_thread("t1").label_ids
    undo_action(rec, client=client, settings=settings, log=log,
                actor="human", context=ctx)
    assert "Recruiter" not in client.get_thread("t1").label_ids


# --- the undo is itself audited ---------------------------------------------

def test_the_undo_is_appended_never_overwrites_the_original(wired):
    """Append-only means append-only. The original record must survive
    untouched - it is the evidence that the action happened at all."""
    client, settings, log, ctx = wired
    rec = act("archive", wired)
    before = len(log.records())
    undo_action(rec, client=client, settings=settings, log=log,
                actor="human", context=ctx)
    after = log.records()
    assert len(after) == before + 1
    assert any(r.id == rec.id and r.action == "archive" for r in after)


def test_the_undo_record_links_back_to_what_it_reversed(wired):
    client, settings, log, ctx = wired
    rec = act("archive", wired)
    undo = undo_action(rec, client=client, settings=settings, log=log,
                       actor="human", context=ctx)
    assert undo.undoes == rec.id
    assert undo.actor == "human"


# --- refusals ---------------------------------------------------------------

def test_undoing_twice_is_refused(wired):
    """The second undo would re-apply the original action's inverse to a state
    that has already been restored."""
    client, settings, log, ctx = wired
    rec = act("archive", wired)
    undo_action(rec, client=client, settings=settings, log=log,
                actor="human", context=ctx)
    with pytest.raises(ValueError, match="already undone"):
        undo_action(rec, client=client, settings=settings, log=log,
                    actor="human", context=ctx)


def test_a_record_with_no_undo_token_cannot_be_undone(wired):
    client, settings, log, ctx = wired
    rec = act("none", wired)
    with pytest.raises(ValueError, match="nothing to undo"):
        undo_action(rec, client=client, settings=settings, log=log,
                    actor="human", context=ctx)


def test_a_refused_action_cannot_be_undone(wired):
    """It never happened. There is nothing to reverse, and pretending there is
    would put a fictional event in the audit trail."""
    client, settings, log, ctx = wired
    try:
        act("send_message", wired)
    except ForbiddenActionError:
        pass
    refused = log.records()[-1]
    with pytest.raises(ValueError, match="did not succeed"):
        undo_action(refused, client=client, settings=settings, log=log,
                    actor="human", context=ctx)


def test_a_dry_run_action_cannot_be_undone(wired):
    """Nothing happened in Gmail, so reversing it would mutate real state to
    undo something that only ever existed as a log line."""
    client, settings, log, ctx = wired
    dry = Settings(**{**settings.__dict__, "dry_run": True})
    rec = execute_action(Action(kind="archive", thread_id="t1"),
                         client=client, settings=dry, log=log,
                         actor="agent", context=ctx)
    with pytest.raises(ValueError, match="simulated"):
        undo_action(rec, client=client, settings=dry, log=log,
                    actor="human", context=ctx)


# --- candidates -------------------------------------------------------------

def test_undo_candidates_excludes_what_is_already_undone(wired):
    client, settings, log, ctx = wired
    a = act("archive", wired)
    b = act("label", wired, label="Keep")
    assert {r.id for r in log.undo_candidates()} == {a.id, b.id}
    undo_action(a, client=client, settings=settings, log=log,
                actor="human", context=ctx)
    assert {r.id for r in log.undo_candidates()} == {b.id}


def test_undo_candidates_never_offers_the_undo_records_themselves(wired):
    client, settings, log, ctx = wired
    rec = act("archive", wired)
    undo_action(rec, client=client, settings=settings, log=log,
                actor="human", context=ctx)
    assert all(r.undoes is None for r in log.undo_candidates())

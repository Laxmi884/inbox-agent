# tests/test_gmail.py
import json
import pytest

from inbox_agent.gmail import SnapshotGmailClient, load_snapshot
from inbox_agent.models import Thread


@pytest.fixture
def snapshot_file(tmp_path):
    data = [
        {"id": "t1", "subject": "Sale 50%", "sender": "deals@shop.com",
         "to": ["me@z.com"], "date": "2026-08-26T10:00:00Z", "snippet": "big sale",
         "body": "Everything half price", "label_ids": ["INBOX", "UNREAD"]},
        {"id": "t2", "subject": "Re: budget", "sender": "boss@work.com",
         "to": ["me@z.com"], "date": "2026-08-26T11:00:00Z", "snippet": "thoughts?",
         "body": "What do you think?", "label_ids": ["INBOX"]},
    ]
    p = tmp_path / "threads.json"
    p.write_text(json.dumps(data))
    return p


def test_load_snapshot_parses_into_threads(snapshot_file):
    threads = load_snapshot(snapshot_file)
    assert len(threads) == 2
    assert all(isinstance(t, Thread) for t in threads)
    assert threads[0].sender_domain == "shop.com"


def test_list_threads_respects_limit(snapshot_file):
    assert len(SnapshotGmailClient(snapshot_file).list_threads(limit=1)) == 1


def test_get_thread_returns_matching_thread(snapshot_file):
    assert SnapshotGmailClient(snapshot_file).get_thread("t2").sender == "boss@work.com"


def test_get_thread_raises_on_unknown_id(snapshot_file):
    with pytest.raises(KeyError):
        SnapshotGmailClient(snapshot_file).get_thread("nope")


def test_mutations_are_recorded_in_memory_not_sent_anywhere(snapshot_file):
    """The snapshot client simulates. It must never claim to have called Gmail."""
    c = SnapshotGmailClient(snapshot_file)
    result = c.archive("t1")
    assert result["simulated"] is True
    assert "INBOX" not in c.get_thread("t1").label_ids


def test_apply_and_remove_label_mutate_local_state(snapshot_file):
    c = SnapshotGmailClient(snapshot_file)
    c.apply_label("t1", "Deals")
    assert "Deals" in c.get_thread("t1").label_ids
    c.remove_label("t1", "Deals")
    assert "Deals" not in c.get_thread("t1").label_ids


def test_trash_is_recorded_and_reversible(snapshot_file):
    c = SnapshotGmailClient(snapshot_file)
    original_labels = set(c.get_thread("t1").label_ids)
    c.trash("t1")
    trashed_labels = c.get_thread("t1").label_ids
    # (a) TRASH is present after trashing
    assert "TRASH" in trashed_labels
    # (b) INBOX is absent after trashing
    assert "INBOX" not in trashed_labels
    # (c) Reversibility: remove TRASH, re-add INBOX, assert label set matches original
    c.remove_label("t1", "TRASH")
    c.apply_label("t1", "INBOX")
    restored_labels = set(c.get_thread("t1").label_ids)
    assert restored_labels == original_labels


def test_create_draft_returns_simulated_and_draft_length(snapshot_file):
    c = SnapshotGmailClient(snapshot_file)
    result = c.create_draft("t1", "This is my draft body")
    assert result["simulated"] is True
    assert result["draft_chars"] == 21


def test_missing_snapshot_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="snapshot"):
        load_snapshot(tmp_path / "absent.json")

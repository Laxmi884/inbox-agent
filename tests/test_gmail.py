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
    c.trash("t1")
    assert "TRASH" in c.get_thread("t1").label_ids


def test_missing_snapshot_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="snapshot"):
        load_snapshot(tmp_path / "absent.json")

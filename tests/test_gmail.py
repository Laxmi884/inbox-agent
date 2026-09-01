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


from inbox_agent.gmail import matches_query


def _thread(i, labels):
    return {"id": f"t{i}", "subject": f"S{i}", "sender": f"s{i}@x.com", "to": [],
            "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "",
            "label_ids": labels}


@pytest.fixture
def mixed_snapshot(tmp_path):
    data = [
        _thread(0, ["INBOX", "UNREAD"]),
        _thread(1, ["INBOX"]),                       # read
        _thread(2, ["INBOX", "UNREAD", "agent/triaged"]),
        _thread(3, ["UNREAD"]),                      # archived
        _thread(4, ["INBOX", "UNREAD"]),
    ]
    p = tmp_path / "threads.json"
    p.write_text(json.dumps(data))
    return p


def test_matches_query_honours_is_unread():
    t = Thread.model_validate(_thread(0, ["INBOX", "UNREAD"]))
    assert matches_query(t, "is:unread")
    r = Thread.model_validate(_thread(1, ["INBOX"]))
    assert not matches_query(r, "is:unread")


def test_matches_query_honours_negated_label():
    t = Thread.model_validate(_thread(2, ["INBOX", "agent/triaged"]))
    assert not matches_query(t, "-label:agent/triaged")
    assert matches_query(t, "label:agent/triaged")


def test_matches_query_is_case_insensitive_on_labels():
    t = Thread.model_validate(_thread(2, ["INBOX", "AGENT/TRIAGED"]))
    assert not matches_query(t, "-label:agent/triaged")


def test_empty_query_matches_everything():
    t = Thread.model_validate(_thread(1, ["INBOX"]))
    assert matches_query(t, "")


def test_unknown_query_term_raises_rather_than_being_ignored():
    """A term the snapshot cannot honour must fail loudly.

    The live client passes `query` to Gmail verbatim, so an unknown term works
    there and would silently do nothing here - which would make every snapshot
    test a false negative for that term.
    """
    t = Thread.model_validate(_thread(0, ["INBOX", "UNREAD"]))
    with pytest.raises(ValueError, match="newer_than:2d"):
        matches_query(t, "is:unread newer_than:2d")


def test_snapshot_client_filters_by_query(mixed_snapshot):
    client = SnapshotGmailClient(mixed_snapshot)
    got = client.list_threads(query="in:inbox is:unread -label:agent/triaged")
    assert [t.id for t in got] == ["t0", "t4"]


def test_snapshot_client_filters_before_applying_the_limit(mixed_snapshot):
    """limit=2 must mean two MATCHING threads, not two candidates then filtered.

    Filtering after the limit is the bug that makes a mailbox with a read run
    of 50 look empty.
    """
    client = SnapshotGmailClient(mixed_snapshot)
    got = client.list_threads(limit=2, query="in:inbox is:unread -label:agent/triaged")
    assert [t.id for t in got] == ["t0", "t4"]


def test_no_query_still_returns_everything_in_order(mixed_snapshot):
    client = SnapshotGmailClient(mixed_snapshot)
    assert len(client.list_threads()) == 5

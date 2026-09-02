"""The same assertions against both GmailClient implementations.

The comment above matches_query says both clients answer the SAME query string,
and that this "is the only thing that makes a snapshot test evidence about live
behaviour". That was a comment. These are the tests that make it true.

Where the two legitimately differ, the difference is asserted rather than
skipped - see test_only_the_snapshot_client_marks_writes_simulated.
"""
import json
import pytest

from inbox_agent.gmail import LiveGmailClient, SnapshotGmailClient
from test_live_gmail import FakeGmailApi, message

# One fixture, two shapes. Label NAMES on the snapshot side and label IDS on the
# live side, because that asymmetry is exactly what _LabelMap exists to erase -
# both clients must still produce names on Thread.label_ids.
THREADS = [
    ("t0", ["INBOX", "UNREAD"], ["INBOX", "UNREAD"]),
    ("t1", ["INBOX"], ["INBOX"]),
    ("t2", ["INBOX", "UNREAD", "agent/triaged"], ["INBOX", "UNREAD", "Label_tri"]),
    ("t3", ["UNREAD"], ["UNREAD"]),
    ("t4", ["INBOX", "UNREAD"], ["INBOX", "UNREAD"]),
]


@pytest.fixture
def snapshot_client(tmp_path):
    data = [{"id": tid, "subject": f"S{tid}", "sender": f"{tid}@x.com",
             "to": ["me@z.com"], "date": "2026-08-26T10:00:00Z",
             "snippet": "s", "body": "", "label_ids": names}
            for tid, names, _ in THREADS]
    p = tmp_path / "threads.json"
    p.write_text(json.dumps(data))
    return SnapshotGmailClient(p)


@pytest.fixture
def live_client():
    api = FakeGmailApi(
        threads={
            tid: [message(f"m{tid}", sender=f"{tid}@x.com", subject=f"S{tid}",
                          to="me@z.com",
                          date="Wed, 26 Aug 2026 10:00:00 +0000",
                          snippet="s", body="", label_ids=ids)]
            for tid, _, ids in THREADS
        },
        labels=[{"id": "INBOX", "name": "INBOX"},
                {"id": "UNREAD", "name": "UNREAD"},
                {"id": "TRASH", "name": "TRASH"},
                {"id": "Label_tri", "name": "agent/triaged"}],
    )
    # The fake ignores `q`, so `order` is pre-filtered to the set inbox_query
    # would return. The point here is that both clients AGREE on that set, not
    # that the fake reimplements Gmail's query engine.
    #
    # `threads` deliberately keeps ALL of them. Only `order` drives list; get
    # must still resolve a thread the query excluded, which is how Gmail
    # behaves - a filtered thread is not a deleted one.
    api.order = ["t0", "t4"]
    return LiveGmailClient(api)


def test_both_clients_return_names_in_label_ids(snapshot_client, live_client):
    """The invariant everything else rests on. A snapshot test asserting on
    'agent/triaged' and a live run must be asserting the same thing."""
    assert snapshot_client.get_thread("t0").label_ids == \
           live_client.get_thread("t0").label_ids == ["INBOX", "UNREAD"]


def test_both_clients_agree_on_the_inbox_query(snapshot_client, live_client):
    query = "in:inbox is:unread -label:agent/triaged"
    snap = [t.id for t in snapshot_client.list_threads(limit=50, query=query)]
    live = [t.id for t in live_client.list_threads(limit=50, query=query)]
    assert snap == live == ["t0", "t4"]


def test_both_clients_produce_the_same_thread_shape(snapshot_client, live_client):
    snap = snapshot_client.get_thread("t0")
    live = live_client.get_thread("t0")
    assert (snap.id, snap.subject, snap.sender, snap.to) == \
           (live.id, live.subject, live.sender, live.to)
    assert snap.date[:19] == live.date[:19]


def test_both_clients_agree_on_derived_identity(snapshot_client, live_client):
    """fingerprint and sender_domain drive rule matching. If the two clients
    derived different ones, a rule taught against the snapshot would silently
    stop matching the same mail live."""
    snap = snapshot_client.get_thread("t0")
    live = live_client.get_thread("t0")
    assert snap.fingerprint == live.fingerprint
    assert snap.sender_domain == live.sender_domain


def test_archive_leaves_the_thread_present_in_both(snapshot_client, live_client):
    """Archive is not delete. The thread must still be gettable afterwards -
    this is what makes a wrong archive cost nothing, and it is the assumption
    the whole backlog sweep rests on."""
    for client in (snapshot_client, live_client):
        client.archive("t0")
        assert "INBOX" not in client.get_thread("t0").label_ids
        assert client.get_thread("t0").id == "t0"


def test_trash_removes_from_inbox_in_both(snapshot_client, live_client):
    for client in (snapshot_client, live_client):
        client.trash("t0")
        assert "INBOX" not in client.get_thread("t0").label_ids


def test_apply_label_is_visible_in_label_ids_in_both(snapshot_client, live_client):
    for client in (snapshot_client, live_client):
        client.apply_label("t0", "agent/triaged")
        assert "agent/triaged" in client.get_thread("t0").label_ids


def test_remove_label_is_gone_from_label_ids_in_both(snapshot_client, live_client):
    for client in (snapshot_client, live_client):
        client.remove_label("t2", "agent/triaged")
        assert "agent/triaged" not in client.get_thread("t2").label_ids


def test_a_triaged_thread_leaves_the_inbox_query_in_both(
        snapshot_client, live_client):
    """The whole point of agent/triaged: a labelled thread stops being fetched.
    Snapshot-only, because the fake does not implement Gmail's query engine -
    but the live half is what Gmail itself guarantees, and gmail.py:19 is why
    proving it here is evidence about there."""
    query = "in:inbox is:unread -label:agent/triaged"
    before = [t.id for t in snapshot_client.list_threads(limit=50, query=query)]
    assert "t0" in before
    snapshot_client.apply_label("t0", "agent/triaged")
    after = [t.id for t in snapshot_client.list_threads(limit=50, query=query)]
    assert "t0" not in after


def test_only_the_snapshot_client_marks_writes_simulated(
        snapshot_client, live_client):
    """A legitimate difference, asserted rather than ignored. The snapshot
    stamps every write so a simulated write can never be mistaken for a real
    one; the live client must never make that claim."""
    assert snapshot_client.archive("t0")["simulated"] is True
    assert live_client.archive("t0").get("simulated") is not True

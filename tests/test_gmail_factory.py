"""One place decides snapshot vs live.

The Protocol means there is exactly one construction site outside tests
(inbox_agent/telegram/__main__.py), which is why this is a factory and not a
migration.
"""
import json
import pytest

from inbox_agent.config import build_gmail_client, load_settings
from inbox_agent.gmail import SnapshotGmailClient


@pytest.fixture
def snapshot_dir(tmp_path):
    data = [{"id": "t1", "subject": "S", "sender": "a@b.com", "to": [],
             "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "",
             "label_ids": ["INBOX", "UNREAD"]}]
    (tmp_path / "threads.json").write_text(json.dumps(data))
    return tmp_path


def test_defaults_to_the_snapshot_client(monkeypatch, snapshot_dir):
    """An unconfigured checkout must not reach the real mailbox."""
    monkeypatch.delenv("INBOX_GMAIL", raising=False)
    monkeypatch.setenv("INBOX_SNAPSHOT_DIR", str(snapshot_dir))
    assert isinstance(build_gmail_client(load_settings()), SnapshotGmailClient)


def test_live_selector_builds_the_live_client(monkeypatch, snapshot_dir):
    monkeypatch.setenv("INBOX_GMAIL", "live")
    monkeypatch.setenv("INBOX_SNAPSHOT_DIR", str(snapshot_dir))
    built = {}

    def fake_live(settings):
        built["called"] = True
        return object()

    monkeypatch.setattr("inbox_agent.config._build_live_gmail_client", fake_live)
    build_gmail_client(load_settings())
    assert built["called"] is True


def test_an_unknown_selector_fails_loudly(monkeypatch, snapshot_dir):
    """A typo must never silently fall back. Falling back to snapshot would look
    like a working run against a mailbox that was never touched; falling back to
    live would touch a mailbox nobody asked it to."""
    monkeypatch.setenv("INBOX_SNAPSHOT_DIR", str(snapshot_dir))
    s = load_settings()
    object.__setattr__(s, "gmail", "livee")   # frozen dataclass; bypass _resolve
    with pytest.raises(ValueError, match="INBOX_GMAIL"):
        build_gmail_client(s)


def test_the_live_client_is_given_a_per_thread_transport(monkeypatch, snapshot_dir):
    """The bug Task 5 found: googleapiclient's service holds ONE httplib2.Http
    and it is not thread-safe, so the hydration pool corrupts its own SSL
    socket. A factory that builds the client without http_factory would
    reintroduce it on the very first live run, and every unit test would still
    pass - the fakes have no sockets."""
    monkeypatch.setenv("INBOX_GMAIL", "live")
    monkeypatch.setenv("INBOX_SNAPSHOT_DIR", str(snapshot_dir))

    sentinel = object()
    monkeypatch.setattr("inbox_agent.google_auth.get_credentials",
                        lambda **k: sentinel)
    # `timeout` is accepted, not ignored: the factory passes it, and a stub
    # that could not would hide the regression this test exists to catch.
    # What the deadline IS belongs to test_http_timeout.
    monkeypatch.setattr("inbox_agent.google_auth.authorized_http",
                        lambda creds, timeout=None: ("http-for", creds))
    monkeypatch.setattr("googleapiclient.discovery.build",
                        lambda *a, **k: object())

    client = build_gmail_client(load_settings())
    assert client._http_factory is not None, "no per-thread transport wired"
    assert client._http() == ("http-for", sentinel)

"""The socket that never returned - 2026-09-11.

The bot sat in `ssl.read` for five hours at 0% CPU, holding MainThread, and
the 23:00 slot never fired. A `py-spy` dump caught it mid-call:

    read (ssl.py:1104)
    _conn_request (httplib2/__init__.py:1396)  host: gmail.googleapis.com
    _with_backoff (inbox_agent/gmail.py:554)   attempt: 1
    get_thread -> _threads -> learn -> _run_triage -> _tick -> run_polling

`attempt: 1` is the whole story. The retry ladder was not exhausted; it never
got a turn, because the call it wraps cannot return. `httplib2.Http()` was
built with no timeout, so a half-open connection blocks forever - and unlike
the idle-keepalive death of 2026-09-03, nothing raises, so retry and
`_reset_http` are both unreachable.

The suite could not have caught this before and still cannot catch it directly:
a fake has no socket, and "blocks forever" is not a thing a unit test may wait
for. What IS testable is every link in the chain that makes the hang
impossible - that the transport is built with a deadline, that the deadline is
configurable and refuses nonsense, that the factory actually passes it, and
that the resulting TimeoutError lands on the retry path rather than escaping.
"""
import socket

import pytest

from inbox_agent import gmail, google_auth
from inbox_agent.config import build_gmail_client, load_settings
from inbox_agent.gmail import LiveGmailClient

from .test_live_gmail import FakeUsers, FakeThreads, _raising, api  # noqa: F401


# --- the transport is built with a deadline ---------------------------------

def test_the_transport_is_built_with_a_deadline(monkeypatch):
    """The one line whose absence cost five hours."""
    seen = {}

    class FakeHttp:
        def __init__(self, timeout=None):
            seen["timeout"] = timeout

    monkeypatch.setitem(__import__("sys").modules, "httplib2",
                        type("m", (), {"Http": FakeHttp}))
    monkeypatch.setitem(__import__("sys").modules, "google_auth_httplib2",
                        type("m", (), {"AuthorizedHttp":
                                       staticmethod(lambda c, http=None: http)}))

    google_auth.authorized_http(object(), timeout=12.5)
    assert seen["timeout"] == 12.5, "the transport can still block forever"


def test_the_deadline_is_not_optional(monkeypatch):
    """A default of None on the parameter would let a caller reintroduce the
    hang silently. Every caller must be made to say a number."""
    seen = {}

    class FakeHttp:
        def __init__(self, timeout=None):
            seen["timeout"] = timeout

    monkeypatch.setitem(__import__("sys").modules, "httplib2",
                        type("m", (), {"Http": FakeHttp}))
    monkeypatch.setitem(__import__("sys").modules, "google_auth_httplib2",
                        type("m", (), {"AuthorizedHttp":
                                       staticmethod(lambda c, http=None: http)}))

    google_auth.authorized_http(object())
    assert isinstance(seen["timeout"], (int, float)), "no default deadline"
    assert seen["timeout"] > 0


# --- the deadline is configurable and refuses nonsense -----------------------

def test_the_default_timeout_is_finite(monkeypatch):
    monkeypatch.delenv("INBOX_HTTP_TIMEOUT", raising=False)
    assert 0 < load_settings().http_timeout < float("inf")


def test_the_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("INBOX_HTTP_TIMEOUT", "30")
    assert load_settings().http_timeout == 30.0


def test_a_timeout_of_zero_is_refused(monkeypatch):
    """0 is a non-blocking socket in httplib2, not "no limit" - every call
    would fail instantly. Refuse it rather than let it look like "off"."""
    monkeypatch.setenv("INBOX_HTTP_TIMEOUT", "0")
    with pytest.raises(ValueError, match="INBOX_HTTP_TIMEOUT"):
        load_settings()


def test_a_negative_timeout_is_refused(monkeypatch):
    monkeypatch.setenv("INBOX_HTTP_TIMEOUT", "-1")
    with pytest.raises(ValueError, match="INBOX_HTTP_TIMEOUT"):
        load_settings()


def test_a_non_numeric_timeout_is_refused(monkeypatch):
    """The schedule's reasoning: a setting that quietly disabled itself on a
    typo is the failure that has no error message naming it."""
    monkeypatch.setenv("INBOX_HTTP_TIMEOUT", "sixty")
    with pytest.raises(ValueError, match="INBOX_HTTP_TIMEOUT"):
        load_settings()


# --- the factory actually passes it -----------------------------------------

def test_the_live_client_transport_carries_the_timeout(monkeypatch, tmp_path):
    """A correct default and a correct authorized_http still hang if the one
    construction site drops the argument on the floor."""
    import json
    (tmp_path / "threads.json").write_text(json.dumps([]))
    monkeypatch.setenv("INBOX_GMAIL", "live")
    monkeypatch.setenv("INBOX_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("INBOX_HTTP_TIMEOUT", "45")

    seen = {}
    monkeypatch.setattr("inbox_agent.google_auth.get_credentials",
                        lambda **k: object())
    monkeypatch.setattr("inbox_agent.google_auth.authorized_http",
                        lambda creds, timeout=None: seen.update(timeout=timeout))
    monkeypatch.setattr("googleapiclient.discovery.build",
                        lambda *a, **k: object())

    build_gmail_client(load_settings())._http()
    assert seen["timeout"] == 45.0, "the deadline never reached the socket"


# --- and the timeout lands on the retry path --------------------------------

def test_a_socket_timeout_counts_as_a_transport_error():
    """socket.timeout IS TimeoutError is a subclass of OSError, so the existing
    tuple already covers it. Pinned because the fix DEPENDS on that being true:
    if _TRANSPORT_ERRORS is ever narrowed to the ssl errors, the timeout stops
    being retried and starts aborting runs instead."""
    assert socket.timeout is TimeoutError
    assert gmail._is_transport_error(TimeoutError("timed out"))


def test_a_timed_out_socket_is_retried_on_a_fresh_transport(api, monkeypatch):
    """What the wedge would have done had the deadline existed: raise, drop the
    half-open connection, reconnect, succeed - instead of blocking forever."""
    monkeypatch.setattr(gmail.time, "sleep", lambda _s: None)
    client = LiveGmailClient(_raising(api, [TimeoutError("timed out")]),
                             http_factory=lambda: object())
    dropped = []
    real = client._reset_http
    monkeypatch.setattr(client, "_reset_http",
                        lambda: (dropped.append(1), real()) and None)

    assert [t.id for t in client.list_threads(limit=10)] == ["t1", "t2"]
    assert dropped, "the retry reused the connection that had just timed out"

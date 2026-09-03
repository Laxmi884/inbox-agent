# tests/test_gmail.py
import json
import pytest

from inbox_agent import gmail
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


# --- the retry policy -------------------------------------------------------
# _with_backoff had no tests at all, which is how this survived: it keyed the
# decision on HTTP status alone, and 403 is the one status where Gmail
# overloads two opposite meanings. A live /triage 20 aborted halfway through
# on 2026-09-03 with 32 actions already applied to the real mailbox, because
# "Quota exceeded ... Units per minute per user" arrives as 403, not 429.

def _http_error(status, reason=None, message="boom"):
    """A real HttpError, built the way googleapiclient builds one, so the
    tests exercise the same attributes the library actually populates."""
    import httplib2
    from googleapiclient.errors import HttpError

    body = {"error": {"code": status, "message": message}}
    if reason:
        body["error"]["errors"] = [{"message": message, "domain": "usageLimits",
                                    "reason": reason}]
    return HttpError(httplib2.Response({"status": status}),
                     json.dumps(body).encode(), uri="https://gmail/x")


@pytest.fixture
def no_sleep(monkeypatch):
    """Record what the backoff WOULD have waited, without waiting."""
    slept = []
    monkeypatch.setattr(gmail.time, "sleep", slept.append)
    return slept


def _flaky(errors):
    """A call that raises each error in turn, then returns "ok"."""
    seq = list(errors)

    def call():
        if seq:
            raise seq.pop(0)
        return "ok"
    return call


def test_a_rate_limit_403_is_retried_not_raised(no_sleep):
    """The bug. Gmail signals "slow down" with 403; the policy read the status,
    saw a 4xx, and treated a transient limit as a permanent refusal."""
    assert gmail._with_backoff(
        _flaky([_http_error(403, "rateLimitExceeded")]), what="get") == "ok"


def test_the_user_rate_limit_403_is_retried_too(no_sleep):
    """The per-user variant of the same signal, and the one a single busy
    mailbox is most likely to hit."""
    assert gmail._with_backoff(
        _flaky([_http_error(403, "userRateLimitExceeded")]), what="get") == "ok"


def test_a_permission_403_still_fails_immediately(no_sleep):
    """The reason the original policy existed, and it stays true: a scope the
    token does not have is not going to appear by waiting, and retrying it
    hides a real misconfiguration behind a delay."""
    from googleapiclient.errors import HttpError

    with pytest.raises(HttpError):
        gmail._with_backoff(
            _flaky([_http_error(403, "insufficientPermissions")]), what="get")
    assert no_sleep == [], "a permission denial was retried"


def test_a_bare_403_with_no_reason_still_fails_immediately(no_sleep):
    """Absent a reason there is nothing to distinguish the two meanings, and
    the safe reading of an ambiguous 403 is the one that surfaces it."""
    from googleapiclient.errors import HttpError

    with pytest.raises(HttpError):
        gmail._with_backoff(_flaky([_http_error(403)]), what="get")


def test_a_rate_limit_waits_long_enough_for_a_per_minute_quota(no_sleep):
    """The limit that fired is per MINUTE. The old schedule gave up after
    1+2+4 = 7 seconds, so even a retrying policy would have failed."""
    from googleapiclient.errors import HttpError

    errors = [_http_error(403, "rateLimitExceeded") for _ in range(20)]
    with pytest.raises(HttpError):
        gmail._with_backoff(_flaky(errors), what="get")
    assert sum(no_sleep) >= 60, f"gave up after only {sum(no_sleep):.0f}s"


def test_a_server_error_keeps_the_short_schedule(no_sleep):
    """5xx is transient and clears fast. Making every 500 wait a minute would
    turn a blip into a visibly hung run."""
    from googleapiclient.errors import HttpError

    with pytest.raises(HttpError):
        gmail._with_backoff(_flaky([_http_error(503) for _ in range(20)]),
                            what="get")
    assert sum(no_sleep) < 30, f"a 5xx waited {sum(no_sleep):.0f}s"


def test_retries_are_jittered(no_sleep, monkeypatch):
    """Five hydrate workers share the quota. Un-jittered, they fail together,
    sleep the identical interval, and collide again on every retry - which is
    exactly the burst that exhausted a per-minute limit in 12 seconds."""
    from googleapiclient.errors import HttpError

    monkeypatch.setattr(gmail.random, "uniform", lambda a, b: b)
    with pytest.raises(HttpError):
        gmail._with_backoff(_flaky([_http_error(403, "rateLimitExceeded")] * 20),
                            what="get")
    high = list(no_sleep)

    no_sleep.clear()
    monkeypatch.setattr(gmail.random, "uniform", lambda a, b: a)
    with pytest.raises(HttpError):
        gmail._with_backoff(_flaky([_http_error(403, "rateLimitExceeded")] * 20),
                            what="get")
    assert high != no_sleep, "the wait does not depend on the jitter at all"


def test_a_successful_call_never_sleeps(no_sleep):
    assert gmail._with_backoff(lambda: "ok", what="get") == "ok"
    assert no_sleep == []


def test_a_non_http_error_is_not_retried(no_sleep):
    """A bug in our own code must not be turned into a slow bug."""
    with pytest.raises(ValueError):
        gmail._with_backoff(_flaky([ValueError("mine")]), what="get")
    assert no_sleep == []

"""LiveGmailClient against a fake of the googleapiclient resource chain.

FakeGmailApi fakes `service.users().threads().list(...).execute()` and friends,
so every test here runs with no network and no credentials. It is the piece
that makes the live client testable at all.
"""
import base64
import pytest

from inbox_agent.gmail import LiveGmailClient


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def message(mid, *, sender, subject, to, date, snippet, body, label_ids):
    return {
        "id": mid,
        "snippet": snippet,
        "labelIds": list(label_ids),
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": sender},
                {"name": "Subject", "value": subject},
                {"name": "To", "value": to},
                {"name": "Date", "value": date},
                {"name": "Message-ID", "value": f"<{mid}@mail>"},
            ],
            "body": {"data": b64(body)},
        },
    }


class _Exec:
    def __init__(self, value):
        self._value = value

    def execute(self, http=None):
        return self._value


class FakeThreads:
    def __init__(self, api):
        self._api = api

    def list(self, userId="me", q="", maxResults=50):
        self._api.queries.append({"q": q, "maxResults": maxResults})
        ids = list(self._api.order)[:maxResults]
        return _Exec({"threads": [{"id": i} for i in ids]})

    def get(self, userId="me", id=None, format="full"):
        self._api.gets.append(id)
        if id not in self._api.threads:
            raise KeyError(id)
        return _Exec({"id": id, "messages": self._api.threads[id]})

    def modify(self, userId="me", id=None, body=None):
        self._api.modifies.append({"id": id, "body": body})
        for msg in self._api.threads[id]:
            labels = set(msg["labelIds"])
            labels |= set((body or {}).get("addLabelIds", []))
            labels -= set((body or {}).get("removeLabelIds", []))
            msg["labelIds"] = sorted(labels)
        return _Exec({"id": id})

    def trash(self, userId="me", id=None):
        self._api.trashed.append(id)
        for msg in self._api.threads[id]:
            labels = set(msg["labelIds"])
            labels.discard("INBOX")
            labels.add("TRASH")
            msg["labelIds"] = sorted(labels)
        return _Exec({"id": id})


class FakeDrafts:
    def __init__(self, api):
        self._api = api

    def create(self, userId="me", body=None):
        self._api.drafts.append(body)
        return _Exec({"id": f"draft_{len(self._api.drafts)}"})


class FakeLabelsResource:
    def __init__(self, api):
        self._api = api

    def list(self, userId="me"):
        self._api.label_list_calls += 1
        return _Exec({"labels": list(self._api.labels)})

    def create(self, userId="me", body=None):
        new = {"id": f"Label_new_{len(self._api.labels)}", "name": body["name"]}
        self._api.labels.append(new)
        self._api.created_labels.append(body["name"])
        return _Exec(new)


class FakeUsers:
    def __init__(self, api):
        self._api = api

    def threads(self):
        return FakeThreads(self._api)

    def drafts(self):
        return FakeDrafts(self._api)

    def labels(self):
        return FakeLabelsResource(self._api)


class FakeGmailApi:
    """Stands in for the object googleapiclient's build() returns."""

    def __init__(self, threads=None, labels=None):
        self.threads = threads or {}
        self.order = list(self.threads)
        self.labels = labels if labels is not None else [
            {"id": "INBOX", "name": "INBOX"},
            {"id": "UNREAD", "name": "UNREAD"},
            {"id": "TRASH", "name": "TRASH"},
            {"id": "Label_1", "name": "Notes"},
            {"id": "Label_6111317184412779502", "name": "Education/AI"},
        ]
        self.queries, self.gets, self.modifies = [], [], []
        self.trashed, self.drafts, self.created_labels = [], [], []
        self.label_list_calls = 0

    def users(self):
        return FakeUsers(self)


@pytest.fixture
def api():
    return FakeGmailApi(threads={
        "t1": [message("m1", sender="deals@shop.com", subject="Sale 50%",
                       to="me@z.com", date="Wed, 26 Aug 2026 10:00:00 +0000",
                       snippet="big sale", body="Everything half price",
                       label_ids=["INBOX", "UNREAD"])],
        "t2": [message("m2", sender="boss@work.com", subject="Re: budget",
                       to="me@z.com", date="Wed, 26 Aug 2026 11:00:00 +0000",
                       snippet="thoughts?", body="What do you think?",
                       label_ids=["INBOX", "Label_1"]),
               message("m3", sender="boss@work.com", subject="Re: budget",
                       to="me@z.com", date="Wed, 26 Aug 2026 12:00:00 +0000",
                       snippet="bump", body="bump",
                       label_ids=["INBOX", "UNREAD"])],
    })


@pytest.fixture
def client(api):
    return LiveGmailClient(api)


# --- reads ------------------------------------------------------------------

def test_query_is_passed_to_gmail_verbatim(client, api):
    """matches_query is never called live. Gmail's q accepts display names for
    label:, verified against the real mailbox with -label:Education/AI, so
    settings.inbox_query needs no translation layer."""
    client.list_threads(limit=10, query="in:inbox is:unread -label:agent/triaged")
    assert api.queries[0]["q"] == "in:inbox is:unread -label:agent/triaged"
    assert api.queries[0]["maxResults"] == 10


def test_list_threads_hydrates_each_id(client, api):
    threads = client.list_threads(limit=10)
    assert {t.id for t in threads} == {"t1", "t2"}
    assert sorted(api.gets) == ["t1", "t2"]


def test_list_threads_preserves_gmail_ordering(client):
    """Gmail returns newest first and the digest renders in that order, so
    ordering is contract, not an accident of thread scheduling."""
    assert [t.id for t in client.list_threads(limit=10)] == ["t1", "t2"]


def test_headers_come_from_the_first_message(client):
    t = client.get_thread("t2")
    assert t.sender == "boss@work.com"
    assert t.subject == "Re: budget"
    assert t.to == ["me@z.com"]


def test_label_ids_are_names_not_ids(client):
    """The contract at gmail.py's matches_query comment. A live run and a
    snapshot test must be asserting the same thing."""
    t = client.get_thread("t2")
    assert "Notes" in t.label_ids
    assert "Label_1" not in t.label_ids


def test_labels_are_unioned_across_messages(client):
    """A thread is UNREAD if any message in it is unread."""
    t = client.get_thread("t2")
    assert "UNREAD" in t.label_ids       # only on m3
    assert "Notes" in t.label_ids        # only on m2


def test_body_is_populated_from_the_payload(client):
    assert client.get_thread("t1").body == "Everything half price"


def test_snippet_is_populated(client):
    assert client.get_thread("t1").snippet == "big sale"


def test_date_is_normalised_to_iso8601(client):
    """Gmail returns RFC 2822. recency.py does date arithmetic on this field
    and the snapshot stores ISO, so the two clients must agree."""
    assert client.get_thread("t1").date.startswith("2026-08-26T10:00:00")


def test_the_label_map_is_built_once_for_a_whole_page(client, api):
    """Five workers racing to build it would issue five labels.list calls."""
    client.list_threads(limit=10)
    assert api.label_list_calls == 1


def test_empty_result_is_an_empty_list_not_an_error(api):
    api.threads, api.order = {}, []
    assert LiveGmailClient(api).list_threads(limit=10) == []


# --- writes -----------------------------------------------------------------

def test_archive_removes_inbox_by_id(client, api):
    client.archive("t1")
    assert api.modifies[0]["body"] == {"removeLabelIds": ["INBOX"]}
    assert "INBOX" not in client.get_thread("t1").label_ids


def test_apply_label_resolves_the_name_to_an_id(client, api):
    client.apply_label("t1", "Notes")
    assert api.modifies[0]["body"] == {"addLabelIds": ["Label_1"]}


def test_apply_label_creates_a_missing_label(client, api):
    """agent/triaged does not exist in the real mailbox - verified."""
    client.apply_label("t1", "agent/triaged")
    assert api.created_labels == ["agent/triaged"]
    assert "agent/triaged" in client.get_thread("t1").label_ids


def test_remove_label_resolves_the_name_to_an_id(client, api):
    client.remove_label("t2", "Notes")
    assert api.modifies[0]["body"] == {"removeLabelIds": ["Label_1"]}


def test_trash_uses_the_trash_endpoint_not_label_manipulation(client, api):
    """threads.trash() is what untrash() reverses, and what gives the owner the
    30-day recovery window. Adding a TRASH label is a different operation."""
    client.trash("t1")
    assert api.trashed == ["t1"]
    assert api.modifies == []


def test_trash_removes_the_thread_from_inbox(client):
    client.trash("t1")
    assert "INBOX" not in client.get_thread("t1").label_ids


def test_create_draft_is_attached_to_the_thread(client, api):
    client.create_draft("t1", "my reply")
    assert api.drafts[0]["message"]["threadId"] == "t1"


def test_create_draft_sets_reply_headers(client, api):
    """threadId alone files it in the right Gmail conversation; In-Reply-To and
    References are what make other mail clients thread it."""
    client.create_draft("t2", "sure")
    raw = base64.urlsafe_b64decode(api.drafts[0]["message"]["raw"]).decode()
    assert "In-Reply-To: <m3@mail>" in raw     # the LAST message, not the first
    assert "References: <m3@mail>" in raw
    assert "To: boss@work.com" in raw
    assert "\r\n\r\nsure" in raw


def test_a_draft_subject_is_not_double_prefixed(client, api):
    """t2's subject is already 'Re: budget'."""
    client.create_draft("t2", "ok")
    raw = base64.urlsafe_b64decode(api.drafts[0]["message"]["raw"]).decode()
    assert "Subject: Re: budget" in raw
    assert "Re: Re:" not in raw


def test_mutations_are_not_marked_simulated(client):
    """SnapshotGmailClient stamps every write {'simulated': True} so a snapshot
    write can never be mistaken for a real one. The live client must not."""
    assert client.archive("t1").get("simulated") is not True


# --- retry ------------------------------------------------------------------
# Not in the plan. Retry logic that is never exercised is where the bugs live,
# and the cost of getting it wrong is asymmetric: retrying a 4xx hides our own
# bug behind a delay, and NOT retrying a 429 fails a run that would have
# succeeded a second later.

class _Resp:
    def __init__(self, status):
        self.status = status


class _Httpish(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.resp = _Resp(status)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("inbox_agent.gmail.time.sleep", lambda s: None)


def _flaky(api, statuses):
    """Make threads.list fail with each status in turn, then succeed."""
    seq = list(statuses)
    real = FakeThreads(api).list

    class Threads(FakeThreads):
        def list(self, userId="me", q="", maxResults=50):
            if seq:
                raise _Httpish(seq.pop(0))
            return real(userId=userId, q=q, maxResults=maxResults)

    class Users(FakeUsers):
        def threads(self):
            return Threads(api)

    api.users = lambda: Users(api)
    return api


def test_a_429_is_retried_and_then_succeeds(api):
    client = LiveGmailClient(_flaky(api, [429, 429]))
    assert [t.id for t in client.list_threads(limit=10)] == ["t1", "t2"]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_are_retried(api, status):
    client = LiveGmailClient(_flaky(api, [status]))
    assert len(client.list_threads(limit=10)) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retried(api, status):
    """A 4xx is a bug in our request - a bad label id, a malformed query.
    Retrying makes the same mistake more slowly while hiding it."""
    client = LiveGmailClient(_flaky(api, [status] * 9))
    with pytest.raises(_Httpish):
        client.list_threads(limit=10)


def test_persistent_retryable_errors_eventually_raise(api):
    """Never swallowed. A silent failure against a real mailbox reports success
    on work that was never done."""
    client = LiveGmailClient(_flaky(api, [503] * 9))
    with pytest.raises(_Httpish):
        client.list_threads(limit=10)


# --- per-thread transport ---------------------------------------------------
# The bug the fakes could not have caught, and the reason this section exists.
# googleapiclient's service holds ONE httplib2.Http, which is not thread-safe.
# Hydrating through a pool that shares it corrupts the SSL socket state and
# surfaces as `ssl.SSLError: WRONG_VERSION_NUMBER` - an error naming nothing
# about threads. Reproduced against the real mailbox: 1 worker always fine,
# 5 workers always failed.

def test_each_thread_gets_its_own_transport(api):
    import threading

    made = []

    def factory():
        h = object()
        made.append(h)
        return h

    client = LiveGmailClient(api, http_factory=factory)
    seen, lock = [], threading.Lock()

    def grab():
        with lock:
            seen.append(client._http())

    ws = [threading.Thread(target=grab) for _ in range(4)]
    for w in ws:
        w.start()
    for w in ws:
        w.join()

    assert len(made) == 4, "one transport per thread, not one shared"
    assert len(set(id(h) for h in seen)) == 4


def test_a_transport_is_reused_within_one_thread(api):
    """Per-thread, not per-call: a new Http per request would drop connection
    pooling and make every call pay a fresh TLS handshake."""
    calls = []
    client = LiveGmailClient(api, http_factory=lambda: calls.append(1) or object())
    first = client._http()
    assert client._http() is first
    assert len(calls) == 1


def test_no_factory_means_the_service_transport(api):
    """HttpRequest.execute falls back to its own http when passed None, which
    is what lets every fake in this file ignore transports entirely."""
    assert LiveGmailClient(api)._http() is None


def test_the_transport_reaches_the_api_call(api):
    """Wiring test: a factory that is never threaded through every execute()
    would leave the bug in place while looking correct."""
    sentinel = object()
    got = []

    class Threads(FakeThreads):
        def list(self, userId="me", q="", maxResults=50):
            outer = self

            class Rec:
                def execute(self, http=None):
                    got.append(http)
                    return {"threads": []}
            return Rec()

    class Users(FakeUsers):
        def threads(self):
            return Threads(api)

    api.users = lambda: Users(api)
    LiveGmailClient(api, http_factory=lambda: sentinel).list_threads(limit=5)
    assert got == [sentinel]

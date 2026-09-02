"""Gmail's modifyThread takes label IDs. Thread.label_ids holds display names.

This map is the only thing that reconciles the two, and it is why a snapshot
test asserting on 'agent/triaged' is evidence about a live run (gmail.py:17-21).
"""
import threading
import time

from inbox_agent.gmail import _LabelMap


class _Exec:
    def __init__(self, value):
        self._value = value

    def execute(self):
        return self._value


class FakeLabels:
    """The users().labels() half of the Gmail service."""

    def __init__(self, labels, list_delay=0.0):
        self._labels = list(labels)
        self.created = []
        self.list_calls = 0
        self._list_delay = list_delay
        self._lock = threading.Lock()

    def list(self, userId="me"):
        with self._lock:
            self.list_calls += 1
        if self._list_delay:
            time.sleep(self._list_delay)
        return _Exec({"labels": list(self._labels)})

    def create(self, userId="me", body=None):
        new = {"id": f"Label_new_{len(self.created)}", "name": body["name"]}
        self.created.append(body["name"])
        self._labels.append(new)
        return _Exec(new)


class FakeService:
    def __init__(self, labels):
        self._labels = labels

    def users(self):
        return self

    def labels(self):
        return self._labels


# Both ID shapes are live in one real account, so no heuristic on ID shape is
# safe and the map must come from labels.list. Taken from the real mailbox.
REAL_LABELS = [
    {"id": "INBOX", "name": "INBOX", "type": "system"},
    {"id": "UNREAD", "name": "UNREAD", "type": "system"},
    {"id": "Label_1", "name": "Notes", "type": "user"},
    {"id": "Label_5", "name": "Property Listings", "type": "user"},
    {"id": "Label_6111317184412779502", "name": "Education/AI", "type": "user"},
    {"id": "Label_7181901001278056114", "name": "Learning", "type": "user"},
]


def _map(labels=None, **kw):
    fake = FakeLabels(labels if labels is not None else REAL_LABELS, **kw)
    return _LabelMap(FakeService(fake)), fake


def test_system_label_ids_are_their_own_names():
    m, _ = _map()
    assert m.to_name("INBOX") == "INBOX"
    assert m.to_id("INBOX") == "INBOX"


def test_short_and_long_user_ids_both_resolve_to_names():
    """Label_1 and Label_6111317184412779502 coexist in one real account."""
    m, _ = _map()
    assert m.to_name("Label_1") == "Notes"
    assert m.to_name("Label_6111317184412779502") == "Education/AI"


def test_names_resolve_back_to_ids():
    m, _ = _map()
    assert m.to_id("Notes") == "Label_1"
    assert m.to_id("Education/AI") == "Label_6111317184412779502"


def test_a_spaced_name_round_trips_to_the_name_not_the_id():
    """Six real labels contain spaces. They can never be a triaged label
    (matches_query splits on whitespace) but they must still read correctly."""
    m, _ = _map()
    assert m.to_name("Label_5") == "Property Listings"
    assert m.to_id("Property Listings") == "Label_5"


def test_the_map_is_built_once_not_per_lookup():
    m, fake = _map()
    m.to_name("INBOX")
    m.to_name("Label_1")
    m.to_id("Notes")
    assert fake.list_calls == 1


def test_unknown_id_refetches_once_then_surfaces_the_raw_id():
    """A dropped label is silent data loss into the classifier's
    'Current labels:' line, so an unresolvable id is surfaced, never dropped."""
    m, fake = _map()
    assert m.to_name("Label_999") == "Label_999"
    assert fake.list_calls == 2  # built once, refetched once on the miss


def test_unknown_id_is_found_after_a_refetch_when_it_exists():
    """A label created in Gmail since the map was built must resolve."""
    m, fake = _map()
    m.to_name("INBOX")  # force the initial build
    fake._labels.append({"id": "Label_42", "name": "Fresh", "type": "user"})
    assert m.to_name("Label_42") == "Fresh"


def test_unknown_name_is_created_rather_than_failing():
    """agent/triaged does not exist in the real mailbox - confirmed against it.
    mark_triaged needs it on the first live run, for every thread processed."""
    m, fake = _map()
    new_id = m.to_id("agent/triaged")
    assert fake.created == ["agent/triaged"]
    assert m.to_name(new_id) == "agent/triaged"


def test_a_created_label_is_not_created_twice():
    m, fake = _map()
    assert m.to_id("agent/triaged") == m.to_id("agent/triaged")
    assert fake.created == ["agent/triaged"]


def test_unknown_name_is_found_after_a_refetch_without_being_created():
    """Created in Gmail (or by another process) since the map was built.
    Creating a duplicate would leave two labels sharing one name."""
    m, fake = _map()
    m.to_id("Notes")  # force the initial build
    fake._labels.append({"id": "Label_77", "name": "agent/triaged"})
    assert m.to_id("agent/triaged") == "Label_77"
    assert fake.created == []


# --- concurrency ------------------------------------------------------------
# Not in the plan. Task 5 hydrates threads through a ThreadPoolExecutor(5), and
# every one of those calls to_name. A refresh that cleared the dicts before
# repopulating them would let a concurrent reader observe a half-built map and
# surface a raw id where a name was available - an intermittent wrong label in
# the digest, which is close to undiagnosable after the fact.

def test_concurrent_readers_never_observe_a_half_built_map():
    m, fake = _map(list_delay=0.01)
    m._ensure()
    errors, barrier = [], threading.Barrier(6)

    def read():
        barrier.wait()
        for _ in range(40):
            if m.to_name("Label_5") != "Property Listings":
                errors.append("saw a partially built map")

    def churn():
        barrier.wait()
        for _ in range(5):
            m.refresh()

    workers = [threading.Thread(target=read) for _ in range(5)]
    workers.append(threading.Thread(target=churn))
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert errors == []


def test_a_concurrent_miss_creates_the_label_only_once():
    """Two threads both missing agent/triaged must not create it twice, which
    would leave two labels with one name and a to_name that depends on dict
    ordering."""
    m, fake = _map(list_delay=0.01)
    m._ensure()
    barrier = threading.Barrier(4)
    got = []

    def resolve():
        barrier.wait()
        got.append(m.to_id("agent/triaged"))

    ws = [threading.Thread(target=resolve) for _ in range(4)]
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    assert fake.created == ["agent/triaged"], fake.created
    assert len(set(got)) == 1, got

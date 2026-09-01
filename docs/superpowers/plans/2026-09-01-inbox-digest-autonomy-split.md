# Inbox Digest — Autonomy Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent act on its own for the reversible, confident majority of
mail, hold only what genuinely needs the owner, and report both in a digest built
for a phone.

**Architecture:** A `partition` node splits proposals into an auto tier that
executes immediately and a held tier that goes to a persistent queue outliving
any single run. The graph's `interrupt()` survives on a conditional edge used
only by `/backlog` (a later plan). The digest renders from the queue plus the
run's audit records, not from a parked checkpoint.

**Tech Stack:** Python 3.11+, LangGraph (StateGraph, InMemoryStore, checkpointer),
Pydantic v2, pytest, raw Telegram Bot API over httpx.

**Spec:** `docs/superpowers/specs/2026-09-01-inbox-digest-design.md`

## Global Constraints

- Telegram hard caps message text at **4096 characters** and `callback_data` at
  **64 bytes**. Protocol, not preference.
- Telegram renders **proportional text and wraps it**. Padded columns
  (`{:<24}`) do not align — they produce a wall. Never pad. A terminal preview
  is not a fair test of a phone layout.
- `LOW_CONFIDENCE = 0.5` (`inbox_agent/render.py:14`). Reuse it; do not
  redefine the threshold.
- Hold precedence, first match wins: **`trash` → `low_confidence` →
  `needs_reply` → `security_alert`**.
- **Trash proposed by a learned rule auto-executes**; trash from the model is
  always held.
- Held items page at **8 per digest page**.
- The fetch query is exactly `in:inbox is:unread -label:agent/triaged`.
- The action chokepoint (`execute_action`), the deny-list, and `INBOX_DRY_RUN`
  are unchanged by this plan. Nothing here grants a new capability.
- Existing tests must keep passing. Run the full suite (`pytest -q`) before
  every commit, not just the new test.

---

### Task 1: `list_threads` gains a query

`GmailClient.list_threads(limit)` can only say "the most recent N". Unread
filtering needs a query the live Gmail API can honour natively, and the snapshot
client has to honour the *same string* — otherwise a passing snapshot test is no
evidence about live behaviour.

**Files:**
- Modify: `inbox_agent/gmail.py:17-24` (protocol), `inbox_agent/gmail.py:48-49`
  (snapshot implementation)
- Test: `tests/test_gmail.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `matches_query(thread: Thread, query: str) -> bool` and
  `GmailClient.list_threads(limit: int = 50, query: str = "") -> list[Thread]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_gmail.py`:

```python
from inbox_agent.gmail import SnapshotGmailClient, matches_query
from inbox_agent.models import Thread
import json
import pytest


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_gmail.py -v`
Expected: FAIL with `ImportError: cannot import name 'matches_query'`

- [ ] **Step 3: Implement**

In `inbox_agent/gmail.py`, add after the imports:

```python
# The subset of Gmail search syntax the snapshot client can honour. Kept
# deliberately tiny: the live client hands `query` to the Gmail API verbatim and
# never calls this function at all. It exists so that both clients answer the
# SAME query string, which is the only thing that makes a snapshot test evidence
# about live behaviour.
def matches_query(thread: Thread, query: str) -> bool:
    """True if `thread` satisfies every term in `query`.

    Raises on a term this parser does not implement, rather than ignoring it. An
    ignored term would pass live (Gmail understands it) and silently do nothing
    here, turning every snapshot test for that term into a false negative.
    """
    labels = {label.upper() for label in thread.label_ids}
    for term in query.split():
        negate = term.startswith("-")
        bare = term[1:] if negate else term
        if bare == "is:unread":
            present = "UNREAD" in labels
        elif bare == "in:inbox":
            present = "INBOX" in labels
        elif bare.startswith("label:"):
            present = bare[len("label:"):].upper() in labels
        else:
            raise ValueError(
                f"{bare!r} is not a query term the snapshot client implements")
        if present == negate:
            return False
    return True
```

Change the protocol at `inbox_agent/gmail.py:18`:

```python
    def list_threads(self, limit: int = 50, query: str = "") -> list[Thread]: ...
```

Replace `SnapshotGmailClient.list_threads`:

```python
    def list_threads(self, limit: int = 50, query: str = "") -> list[Thread]:
        # Filter BEFORE limiting. Limiting first would mean "the 50 newest, of
        # which some are unread", so an inbox with a long read run reads as empty.
        threads = (self._threads[i] for i in self._order)
        if query:
            threads = (t for t in threads if matches_query(t, query))
        out = []
        for thread in threads:
            out.append(thread)
            if len(out) >= limit:
                break
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_gmail.py -v && pytest -q`
Expected: PASS, and the whole suite still green (the new parameter is keyword-defaulted, so every existing call site is unaffected).

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/gmail.py tests/test_gmail.py
git commit -m "Widen GmailClient.list_threads with a query, honoured by the snapshot

The protocol could only say 'the most recent N', which cannot express unread.
Widening it before LiveGmailClient is written is materially cheaper than after.

matches_query implements only is:unread, in:inbox and label:/-label:, and
RAISES on anything else. An ignored term would pass live and silently do
nothing on the snapshot, which would make every snapshot test for that term a
false negative. Filtering happens before the limit, so limit=20 means twenty
matching threads rather than twenty candidates then filtered."
```

---

### Task 2: The partition rule

The autonomy ladder decides which items the agent may act on alone. This is that
decision as a pure function, so it is testable without a graph.

**Files:**
- Create: `inbox_agent/partition.py`
- Test: `tests/test_partition.py`

**Interfaces:**
- Consumes: `ReviewItem` (`inbox_agent/models.py:93`), `LOW_CONFIDENCE`
  (`inbox_agent/render.py:14`).
- Produces:
  - `HoldReason = Literal["trash", "low_confidence", "needs_reply", "security_alert"]`
  - `hold_reason(item: ReviewItem) -> Optional[HoldReason]`
  - `partition(items: list[ReviewItem]) -> tuple[list[ReviewItem], list[tuple[ReviewItem, HoldReason]]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_partition.py`:

```python
"""The autonomy ladder as code. Pure, so it needs no graph and no LLM."""
from inbox_agent.models import Action, ReviewItem
from inbox_agent.partition import hold_reason, partition


def item(kind="archive", *, category="promotion", conf=0.9, source="model",
         rule_id=None, tid="t1"):
    return ReviewItem(
        thread_id=tid, category=category, subject="S", sender="a@b.com",
        snippet="s", proposed=[Action(kind=kind, thread_id=tid)],
        reason="because", confidence=conf, source=source, rule_id=rule_id)


def test_reversible_confident_actions_are_not_held():
    for kind in ("label", "archive", "draft", "none"):
        assert hold_reason(item(kind)) is None


def test_model_proposed_trash_is_held():
    assert hold_reason(item("trash")) == "trash"


def test_rule_authorised_trash_is_not_held():
    """The ladder's 'learned rule, else human approval' row. Without this
    clause trash can never graduate, and the row never does anything."""
    assert hold_reason(item("trash", source="rule", rule_id="r-1")) is None


def test_low_confidence_is_held_even_for_a_reversible_action():
    assert hold_reason(item("archive", conf=0.41)) == "low_confidence"


def test_confidence_exactly_at_the_threshold_is_not_low():
    """LOW_CONFIDENCE is 0.5 and the comparison is strict, matching render.py."""
    assert hold_reason(item("archive", conf=0.5)) is None


def test_needs_reply_is_held_for_attention():
    assert hold_reason(item("draft", category="needs_reply")) == "needs_reply"


def test_security_alert_is_held_for_attention():
    assert hold_reason(item("label", category="security_alert")) == "security_alert"


def test_trash_outranks_low_confidence():
    assert hold_reason(item("trash", conf=0.2)) == "trash"


def test_low_confidence_outranks_needs_reply():
    """A low-confidence needs_reply must never be one-tap approvable: the
    one-tap button covers the attention tier only."""
    assert hold_reason(item("draft", category="needs_reply", conf=0.2)) == "low_confidence"


def test_an_item_with_several_actions_is_held_if_any_is_trash():
    multi = ReviewItem(
        thread_id="t9", category="promotion", subject="S", sender="a@b.com",
        snippet="s",
        proposed=[Action(kind="label", thread_id="t9"),
                  Action(kind="trash", thread_id="t9")],
        reason="r", confidence=0.9, source="model")
    assert hold_reason(multi) == "trash"


def test_partition_splits_and_preserves_order():
    items = [item(tid="t0"), item("trash", tid="t1"), item(tid="t2"),
             item("archive", conf=0.1, tid="t3")]
    auto, held = partition(items)
    assert [i.thread_id for i in auto] == ["t0", "t2"]
    assert [(i.thread_id, r) for i, r in held] == [("t1", "trash"),
                                                   ("t3", "low_confidence")]


def test_partition_of_an_empty_list_is_two_empty_lists():
    assert partition([]) == ([], [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_partition.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inbox_agent.partition'`

- [ ] **Step 3: Implement**

Create `inbox_agent/partition.py`:

```python
"""The autonomy ladder as code (spec section 2.2).

The Stage A spec grants `label`, `archive` and `draft` "always" authority and
gates only `trash`. The implementation gated everything, which is a deviation
from our own design rather than a conservative reading of it: a gate that fires
on every item is rubber-stamped, and a rubber-stamped gate reviews nothing.

Pure by intent. Whether the agent may act alone is the highest-consequence
judgment in the system, so it is a function over a ReviewItem with no graph, no
store and no network anywhere near it.
"""
from __future__ import annotations

from typing import Literal, Optional

from .models import ReviewItem
from .render import LOW_CONFIDENCE

HoldReason = Literal["trash", "low_confidence", "needs_reply", "security_alert"]

# Held because the thread wants a person, not because the action is risky. The
# proposed action for these is typically a draft or a label - harmless - so they
# are the tier the one-tap approve button covers.
ATTENTION_CATEGORIES = ("needs_reply", "security_alert")


def hold_reason(item: ReviewItem) -> Optional[HoldReason]:
    """Why this item must wait for the owner, or None if the agent may act.

    Order is precedence, first match wins: trash, low_confidence, needs_reply,
    security_alert. Authorisation always outranks attention, so a low-confidence
    needs_reply is held as low_confidence and never becomes one-tap approvable.
    """
    kinds = {action.kind for action in item.proposed}
    if "trash" in kinds:
        # "trash: learned rule, else human approval at interrupt". A rule the
        # owner taught IS the authorisation; without this clause trash can never
        # graduate and that row of the ladder never does anything.
        if not (item.source == "rule" and item.rule_id):
            return "trash"
    if item.confidence < LOW_CONFIDENCE:
        return "low_confidence"
    if item.category in ATTENTION_CATEGORIES:
        return item.category  # type: ignore[return-value]
    return None


def partition(
    items: list[ReviewItem],
) -> tuple[list[ReviewItem], list[tuple[ReviewItem, HoldReason]]]:
    """Split into (act now, wait for the owner), preserving input order."""
    auto: list[ReviewItem] = []
    held: list[tuple[ReviewItem, HoldReason]] = []
    for item in items:
        reason = hold_reason(item)
        if reason is None:
            auto.append(item)
        else:
            held.append((item, reason))
    return auto, held
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_partition.py -v && pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/partition.py tests/test_partition.py
git commit -m "Add the partition rule: the autonomy ladder as a pure function

propose() put every decision into the review list, so the spec's ladder -
label/archive/draft at 'always' authority - was implemented nowhere. This is
that ladder, as a function over a ReviewItem with no graph or store near it.

Precedence is trash, low_confidence, needs_reply, security_alert. Authorisation
outranks attention so a low-confidence needs_reply is never one-tap approvable.
Rule-authorised trash auto-executes, which is the clause that lets trash
graduate."
```

---

### Task 3: The held queue

Held items outlive the run that produced them, so they cannot live in a
checkpoint. This is the store that makes carryover possible.

**Files:**
- Modify: `inbox_agent/models.py` (add `HeldItem` after `ReviewRequest`)
- Modify: `inbox_agent/store.py` (add `HELD_NS` and `HeldQueue`)
- Test: `tests/test_held_queue.py`

**Interfaces:**
- Consumes: `ReviewItem`, `HoldReason`, a LangGraph `BaseStore` from
  `build_store()` (`inbox_agent/store.py:32`).
- Produces:
  - `HeldItem` with fields `thread_id: str`, `run_id: str`,
    `first_held_at: datetime`, `hold_reason: str`, `item: ReviewItem`
  - `HeldQueue(store)` with `add(item, *, run_id, reason, now=None) -> HeldItem`,
    `all() -> list[HeldItem]`, `get(thread_id) -> Optional[HeldItem]`,
    `remove(thread_id) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_held_queue.py`:

```python
"""The queue that makes held items outlive the run that produced them."""
from datetime import datetime, timedelta, timezone

from inbox_agent.models import Action, HeldItem, ReviewItem
from inbox_agent.store import HeldQueue, build_store

T0 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def item(tid="t1", conf=0.9):
    return ReviewItem(
        thread_id=tid, category="promotion", subject=f"Subject {tid}",
        sender="a@b.com", snippet="s",
        proposed=[Action(kind="trash", thread_id=tid)],
        reason="because", confidence=conf, source="model")


def queue():
    return HeldQueue(build_store())


def test_add_then_get_round_trips():
    q = queue()
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    got = q.get("t1")
    assert got is not None
    assert got.thread_id == "t1"
    assert got.hold_reason == "trash"
    assert got.run_id == "r1"
    assert got.first_held_at == T0
    assert got.item.subject == "Subject t1"


def test_get_returns_none_for_an_unknown_thread():
    assert queue().get("nope") is None


def test_all_is_ordered_oldest_first():
    q = queue()
    q.add(item("t2"), run_id="r2", reason="trash", now=T0 + timedelta(hours=10))
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    assert [h.thread_id for h in q.all()] == ["t1", "t2"]


def test_re_adding_a_held_thread_keeps_the_ORIGINAL_first_held_at():
    """This is what makes 'waiting since Tue 8:00' true.

    An item held at 8am and still held at 6pm is one item that has been waiting
    ten hours, not a fresh one. Overwriting first_held_at would reset the age on
    every run and the queue would never look old, which is the entire signal.
    """
    q = queue()
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    q.add(item("t1"), run_id="r2", reason="trash", now=T0 + timedelta(hours=10))
    assert q.get("t1").first_held_at == T0
    assert len(q.all()) == 1


def test_re_adding_refreshes_the_item_and_the_reason():
    """The age is sticky; the content is not. A re-classified thread should
    show its current proposal and current reason for being held."""
    q = queue()
    q.add(item("t1", conf=0.9), run_id="r1", reason="trash", now=T0)
    q.add(item("t1", conf=0.2), run_id="r2", reason="low_confidence",
          now=T0 + timedelta(hours=10))
    got = q.get("t1")
    assert got.hold_reason == "low_confidence"
    assert got.item.confidence == 0.2
    assert got.run_id == "r2"


def test_remove_takes_it_out_of_the_queue():
    q = queue()
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    q.remove("t1")
    assert q.get("t1") is None
    assert q.all() == []


def test_removing_an_absent_thread_is_a_no_op():
    queue().remove("never-held")


def test_all_reads_past_the_default_search_page():
    """BaseStore.search() defaults to limit=10. The same truncation trap that
    rules() had to paginate around applies here."""
    q = queue()
    for i in range(25):
        q.add(item(f"t{i:02d}"), run_id="r1", reason="trash",
              now=T0 + timedelta(minutes=i))
    assert len(q.all()) == 25


def test_held_item_is_json_serialisable():
    """It crosses the same boundary ReviewRequest does."""
    q = queue()
    q.add(item("t1"), run_id="r1", reason="trash", now=T0)
    dumped = q.get("t1").model_dump(mode="json")
    assert HeldItem.model_validate(dumped).thread_id == "t1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_held_queue.py -v`
Expected: FAIL with `ImportError: cannot import name 'HeldItem'`

- [ ] **Step 3: Implement**

In `inbox_agent/models.py`, add immediately after the `ReviewRequest` class:

```python
class HeldItem(BaseModel):
    """One proposal waiting on the owner, persisted outside any single run.

    `first_held_at` is what lets the digest say "waiting since Tue 8:00". It is
    set once and never refreshed, so an item held this morning and still held
    this evening reads as ten hours old rather than brand new - the ageing IS
    the pressure to deal with the queue.

    Embeds the whole ReviewItem rather than flattening its fields: ReviewItem is
    already the renderer's contract, and re-declaring it here would give the
    digest two shapes to render instead of one.
    """
    thread_id: str
    run_id: str
    first_held_at: datetime
    hold_reason: str
    item: ReviewItem
```

In `inbox_agent/store.py`, add next to the other namespaces (after line 21):

```python
HELD_NS = ("held", "items")
```

Add `HeldItem` to the models import at the top of `store.py`:

```python
from .models import ActionKind, HeldItem, Rule, Thread
```

Append the `HeldQueue` class to `inbox_agent/store.py`:

```python
class HeldQueue:
    """Proposals waiting on the owner, across runs.

    Separate from PreferenceStore because the lifetimes differ: a rule is
    permanent knowledge, a held item is a piece of work in flight. Same backing
    BaseStore, different namespace, so there is still exactly one thing to
    persist later.
    """

    def __init__(self, store):
        self._store = store

    def add(self, item: ReviewItem, *, run_id: str, reason: str,
            now: Optional[datetime] = None) -> HeldItem:
        """Hold `item`, preserving the original wait time if already held.

        Idempotent on thread_id: a thread the agent holds twice is one item that
        has been waiting since the first time, not two items. The content and
        the reason ARE refreshed, so a re-classified thread shows its current
        proposal.
        """
        existing = self.get(item.thread_id)
        held = HeldItem(
            thread_id=item.thread_id,
            run_id=run_id,
            first_held_at=existing.first_held_at if existing
            else (now or datetime.now(timezone.utc)),
            hold_reason=reason,
            item=item,
        )
        self._store.put(HELD_NS, held.thread_id,
                        {"held": held.model_dump(mode="json")})
        return held

    def get(self, thread_id: str) -> Optional[HeldItem]:
        entry = self._store.get(HELD_NS, thread_id)
        return HeldItem.model_validate(entry.value["held"]) if entry else None

    def remove(self, thread_id: str) -> None:
        """Absent is not an error: a double-tap must not raise at the transport."""
        self._store.delete(HELD_NS, thread_id)

    def all(self) -> list[HeldItem]:
        """Everything held, oldest first.

        Paginates for the same reason rules() does: BaseStore.search() defaults
        to limit=10, and a silently truncated queue would hide work the owner is
        waiting to do - the exact invisible failure this system is built against.
        """
        out: list[HeldItem] = []
        offset = 0
        while True:
            page = self._store.search(HELD_NS, limit=_SEARCH_PAGE_SIZE, offset=offset)
            out.extend(HeldItem.model_validate(entry.value["held"]) for entry in page)
            if len(page) < _SEARCH_PAGE_SIZE:
                break
            offset += _SEARCH_PAGE_SIZE
        return sorted(out, key=lambda h: h.first_held_at)
```

Add the `ReviewItem` import to `store.py`'s models import line:

```python
from .models import ActionKind, HeldItem, ReviewItem, Rule, Thread
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_held_queue.py -v && pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/models.py inbox_agent/store.py tests/test_held_queue.py
git commit -m "Add HeldItem and HeldQueue: work in flight, persisted across runs

Held items outlive the run that produced them, so they cannot live in a
checkpoint. add() is idempotent on thread_id and keeps the ORIGINAL
first_held_at, which is what makes 'waiting since Tue 8:00' true - refreshing
it would reset the age on every run and the queue would never look old, which
is the whole signal.

all() paginates for the same reason rules() does: search() defaults to
limit=10, and a silently truncated queue hides work the owner is waiting on."
```

---

### Task 4: Rewire the graph around the partition

The graph gains three nodes and a conditional edge. `interrupt()` stays on the
`/backlog` branch, which no caller uses yet — this task builds the fork and
proves both sides of it.

**Files:**
- Modify: `inbox_agent/graph.py:35-56` (`TriageState`), `inbox_agent/graph.py:120-288`
  (`build_graph`)
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `partition()` (Task 2), `HeldQueue` (Task 3).
- Produces:
  - `build_graph(..., held: HeldQueue)` — a new **required** keyword argument.
  - `TriageState` gains `mode: str` (`"incremental"` default, or `"backlog"`),
    `auto: list[dict]`, `held: list[dict]`.
  - Graph result gains `executed` on the incremental path with no resume needed.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph.py`. Note the `wiring` fixture must gain a `held` key —
update it first:

```python
from inbox_agent.store import HeldQueue, PreferenceStore, build_store

# In the existing `wiring` fixture, add to the returned dict:
#     held=HeldQueue(build_store()),
```

Then add the tests:

```python
def _snapshot(tmp_path, threads):
    p = tmp_path / "threads.json"
    p.write_text(json.dumps(threads))
    return p


def _row(tid, subject="Sale", sender="deals@shop.com"):
    return {"id": tid, "subject": subject, "sender": sender, "to": [],
            "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "",
            "label_ids": ["INBOX", "UNREAD"]}


def test_incremental_run_completes_without_an_interrupt(wiring):
    """Act-then-report: the run does not wait. It acts and finishes."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-a"}})
    assert "__interrupt__" not in result


def test_incremental_run_executes_the_auto_tier(wiring):
    """FakeLLM proposes archive at 0.9, which is 'always' authority."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-b"}})
    assert len(result["executed"]) == 1
    assert result["executed"][0]["action"] == "archive"


def test_incremental_run_holds_trash_instead_of_executing_it(tmp_path, wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    wiring = dict(wiring)
    wiring["llm"] = FakeLLM(ThreadJudgment(
        category="newsletter_noise", action="trash", label=None,
        reason="junk", confidence=0.9))
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-c"}})
    assert result["executed"] == []
    assert [h.hold_reason for h in wiring["held"].all()] == ["trash"]


def test_held_items_reach_the_queue_with_the_run_id(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    wiring = dict(wiring)
    wiring["llm"] = FakeLLM(ThreadJudgment(
        category="other", action="archive", label=None,
        reason="not sure", confidence=0.2))
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-d"}})
    queued = wiring["held"].all()
    assert len(queued) == 1
    assert queued[0].hold_reason == "low_confidence"
    assert queued[0].run_id


def test_two_runs_accumulate_held_items_rather_than_replacing_them(tmp_path):
    """Carryover. The evening digest must still show the morning's held items."""
    from langgraph.checkpoint.memory import InMemorySaver
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    snap = _snapshot(tmp_path, [_row("t1"), _row("t2")])
    held = HeldQueue(build_store())
    graph = build_graph(
        client=SnapshotGmailClient(snap), prefs=PreferenceStore(build_store()),
        policy=Policy(text="T", version="local:test", source="local"),
        llm=FakeLLM(ThreadJudgment(category="other", action="archive", label=None,
                                   reason="unsure", confidence=0.2)),
        settings=settings, log=AuditLog(settings.audit_log), held=held,
        checkpointer=InMemorySaver())

    graph.invoke({"limit": 1}, {"configurable": {"thread_id": "run-1"}})
    assert len(held.all()) == 1
    graph.invoke({"limit": 2}, {"configurable": {"thread_id": "run-2"}})
    assert len(held.all()) == 2


def test_backlog_mode_still_suspends_at_the_interrupt(wiring):
    """interrupt() survives, scoped to the one job that genuinely waits."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10, "mode": "backlog"},
                          {"configurable": {"thread_id": "run-e"}})
    assert "__interrupt__" in result


def test_backlog_mode_executes_nothing_before_approval(wiring):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    graph.invoke({"limit": 10, "mode": "backlog"},
                 {"configurable": {"thread_id": "run-f"}})
    assert wiring["log"].records() == []
    assert wiring["held"].all() == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL — `build_graph() got an unexpected keyword argument 'held'`

- [ ] **Step 3: Implement**

In `inbox_agent/graph.py`, extend `TriageState` (after line 55):

```python
    # "incremental" (default) or "backlog". Selects the edge out of partition:
    # incremental acts then reports, backlog previews then commits. The two
    # differ in risk, not in classification - a bad rule applied across 500
    # historical threads is not something per-item undo repairs comfortably.
    mode: str
    auto: list[dict]
    held: list[dict]
```

Add the imports:

```python
from .partition import partition
from .store import HeldQueue
```

Change the signature of `build_graph` (line 120) to take the queue:

```python
def build_graph(
    *,
    client,
    prefs: PreferenceStore,
    policy: Policy,
    llm,
    settings: Settings,
    log: AuditLog,
    held: HeldQueue,
    checkpointer=None,
):
```

Extract the execution loop out of `execute` so both paths share one chokepoint.
Add this module-level helper above `build_graph`:

```python
def _run_actions(actions, *, decision, verdict, client, settings, log, context):
    """Push one item's actions through the chokepoint. Returns (executed, refused).

    Lifted out of execute() so auto_execute and the interrupt path cannot drift
    apart. There must be exactly one place where an Action reaches Gmail.
    """
    executed, refused = [], []
    for action in actions:
        if action.kind == "none":
            continue
        try:
            record = execute_action(
                action, client=client, settings=settings, log=log,
                actor="human" if verdict == "edit" else (
                    f"rule:{decision.rule_id}" if decision.rule_id else "agent"),
                context=context,
                rule_provenance=decision.reason if decision.rule_id else None,
            )
            executed.append(record.model_dump(mode="json"))
        except ForbiddenActionError as exc:
            refused.append({"thread_id": decision.thread_id,
                            "kind": action.kind, "error": str(exc)})
    return executed, refused
```

Replace the body of `execute`'s verdict loop so it calls `_run_actions` instead
of inlining the try/except. Every existing guard stays — the "not part of the
reviewed batch" skip is the interrupt's trust boundary and must not move:

```python
        for thread_id, verdict in response.decisions.items():
            if verdict == "reject":
                continue

            # The interrupt's whole purpose is a trust boundary: the executed
            # set must be a subset of what the human was actually shown. A
            # resume payload naming a thread that never appeared in this batch -
            # stale, replayed, or forged - must be skipped before any indexing
            # happens, on every verdict branch.
            decision = decisions.get(thread_id)
            if decision is None:
                skipped.append({"thread_id": thread_id, "verdict": verdict,
                                "reason": "not part of the reviewed batch"})
                continue

            actions = (response.edits.get(thread_id)
                       if verdict == "edit" else decision.actions) or []
            ran, refused_here = _run_actions(
                actions, decision=decision, verdict=verdict,
                client=client, settings=settings, log=log, context=context)
            executed += ran
            refused += refused_here

        return {"executed": executed, "refused": refused, "skipped": skipped}
```

Add the three new nodes inside `build_graph`, after `propose`:

```python
    def partition_node(state: TriageState) -> dict:
        request = ReviewRequest.model_validate(state["review"])
        auto, held_pairs = partition(request.items)
        return {
            "auto": [i.model_dump(mode="json") for i in auto],
            "held": [{"item": i.model_dump(mode="json"), "reason": r}
                     for i, r in held_pairs],
        }

    def route_after_partition(state: TriageState) -> str:
        # Backlog previews before committing; everything else acts then reports.
        return "review" if state.get("mode") == "backlog" else "auto_execute"

    def auto_execute(state: TriageState, config: RunnableConfig) -> dict:
        context = _context(config)
        decisions = {d["thread_id"]: Decision.model_validate(d)
                     for d in state["decisions"]}
        executed, refused = [], []
        for raw in state.get("auto", []):
            item = ReviewItem.model_validate(raw)
            decision = decisions.get(item.thread_id)
            if decision is None:
                continue
            ran, refused_here = _run_actions(
                decision.actions, decision=decision, verdict="approve",
                client=client, settings=settings, log=log, context=context)
            executed += ran
            refused += refused_here
        return {"executed": executed, "refused": refused}

    def enqueue_held(state: TriageState) -> dict:
        run_id = ReviewRequest.model_validate(state["review"]).run_id
        for raw in state.get("held", []):
            held.add(ReviewItem.model_validate(raw["item"]),
                     run_id=run_id, reason=raw["reason"])
        return {}
```

`execute` currently builds its `ExecutionContext` inline. Lift it into a closure
inside `build_graph` — a closure rather than a module-level function because it
needs `llm`, `settings` and `policy`, which are graph-scoped — so `auto_execute`,
`execute` and `mark_triaged` all build it identically. Place it above `fetch`
and delete the inline construction from `execute`:

```python
    def _context(config: Optional[RunnableConfig]) -> ExecutionContext:
        """Per-invocation, not per-graph: checkpoint_id and langsmith_run_id are
        run-scoped and only exist once a run is actually underway.

        Accepts None so nodes without a RunnableConfig can still build one.

        checkpoint_id: verified against a real config dict at runtime (langgraph
        1.2.11) - config["configurable"]["checkpoint_id"] exists as a key but is
        None on both the initial and the resumed invocation of a node; it is not
        populated by ordinary forward execution in this version. thread_id IS
        reliably present on every invocation, so that is what gets recorded
        (field left named checkpoint_id per the audit schema).
        """
        configurable = config.get("configurable", {}) if config else {}

        # Only present when tracing is actually active. Tracing is OFF by
        # default locally, so this must never raise or add latency when
        # LangSmith is not configured.
        langsmith_run_id = None
        try:
            run_tree = get_current_run_tree()
            if run_tree is not None:
                langsmith_run_id = str(run_tree.id)
        except Exception:
            langsmith_run_id = None

        return ExecutionContext(
            model=getattr(llm, "model", None) or getattr(llm, "model_name", None),
            backend=settings.backend,
            policy_version=policy.version,
            checkpoint_id=configurable.get("thread_id"),
            langsmith_run_id=langsmith_run_id,
        )
```

`execute` then begins with `context = _context(config)` and keeps the rest of
its body unchanged.

Rewire the edges at the bottom of `build_graph`:

```python
    for name, fn in (("fetch", fetch), ("triage", triage), ("propose", propose),
                     ("partition", partition_node), ("auto_execute", auto_execute),
                     ("enqueue_held", enqueue_held),
                     ("review", review), ("execute", execute), ("learn", learn)):
        builder.add_node(name, fn)

    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "triage")
    builder.add_edge("triage", "propose")
    builder.add_edge("propose", "partition")
    builder.add_conditional_edges("partition", route_after_partition,
                                  {"auto_execute": "auto_execute", "review": "review"})
    builder.add_edge("auto_execute", "enqueue_held")
    builder.add_edge("enqueue_held", "learn")
    builder.add_edge("review", "execute")
    builder.add_edge("execute", "learn")
    builder.add_edge("learn", END)
```

`learn` reads `state.get("response")` and returns empty results when there is
none, so the incremental path passes through it harmlessly — corrections on the
incremental path are learned at correction time, not at run time (Plan 2).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph.py -v && pytest -q`
Expected: PASS. Existing interrupt tests must now pass `mode="backlog"`; update
those call sites in `tests/test_graph.py` and `tests/test_tg_bot.py` as part of
this step.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/graph.py tests/test_graph.py tests/test_tg_bot.py
git commit -m "Split the graph: act-then-report by default, interrupt for backlog

partition routes to auto_execute (incremental) or review (backlog). The
incremental run no longer waits - it acts on the 'always' authority tier,
queues the rest, and finishes.

interrupt() survives on the backlog branch. Carryover made held items outlive
runs, so review is no longer run-scoped and the primitive stopped fitting the
incremental path; a 500-thread historical sweep genuinely waits, and
preview-then-commit is exactly what it is for.

_run_actions is lifted out of execute() so both paths share one chokepoint.
There must be exactly one place where an Action reaches Gmail."
```

---

### Task 5: Fetch only unread, and mark what was triaged

Without a marker, a thread the agent labels but leaves in the inbox — and every
thread it decides `none` on — stays `INBOX + UNREAD` forever and is re-triaged,
re-charged and re-reported in every digest.

**Files:**
- Modify: `inbox_agent/config.py:35-51` (`Settings`), `inbox_agent/config.py:62-75`
  (`load_settings`)
- Modify: `inbox_agent/graph.py` (`fetch`, plus a new `mark_triaged` node)
- Test: `tests/test_graph.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `list_threads(limit, query)` (Task 1), the graph from Task 4.
- Produces: `Settings.triaged_label: str = "agent/triaged"`, read from
  `INBOX_TRIAGED_LABEL`; `Settings.inbox_query` property returning
  `f"in:inbox is:unread -label:{triaged_label}"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_triaged_label_defaults_and_is_overridable(monkeypatch):
    from inbox_agent.config import load_settings
    monkeypatch.delenv("INBOX_TRIAGED_LABEL", raising=False)
    assert load_settings().triaged_label == "agent/triaged"
    monkeypatch.setenv("INBOX_TRIAGED_LABEL", "bot/seen")
    assert load_settings().triaged_label == "bot/seen"


def test_inbox_query_excludes_read_and_already_triaged_mail(monkeypatch):
    from inbox_agent.config import load_settings
    monkeypatch.setenv("INBOX_TRIAGED_LABEL", "agent/triaged")
    q = load_settings().inbox_query
    assert "in:inbox" in q and "is:unread" in q and "-label:agent/triaged" in q
```

Add to `tests/test_graph.py`:

```python
def test_fetch_only_picks_unread_untriaged_inbox_mail(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    rows = [
        _row("unread") | {"label_ids": ["INBOX", "UNREAD"]},
        _row("read") | {"label_ids": ["INBOX"]},
        _row("done") | {"label_ids": ["INBOX", "UNREAD", "agent/triaged"]},
        _row("archived") | {"label_ids": ["UNREAD"]},
    ]
    snap = _snapshot(tmp_path, rows)
    graph = build_graph(
        client=SnapshotGmailClient(snap), prefs=PreferenceStore(build_store()),
        policy=Policy(text="T", version="local:test", source="local"),
        llm=FakeLLM(), settings=settings, log=AuditLog(settings.audit_log),
        held=HeldQueue(build_store()), checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10}, {"configurable": {"thread_id": "run-q"}})
    assert result["thread_ids"] == ["unread"]


def test_every_processed_thread_gets_the_triaged_label(tmp_path):
    """Both tiers. A held item is processed too - it is in the queue, and
    leaving it unlabelled would re-triage it on the next run."""
    from langgraph.checkpoint.memory import InMemorySaver
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    snap = _snapshot(tmp_path, [_row("t1"), _row("t2")])
    client = SnapshotGmailClient(snap)
    graph = build_graph(
        client=client, prefs=PreferenceStore(build_store()),
        policy=Policy(text="T", version="local:test", source="local"),
        llm=FakeLLM(ThreadJudgment(category="other", action="archive", label=None,
                                   reason="unsure", confidence=0.2)),
        settings=settings, log=AuditLog(settings.audit_log),
        held=HeldQueue(build_store()), checkpointer=InMemorySaver())
    graph.invoke({"limit": 2}, {"configurable": {"thread_id": "run-m"}})
    for tid in ("t1", "t2"):
        assert settings.triaged_label in client.get_thread(tid).label_ids
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_config.py tests/test_graph.py -v`
Expected: FAIL — `Settings.__init__() got an unexpected keyword argument` /
`AttributeError: 'Settings' object has no attribute 'inbox_query'`

- [ ] **Step 3: Implement**

In `inbox_agent/config.py`, add to `Settings` after `stale_after_days`:

```python
    # Applied to every thread the agent has processed, so it leaves the fetch
    # query. Without it, threads that were labelled but left in the inbox - and
    # everything decided `none` - stay INBOX+UNREAD forever and are re-triaged,
    # re-charged and re-reported in every single digest until read by hand.
    #
    # In Gmail rather than a local set on purpose: it is visible, so "why did it
    # ignore this?" has an answer you can see in the mailbox; it survives losing
    # the local store; and `label` is already at "always" authority, so it grants
    # no new capability. Marking as READ would have worked too and was rejected -
    # it destroys unread as a signal for the human and is not on the ladder.
    triaged_label: str = "agent/triaged"

    @property
    def inbox_query(self) -> str:
        return f"in:inbox is:unread -label:{self.triaged_label}"
```

In `load_settings()`, add:

```python
        triaged_label=os.getenv("INBOX_TRIAGED_LABEL", "agent/triaged").strip(),
```

In `inbox_agent/graph.py`, change `fetch`:

```python
    def fetch(state: TriageState) -> dict:
        threads = client.list_threads(
            limit=state.get("limit", settings.snapshot_size),
            query=settings.inbox_query)
        return {"thread_ids": [t.id for t in threads]}
```

Add a `mark_triaged` node after `enqueue_held`:

```python
    def mark_triaged(state: TriageState) -> dict:
        """Label every thread this run processed, both tiers.

        Held items are marked too: they are in the queue and will be shown from
        there, so leaving them unlabelled would re-triage them on every run
        while they wait - which is exactly the flood this label exists to stop.

        Goes through execute_action like anything else, so it is audited, and is
        refused by the deny-list and skipped by dry-run on the same terms.
        """
        context = _context(None)
        for thread_id in state.get("thread_ids", []):
            action = Action(kind="label", thread_id=thread_id,
                            params={"label": settings.triaged_label})
            try:
                execute_action(action, client=client, settings=settings, log=log,
                               actor="agent", context=context)
            except ForbiddenActionError:
                # Configured out. Not fatal: the run's real work already happened.
                log_module.getLogger(__name__).warning(
                    "triaged label refused by the deny-list; threads will be re-triaged")
        return {}
```

Add `import logging as log_module` at the top of `graph.py` (the name `log` is
already taken by the `AuditLog` parameter), and `Action` to the models import.

Rewire so both branches pass through it:

```python
    builder.add_edge("enqueue_held", "mark_triaged")
    builder.add_edge("mark_triaged", "learn")
    builder.add_edge("execute", "mark_triaged")
```

and remove the old `builder.add_edge("execute", "learn")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_config.py tests/test_graph.py -v && pytest -q`
Expected: PASS. Existing `test_gmail.py` snapshot fixtures use `label_ids:
["INBOX"]` with no `UNREAD`; any graph test that expects threads to be fetched
must add `"UNREAD"`. Fix those fixtures in this step.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/config.py inbox_agent/graph.py tests/
git commit -m "Fetch only unread untriaged inbox mail, and mark what was processed

Archive and trash remove INBOX so those items leave the query naturally. The
ones that do not: threads labelled but left in the inbox, and everything decided
'none'. Those stay INBOX+UNREAD forever and would be re-triaged, re-charged and
re-reported in every digest until read by hand.

The marker lives in Gmail rather than a local set because it is visible - 'why
did it ignore this?' has an answer you can see in the mailbox - it survives
losing the store, and label is already 'always' authority so it grants no new
capability. Marking as read was rejected: it destroys unread as a human signal
and is not on the ladder.

Held items are marked too. They are in the queue and rendered from there;
leaving them unlabelled would re-triage them on every run while they wait."
```

---

### Task 6: Callback identity, so a stale tap cannot hit the wrong thread

Today an index resolves against a parked checkpoint, so it always names the list
the human saw. With a persistent queue, positions shift between digests and
yesterday's tap lands on today's item 3.

**Files:**
- Modify: `inbox_agent/telegram/callbacks.py:29-45` (kinds), `:56-100` (encode/decode)
- Test: `tests/test_tg_callbacks.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `encode(kind, index=None, label_index=None, *, digest_id="") -> str`
  - `Intent` gains `digest_id: str = ""`
  - Two new kinds: `done` (code `D`), `approve_attention` (code `T`)
  - `DIGEST_ID_LEN = 4`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tg_callbacks.py`:

```python
from inbox_agent.telegram.callbacks import CB_MAX_BYTES, Intent, decode, encode


def test_digest_id_round_trips_for_every_kind():
    for kind, args in (("open", (3,)), ("approve", (3,)), ("reject", (3,)),
                       ("label", (3, 2)), ("prev", ()), ("next", ()),
                       ("done", ()), ("approve_attention", ()), ("list", ())):
        data = encode(kind, *args, digest_id="7f2a")
        got = decode(data)
        assert got.kind == kind, data
        assert got.digest_id == "7f2a", data


def test_index_and_label_index_survive_alongside_the_digest_id():
    got = decode(encode("label", 3, 2, digest_id="7f2a"))
    assert (got.index, got.label_index) == (3, 2)


def test_encoding_stays_inside_the_64_byte_cap():
    for kind, args in (("label", (9999, 99)), ("open", (9999,))):
        assert len(encode(kind, *args, digest_id="7f2a").encode()) <= CB_MAX_BYTES


def test_a_callback_without_a_digest_id_decodes_with_an_empty_one():
    """Backwards compatible: a message sent before this change still decodes,
    and the bot's staleness check treats an empty id as not-current."""
    assert decode("a:3").digest_id == ""


def test_garbage_is_still_a_noop():
    for data in ("", "zzz", "a:", "a:x:7f2a", "l:1:7f2a", "a:1:2:3:4"):
        assert decode(data).kind == "noop"


def test_a_digest_id_that_is_not_hex_is_rejected():
    """Bounds what we will parse at all, the same reasoning as _MAX_PARSED_INDEX."""
    assert decode("a:3:zz//").kind == "noop"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_tg_callbacks.py -v`
Expected: FAIL — `encode() got an unexpected keyword argument 'digest_id'`

- [ ] **Step 3: Implement**

In `inbox_agent/telegram/callbacks.py`:

```python
# Four hex characters, minted per digest. Positions used to resolve against a
# parked checkpoint, so an index always named the list the human was shown. With
# a queue that outlives runs, positions shift between digests and a tap on
# yesterday's message would land on today's item 3. The id makes the message
# itself identify which list it belongs to.
DIGEST_ID_LEN = 4
_HEX = set("0123456789abcdef")

Kind = Literal["approve", "reject", "label", "prev", "next", "approve_all",
               "open", "list", "done", "approve_attention", "noop"]

_CODE_TO_KIND: dict[str, Kind] = {
    "a": "approve", "r": "reject", "l": "label",
    "p": "prev", "n": "next", "A": "approve_all",
    "o": "open", "L": "list",
    # Opening the run's audit records, and the attention-tier one-tap approve.
    "D": "done", "T": "approve_attention",
}
```

Add `digest_id` to `Intent`:

```python
@dataclass(frozen=True)
class Intent:
    kind: Kind
    index: Optional[int] = None
    label_index: Optional[int] = None
    digest_id: str = ""
```

Rewrite `encode` and `decode`. The digest id is always the **last** field, for
every kind, so the arity per kind stays unambiguous:

```python
def encode(kind: Kind, index: Optional[int] = None,
           label_index: Optional[int] = None, *, digest_id: str = "") -> str:
    parts = [_KIND_TO_CODE[kind]]
    if index is not None:
        parts.append(str(index))
    if label_index is not None:
        parts.append(str(label_index))
    if digest_id:
        parts.append(digest_id)
    return ":".join(parts)


def decode(data: str) -> Intent:
    """Parse callback data. NEVER raises - hostile input becomes a no-op."""
    if not isinstance(data, str) or not data:
        return Intent("noop")
    parts = data.split(":")
    kind = _CODE_TO_KIND.get(parts[0])
    if kind is None:
        return Intent("noop")
    rest = parts[1:]

    digest_id = ""
    if rest and _is_digest_id(rest[-1]):
        digest_id = rest[-1]
        rest = rest[:-1]

    if kind in ("prev", "next", "approve_all", "list", "done", "approve_attention"):
        return Intent(kind, digest_id=digest_id) if not rest else Intent("noop")

    if kind in ("approve", "reject", "open"):
        if len(rest) != 1:
            return Intent("noop")
        index = _parse(rest[0])
        return (Intent(kind, index, digest_id=digest_id)
                if index is not None else Intent("noop"))

    if len(rest) != 2:
        return Intent("noop")
    index, label_index = _parse(rest[0]), _parse(rest[1])
    if index is None or label_index is None:
        return Intent("noop")
    return Intent("label", index, label_index, digest_id=digest_id)


def _is_digest_id(raw: str) -> bool:
    """A digest id is exactly DIGEST_ID_LEN lowercase hex characters.

    Length plus alphabet is what keeps it distinguishable from an index: an
    index is short and decimal, so "7f2a" can never be one. Bounded for the same
    reason _MAX_PARSED_INDEX is - refuse absurd input before parsing it.
    """
    return len(raw) == DIGEST_ID_LEN and all(c in _HEX for c in raw)


def _parse(raw: str) -> Optional[int]:
    # str.isdigit() rejects "-1", "1e5", "" and anything non-ASCII-numeric, so a
    # negative index cannot be constructed here at all.
    if not raw.isdigit() or len(raw) > 5:
        return None
    value = int(raw)
    return value if value <= _MAX_PARSED_INDEX else None
```

Delete the old nested `parse` function inside `decode` (it is now `_parse` at
module level).

**Note the ambiguity this creates and the test that pins it:** a four-digit
index like `1234` is also valid hex, so `decode("a:1234")` reads the `1234` as a
digest id and yields `noop`. That is the safe direction — a bare index with no
digest id is exactly the stale shape we want rejected — and
`test_a_callback_without_a_digest_id_decodes_with_an_empty_one` uses `a:3` to
stay unambiguous. Always pass `digest_id` when encoding.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_tg_callbacks.py -v && pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/callbacks.py tests/test_tg_callbacks.py
git commit -m "Put a digest id in callback data so a stale tap cannot act

Indices used to resolve against a parked checkpoint, so an index always named
the list the human was shown. A queue that outlives runs breaks that: positions
shift between digests and a tap on yesterday's message lands on today's item 3.

Four hex characters, always the last field for every kind, well inside the
64-byte cap. Thread ids still never travel in callback data, so the structural
guarantee - there is no callback string that can name a thread outside the
batch - is unchanged."
```

---

### Task 7: The digest renderer

Structure C: counts on top, held items in full grouped by why they are held, done
items reduced to counts behind a button.

**Files:**
- Modify: `inbox_agent/telegram/render_tg.py` (replace `digest`, keep `paged`)
- Test: `tests/test_tg_render.py`

**Interfaces:**
- Consumes: `HeldItem` (Task 3), `encode(..., digest_id=...)` and the `done` /
  `approve_attention` kinds (Task 6).
- Produces:
  - `DigestView` dataclass: `run_at: datetime`, `total: int`,
    `done_by_kind: dict[str, int]`, `rule_decided: int`,
    `held: list[HeldItem]`, `digest_id: str`
  - `digest(view: DigestView, page: int = 0) -> tuple[str, list]`
  - `HELD_PAGE_SIZE = 8`, `SECTIONS: tuple[tuple[str, str], ...]`

- [ ] **Step 1: Write the failing tests**

Replace the digest tests in `tests/test_tg_render.py` (keep the header and
`paged` tests):

```python
from datetime import datetime, timedelta, timezone

from inbox_agent.models import HeldItem
from inbox_agent.telegram.render_tg import (
    DigestView, HELD_PAGE_SIZE, TG_MAX_TEXT, digest,
)

NOW = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def held(tid, reason, *, conf=0.9, subject=None, held_at=NOW, reason_text="because"):
    return HeldItem(
        thread_id=tid, run_id="r1", first_held_at=held_at, hold_reason=reason,
        item=ReviewItem(
            thread_id=tid, category="promotion", subject=subject or f"Subject {tid}",
            sender=f"{tid}@example.com", snippet="s",
            proposed=[Action(kind="trash", thread_id=tid)],
            reason=reason_text, confidence=conf, source="model"))


def view(held_items=(), *, total=22, done=None, rule_decided=12):
    return DigestView(
        run_at=NOW, total=total,
        done_by_kind=done if done is not None else {"archive": 9, "label": 6, "draft": 3},
        rule_decided=rule_decided, held=list(held_items), digest_id="7f2a")


def test_the_stat_line_reports_what_was_done_and_what_waits():
    text, _ = digest(view([held("t1", "trash")]))
    assert "22" in text
    assert "9" in text and "6" in text and "3" in text
    assert "1 waiting" in text


def test_held_items_are_grouped_under_their_reason():
    text, _ = digest(view([held("t1", "trash"), held("t2", "needs_reply"),
                           held("t3", "security_alert"), held("t4", "low_confidence")]))
    assert "TRASH" in text
    assert "NEEDS REPLY" in text
    assert "SECURITY" in text
    assert "NOT SURE" in text


def test_an_empty_section_is_omitted_entirely():
    text, _ = digest(view([held("t1", "trash")]))
    assert "NEEDS REPLY" not in text
    assert "SECURITY" not in text


def test_each_held_item_shows_subject_sender_and_the_agents_reason():
    text, _ = digest(view([held("t1", "trash", subject="Quartz or Mechanical?",
                                reason_text="Marketing mail, no order reference.")]))
    assert "Quartz or Mechanical?" in text
    assert "t1@example.com" in text
    assert "Marketing mail, no order reference." in text


def test_low_confidence_is_shown_numerically():
    text, _ = digest(view([held("t1", "low_confidence", conf=0.41)]))
    assert "0.41" in text


def test_a_carried_over_item_shows_how_long_it_has_waited():
    old = held("t1", "trash", held_at=NOW - timedelta(hours=10))
    text, _ = digest(view([old]))
    assert "waiting since" in text.lower()


def test_an_item_held_in_this_run_shows_no_waiting_since():
    text, _ = digest(view([held("t1", "trash", held_at=NOW)]))
    assert "waiting since" not in text.lower()


def test_carried_items_sort_above_fresh_ones_within_a_section():
    fresh = held("fresh", "trash", held_at=NOW)
    old = held("old", "trash", held_at=NOW - timedelta(hours=10))
    text, _ = digest(view([fresh, old]))
    assert text.index("Subject old") < text.index("Subject fresh")


def test_done_items_are_counts_not_a_list():
    text, _ = digest(view([held("t1", "trash")]))
    assert "DONE" in text
    assert "18" in text
    assert "12" in text and "taught" in text.lower()


def test_the_rule_clause_is_omitted_when_nothing_was_rule_decided():
    text, _ = digest(view([held("t1", "trash")], rule_decided=0))
    assert "taught" not in text.lower()


def test_never_pads_columns():
    """Telegram wraps proportional text; padding produces a wall, not a table.
    Verified on a real phone once already - do not reintroduce it."""
    text, _ = digest(view([held(f"t{i}", "trash") for i in range(4)]))
    assert "   " not in text


def test_a_button_per_held_item_on_this_page():
    _, kb = digest(view([held(f"t{i}", "trash") for i in range(4)]))
    flat = [label for row in kb for (label, _) in row]
    assert [l for l in flat if l in {"1", "2", "3", "4"}] == ["1", "2", "3", "4"]


def test_the_one_tap_button_names_the_attention_tier_only():
    _, kb = digest(view([held("t1", "trash"), held("t2", "needs_reply")]))
    labels = [label for row in kb for (label, _) in row]
    assert any("Approve" in l for l in labels)


def test_there_is_no_one_tap_button_when_nothing_is_held_for_attention():
    """A blanket approve must never be able to reach trash or a guess."""
    _, kb = digest(view([held("t1", "trash"), held("t2", "low_confidence")]))
    labels = [label for row in kb for (label, _) in row]
    assert not any("Approve" in l for l in labels)


def test_held_items_page_at_the_limit():
    items = [held(f"t{i:02d}", "trash") for i in range(HELD_PAGE_SIZE + 3)]
    text, kb = digest(view(items), page=0)
    assert "Subject t00" in text
    assert f"Subject t{HELD_PAGE_SIZE:02d}" not in text
    labels = [label for row in kb for (label, _) in row]
    assert any("Next" in l for l in labels)


def test_page_two_shows_the_remainder_and_keeps_absolute_numbering():
    items = [held(f"t{i:02d}", "trash") for i in range(HELD_PAGE_SIZE + 3)]
    text, _ = digest(view(items), page=1)
    assert f"Subject t{HELD_PAGE_SIZE:02d}" in text
    assert f"{HELD_PAGE_SIZE + 1}." in text


def test_an_out_of_range_page_clamps_rather_than_raising():
    """Reached from a callback. A stale one must land somewhere sane."""
    text, _ = digest(view([held("t1", "trash")]), page=99)
    assert "Subject t1" in text


def test_an_empty_queue_still_renders_the_report():
    text, kb = digest(view([]))
    assert "DONE" in text
    assert "0 waiting" in text or "waiting" not in text


def test_never_exceeds_the_telegram_cap():
    items = [held(f"t{i:02d}", "trash", subject="x" * 200, reason_text="y" * 400)
             for i in range(HELD_PAGE_SIZE)]
    text, _ = digest(view(items))
    assert len(text) <= TG_MAX_TEXT
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_tg_render.py -v`
Expected: FAIL with `ImportError: cannot import name 'DigestView'`

- [ ] **Step 3: Implement**

Replace `DIGEST_PAGE_SIZE`, `MAX_ITEM_BUTTONS` and the `digest` function in
`inbox_agent/telegram/render_tg.py`:

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..models import HeldItem

# A phone screen, roughly. Beyond this the queue is scrolling, not scanning.
HELD_PAGE_SIZE = 8

# Section order is hold-reason precedence order, so the most consequential
# things are nearest the top of the message where they are read first.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("trash", "🗑 TRASH — needs your OK"),
    ("needs_reply", "✉️ NEEDS REPLY"),
    ("security_alert", "🔒 SECURITY"),
    ("low_confidence", "❓ NOT SURE"),
)

# Held for attention rather than authorisation: the proposed action is harmless
# and the thread simply wants a person. Only these are one-tap approvable.
ATTENTION_REASONS = frozenset({"needs_reply", "security_alert"})

_SUBJECT_CAP = 70
_REASON_CAP = 160


@dataclass
class DigestView:
    """Everything the digest renders, assembled by the caller.

    A dataclass rather than a ReviewRequest because the digest no longer reports
    one run: it reports what this run DID (audit records) alongside what is
    still waiting from any run (the queue). Keeping the renderer pure over this
    view is what kept render.py's contract worth having.
    """
    run_at: datetime
    total: int
    done_by_kind: dict[str, int]
    rule_decided: int
    held: list[HeldItem] = field(default_factory=list)
    digest_id: str = ""


def _age(item: HeldItem, now: datetime) -> str:
    """'waiting since Tue 8:00', or empty for something held in this run.

    Only carried-over items say it. Printing it on everything would make the
    phrase meaningless, and its whole job is to make an ignored queue look
    ignored.
    """
    held_at = item.first_held_at
    if held_at.tzinfo is None:
        held_at = held_at.replace(tzinfo=timezone.utc)
    if (now - held_at).total_seconds() < 3600:
        return ""
    return f" · waiting since {held_at.astimezone().strftime('%a %H:%M')}"


def _held_line(number: int, item: HeldItem, now: datetime) -> str:
    """Three lines: what it is, who sent it, and why the agent wants this.

    `reason` is here because these are the items the agent deliberately would
    not decide alone - it is what turns a rejection into training data rather
    than a shrug. The one-line form that was right for a fifty-item list is
    wrong for a list of four.
    """
    conf = (f" · {item.item.confidence:.2f}"
            if item.item.confidence < LOW_CONFIDENCE else "")
    why = (item.item.reason or NO_REASON).strip()[:_REASON_CAP]
    return (f"{number}. {item.item.subject[:_SUBJECT_CAP]}\n"
            f"{item.item.sender}{conf}{_age(item, now)}\n"
            f"{why}")


def digest(view: DigestView, page: int = 0) -> tuple[str, list]:
    """Counts on top, held items in full, done items behind a button.

    Grouped by why an item is held rather than by what the agent proposes: the
    sections ARE the queue, and the stat line is the report.
    """
    # Carried-over items first within each section: an ignored queue should read
    # as one. all() is already oldest-first, so this is stable.
    ordered = sorted(view.held, key=lambda h: h.first_held_at)
    pages = max(1, (len(ordered) + HELD_PAGE_SIZE - 1) // HELD_PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    start = page * HELD_PAGE_SIZE
    window = ordered[start:start + HELD_PAGE_SIZE]

    # Absolute numbering, computed once over the whole queue: item 9 is item 9
    # on page 2, so a number the owner reads means the same thing on every page.
    numbers = {h.thread_id: n for n, h in enumerate(ordered, start=1)}

    stat = " · ".join(f"{count} {kind}"
                      for kind, count in sorted(view.done_by_kind.items()) if count)
    lines = [f"Inbox · {view.run_at.astimezone().strftime('%H:%M')} · "
             f"{view.total} threads"]
    lines.append(f"{stat} · {len(ordered)} waiting" if stat
                 else f"{len(ordered)} waiting")
    if pages > 1:
        lines[0] += f"  (page {page + 1}/{pages})"

    for reason, title in SECTIONS:
        section = [h for h in window if h.hold_reason == reason]
        if not section:
            continue
        lines.append("")
        lines.append(f"{title} ({len(section)})")
        for held_item in section:
            lines.append(_held_line(numbers[held_item.thread_id],
                                    held_item, view.run_at))

    done_total = sum(view.done_by_kind.values())
    lines.append("")
    lines.append(f"✓ DONE ({done_total})")
    if view.rule_decided:
        # Make the learning visible. As rules accumulate this climbs and the
        # sections above shrink. Suppressed at zero: "0 came from rules you
        # taught me" reads as a failure rather than as a not-yet.
        lines.append(f"{view.rule_decided} came from rules you taught me")

    text = "\n".join(lines)[:TG_MAX_TEXT]

    keyboard: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    for held_item in window:
        row.append((str(numbers[held_item.thread_id]),
                    encode("open", numbers[held_item.thread_id] - 1,
                           digest_id=view.digest_id)))
        if len(row) == 4:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    if done_total:
        keyboard.append([(f"📋 Show the {done_total} done",
                          encode("done", digest_id=view.digest_id))])

    attention = [h for h in ordered if h.hold_reason in ATTENTION_REASONS]
    if attention:
        # Attention tier only. A blanket button that could reach trash or a
        # low-confidence guess would rubber-stamp exactly the set this design
        # isolated to avoid rubber-stamping.
        keyboard.append([(f"✅ Approve {len(attention)} replies & alerts",
                          encode("approve_attention", digest_id=view.digest_id))])

    nav: list[tuple[str, str]] = []
    if page > 0:
        nav.append(("◀ Prev", encode("prev", digest_id=view.digest_id)))
    if page < pages - 1:
        nav.append(("Next ▶", encode("next", digest_id=view.digest_id)))
    if nav:
        keyboard.append(nav)

    return text, keyboard
```

Delete the now-unused `DIGEST_PAGE_SIZE` and `MAX_ITEM_BUTTONS` constants, and
the `categories` parameter from `digest`'s old signature. `paged()`, `header()`
and `rule_decided_count()` stay as they are — `paged` is still the single-item
view the interrupt path uses in Plan 3.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_tg_render.py -v && pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/render_tg.py tests/test_tg_render.py
git commit -m "Rebuild the digest: counts on top, held items in full, done behind a button

Grouped by why an item is held rather than by what the agent proposes - the
sections are the queue, the stat line is the report.

Three lines per held item, including the agent's reason. The one-line form was
right for a fifty-item list; this list is the handful the agent deliberately
would not decide alone, and for those `reason` is the whole point - it is what
turns a rejection into training data rather than a shrug.

Carried-over items sort first and say how long they have waited, so an ignored
queue reads as ignored. The one-tap approve button appears only when something
is held for ATTENTION, and never when the queue is only trash or guesses.

category is deliberately not shown: under a header that already says TRASH or
NEEDS REPLY it says the same thing twice."
```

---

### Task 8: Wire the bot to the queue

The bot currently reads its review payload from a parked checkpoint. There is no
longer one on the incremental path.

**Files:**
- Modify: `inbox_agent/telegram/bot.py:29-115` (state and rendering),
  `:118-190` (commands), `:190-220` (callbacks)
- Modify: `inbox_agent/telegram/__main__.py` (construct and inject `HeldQueue`)
- Test: `tests/test_tg_bot.py`

**Interfaces:**
- Consumes: `HeldQueue` (Task 3), `Intent.digest_id` (Task 6),
  `DigestView`/`digest` (Task 7).
- Produces: `Bot(..., held: HeldQueue)`; `Bot._view() -> DigestView`;
  `/held` command.

- [ ] **Step 1: Write the failing tests**

`tests/test_tg_bot.py` already has a `bot` fixture returning `(bot, transport,
log)`, plus `msg()` and `cb()` update builders. Two changes to it first:

```python
# 1. snapshot_file must produce UNREAD mail, or Task 5's query fetches nothing:
    data = [{"id": f"t{i}", "subject": f"Sale {i}", "sender": f"deals{i}@shop.com",
             "to": [], "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "",
             "label_ids": ["INBOX", "UNREAD"]} for i in range(4)]

# 2. the fixture builds and injects one HeldQueue, shared by graph and bot:
    held = HeldQueue(build_store())
    graph = build_graph(client=SnapshotGmailClient(snapshot_file),
                        prefs=PreferenceStore(build_store()),
                        policy=Policy(text="P", version="local:t", source="local"),
                        llm=FakeLLM(), settings=settings, log=log, held=held,
                        checkpointer=InMemorySaver())
    t = FakeTransport()
    return Bot(transport=t, graph=graph, settings=settings, held=held,
               categories=["recruiter", "promotion"]), t, log
```

Then add the tests:

```python
def test_triage_sends_a_digest_built_from_the_queue(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    assert t.sent, "the digest was never sent"
    assert "DONE" in t.sent[-1]["text"]


def test_held_command_shows_the_queue_without_running_a_triage(bot):
    """The queue outlives runs, so looking at it must not produce more work."""
    b, t, _ = bot
    before = len(t.sent)
    b.handle_update(msg("/held"))
    assert len(t.sent) == before + 1
    assert b._runs_started == 0


def test_a_callback_from_a_previous_digest_is_ignored(bot):
    """The whole point of the digest id."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    before = len(t.edited)
    b.handle_update(cb(encode("open", 0, digest_id="dead")))
    assert len(t.edited) == before


def test_a_callback_from_the_current_digest_is_honoured(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    assert t.edited


def test_an_unauthorised_update_is_dropped_before_anything_runs(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage", chat_id=999999))
    assert b.rejected_updates == 1
    assert b._runs_started == 0


def test_a_failing_triage_says_so_rather_than_going_quiet(bot, monkeypatch):
    """Silence is indistinguishable from an empty inbox.

    A model that is down must not look like a morning with no mail - that is
    the failure the owner would trust for days without noticing.
    """
    b, t, _ = bot

    def boom(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(b.graph, "invoke", boom)
    b.handle_update(msg("/triage 4"))
    assert "failed" in t.sent[-1]["text"].lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_tg_bot.py -v`
Expected: FAIL — `AttributeError: 'Bot' object has no attribute '_digest_id'`

- [ ] **Step 3: Implement**

In `inbox_agent/telegram/bot.py`:

Add `held: HeldQueue` to `Bot.__init__` and store it. Replace the `_page`/`_view`
state with:

```python
        self._page = 0
        self._digest_id = ""
        # Counts runs actually started, so a test can assert that /held ran none.
        self._runs_started = 0
        self._last_run: Optional[dict] = None   # the graph result for the digest
```

Replace `_request()`/`_render()` with a view builder:

```python
    def _new_digest_id(self) -> str:
        return uuid.uuid4().hex[:DIGEST_ID_LEN]

    def _view(self) -> DigestView:
        """Assemble what the digest renders: this run's work plus the queue.

        The done counts come from the audit records the run wrote, not from
        graph state: the audit log is the durable record of what actually
        reached Gmail, and it is the same source the undo path will read.
        """
        result = self._last_run or {}
        done_by_kind: dict[str, int] = {}
        rule_decided = 0
        for record in result.get("executed", []):
            if record.get("action") == "label" and \
                    record.get("params", {}).get("label") == self.settings.triaged_label:
                continue  # bookkeeping, not work the owner cares about
            done_by_kind[record["action"]] = done_by_kind.get(record["action"], 0) + 1
            if str(record.get("actor", "")).startswith("rule:"):
                rule_decided += 1
        return DigestView(
            run_at=datetime.now(timezone.utc),
            total=len(result.get("thread_ids", [])),
            done_by_kind=done_by_kind,
            rule_decided=rule_decided,
            held=self.held.all(),
            digest_id=self._digest_id,
        )

    def _show(self, *, edit: bool) -> None:
        text, keyboard = digest(self._view(), self._page)
        if edit and self._message_id is not None:
            self.transport.edit_message(self.chat_id, self._message_id, text, keyboard)
        else:
            sent = self.transport.send_message(self.chat_id, text, keyboard)
            self._message_id = (sent or {}).get("message_id")
```

Rewrite `_start`:

```python
    def _start(self, limit: int) -> None:
        self._run += 1
        self._runs_started += 1
        self._page = 0
        self._message_id = None
        self._digest_id = self._new_digest_id()

        started = time.monotonic()
        try:
            self._last_run = self.graph.invoke(
                {"limit": limit, "mode": "incremental"}, self._config)
        except Exception as exc:
            # Silence is indistinguishable from an empty inbox, which is a
            # failure the owner would trust for days without noticing. Say so.
            log.exception("triage failed")
            self.transport.send_message(
                self.chat_id, f"Triage failed: {type(exc).__name__}. "
                              f"Nothing was executed. /triage to retry.")
            return
        elapsed = time.monotonic() - started
        log.info("triage done in %.1fs: %s executed, %s held", elapsed,
                 len(self._last_run.get("executed", [])), len(self.held.all()))
        self._show(edit=False)
```

Add the `/held` command to `_on_message`'s dispatch:

```python
        elif command == "/held":
            # Shows the queue without running anything: the queue outlives runs,
            # so looking at it must not require producing more work.
            self._page = 0
            self._message_id = None
            self._digest_id = self._new_digest_id()
            self._show(edit=False)
```

and update the help text to
`"Commands: /triage [n] · /held · /status · /cancel"`.

Add the staleness guard at the top of `_on_callback`, immediately after
`decode`:

```python
        intent = decode(query.get("data", ""))
        self.transport.answer_callback(query.get("id", ""))
        if intent.kind == "noop":
            return
        if intent.digest_id != self._digest_id:
            # A tap on a superseded digest. Positions have shifted since that
            # message was drawn, so acting on it would act on the wrong thread.
            log.info("ignored a callback from digest %r (current %r)",
                     intent.digest_id, self._digest_id)
            return
```

Handle `prev`/`next` by paging and `open` by delegating to the paged view (the
held-item detail view and its verdicts are Plan 2 — for now `open` re-renders
the digest so the callback path is exercised end to end):

```python
        if intent.kind in ("next", "prev"):
            self._page = max(0, self._page + (1 if intent.kind == "next" else -1))
            self._show(edit=True)
            return
        if intent.kind in ("open", "done", "approve_attention"):
            # Plan 2 gives these their real behaviour. Re-rendering keeps the
            # message live rather than silently doing nothing.
            self._show(edit=True)
            return
```

Update the imports at the top of `bot.py`:

```python
import uuid
from datetime import datetime, timezone

from ..store import HeldQueue
from .callbacks import DIGEST_ID_LEN, decode
from .render_tg import DigestView, digest
```

In `inbox_agent/telegram/__main__.py`, construct the queue once and pass it to
both `build_graph` and `Bot`, using the same `build_store()` the
`PreferenceStore` uses so there is one thing to persist later.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_tg_bot.py -v && pytest -q`
Expected: PASS. The whole suite must be green; existing bot tests that assert
on the old digest text need updating to the new structure in this step.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/bot.py inbox_agent/telegram/__main__.py tests/test_tg_bot.py
git commit -m "Render the digest from the queue and the audit log, not a checkpoint

The bot read its payload from a parked run. On the incremental path there is no
longer one: the run completes, so what is still outstanding lives in the queue
and what was done lives in the audit records.

Done counts come from the audit log rather than graph state because that is the
durable record of what actually reached Gmail, and it is the same source the
undo path will read. The triaged label is filtered out of the counts -
bookkeeping, not work the owner cares about.

Adds /held, which shows the queue without running a triage: the queue outlives
runs, so looking at it must not require producing more work. Callbacks whose
digest id is not current are ignored."
```

---

## What this plan does not cover

Deliberately deferred to later plans, each of which produces working software on
its own:

- **Plan 2 — corrections and schedule.** The held-item detail view, approve /
  not this / relabel verdicts, the modal free-text correction, and the
  in-process timer with catch-up on wake. Task 8 leaves `open` and
  `approve_attention` as re-renders so the callback path is live but inert.
- **Plan 3 — backlog sweep.** `/backlog`, group-level approval of the interrupt
  payload, and `demote_stale` finally having a job. Task 4 builds and tests the
  `mode="backlog"` branch; nothing invokes it yet.
- **Plan 4 — undo and learning from done.** The done panel, `undo_action()`
  wiring, `record_override()`, and learning from an undo. Blocked on
  `INBOX_DRY_RUN=false` against live Gmail, since `undo_action()` correctly
  refuses on dry-run records. Task 8 leaves the `done` callback as a re-render.

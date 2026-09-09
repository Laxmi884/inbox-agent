# The durable done view — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make what a run did outlive the run after it, so a correction on any
of the last ten runs still teaches the agent.

**Architecture:** A `RunReport` per run, written by the graph into the store
`HeldQueue` already uses (its own namespace, not a third sqlite file), addressed
by `ReviewRequest.run_id`, retained ten deep. The bot's five report readers stop
reading `Bot._last_run` and read that record instead; `/done` adds a run list
above the existing done panel and item screen, which are reused unchanged.

**Tech Stack:** Python 3, pydantic v2, LangGraph `BaseStore`/`SqliteStore`,
pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-08-durable-done-view-design.md`
(committed `dbf9740`, approved by the owner 2026-09-08). Read it before Task 1;
every "why" below is short because the spec carries it.

## Global Constraints

- **Branch, not main.** The owner's explicit instruction: this work goes on its
  own branch (worktree flow — `superpowers:using-git-worktrees`). The session
  cannot merge into main itself; hand over the SHA at the end.
- **`python -m pytest` is the whole feedback loop.** 917 tests, ~5s, offline, no
  credentials. Green after every task; a change is not done until it is.
- **A bug fix lands with the test that would have caught it.** Same for this
  feature: the restart test in Task 3 is the one that would have caught the
  original defect.
- **Never edit `inbox_agent_stage_a_explained.ipynb` by hand.** It is generated
  by `build_teaching_notebook.py`. Task 2 changes the builder and re-runs it.
- Commit messages: one line, imperative, saying what changed and why.
- Retention is exactly **10** reports, pruned oldest-first by `ran_at`, **on
  write only** (spec 2.2).
- `INBOX_BODY_BUDGET`, `.env` and the live bot are untouched by this work. Do
  not start a bot by hand; launchd owns the live one.
- Working tree note: `inbox_agent_stage_a_explained.ipynb` is dirty on `main`
  with the known Jupyter unicode-escape noise. Working in a worktree off `HEAD`
  leaves it alone — do not commit it, do not `checkout` it out from under the
  owner.

---

### Task 1: `RunReport`, `DoneRecord` and `DoneStore`

**Files:**
- Modify: `inbox_agent/models.py` (after `HeldItem`, ~line 195)
- Modify: `inbox_agent/store.py` (namespace constant near `HELD_NS` line 26; new
  class after `HeldQueue`, end of file)
- Create: `tests/test_done_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `inbox_agent.models.DoneRecord(thread_id: str, item: ReviewItem, actions: list[tuple[str, Optional[str]]] = [], rule_id: Optional[str] = None)`
  - `inbox_agent.models.RunReport(run_id: str, ran_at: datetime, total: int = 0, remaining: int = 0, done: list[DoneRecord] = [])`
  - `inbox_agent.store.DONE_NS = ("done", "reports")`, `inbox_agent.store.MAX_REPORTS = 10`
  - `inbox_agent.store.DoneStore(store)` with `record(report: RunReport) -> RunReport`,
    `get(run_id: str) -> Optional[RunReport]`, `recent(limit: int = MAX_REPORTS) -> list[RunReport]`
    (newest first).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_done_store.py`:

```python
"""The record of what a run did, kept past the run after it."""
from datetime import datetime, timedelta, timezone

from inbox_agent.models import Action, DoneRecord, ReviewItem, RunReport
from inbox_agent.store import MAX_REPORTS, DoneStore, build_store

T0 = datetime(2026, 9, 8, 8, 0, tzinfo=timezone.utc)


def item(tid="t1"):
    return ReviewItem(
        thread_id=tid, category="promotion", subject=f"Subject {tid}",
        sender="deals@shop.com", snippet="s",
        proposed=[Action(kind="archive", thread_id=tid)],
        reason="a sale", confidence=0.9, source="model")


def report(run_id="r1", at=T0, threads=("t1",)):
    return RunReport(
        run_id=run_id, ran_at=at, total=len(threads), remaining=3,
        done=[DoneRecord(thread_id=t, item=item(t),
                         actions=[("archive", None), ("label", "promo")],
                         rule_id="r-abc")
              for t in threads])


def store():
    return DoneStore(build_store())


def test_a_report_round_trips_with_every_field():
    s = store()
    s.record(report())
    got = s.get("r1")
    assert got is not None
    assert got.run_id == "r1"
    assert got.ran_at == T0
    assert got.total == 1 and got.remaining == 3
    row = got.done[0]
    assert row.thread_id == "t1"
    # The whole ReviewItem, not a flattened subject: category and reason are
    # what the correction path reads off it.
    assert row.item.subject == "Subject t1"
    assert row.item.category == "promotion"
    assert row.item.reason == "a sale"
    assert row.actions == [("archive", None), ("label", "promo")]
    assert row.rule_id == "r-abc"


def test_get_returns_none_for_an_unknown_run():
    assert store().get("nope") is None


def test_recent_is_newest_first():
    s = store()
    s.record(report("old", T0))
    s.record(report("new", T0 + timedelta(hours=4)))
    assert [r.run_id for r in s.recent()] == ["new", "old"]


def test_an_eleventh_report_prunes_the_oldest():
    s = store()
    for i in range(MAX_REPORTS + 1):
        s.record(report(f"r{i}", T0 + timedelta(minutes=i)))
    kept = [r.run_id for r in s.recent()]
    assert len(kept) == MAX_REPORTS
    assert "r0" not in kept, "the oldest report survived the prune"
    assert s.get("r0") is None
    assert kept[0] == f"r{MAX_REPORTS}"


def test_reading_never_prunes():
    """Pruning on read would make /done mutate the record it is showing."""
    s = store()
    for i in range(MAX_REPORTS + 1):
        s.record(report(f"r{i}", T0 + timedelta(minutes=i)))
    s._store.put(("done", "reports"), "extra",
                 {"report": report("extra", T0).model_dump(mode="json")})
    assert len(s.recent(limit=99)) == MAX_REPORTS + 1
    assert s.get("extra") is not None


def test_a_run_that_did_nothing_still_records_a_report():
    """"It ran and did nothing" must be distinguishable from "it never ran"."""
    s = store()
    s.record(RunReport(run_id="quiet", ran_at=T0, total=0, remaining=0, done=[]))
    got = s.get("quiet")
    assert got is not None and got.done == []


def test_a_naive_timestamp_is_read_back_as_utc():
    """Reports are sorted against each other; naive vs aware raises."""
    s = store()
    s.record(RunReport(run_id="naive", ran_at=datetime(2026, 9, 8, 8, 0)))
    s.record(report("aware", T0 + timedelta(hours=1)))
    assert [r.run_id for r in s.recent()] == ["aware", "naive"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_done_store.py -q`
Expected: collection error — `ImportError: cannot import name 'DoneRecord'`.

- [ ] **Step 3: Add the models**

In `inbox_agent/models.py`, extend the pydantic import:

```python
from pydantic import BaseModel, Field, field_validator, model_validator
```

and add after `HeldItem` (before `ReviewResponse`):

```python
class DoneRecord(BaseModel):
    """One thread a run acted on, and what it did to it.

    Embeds the whole ReviewItem for the reason HeldItem does: ReviewItem is
    already the renderer's contract, and re-declaring subject and sender here
    would give the report two shapes to render instead of one. It also happens
    to carry every field the correction path reads - `category` and `reason`
    come free with it, and `rule_id` says which rule to demote when the owner
    overrules it.

    `actions` is (kind, label) pairs, the shape DoneItem already renders and
    counts, so the panel does not parse a string it just formatted.
    """
    thread_id: str
    item: ReviewItem
    actions: list[tuple[str, Optional[str]]] = Field(default_factory=list)
    rule_id: Optional[str] = None


class RunReport(BaseModel):
    """What one run did, addressed by the run's own id.

    `total` and `remaining` live here rather than being left in graph state so
    a past run's digest header renders identically to a live one, with no
    special case for "this run is not the current one".
    """
    run_id: str
    ran_at: datetime
    total: int = 0
    remaining: int = 0
    done: list[DoneRecord] = Field(default_factory=list)

    @field_validator("ran_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        """Naive in, UTC out. Reports are sorted against each other by this
        field, and comparing a naive datetime to an aware one raises - a crash
        in the one place whose whole job is to still be there after a restart.
        """
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
```

`models.py` imports `datetime` but not `timezone`; change that import to:

```python
from datetime import datetime, timezone
```

- [ ] **Step 4: Add `DoneStore`**

In `inbox_agent/store.py`, extend the models import:

```python
from .models import (ActionKind, ActionTemplate, HeldItem, ReviewItem, RunReport,
                     Rule, Thread)
```

Add beside the other namespaces (after `HELD_NS`):

```python
DONE_NS = ("done", "reports")

# Ten runs, not a time window: at five slots a day that is two days, and a count
# is robust to the bot being off in a way a window is not - after a quiet
# weekend ten runs are still ten runs where "the last 48 hours" is empty.
MAX_REPORTS = 10
```

Add at the end of the file:

```python
class DoneStore:
    """What each run did, kept past the run after it.

    Same bargain as HeldQueue, and it shares HeldQueue's store: store-agnostic,
    its own namespace, no embedding index. A run report has no `text` field for
    a semantic index to key off, so it belongs beside the held items rather than
    beside the rules - and a third sqlite file would be a third object threaded
    through build_graph and Bot for no gain.

    Retention is MAX_REPORTS deep, pruned oldest-first on WRITE. Never on read:
    the write already touches the store, and a read that mutates would make
    /done surprising to reason about.
    """

    def __init__(self, store):
        self._store = store

    def record(self, report: RunReport) -> RunReport:
        self._store.put(DONE_NS, report.run_id,
                        {"report": report.model_dump(mode="json")})
        for stale in self._all()[MAX_REPORTS:]:
            self._store.delete(DONE_NS, stale.run_id)
        return report

    def get(self, run_id: str) -> Optional[RunReport]:
        entry = self._store.get(DONE_NS, run_id)
        return RunReport.model_validate(entry.value["report"]) if entry else None

    def recent(self, limit: int = MAX_REPORTS) -> list[RunReport]:
        """The last runs, newest first - the order /done lists them in."""
        return self._all()[:limit]

    def _all(self) -> list[RunReport]:
        page = _search_all(self._store, DONE_NS)
        reports = [RunReport.model_validate(entry.value["report"])
                   for entry in page]
        return sorted(reports, key=lambda r: r.ran_at, reverse=True)
```

- [ ] **Step 5: Run the new tests, then the suite**

Run: `python -m pytest tests/test_done_store.py -q`
Expected: PASS (8 tests).
Run: `python -m pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add inbox_agent/models.py inbox_agent/store.py tests/test_done_store.py
git commit -m "Keep a run's done record in the store, ten runs deep"
```

---

### Task 2: The graph writes a report for every run

**Files:**
- Modify: `inbox_agent/graph.py` (imports; new module-level `run_report_from_state`
  before `build_graph`; `build_graph` signature ~line 190; `enqueue_held` ~line 387)
- Modify: `inbox_agent/telegram/__main__.py` (~line 125 store wiring, ~line 134 `build_graph`)
- Modify: `build_teaching_notebook.py` (~line 1268 cell source), then re-run it
- Modify: `tests/test_graph.py` (the `wiring` fixture line 49, and the six
  explicit-kwarg `build_graph(` calls at lines 419, 463, 564, 640, 666, 697)
- Modify: `tests/test_tg_bot.py` (fixture, line 83-95)
- Create: `tests/test_run_report.py`

**Interfaces:**
- Consumes: `RunReport`, `DoneRecord`, `DoneStore` from Task 1.
- Produces:
  - `inbox_agent.graph.run_report_from_state(state, *, triaged_label: str, now: Optional[datetime] = None) -> RunReport`
  - `build_graph(..., held: HeldQueue, done: DoneStore, checkpointer=None)` —
    `done` is REQUIRED, positioned after `held`, exactly as `held` itself was
    made required rather than defaulted.
  - `enqueue_held` writes one report per incremental run.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_report.py`:

```python
"""Every run leaves a record of what it did, written by the graph."""
import json
from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from inbox_agent.audit import AuditLog
from inbox_agent.classify import ThreadJudgment
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.graph import build_graph, run_report_from_state
from inbox_agent.models import Action, ReviewItem, ReviewRequest
from inbox_agent.policy import Policy
from inbox_agent.store import DoneStore, HeldQueue, PreferenceStore, build_store


class FakeLLM:
    def with_structured_output(self, schema): return self
    def invoke(self, messages):
        return ThreadJudgment(category="promotion", action="archive",
                              reason="a sale", confidence=0.9)


@pytest.fixture
def wiring(tmp_path):
    data = [{"id": f"t{i}", "subject": f"Sale {i}", "sender": "deals@shop.com",
             "to": [], "date": "2026-09-08T10:00:00Z", "snippet": "s", "body": "",
             "label_ids": ["INBOX", "UNREAD"]} for i in range(2)]
    snap = tmp_path / "threads.json"
    snap.write_text(json.dumps(data))
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev")
    return dict(client=SnapshotGmailClient(snap),
                prefs=PreferenceStore(build_store()),
                policy=Policy(text="P", version="local:t", source="local"),
                llm=FakeLLM(), settings=settings,
                log=AuditLog(settings.audit_log),
                held=HeldQueue(build_store()), done=DoneStore(build_store()))


def _state(executed, *, items, triaged="agent/triaged", thread_ids=("t1",),
           remaining=4):
    request = ReviewRequest(run_id="abc12345", policy_version="local:t",
                            items=items)
    return {"review": request.model_dump(mode="json"),
            "executed": executed, "thread_ids": list(thread_ids),
            "remaining": remaining}


def _item(tid="t1"):
    return ReviewItem(thread_id=tid, category="promotion", subject=f"Sale {tid}",
                      sender="deals@shop.com", snippet="s",
                      proposed=[Action(kind="archive", thread_id=tid)],
                      reason="a sale", confidence=0.9, source="model")


# --- the join ---------------------------------------------------------------

def test_the_report_joins_audit_records_to_their_proposals():
    state = _state([{"action": "archive", "thread_id": "t1", "actor": "model"},
                    {"action": "label", "thread_id": "t1",
                     "params": {"label": "promo"}, "actor": "rule:r-7"}],
                   items=[_item("t1")])
    report = run_report_from_state(state, triaged_label="agent/triaged")
    assert report.run_id == "abc12345"
    assert report.total == 1 and report.remaining == 4
    assert len(report.done) == 1
    row = report.done[0]
    assert row.item.subject == "Sale t1"
    assert row.item.reason == "a sale"
    assert row.actions == [("archive", None), ("label", "promo")]
    assert row.rule_id == "r-7", "the rule that decided it was not recorded"


def test_the_bookkeeping_label_is_not_reported_as_work():
    state = _state([{"action": "label", "thread_id": "t1",
                     "params": {"label": "agent/triaged"}, "actor": "model"},
                    {"action": "archive", "thread_id": "t1", "actor": "model"}],
                   items=[_item("t1")])
    report = run_report_from_state(state, triaged_label="agent/triaged")
    assert report.done[0].actions == [("archive", None)]


def test_a_record_with_no_proposal_is_still_reported_by_id():
    """An action with no visible subject is strange; hiding it would be worse."""
    state = _state([{"action": "archive", "thread_id": "ghost", "actor": "model"}],
                   items=[])
    report = run_report_from_state(state, triaged_label="agent/triaged")
    assert report.done[0].thread_id == "ghost"
    assert report.done[0].item.subject == "ghost"


def test_a_malformed_record_does_not_raise():
    """This runs after the graph acted: an exception costs the owner the report
    for work that already reached Gmail."""
    state = _state([{"actor": "model"}, {"action": "archive"}],
                   items=[_item("t1")])
    assert run_report_from_state(state, triaged_label="agent/triaged").done == []


# --- wired into the graph ---------------------------------------------------

def test_a_run_records_its_report_under_its_own_run_id(wiring):
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-1"}})
    run_id = result["review"]["run_id"]
    report = wiring["done"].get(run_id)
    assert report is not None, "the run left no record of what it did"
    assert report.total == 2
    assert {r.thread_id for r in report.done} == {"t0", "t1"}
    assert all(("archive", None) in r.actions for r in report.done)


def test_a_run_that_executed_nothing_still_writes_a_report(wiring, tmp_path):
    """"It ran and did nothing" must be distinguishable from "it never ran"."""
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    wiring["client"] = SnapshotGmailClient(empty)
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 10, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-1"}})
    report = wiring["done"].get(result["review"]["run_id"])
    assert report is not None and report.done == []


def test_two_runs_leave_two_reports(wiring):
    graph = build_graph(**wiring, checkpointer=InMemorySaver())
    first = graph.invoke({"limit": 10, "mode": "incremental"},
                         {"configurable": {"thread_id": "run-1"}})
    second = graph.invoke({"limit": 10, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-2"}})
    ids = [r.run_id for r in wiring["done"].recent()]
    assert set(ids) == {first["review"]["run_id"], second["review"]["run_id"]}
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_run_report.py -q`
Expected: collection error — `ImportError: cannot import name 'run_report_from_state'`.

- [ ] **Step 3: Write the report builder**

In `inbox_agent/graph.py`, extend the imports:

```python
from .models import (Action, ActionKind, AuditRecord, Decision, DoneRecord,
                     ExecutionContext, ReviewItem, ReviewRequest, ReviewResponse,
                     RunReport, Thread)
from .store import DoneStore, HeldQueue, PreferenceStore, rule_from_correction
```

(keep whatever names the existing import lines already carry; add `DoneRecord`,
`RunReport` and `DoneStore` to them, and make sure `datetime`/`timezone` and
`Optional` are imported — they are used below.)

Add above `build_graph`:

```python
def run_report_from_state(state: TriageState, *, triaged_label: str,
                          now: Optional[datetime] = None) -> RunReport:
    """What this run did, joined into one durable record.

    Two sources, the same two the digest panel has always used. The audit
    records say what actually went through the chokepoint - the only honest
    answer to "what did you do" - but they know a thread by id, which is not
    something the owner can read. The proposals carry the subject, the sender,
    the category and the model's reason. A record whose thread is missing from
    the proposals still gets a row, named by its id: an action with no visible
    subject is strange, and hiding it would be worse.

    Read off `review.items` rather than `auto`: in an incremental run the
    executed set is a subset of `auto` and the two agree, and taking the whole
    proposal list means a record can never lose its subject to a partition
    detail.

    .get() throughout, and never raising: this runs after the actions have
    reached Gmail, so an exception here would cost the owner the report for
    work that already happened.
    """
    request = ReviewRequest.model_validate(state["review"])
    proposals = {i.thread_id: i for i in request.items}

    rows: dict[str, DoneRecord] = {}
    for record in state.get("executed", []):
        if not isinstance(record, dict):
            continue
        kind = record.get("action")
        thread_id = record.get("thread_id")
        if not kind or not thread_id:
            continue
        label = (record.get("params") or {}).get("label")
        if kind == "label" and label == triaged_label:
            continue    # bookkeeping on every thread, not work to report
        row = rows.get(thread_id)
        if row is None:
            item = proposals.get(thread_id) or ReviewItem(
                thread_id=thread_id, subject=thread_id, sender="", snippet="",
                proposed=[], reason="", confidence=0.0, source="model")
            row = DoneRecord(thread_id=thread_id, item=item)
            rows[thread_id] = row
        row.actions.append((kind, label))
        actor = str(record.get("actor", ""))
        if actor.startswith("rule:"):
            row.rule_id = actor.split(":", 1)[1]

    return RunReport(run_id=request.run_id,
                     ran_at=now or datetime.now(timezone.utc),
                     total=len(state.get("thread_ids", [])),
                     remaining=int(state.get("remaining", 0) or 0),
                     done=list(rows.values()))
```

- [ ] **Step 4: Wire it into the graph**

`build_graph`'s signature gains `done`, right after `held`:

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
    done: DoneStore,
    checkpointer=None,
):
```

and at the end of `enqueue_held`, replacing `return {}`:

```python
        # The report goes here rather than in the bot because this node already
        # holds `executed`, the proposals and the run id together - and a run
        # gets its record whether or not Telegram drove it.
        done.record(run_report_from_state(
            state, triaged_label=settings.triaged_label))
        return {}
```

Extend `enqueue_held`'s docstring with one line: `Also writes this run's
RunReport: same node, same join, and the record outlives the run.`

Note for the reviewer: the backlog path (`review -> execute -> mark_triaged`)
does not pass through `enqueue_held`, so it writes no report. That is correct
for now — `/backlog` is unwired and stays unwired (spec 8).

- [ ] **Step 5: Update every call site**

`inbox_agent/telegram/__main__.py` — build the work store once and hand it to
both, rather than reaching into `HeldQueue` for the store it holds:

```python
    work_store = open_store(settings.store_dir / "held.sqlite")
    held = HeldQueue(work_store)
    # Same store, own namespace (spec 2.1): a run report wants an embedding
    # index no more than a held item does, and a third sqlite file would be a
    # third object threaded through build_graph and Bot for no gain.
    done = DoneStore(work_store)
```

and pass `done=done` to both `build_graph(...)` and `Bot(...)` (the `Bot`
keyword lands in Task 3; add it there, not here). Import `DoneStore` from
`..store`.

`build_teaching_notebook.py` (~line 1268), inside the cell source:

```python
from inbox_agent.store import DoneStore, HeldQueue
...
teach_held = HeldQueue(build_store())
# Own namespace in the same store: the run report that makes /done survive a
# restart (see DoneStore in store.py).
teach_done = DoneStore(build_store())

g = build_graph(client=client, prefs=teach_prefs, policy=pol, llm=llm,
                settings=settings, log=log, held=teach_held, done=teach_done,
                checkpointer=checkpointer)
```

then regenerate — never edit the `.ipynb`:

```bash
python build_teaching_notebook.py
```

`tests/test_graph.py`: add `done=DoneStore(build_store())` to the `wiring`
fixture dict and to each of the six explicit-kwarg calls (lines 419, 463, 564,
640, 666, 697), and add `DoneStore` to its `inbox_agent.store` import.

`tests/test_tg_bot.py`: in the fixture, build one store for both:

```python
    from inbox_agent.store import DoneStore
    work = build_store()
    held = HeldQueue(work)
    # The report store the graph writes and the bot reads. One instance, like
    # the queue: two would let the bot show an empty /done while the graph
    # filled another.
    done = DoneStore(work)
```

and pass `done=done` to `build_graph(...)` (the `Bot(...)` keyword arrives in
Task 3).

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_run_report.py -q`
Expected: PASS (7 tests).
Run: `python -m pytest -q`
Expected: PASS. Any `TypeError: build_graph() missing 1 required keyword-only
argument: 'done'` is a call site Step 5 missed — fix it there, do not default
the parameter.

- [ ] **Step 7: Commit**

```bash
git add inbox_agent/graph.py inbox_agent/telegram/__main__.py \
        build_teaching_notebook.py inbox_agent_stage_a_explained.ipynb \
        tests/test_graph.py tests/test_tg_bot.py tests/test_run_report.py
git commit -m "Write a durable report of what each run did, in the graph"
```

---

### Task 3: The bot reports from the store, not from `_last_run`

**Files:**
- Modify: `inbox_agent/telegram/bot.py` (constructor ~line 50; state ~line 114;
  `_view` ~line 207; `_done_items` ~line 251; `_item_screen` ~line 360;
  `_category_of` ~line 724; `_rule_id_of` ~line 754; `_run_triage` ~line 948)
- Modify: `inbox_agent/telegram/__main__.py` (`Bot(...)` ~line 144)
- Modify: `tests/test_tg_bot.py` (fixture returns the store; new tests)

**Interfaces:**
- Consumes: `DoneStore`, `RunReport` (Task 1); the report the graph writes (Task 2).
- Produces:
  - `Bot(..., held: HeldQueue, done: DoneStore, ...)` — required keyword.
  - `Bot._report() -> Optional[RunReport]` — the report the screen is currently
    about. Task 4 extends it to past runs; here it resolves the current run only.
  - `Bot._report_run_id: Optional[str]`, `Bot._done_run: Optional[int]` (the
    latter stays `None` until Task 4).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tg_bot.py`. The fixture must expose the store, so change its
return to `Bot(...), t, log` with `done=done` passed in, and add these tests:

```python
def test_the_done_panel_does_not_read_the_graph_result(bot):
    """_last_run is not what the report is built from any more."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b._last_run = None                      # the restart, simulated in place
    t.edited.clear()
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    panel = t.edited[-1]["text"]
    assert "Sale 0" in panel, "the done panel came back empty without _last_run"


def test_the_digest_header_counts_come_from_the_report(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b._last_run = None
    t.sent.clear()
    b._show(edit=False)
    digest_text = t.sent[-1]["text"]
    # "Inbox · 15:00 · 4 threads" - the header the report now supplies.
    assert "4 threads" in digest_text, "the run's thread count did not survive"


def test_a_fresh_bot_over_the_same_store_still_reports_the_run(bot):
    """The test that would have caught the original defect.

    A restart empties _last_run. Under the old code the panel is empty and the
    correction below is impossible.
    """
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    run_id = b._report_run_id
    assert run_id and b.done.get(run_id) is not None

    reborn = Bot(transport=FakeTransport(), graph=b.graph, settings=b.settings,
                 held=b.held, prefs=b.prefs, done=b.done, client=b.client,
                 log=b.log, categories=b.categories)
    report = reborn.done.get(run_id)
    assert report is not None
    assert {r.thread_id for r in report.done} == {"t0", "t1", "t2", "t3"}
    # And the fields a correction needs are all on it.
    row = report.done[0]
    assert row.item.category == "promotion"
    assert row.item.reason


def test_a_correction_still_works_when_last_run_is_gone(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b._last_run = None
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    assert b.prefs.rules(), "the correction taught nothing"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_tg_bot.py -q -k "report or last_run or reborn or correction_still"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'done'`
and, once that is added, empty panels.

- [ ] **Step 3: Take the report from the store**

In `inbox_agent/telegram/bot.py`:

Constructor — add the parameter after `held` and the attribute beside it:

```python
    def __init__(self, *, transport, graph, settings: Settings,
                 held: HeldQueue, done: DoneStore, prefs: PreferenceStore,
                 client=None, log: Optional[AuditLog] = None,
                 categories: Sequence[str] = (), mode: Optional[str] = None,
                 policy_version: Optional[str] = None,
                 on_run: Optional[Callable[[datetime], None]] = None):
```

```python
        # What each run did, outliving the run after it. Injected for the same
        # reason the queue is: the graph writes it and the bot reads it, and two
        # instances would let /done be permanently empty while runs filled
        # another store.
        self.done = done
```

Import it: `from ..store import DoneStore, HeldQueue, PreferenceStore, rule_from_correction`
(match the existing import line).

State, beside `_last_run`:

```python
        self._last_run: Optional[dict] = None   # the graph result, for the run flow
        # Which run the screen is reporting. The report itself lives in the
        # store, so this is an id and not a payload - that is what makes the
        # DONE panel survive a later run and a restart.
        self._report_run_id: Optional[str] = None
        # Which past run is open, as a position in DoneStore.recent(). None
        # means the screen is about the run this process just did.
        self._done_run: Optional[int] = None
```

The lookup, next to `_view`:

```python
    def _report(self) -> Optional[RunReport]:
        """The run the screen is currently about.

        Resolved on every read rather than cached: the store is the source of
        truth, and a report held in an attribute is exactly the failure this
        change exists to remove.
        """
        if self._done_run is not None:
            runs = self.done.recent()
            if 0 <= self._done_run < len(runs):
                return runs[self._done_run]
            return None
        if not self._report_run_id:
            return None
        return self.done.get(self._report_run_id)
```

`_view` — replace the `result = self._last_run ...` block and the counting loop:

```python
        report = self._report() if run_report else None
        done_by_kind: dict[str, int] = {}
        rule_decided = 0
        for record in (report.done if report else []):
            for kind, _label in record.actions:
                done_by_kind[kind] = done_by_kind.get(kind, 0) + 1
            if record.rule_id:
                # Counted per action, exactly as the audit records were: a
                # thread a rule labelled AND archived is two pieces of work the
                # rule did, and the header has always said so.
                rule_decided += len(record.actions)
        return DigestView(
            run_at=report.ran_at if report else datetime.now(timezone.utc),
            total=report.total if report else 0,
            done_by_kind=done_by_kind,
            rule_decided=rule_decided,
            held=self.held.all(),
            digest_id=self._digest_id,
            dry_run=bool(self.settings.dry_run),
            run_report=run_report,
            remaining=report.remaining if report else 0,
        )
```

Update `_view`'s docstring: the counts come from the run's persisted report,
which is itself built from the audit records the run wrote.

`_done_items` — the whole body becomes:

```python
    def _done_items(self) -> list[DoneItem]:
        """What the run did, per thread, for the panel behind the button.

        The join happened when the report was written (graph.run_report_from_state);
        this is the screen's shape of it. The rule note is resolved here rather
        than stored, because a rule can be corrected after the run and what the
        owner needs to judge is what it says NOW.
        """
        report = self._report()
        items: list[DoneItem] = []
        for record in (report.done if report else []):
            item = DoneItem(thread_id=record.thread_id,
                            subject=record.item.subject or record.thread_id,
                            sender=record.item.sender,
                            snippet=record.item.snippet,
                            actions=[tuple(a) for a in record.actions])
            if record.rule_id:
                item.from_rule = True
                item.rule_id = record.rule_id
                rule = self._rule(record.rule_id)
                item.rule_note = (f"{rule.scope} {rule.pattern} → {rule.summary}"
                                  if rule else "")
            items.append(item)
        return items
```

`_item_screen`'s `why` lookup:

```python
        report = self._report()
        why = ""
        for record in (report.done if report else []):
            if record.thread_id == item.thread_id:
                why = record.item.reason or ""
```

`_category_of`:

```python
    def _category_of(self, thread_id: str) -> str:
        report = self._report()
        for record in (report.done if report else []):
            if record.thread_id == thread_id:
                return record.item.category or "other"
        # Still the queue's job when the thread was held rather than done.
        held = self.held.get(thread_id)
        return (held.item.category if held else "other") or "other"
```

`_rule_id_of`:

```python
    def _rule_id_of(self, item: DoneItem) -> Optional[str]:
        report = self._report()
        for record in (report.done if report else []):
            if record.thread_id == item.thread_id and record.rule_id:
                return record.rule_id
        # A held item has no executed record to read an actor out of - being
        # held is precisely what stopped it executing - so it carries the rule
        # id on the item instead. _open_item puts it there.
        return item.rule_id or None
```

`_run_triage`, in the reset block at the top:

```python
        self._done_run = None
        self._report_run_id = None
```

and immediately after a successful `graph.invoke`:

```python
        # The run's own id, not the graph result: the report is in the store and
        # this is how the screen addresses it.
        self._report_run_id = ((self._last_run or {}).get("review") or {}).get("run_id")
```

Add `RunReport` to the models import in `bot.py`.

`inbox_agent/telegram/__main__.py`: pass `done=done` to `Bot(...)`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_tg_bot.py -q`
Expected: PASS.
Run: `python -m pytest -q`
Expected: PASS. `tests/test_tg_render.py` builds `DigestView` directly and is
unaffected; if a digest assertion moved, the counting rule above is the thing to
check first.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/bot.py inbox_agent/telegram/__main__.py tests/test_tg_bot.py
git commit -m "Report from the stored run record, so a restart cannot empty the done panel"
```

---

### Task 4: `/done` — the run list, and corrections on a past run

**Files:**
- Modify: `inbox_agent/telegram/callbacks.py` (`Kind`, `_CODE_TO_KIND`, `decode`)
- Modify: `inbox_agent/telegram/render_tg.py` (new `runs_panel`, `done_panel`
  back button)
- Modify: `inbox_agent/telegram/bot.py` (`/done` command, `_show_runs`, `_show`,
  dispatch)
- Modify: `tests/test_tg_callbacks.py`, `tests/test_tg_render.py`, `tests/test_tg_bot.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3, `Bot._report()` and `Bot._done_run`.
- Produces:
  - callbacks: kind `"runs"` (code `u`, no index) and `"run"` (code `U`, one index)
  - `render_tg.runs_panel(reports: list[RunReport], *, digest_id: str, now: datetime) -> tuple[str, list]`
  - `render_tg.done_panel(view, page=0, *, back_to_runs: bool = False)`
  - `Bot._show_runs()`

- [ ] **Step 1: Write the failing callback and renderer tests**

Add to `tests/test_tg_callbacks.py`:

```python
def test_runs_and_run_round_trip():
    from inbox_agent.telegram.callbacks import decode, encode
    assert decode(encode("runs", digest_id="ab12")).kind == "runs"
    intent = decode(encode("run", 2, digest_id="ab12"))
    assert (intent.kind, intent.index, intent.digest_id) == ("run", 2, "ab12")


def test_a_run_callback_without_an_index_is_a_noop():
    from inbox_agent.telegram.callbacks import decode
    assert decode("U:ab12").kind == "noop"


def test_a_runs_callback_carrying_an_index_is_a_noop():
    from inbox_agent.telegram.callbacks import decode
    assert decode("u:1:ab12").kind == "noop"
```

Add to `tests/test_tg_render.py`:

```python
def test_runs_panel_lists_runs_newest_first_with_their_counts():
    from datetime import datetime, timezone
    from inbox_agent.models import DoneRecord, ReviewItem, RunReport
    from inbox_agent.telegram.render_tg import runs_panel

    now = datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc)

    def row(tid, kinds):
        return DoneRecord(
            thread_id=tid,
            item=ReviewItem(thread_id=tid, subject="s", sender="a@b.com",
                            snippet="", proposed=[], reason="r",
                            confidence=0.9, source="model"),
            actions=[(k, None) for k in kinds])

    reports = [
        RunReport(run_id="aaa", ran_at=now.replace(hour=15), total=22,
                  done=[row("t1", ["archive"]), row("t2", ["archive", "label"])]),
        RunReport(run_id="bbb", ran_at=now.replace(hour=12), total=9, done=[]),
    ]
    text, keyboard = runs_panel(reports, digest_id="ab12", now=now)
    lines = text.splitlines()
    assert lines[0].startswith("Done · last")
    assert "22 threads" in lines[1] and "2 archive" in lines[1]
    assert "1 label" in lines[1]
    assert "9 threads" in lines[2] and "nothing" in lines[2]
    # One numbered button per run, plus a way out.
    flat = [b for row_ in keyboard for b in row_]
    assert ("1", "U:0:ab12") in flat
    assert ("2", "U:1:ab12") in flat
    assert any("Back" in label for label, _ in flat)


def test_runs_panel_says_so_when_there_are_no_runs():
    from datetime import datetime, timezone
    from inbox_agent.telegram.render_tg import runs_panel
    text, keyboard = runs_panel([], digest_id="ab12",
                                now=datetime(2026, 9, 8, tzinfo=timezone.utc))
    assert "No runs" in text
    assert keyboard, "a screen with no way out is a trap on a phone"


def test_runs_panel_shows_at_most_ten_runs():
    from datetime import datetime, timedelta, timezone
    from inbox_agent.models import RunReport
    from inbox_agent.telegram.render_tg import RUNS_PAGE_SIZE, runs_panel
    now = datetime(2026, 9, 8, 16, 0, tzinfo=timezone.utc)
    reports = [RunReport(run_id=f"r{i}", ran_at=now - timedelta(hours=i), total=i)
               for i in range(RUNS_PAGE_SIZE + 2)]
    text, keyboard = runs_panel(reports, digest_id="ab12", now=now)
    numbered = [line for line in text.splitlines() if line[:1].isdigit()]
    assert len(numbered) == RUNS_PAGE_SIZE
    assert numbered[0].startswith("1. ")


def test_done_panel_can_go_back_to_the_run_list():
    # done_view() and done() are this file's existing helpers, above.
    _text, keyboard = done_panel(done_view([done("t1")]), 0, back_to_runs=True)
    assert keyboard[-1][0][1].startswith("u:"), "Back did not return to the runs"
    assert "runs" in keyboard[-1][0][0]


def test_done_panel_still_goes_back_to_the_digest_by_default():
    _text, keyboard = done_panel(done_view([done("t1")]))
    assert keyboard[-1][0][1].startswith("L:"), "the live panel lost its way back"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_tg_callbacks.py tests/test_tg_render.py -q`
Expected: FAIL — `KeyError: 'runs'` from `encode`, `ImportError` for `runs_panel`.

- [ ] **Step 3: Add the two callback kinds**

In `inbox_agent/telegram/callbacks.py`, extend `Kind`:

```python
               "keep", "relabel", "teach_trash",
               # The run list, and opening one run in it. Index-based like every
               # other position here: an 8-character run_id plus a digest id
               # plus a kind does not reliably fit in 64 bytes, and the
               # digest_id guard already refuses a tap from a superseded list -
               # which is exactly what a shifted run list is.
               "runs", "run",
```

`_CODE_TO_KIND`: `"u": "runs", "U": "run",`

`decode`: add `"runs"` to the no-index tuple, and `"run"` to the one-index tuple.

- [ ] **Step 4: Add the renderer**

In `inbox_agent/telegram/render_tg.py`, import `RunReport` from `..models` and add:

```python
# One tap reaches ten runs; more rows than that is a wall of numbers on a phone.
RUNS_PAGE_SIZE = 10


def _run_when(ran_at: datetime, now: datetime) -> str:
    """"today 15:00", or "6 Sep 15:00" once it is not today any more."""
    local = ran_at.astimezone()
    if local.date() == now.astimezone().date():
        return f"today {local.strftime('%H:%M')}"
    return local.strftime("%-d %b %H:%M")


def _run_summary(report: RunReport) -> str:
    """"22 threads · 18 archive · 2 label" - the digest header, one line long."""
    by_kind: dict[str, int] = {}
    for record in report.done:
        for kind, _label in record.actions:
            by_kind[kind] = by_kind.get(kind, 0) + 1
    counts = " · ".join(f"{count} {kind}"
                        for kind, count in sorted(by_kind.items(),
                                                  key=lambda kv: -kv[1]))
    threads = f"{report.total} thread" + ("" if report.total == 1 else "s")
    # Never an empty tail: a run that did nothing must not read as a run whose
    # report went missing.
    return f"{threads} · {counts}" if counts else f"{threads} · did nothing"


def runs_panel(reports: Sequence[RunReport], *, digest_id: str,
               now: datetime) -> tuple[str, list]:
    """The last runs, newest first, one tap from what each of them did.

    The only new screen in this feature: everything below it is the done panel
    and the item screen that already existed, pointed at a stored run instead of
    at the last one this process happened to do.
    """
    window = list(reports)[:RUNS_PAGE_SIZE]
    if not window:
        return ("No runs recorded yet.\n\n"
                "The next scheduled run will leave one here.",
                [[("↩ Back to the digest", encode("list", digest_id=digest_id))]])

    lines = [f"Done · last {len(window)} run" + ("" if len(window) == 1 else "s")]
    for number, report in enumerate(window, start=1):
        lines.append(f"{number}. {_run_when(report.ran_at, now)} · "
                     f"{_run_summary(report)}")

    keyboard: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    for offset in range(len(window)):
        row.append((str(offset + 1), encode("run", offset, digest_id=digest_id)))
        if len(row) == 5:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([("↩ Back to the digest", encode("list", digest_id=digest_id))])
    return "\n".join(lines)[:TG_MAX_TEXT], keyboard
```

`done_panel` gains the parameter and uses it for its last row:

```python
def done_panel(view: DigestView, page: int = 0, *,
               back_to_runs: bool = False) -> tuple[str, list]:
```

```python
    # Always last, always present: a screen with no way out is a trap on a
    # phone, where there is no Escape key. It returns where the owner came
    # from - the run list for a past run, the digest for the live one.
    if back_to_runs:
        keyboard.append([("↩ Back to the runs",
                          encode("runs", digest_id=view.digest_id))])
    else:
        keyboard.append([("↩ Back to the digest",
                          encode("list", digest_id=view.digest_id))])
```

- [ ] **Step 5: Run the renderer and callback tests**

Run: `python -m pytest tests/test_tg_callbacks.py tests/test_tg_render.py -q`
Expected: PASS.

- [ ] **Step 6: Write the failing bot tests**

Add to `tests/test_tg_bot.py`:

```python
def test_done_with_no_runs_says_so(bot):
    b, t, _ = bot
    b.handle_update(msg("/done"))
    assert "No runs" in t.sent[-1]["text"]


def test_done_lists_the_run_that_just_happened(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    t.sent.clear()
    b.handle_update(msg("/done"))
    text = t.sent[-1]["text"]
    assert text.startswith("Done · last 1 run")
    assert "4 threads" in text


def test_opening_a_past_run_shows_what_it_did(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(msg("/done"))
    b.handle_update(cb(encode("run", 0, digest_id=b._digest_id)))
    panel = t.edited[-1]["text"]
    assert "Sale 0" in panel and "archive" in panel


def test_back_from_a_past_run_returns_to_the_run_list(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(msg("/done"))
    b.handle_update(cb(encode("run", 0, digest_id=b._digest_id)))
    assert any("Back to the runs" in label
               for row in t.edited[-1]["keyboard"] for label, _ in row)
    b.handle_update(cb(encode("runs", digest_id=b._digest_id)))
    assert t.edited[-1]["text"].startswith("Done · last")


def test_a_correction_on_a_past_run_teaches_the_same_rule(bot):
    """The whole point: a run three slots ago is still correctable."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    # A second run, so the first is no longer the live one.
    b.handle_update(msg("/triage 4"))
    b.handle_update(msg("/done"))
    runs = b.done.recent()
    assert len(runs) == 2
    oldest = len(runs) - 1
    b.handle_update(cb(encode("run", oldest, digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rules = b.prefs.rules()
    assert rules, "correcting a past run taught nothing"
    assert "Learned" in t.edited[-1]["text"]


def test_a_past_runs_item_screen_shows_the_model_reason(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(msg("/done"))
    b.handle_update(cb(encode("run", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    assert "a sale" in t.edited[-1]["text"]


def test_a_tap_from_a_superseded_run_list_is_refused(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(msg("/done"))
    stale = b._digest_id
    b.handle_update(msg("/done"))            # a new list, new digest id
    t.sent.clear()
    b.handle_update(cb(encode("run", 0, digest_id=stale)))
    assert t.answered[-1]["text"].startswith("That digest is out of date")


def test_corrections_from_a_past_run_never_touch_gmail(bot):
    """A purged thread must report plainly, not raise. Nothing here re-fetches:
    _thread_for builds the Thread from what is on screen."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))

    class Exploding:
        def get_thread(self, *a, **k): raise AssertionError("re-fetched Gmail")
        def __getattr__(self, name): raise AssertionError("touched Gmail")

    b.client = Exploding()
    b.handle_update(msg("/done"))
    b.handle_update(cb(encode("run", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    assert b.prefs.rules()
```

- [ ] **Step 7: Run them and watch them fail**

Run: `python -m pytest tests/test_tg_bot.py -q -k "done or run"`
Expected: FAIL — `/done` falls through to the commands help.

- [ ] **Step 8: Wire the screen into the bot**

In `inbox_agent/telegram/bot.py`:

Import `runs_panel` alongside `done_panel` and friends.

`_on_message` — a new command, and the help line that names it:

```python
        elif command == "/done":
            self._show_runs()
```

```python
                "Commands: /triage [n] · /held · /done · /status · /cancel")
```

A new screen opener beside `_show_queue`:

```python
    def _show_runs(self) -> None:
        """The last runs, as a new message, running nothing.

        A new message rather than an edit for the reason /held is one: editing
        would replace what an earlier run reported, and the owner scrolling back
        would find a different run in its place.
        """
        self._page = 0
        self._message_id = None
        self._digest_id = self._new_digest_id()
        self._run_report = False
        self._done_run = None
        self._panel = "runs"
        self._show(edit=False)
```

`_show` gains the fourth state, and tells the done panel where Back goes:

```python
    def _show(self, *, edit: bool) -> None:
        view = self._view(run_report=self._run_report)
        if self._panel == "item":
            text, keyboard = self._item_screen()
        elif self._panel == "runs":
            text, keyboard = runs_panel(self.done.recent(),
                                        digest_id=self._digest_id,
                                        now=datetime.now(timezone.utc))
        elif self._panel == "done":
            view.done = self._done_items()
            text, keyboard = done_panel(view, self._done_page,
                                        back_to_runs=self._done_run is not None)
        else:
            text, keyboard = digest(view, self._page)
```

Dispatch, beside the existing `done` and `list` branches:

```python
        if intent.kind == "runs":
            self._panel = "runs"
            self._done_run = None
            self._show(edit=True)
            return
        if intent.kind == "run":
            # A position in the list that was drawn, never a run id: the same
            # index -> identity boundary every other button here respects.
            self._done_run = intent.index or 0
            self._done_page = 0
            self._panel = "done"
            self._show(edit=True)
            return
```

and `list` returns to the live digest, which means leaving the past run behind:

```python
        if intent.kind == "list":
            self._panel = "digest"
            # Back to the digest is back to now: leaving a past run selected
            # would stamp the live screen with an old run's counts.
            self._done_run = None
            self._show(edit=True)
            return
```

`_view`'s `run_report=False` path already returns zeroed counts, so the run list
and the digest under it stay honest about having run nothing.

- [ ] **Step 9: Run the tests**

Run: `python -m pytest tests/test_tg_bot.py -q`
Expected: PASS.
Run: `python -m pytest -q`
Expected: PASS — the full suite, which is the only completion claim worth making.

- [ ] **Step 10: Commit**

```bash
git add inbox_agent/telegram/callbacks.py inbox_agent/telegram/render_tg.py \
        inbox_agent/telegram/bot.py tests/test_tg_bot.py \
        tests/test_tg_callbacks.py tests/test_tg_render.py
git commit -m "Add /done, so the last ten runs stay correctable"
```

---

## After the last task

- `python -m pytest` green, and say the count.
- `inbox-agent doctor` still clean (it reads settings, not the new store, so this
  is a smoke check rather than a gate).
- The branch does NOT get merged from this session — hand the owner the branch
  name and the head SHA.
- `INBOX_SCHEDULE` can stay as it is; nothing here changes scheduling. What
  changes is that the four runs a day the owner was not watching are now
  correctable, which was the condition on the schedule going live.
- Worth telling the owner: the backlog path writes no report (it does not pass
  through `enqueue_held`), and `/backlog` remains unwired.

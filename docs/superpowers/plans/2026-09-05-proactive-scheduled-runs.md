# Proactive Scheduled Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the inbox agent triage on its own schedule, deferring to a review in progress, surviving a sleeping laptop, and saying when a run did not get to everything.

**Architecture:** A pure `Trigger` in a new `inbox_agent/schedule.py` answers "is a slot owed?" from values the caller passes in — no clock, no filesystem. The Telegram polling loop asks it once per iteration and calls the same run path `/triage` uses. Serialisation is free: `run_polling` is single-threaded, so a scheduled run can only begin between updates, never during one.

**Tech Stack:** Python 3.11+, stdlib `dataclasses`/`datetime`/`json`, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-05-proactive-scheduled-runs-design.md`

## Global Constraints

- **Stdlib only in `inbox_agent/schedule.py`.** No new third-party dependency is introduced by this plan.
- **`python -m pytest` must stay offline.** No network, no credentials, ~5s. 848 tests pass today; every task ends green.
- **`Trigger` is I/O-free and clock-free.** It never calls `datetime.now()` and never touches the filesystem. Callers pass `now`. This is what makes section 3 of the spec testable without a fake clock or a `tmp_path`.
- **All schedule datetimes are naive local.** `datetime.now()` with no tzinfo, throughout `schedule.py` and the loop. Do not mix in `timezone.utc` — `Attempt.at` is compared directly against slot instants derived from `now`.
- **`Attempt` is not `Bot._last_run`.** `Bot._last_run` is the previous graph result held for the digest. They are unrelated despite the names. Do not merge them.
- **Never relax the `digest_id` check** in `_on_callback`. Buttons address items by position; honouring a superseded tap acts on the wrong thread. Task 6 changes what the refusal *does*, never whether it refuses.
- **Defaults on every new `Settings` field and every new constructor argument**, so existing constructions in tests and the notebook keep working unchanged. This is the established pattern (`config.py:47` onward).
- **A bug fix lands with the test that would have caught it** (CLAUDE.md). Every task here is test-first.
- **Commit messages: one line, imperative, saying what changed and why.**

---

### Task 1: `Trigger` and `Attempt` — the scheduling rule

The whole of spec section 3, as pure functions. Nothing else in this plan can be tested until this exists.

**Files:**
- Create: `inbox_agent/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Attempt(at: datetime, slot: datetime | None = None, count: int = 0, failed: bool = False)` — frozen dataclass.
  - `Trigger(slots: tuple[time, ...] = (), grace: timedelta = 2h, cooldown: timedelta = 15m, backoff: timedelta = 5m, max_attempts: int = 2)` — frozen dataclass.
  - `Trigger.owed(now: datetime, last: Attempt | None) -> datetime | None`
  - `Trigger.latest_slot(now: datetime) -> datetime | None`
  - `Trigger.next_slot(now: datetime) -> datetime | None`
  - `scheduled_attempt(slot: datetime, now: datetime, previous: Attempt | None, *, failed: bool) -> Attempt`
  - `manual_attempt(now: datetime) -> Attempt`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schedule.py`:

```python
"""When a scheduled run is owed.

Every test here passes `now` and the previous `Attempt` explicitly: Trigger has
no clock and no filesystem, which is the whole reason section 3 of the spec is
checkable this cheaply.
"""
from datetime import datetime, time, timedelta

from inbox_agent.schedule import (Attempt, Trigger, manual_attempt,
                                  scheduled_attempt)

SLOTS = (time(9, 0), time(12, 0), time(18, 0))
TRIGGER = Trigger(slots=SLOTS)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute)


def test_a_passed_slot_with_no_prior_attempt_is_owed():
    assert TRIGGER.owed(at(7, 9, 5), None) == at(7, 9, 0)


def test_no_slots_never_owes_anything():
    assert Trigger(slots=()).owed(at(7, 9, 5), None) is None


def test_the_most_recent_slot_wins_so_three_missed_ones_collapse_to_one():
    """A laptop asleep from 08:00 to 18:30 wakes owing 18:00 alone, not 9, 12
    and 18. Collapsing is not a feature; it falls out of only ever asking
    about the latest slot."""
    assert TRIGGER.owed(at(7, 18, 30), None) == at(7, 18, 0)


def test_before_the_first_slot_of_the_day_yesterdays_last_is_the_candidate():
    """The candidate wraps to yesterday - but at 00:30 that slot is 6.5 hours
    old, so grace then rejects it. Both halves are asserted: picking the
    candidate and refusing to act on it are different steps."""
    assert TRIGGER.latest_slot(at(7, 0, 30)) == at(6, 18, 0)
    assert TRIGGER.owed(at(7, 0, 30), None) is None


def test_yesterdays_last_slot_is_owed_when_it_is_still_inside_grace():
    """Just after midnight is too late; 19:00 the evening before is not."""
    assert TRIGGER.owed(at(7, 19, 30), None) == at(7, 18, 0)


def test_a_slot_older_than_grace_is_not_owed():
    """08:30's triage must not arrive at 23:00."""
    assert TRIGGER.owed(at(7, 11, 30), None) is None


def test_an_attempted_slot_is_not_owed_again():
    last = scheduled_attempt(at(7, 9, 0), at(7, 9, 1), None, failed=False)
    assert TRIGGER.owed(at(7, 9, 5), last) is None


def test_a_run_inside_cooldown_before_the_slot_covers_it():
    """A /triage typed at 08:55 swept the same untriaged backlog the 09:00 slot
    would sweep, so 09:00 must not fire six minutes later."""
    assert TRIGGER.owed(at(7, 9, 1), manual_attempt(at(7, 8, 55))) is None


def test_a_run_older_than_cooldown_does_not_cover_the_slot():
    assert TRIGGER.owed(at(7, 9, 1), manual_attempt(at(7, 8, 40))) == at(7, 9, 0)


def test_a_failed_attempt_is_not_owed_again_before_backoff():
    last = scheduled_attempt(at(7, 9, 0), at(7, 9, 1), None, failed=True)
    assert TRIGGER.owed(at(7, 9, 3), last) is None


def test_a_failed_attempt_is_owed_again_once_backoff_has_elapsed():
    last = scheduled_attempt(at(7, 9, 0), at(7, 9, 1), None, failed=True)
    assert TRIGGER.owed(at(7, 9, 7), last) == at(7, 9, 0)


def test_a_second_failure_exhausts_the_slot():
    first = scheduled_attempt(at(7, 9, 0), at(7, 9, 1), None, failed=True)
    second = scheduled_attempt(at(7, 9, 0), at(7, 9, 7), first, failed=True)
    assert second.count == 2
    assert TRIGGER.owed(at(7, 9, 20), second) is None


def test_a_successful_attempt_is_never_owed_again_despite_backoff():
    last = scheduled_attempt(at(7, 9, 0), at(7, 9, 1), None, failed=False)
    assert TRIGGER.owed(at(7, 9, 30), last) is None


def test_grace_outranks_backoff():
    """A retry that would land outside the grace window is not owed: staleness
    beats the retry budget, or a failure at 10:58 reopens the slot at 11:03."""
    last = scheduled_attempt(at(7, 9, 0), at(7, 10, 58), None, failed=True)
    assert TRIGGER.owed(at(7, 11, 3), last) is None


def test_a_failure_at_a_different_slot_does_not_grant_a_retry_here():
    """count belongs to a slot. Yesterday's exhausted 18:00 must not stop 09:00
    from running, and must not hand it a used-up budget either."""
    last = scheduled_attempt(at(6, 18, 0), at(6, 18, 1), None, failed=True)
    assert TRIGGER.owed(at(7, 9, 5), last) == at(7, 9, 0)


def test_scheduled_attempt_counts_within_a_slot_and_resets_across_slots():
    first = scheduled_attempt(at(7, 9, 0), at(7, 9, 1), None, failed=True)
    assert first.count == 1
    same = scheduled_attempt(at(7, 9, 0), at(7, 9, 7), first, failed=True)
    assert same.count == 2
    later = scheduled_attempt(at(7, 12, 0), at(7, 12, 1), same, failed=False)
    assert later.count == 1


def test_manual_attempt_belongs_to_no_slot():
    a = manual_attempt(at(7, 8, 55))
    assert a.slot is None and a.count == 0 and a.failed is False


def test_a_repeated_local_hour_on_a_dst_day_does_not_run_twice():
    """These are naive wall-clock times, so on the autumn shift 01:30 happens
    twice. The second one is covered by the attempt the first one recorded -
    rule 3 absorbs it, and no special DST handling is needed."""
    trigger = Trigger(slots=(time(1, 30),))
    first = scheduled_attempt(at(7, 1, 30), at(7, 1, 31), None, failed=False)
    assert trigger.owed(at(7, 1, 35), first) is None


def test_a_slot_skipped_by_the_spring_shift_is_absorbed_by_grace():
    """02:30 never happens on the spring shift day. The clock jumps to 03:00,
    the slot is missed, and grace decides whether it is still worth running -
    the same rule a sleeping laptop uses."""
    trigger = Trigger(slots=(time(2, 30),))
    assert trigger.owed(at(7, 3, 0), None) == at(7, 2, 30)   # inside grace
    assert trigger.owed(at(7, 5, 0), None) is None           # outside it


def test_next_slot_is_today_when_one_remains_and_tomorrow_otherwise():
    assert TRIGGER.next_slot(at(7, 10, 0)) == at(7, 12, 0)
    assert TRIGGER.next_slot(at(7, 19, 0)) == at(8, 9, 0)
    assert Trigger(slots=()).next_slot(at(7, 10, 0)) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_schedule.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'inbox_agent.schedule'`

- [ ] **Step 3: Write the implementation**

Create `inbox_agent/schedule.py`:

```python
"""When a scheduled run is owed, and nothing else.

Pure logic: no clock, no filesystem, no bot. Every decision is made from values
the caller passes in, which is what makes the whole scheduling rule testable
without a fake clock or a temporary directory - and what keeps the rule out of
the polling loop, where it would only ever be exercised by running the bot.

All datetimes here are NAIVE LOCAL. Slots are wall-clock times of day, so the
comparison that matters is against the clock on the wall; mixing in an aware
UTC value would compare a slot to an instant eight hours away.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Attempt:
    """What the schedule remembers about the last run of any kind.

    NOT Bot._last_run, which is the previous graph result held for the digest.
    The two are unrelated despite the names.
    """
    at: datetime                    # when it was attempted
    slot: Optional[datetime] = None  # the slot it was for; None for a typed run
    count: int = 0                  # attempts against `slot` so far
    failed: bool = False            # did it raise


@dataclass(frozen=True)
class Trigger:
    """The schedule as a rule, asked once per polling iteration."""
    slots: tuple[time, ...] = ()
    grace: timedelta = timedelta(hours=2)
    cooldown: timedelta = timedelta(minutes=15)
    backoff: timedelta = timedelta(minutes=5)
    max_attempts: int = 2

    def latest_slot(self, now: datetime) -> Optional[datetime]:
        """The most recent slot instant at or before `now`.

        Only ever the most recent one. That single choice is what makes a
        laptop asleep through three slots wake owing exactly one run.
        """
        if not self.slots:
            return None
        today = [datetime.combine(now.date(), t, tzinfo=now.tzinfo)
                 for t in self.slots]
        passed = [d for d in today if d <= now]
        if passed:
            return max(passed)
        return datetime.combine(now.date() - timedelta(days=1), max(self.slots),
                                tzinfo=now.tzinfo)

    def next_slot(self, now: datetime) -> Optional[datetime]:
        """The next slot at or after `now`. For the banner and /status only."""
        if not self.slots:
            return None
        ahead = [d for d in (datetime.combine(now.date(), t, tzinfo=now.tzinfo)
                             for t in self.slots) if d > now]
        if ahead:
            return min(ahead)
        return datetime.combine(now.date() + timedelta(days=1), min(self.slots),
                                tzinfo=now.tzinfo)

    def owed(self, now: datetime, last: Optional[Attempt]) -> Optional[datetime]:
        """The slot to run now, or None."""
        slot = self.latest_slot(now)
        if slot is None:
            return None
        # Staleness is checked before anything else, so it outranks the retry
        # budget: a failure at 10:58 does not reopen a 09:00 slot at 11:03.
        if now - slot > self.grace:
            return None
        if last is not None and last.at >= slot - self.cooldown:
            # Something already covered this slot - either an attempt at it, or
            # a run close enough before it to have swept the same backlog. Only
            # a FAILED attempt at THIS slot, with budget left and its backoff
            # elapsed, reopens it.
            if not (last.failed and last.slot == slot):
                return None
            if last.count >= self.max_attempts:
                return None
            if now < last.at + self.backoff:
                return None
        return slot


def scheduled_attempt(slot: datetime, now: datetime,
                      previous: Optional[Attempt], *, failed: bool) -> Attempt:
    """Record of a scheduled run. `count` belongs to the slot, not to the day."""
    carried = previous.count if (previous and previous.slot == slot) else 0
    return Attempt(at=now, slot=slot, count=carried + 1, failed=failed)


def manual_attempt(now: datetime) -> Attempt:
    """Record of a run the owner typed. It belongs to no slot, but it still
    covers one within `cooldown` - it swept the same untriaged backlog."""
    return Attempt(at=now)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_schedule.py -q`
Expected: 20 passed

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/schedule.py tests/test_schedule.py
git commit -m "Decide when a scheduled run is owed, without a clock or a disk"
```

---

### Task 2: `ScheduleStore` — the attempt across restarts

launchd restarts the process. Catch-up and the retry budget are both worthless if the record dies with it: losing `count` is the retry storm again, one restart at a time.

**Files:**
- Modify: `inbox_agent/schedule.py` (append)
- Test: `tests/test_schedule.py` (append)

**Interfaces:**
- Consumes: `Attempt` from Task 1.
- Produces:
  - `ScheduleStore(path: Path | str)` with `.last -> Attempt | None` and `.record(attempt: Attempt) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schedule.py`:

```python
from inbox_agent.schedule import ScheduleStore


def test_an_attempt_survives_a_restart_with_its_count(tmp_path):
    """A restart must not hand a failing slot a fresh retry budget - that is the
    retry storm again, one restart at a time."""
    path = tmp_path / "schedule.json"
    first = scheduled_attempt(at(7, 9, 0), at(7, 9, 1), None, failed=True)
    second = scheduled_attempt(at(7, 9, 0), at(7, 9, 7), first, failed=True)
    ScheduleStore(path).record(second)

    reopened = ScheduleStore(path)          # a new process
    assert reopened.last == second
    assert reopened.last.count == 2
    assert TRIGGER.owed(at(7, 9, 20), reopened.last) is None


def test_a_manual_attempt_round_trips_with_no_slot(tmp_path):
    path = tmp_path / "schedule.json"
    ScheduleStore(path).record(manual_attempt(at(7, 8, 55)))
    assert ScheduleStore(path).last == manual_attempt(at(7, 8, 55))


def test_a_missing_file_is_no_previous_attempt(tmp_path):
    assert ScheduleStore(tmp_path / "nothing.json").last is None


def test_a_corrupt_file_is_no_previous_attempt_and_is_logged(tmp_path, caplog):
    """At worst one extra run. Refusing to start over an unreadable scheduling
    hint would be a far worse trade."""
    path = tmp_path / "schedule.json"
    path.write_text("{ this is not json")
    with caplog.at_level("WARNING"):
        assert ScheduleStore(path).last is None
    assert "schedule.json" in caplog.text


def test_recording_creates_the_directory(tmp_path):
    path = tmp_path / "store" / "schedule.json"
    ScheduleStore(path).record(manual_attempt(at(7, 8, 55)))
    assert path.exists()


def test_last_is_updated_in_memory_without_a_reread(tmp_path):
    store = ScheduleStore(tmp_path / "schedule.json")
    store.record(manual_attempt(at(7, 8, 55)))
    assert store.last == manual_attempt(at(7, 8, 55))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_schedule.py -q`
Expected: FAIL — `ImportError: cannot import name 'ScheduleStore'`

- [ ] **Step 3: Write the implementation**

Append to `inbox_agent/schedule.py`:

```python
class ScheduleStore:
    """The last attempt, across restarts.

    Separate from Trigger on purpose: Trigger stays pure, and this is the only
    thing here that touches a disk.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._last = self._read()

    @property
    def last(self) -> Optional[Attempt]:
        return self._last

    def _read(self) -> Optional[Attempt]:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text())
            return Attempt(
                at=datetime.fromisoformat(raw["at"]),
                slot=(datetime.fromisoformat(raw["slot"])
                      if raw.get("slot") else None),
                count=int(raw.get("count", 0)),
                failed=bool(raw.get("failed", False)))
        except Exception:
            # One extra run is the cost of not understanding this file. Refusing
            # to start would be the expensive failure.
            log.warning("could not read %s; treating it as no previous attempt",
                        self.path, exc_info=True)
            return None

    def record(self, attempt: Attempt) -> None:
        self._last = attempt
        payload = {"at": attempt.at.isoformat(),
                   "slot": attempt.slot.isoformat() if attempt.slot else None,
                   "count": attempt.count,
                   "failed": attempt.failed}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written through a temporary file and renamed: a half-written record
        # read back as corrupt is a lost retry budget, and the process this
        # runs in is one launchd will restart mid-write given the chance.
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_schedule.py -q`
Expected: 26 passed

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/schedule.py tests/test_schedule.py
git commit -m "Keep the last attempt across restarts, so a retry budget survives one"
```

---

### Task 3: `INBOX_SCHEDULE` configuration

**Files:**
- Modify: `inbox_agent/config.py` (add `schedule` to `Settings`, add `_resolve_schedule`, wire into `load_settings`)
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Settings.schedule: tuple[time, ...]`, defaulting to `()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
from datetime import time as _time

import pytest

from inbox_agent.config import load_settings


def test_schedule_is_empty_by_default(monkeypatch):
    """Off unless explicitly turned on. A proactive agent nobody asked for is
    the one change here that acts on a real mailbox unprompted."""
    monkeypatch.delenv("INBOX_SCHEDULE", raising=False)
    assert load_settings().schedule == ()


def test_schedule_parses_sorts_and_deduplicates(monkeypatch):
    monkeypatch.setenv("INBOX_SCHEDULE", "18:00, 09:00,12:00 , 09:00")
    assert load_settings().schedule == (_time(9, 0), _time(12, 0), _time(18, 0))


def test_a_malformed_schedule_raises_and_names_the_setting(monkeypatch):
    """A schedule that silently disabled itself on a typo would be a proactive
    agent that is not proactive and does not say so."""
    monkeypatch.setenv("INBOX_SCHEDULE", "08:30,noon")
    with pytest.raises(ValueError, match="INBOX_SCHEDULE"):
        load_settings()


def test_an_out_of_range_time_raises(monkeypatch):
    monkeypatch.setenv("INBOX_SCHEDULE", "25:00")
    with pytest.raises(ValueError, match="INBOX_SCHEDULE"):
        load_settings()


def test_trailing_separators_are_tolerated(monkeypatch):
    monkeypatch.setenv("INBOX_SCHEDULE", "09:00,")
    assert load_settings().schedule == (_time(9, 0),)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q -k schedule`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'schedule'`

- [ ] **Step 3: Write the implementation**

In `inbox_agent/config.py`, add `time` to the `datetime` import, then add this field to `Settings` after `tg_mode` (defaulted, like every field after `stale_after_days`, so existing constructions keep working):

```python
    # Local wall-clock times at which the bot triages on its own. Empty means it
    # only ever runs when the owner types /triage, which is the default: this is
    # the one setting that makes the agent act on a real mailbox unprompted.
    schedule: tuple[time, ...] = ()
```

Add the resolver beside the other `_resolve_*` functions:

```python
def _resolve_schedule() -> tuple[time, ...]:
    """Local times of day, sorted and de-duplicated.

    Raises rather than falling back, for the reason _resolve_triaged_label
    raises: a schedule that quietly disabled itself on a typo is a proactive
    agent that is not proactive and does not say so - and unlike a bad label,
    nothing downstream would ever produce an error naming this setting.
    """
    raw = os.getenv("INBOX_SCHEDULE", "").strip()
    if not raw:
        return ()
    slots: list[time] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue            # a trailing comma is a typo, not a failure
        hh, sep, mm = part.partition(":")
        try:
            if not sep:
                raise ValueError
            slots.append(time(int(hh), int(mm)))
        except ValueError:
            raise ValueError(
                f"INBOX_SCHEDULE contains {part!r}, which is not an HH:MM "
                f"time of day. Use a comma-separated list like "
                f"08:30,13:00,18:00, or leave it empty to run only when you "
                f"send /triage.") from None
    return tuple(sorted(set(slots)))
```

And in `load_settings`, alongside the other resolved fields:

```python
        schedule=_resolve_schedule(),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_config.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/config.py tests/test_config.py
git commit -m "Read a schedule from the environment, and refuse a typo instead of ignoring it"
```

---

### Task 4: Say what the run did not get to

The digest already reports what was scanned. A run that scanned 50 of 50 and one that scanned the first 50 of 80 render identically today, and only one of them means the owner is caught up.

**Files:**
- Modify: `inbox_agent/graph.py` (`TriageState`, `fetch`)
- Modify: `inbox_agent/telegram/bot.py:196` (`_view`)
- Modify: `inbox_agent/telegram/render_tg.py:131` (`DigestView`), `:270` (header)
- Test: `tests/test_graph.py`, `tests/test_tg_render.py`

**Interfaces:**
- Consumes: `GmailClient.list_thread_ids(query, max_ids)` — already on the protocol and both clients (`gmail.py:62`, `:107`, `:681`). No new client method.
- Produces: `TriageState["remaining"]: int`, `DigestView.remaining: int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tg_render.py` (import `DigestView` and `digest` the way that file already does):

```python
def test_the_header_reports_what_the_cap_left_behind():
    """Scanning 50 of 50 and scanning the first 50 of 80 must not look the
    same. Only one of them means the owner is caught up."""
    view = DigestView(run_at=NOW, total=50, done_by_kind={"archive": 50},
                      rule_decided=0, held=[], digest_id="ab12cd34",
                      dry_run=False, run_report=True, remaining=30)
    text, _ = digest(view)
    assert "50 threads" in text
    assert "30 more waiting" in text


def test_the_header_is_silent_when_the_cap_did_not_bind():
    """A permanently present '0 more waiting' is read for a week and then never
    again. A line that appears only when it means something keeps its meaning."""
    view = DigestView(run_at=NOW, total=12, done_by_kind={"archive": 12},
                      rule_decided=0, held=[], digest_id="ab12cd34",
                      dry_run=False, run_report=True, remaining=0)
    text, _ = digest(view)
    assert "more waiting" not in text


def test_held_reports_no_remainder_because_it_ran_nothing():
    view = DigestView(run_at=NOW, total=0, done_by_kind={}, rule_decided=0,
                      held=[], digest_id="ab12cd34", dry_run=False,
                      run_report=False, remaining=30)
    text, _ = digest(view)
    assert "more waiting" not in text
```

Append to `tests/test_graph.py`. The file's own `snapshot_file` fixture holds a
single thread, which cannot show a cap binding, so these bring their own:

```python
@pytest.fixture
def three_threads(tmp_path):
    data = [{"id": f"r{i}", "subject": f"Sale {i}", "sender": "deals@shop.com",
             "to": [], "date": "2026-08-26T10:00:00Z", "snippet": "s",
             "body": "b", "label_ids": ["INBOX", "UNREAD"]} for i in range(3)]
    p = tmp_path / "three.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def wiring_of_three(wiring, three_threads):
    return {**wiring, "client": SnapshotGmailClient(three_threads)}


def test_fetch_reports_how_many_the_limit_left_behind(wiring_of_three):
    """The probe is ids-only: it answers 'was the cap the reason this run
    stopped?' without paying to read the answer."""
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring_of_three, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 2, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-remaining"}})
    assert len(result["thread_ids"]) == 2
    assert result["remaining"] == 1


def test_fetch_reports_no_remainder_when_the_limit_was_not_reached(
        wiring_of_three):
    from langgraph.checkpoint.memory import InMemorySaver
    graph = build_graph(**wiring_of_three, checkpointer=InMemorySaver())
    result = graph.invoke({"limit": 50, "mode": "incremental"},
                          {"configurable": {"thread_id": "run-no-remaining"}})
    assert result["remaining"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tg_render.py tests/test_graph.py -q`
Expected: FAIL — `TypeError: DigestView.__init__() got an unexpected keyword argument 'remaining'`, and `KeyError: 'remaining'`

- [ ] **Step 3: Write the implementation**

In `inbox_agent/graph.py`, add to `TriageState`:

```python
    # How many more threads matched the query than this run took. Not a count
    # of the mailbox: the probe asks for limit + 1 ids and stops, so this says
    # "the cap bound, and by at least this much".
    remaining: int
```

Replace `fetch` (`graph.py:235`):

```python
    def fetch(state: TriageState) -> dict:
        limit = state.get("limit", settings.snapshot_size)
        threads = client.list_threads(limit=limit, query=settings.inbox_query)
        # Ids only, one page, nothing hydrated. list_thread_ids costs 5 quota
        # units per page of 500 against 10 to hydrate a single thread, so
        # asking whether the cap bound is far cheaper than reading what it cut.
        # It belongs here rather than in the bot: fetch already owns the query
        # and the limit, and a second caller deciding a run's corpus is a second
        # place for the two to drift.
        ids = client.list_thread_ids(query=settings.inbox_query,
                                     max_ids=limit + 1)
        return {"thread_ids": [t.id for t in threads],
                "remaining": max(0, len(ids) - len(threads))}
```

In `inbox_agent/telegram/render_tg.py`, add to `DigestView` (`:131`), defaulted so every existing construction keeps working:

```python
    remaining: int = 0
```

And in `digest` (`:270`), replace the `run_report` header branch:

```python
    if view.run_report:
        stat = " · ".join(f"{count} {kind}"
                          for kind, count in sorted(view.done_by_kind.items())
                          if count)
        # The remainder only when the cap actually bound. A permanent
        # "0 more waiting" is read for a week and then never again.
        head = [f"Inbox · {clock} · {view.total} threads"
                + (f" · {view.remaining} more waiting" if view.remaining else ""),
                f"{stat} · {len(ordered)} waiting" if stat
                else f"{len(ordered)} waiting"]
```

In `inbox_agent/telegram/bot.py`, in `_view`'s return (`:225`), add:

```python
            remaining=int((result or {}).get("remaining", 0)),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tg_render.py tests/test_graph.py tests/test_tg_bot.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/graph.py inbox_agent/telegram/render_tg.py inbox_agent/telegram/bot.py tests/test_tg_render.py tests/test_graph.py
git commit -m "Say how many threads the limit left behind, so a full run is not mistaken for a finished one"
```

---

### Task 5: Split `_start`, and add `run_scheduled`

**Files:**
- Modify: `inbox_agent/telegram/bot.py:50` (`__init__`), `:870` (`_start`)
- Test: `tests/test_tg_bot.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. `Bot` stays free of `schedule` imports — the loop owns those types (Task 7).
- Produces:
  - `Bot._run_triage(limit: int, *, failure_suffix: str = "") -> bool` — the run, no announcement, returns whether it finished.
  - `Bot.run_scheduled(slot: datetime, *, retry_in: timedelta | None = None) -> bool`
  - `Bot.on_run: Callable[[datetime], None] | None` — constructor arg, default `None`, called after a run the owner typed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tg_bot.py`:

```python
def boom(*a, **kw):
    """A run that dies partway, the way an expired token makes it."""
    raise RuntimeError("connection refused")


SLOT = datetime(2026, 9, 7, 9, 0)


def test_a_scheduled_run_sends_no_pre_notice(bot):
    """The 'this takes a few minutes' line exists because the owner typed
    something and was watching. Nobody is watching a scheduled run, and a second
    unprompted ping per slot is noise."""
    b, t, _ = bot
    b.run_scheduled(SLOT)
    assert not any("takes a few minutes" in m["text"] for m in t.sent)


def test_a_scheduled_run_still_sends_the_digest(bot):
    b, t, _ = bot
    assert b.run_scheduled(SLOT) is True
    assert any("Inbox ·" in m["text"] for m in t.sent)


def test_a_typed_triage_still_announces_itself(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    assert any("takes a few minutes" in m["text"] for m in t.sent)


def test_a_failed_scheduled_run_returns_false_and_names_the_retry(
        bot, monkeypatch):
    b, t, _ = bot
    monkeypatch.setattr(b.graph, "invoke", boom)
    assert b.run_scheduled(SLOT, retry_in=timedelta(minutes=5)) is False
    assert any("Retrying in 5 minutes" in m["text"] for m in t.sent)


def test_a_failed_scheduled_run_with_no_retry_left_says_so(bot, monkeypatch):
    """The two failure messages have to differ, or a repeat reads as a stutter
    rather than as the slot being abandoned."""
    b, t, _ = bot
    monkeypatch.setattr(b.graph, "invoke", boom)
    assert b.run_scheduled(SLOT, retry_in=None) is False
    text = " ".join(m["text"] for m in t.sent)
    assert "Retrying" not in text
    assert "next scheduled run" in text


def test_a_typed_run_notifies_on_run_and_a_scheduled_one_does_not(bot):
    """A typed /triage marks the slot too - it swept the same backlog. The loop
    records scheduled runs itself, so doing it here as well would reset the
    retry count on every attempt."""
    b, t, _ = bot
    marks = []
    b.on_run = marks.append
    b.handle_update(msg("/triage 4"))
    assert len(marks) == 1
    b.run_scheduled(SLOT)
    assert len(marks) == 1


def test_on_run_is_called_even_when_the_run_raised(bot, monkeypatch):
    """Recording the attempt is what bounds the retry. If it only happened on
    success, a failing slot would be owed again 50 seconds later, forever."""
    b, t, _ = bot
    monkeypatch.setattr(b.graph, "invoke", boom)
    marks = []
    b.on_run = marks.append
    b.handle_update(msg("/triage 4"))
    assert len(marks) == 1
```

Add `timedelta` to that file's `datetime` import.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tg_bot.py -q`
Expected: FAIL — `AttributeError: 'Bot' object has no attribute 'run_scheduled'`

- [ ] **Step 3: Write the implementation**

In `Bot.__init__` (`bot.py:50`), add the parameter (defaulted, so every existing construction keeps working) and the attribute:

```python
                 policy_version: Optional[str] = None,
                 on_run: Optional[Callable[[datetime], None]] = None):
```

```python
        # Called after a run the owner TYPED, so the schedule can mark the slot
        # it covered. Scheduled runs are recorded by the loop that started them;
        # firing this for those too would reset the retry count every attempt.
        self.on_run = on_run
```

Add `Callable` to the `typing` import at the top of the file.

Replace `_start` (`bot.py:870`) with three methods. `_start` keeps only the announcement and the marking:

```python
    def _start(self, limit: int) -> None:
        # Say something before the four minutes of silence, not after. A run is
        # one blocking graph.invoke: classify is seconds per thread and the two
        # Gmail phases are seconds per action, so a twenty-thread run is minutes
        # long and, until this line, sent nothing at all until the digest. The
        # owner cannot tell that from a bot that has died, and asked.
        self.transport.send_message(
            self.chat_id, f"Triaging up to {limit} threads. This takes a few "
                          f"minutes; the digest arrives when it is done.")
        self._run_triage(limit)
        # A typed run swept the same untriaged backlog a slot would have, so it
        # covers one. Marked even when the run raised: recording the attempt is
        # what bounds the retry.
        if self.on_run is not None:
            try:
                self.on_run(datetime.now())
            except Exception:
                log.exception("could not record the run against the schedule")

    def run_scheduled(self, slot: datetime, *,
                      retry_in: Optional[timedelta] = None) -> bool:
        """A run nobody typed. Returns whether it finished.

        No pre-notice: that line is owed to someone watching a wait they asked
        for. The caller records the attempt - it holds the retry budget.
        """
        if retry_in is not None:
            suffix = (f" Retrying in {int(retry_in.total_seconds() // 60)} "
                      f"minutes.")
        else:
            suffix = " Not retrying; the next scheduled run is the next attempt."
        log.info("scheduled triage for slot %s", slot.isoformat())
        return self._run_triage(self.settings.snapshot_size,
                                failure_suffix=suffix)

    def _run_triage(self, limit: int, *, failure_suffix: str = "") -> bool:
        """The run itself, with no announcement and no schedule bookkeeping."""
        self._run += 1
        self._runs_started += 1
        self._page = 0
        self._intents = {}
        self._message_id = None
        self._digest_id = self._new_digest_id()
        self._run_report = True
        self._panel = "digest"
        self._done_page = 0

        log.info("triage start: limit=%s run=%s", limit, self._run)
        started = time.monotonic()
        # mode="incremental": a run ACTS. The confident, reversible majority is
        # executed and only what genuinely needs the owner goes to the queue.
        try:
            self._last_run = self.graph.invoke(
                {"limit": limit, "mode": "incremental"},
                self._trace_config(limit))
        except Exception as exc:
            # Silence is indistinguishable from an empty inbox, which is a
            # failure the owner would trust for days without noticing. Say so.
            log.exception("triage failed")
            # NOT "nothing was executed" - the run can raise anywhere, including
            # after auto_execute has already pushed actions through the
            # chokepoint, so the honest claim is that it did not finish.
            self.transport.send_message(
                self.chat_id, f"Triage failed: {type(exc).__name__}. "
                              f"The run did not finish; some actions may already "
                              f"have run. /held to see the queue, /triage to "
                              f"retry." + failure_suffix)
            return False
        elapsed = time.monotonic() - started
        log.info("triage done in %.1fs: %s executed, %s held", elapsed,
                 len(self._last_run.get("executed", [])), len(self.held.all()))
        self._show(edit=False)
        self._send_health_alerts()
        return True
```

Add `timedelta` to the `datetime` import in `bot.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tg_bot.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/bot.py tests/test_tg_bot.py
git commit -m "Let a run start without anyone typing it, and say whether a retry follows"
```

---

### Task 6: The quiet gate, and a stale tap that re-renders

**Files:**
- Modify: `inbox_agent/telegram/bot.py:50` (`__init__`), `:833` (`handle_update`), `:841` (`_on_message`, `/held`), `:1018` (`_on_callback`)
- Test: `tests/test_tg_bot.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Bot.idle_for(now: datetime) -> timedelta`, `Bot._show_queue() -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tg_bot.py`:

```python
def test_idle_for_is_unbounded_before_any_update(bot):
    """A bot nobody has touched is not mid-review, so an owed run should not
    wait five minutes for a conversation that never started."""
    b, _t, _ = bot
    assert b.idle_for(datetime(2026, 9, 7, 9, 0)) > timedelta(days=365)


def test_handling_an_update_stamps_the_touch(bot):
    b, _t, _ = bot
    b.handle_update(msg("/status"))
    assert b.idle_for(datetime.now()) < timedelta(seconds=5)
    assert b.idle_for(datetime.now() + timedelta(minutes=5)) >= timedelta(minutes=5)


def test_an_unauthorised_update_does_not_count_as_the_owner_reviewing(bot):
    b, _t, _ = bot
    b.handle_update(msg("/status", chat_id=999))
    assert b.idle_for(datetime(2026, 9, 7, 9, 0)) > timedelta(days=365)


def test_a_stale_tap_sends_the_current_queue_instead_of_asking_for_a_command(bot):
    """With scheduled runs every digest but the newest is stale, so this stops
    being an edge case and becomes how an absent owner comes back to the phone.
    A toast is a banner that vanishes; the queue is on the screen."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    before = len(t.sent)
    b.handle_update(cb(encode("open", 1, digest_id="deadbeef")))
    assert len(t.sent) > before                  # a new message, not a toast
    assert "waiting" in t.sent[-1]["text"]


def test_a_stale_tap_does_not_edit_the_message_it_came_from(bot):
    """Editing would silently replace what that run reported, and the owner
    scrolling back later would find a different run in its place."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    edits = len(t.edited)
    b.handle_update(cb(encode("open", 1, digest_id="deadbeef")))
    assert len(t.edited) == edits


def test_a_stale_tap_starts_no_run(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    runs = b._runs_started
    b.handle_update(cb(encode("open", 1, digest_id="deadbeef")))
    assert b._runs_started == runs
    assert "DONE" not in t.sent[-1]["text"]      # no run report without a run


def test_a_noop_intent_only_answers_and_sends_nothing(bot):
    """Data too old or malformed to decode has no digest to be stale relative
    to, so there is nothing to re-render."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    before = len(t.sent)
    b.handle_update(cb("a:1234"))
    assert len(t.sent) == before
    assert t.answered[-1]["text"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tg_bot.py -q`
Expected: FAIL — `AttributeError: 'Bot' object has no attribute 'idle_for'`

- [ ] **Step 3: Write the implementation**

In `Bot.__init__`, beside the other UI state:

```python
        # When the owner last touched the bot. The scheduler's quiet gate reads
        # it: taps are the only honest signal of "mid-review" there is, because
        # nothing emits a done-reviewing event and a queue the owner is
        # deliberately ignoring would read as a review that never ends.
        self._last_touch: Optional[datetime] = None
```

Replace `handle_update` (`bot.py:833`):

```python
    def handle_update(self, update: dict) -> None:
        if not self._authorised(update):
            return
        # Only authorised updates count: a rejected update is not the owner
        # reviewing anything.
        self._last_touch = datetime.now()
        if "callback_query" in update:
            self._on_callback(update["callback_query"])
        elif "message" in update:
            self._on_message(update["message"])

    def idle_for(self, now: datetime) -> timedelta:
        """How long since the owner last touched the bot."""
        if self._last_touch is None:
            return timedelta.max
        return now - self._last_touch
```

Extract the `/held` body into a method, and call it from both places. Add beside `_show`:

```python
    def _show_queue(self) -> None:
        """The current queue, as a new message, running nothing.

        The queue outlives runs, so looking at it must not require producing
        more work - and with no run, no run report (see _view's run_report).
        """
        self._page = 0
        self._message_id = None
        self._digest_id = self._new_digest_id()
        self._run_report = False
        self._panel = "digest"
        self._show(edit=False)
```

In `_on_message`, replace the `/held` branch body with `self._show_queue()`.

In `_on_callback`, replace the stale-digest branch (`bot.py:1018`):

```python
        if not self._digest_id or intent.digest_id != self._digest_id:
            # A tap on a superseded digest. Positions have shifted since that
            # message was drawn, so acting on it would act on the wrong thread -
            # this check stays exactly as strict as it was.
            #
            # What changed is the refusal. Telling the owner to send /held was
            # tuned for a rare event; with scheduled runs every digest but the
            # newest is stale, so this is simply how an absent owner comes back
            # to their phone. A toast is a banner that vanishes, so send the
            # queue instead of asking for a command.
            #
            # A NEW message, never an edit: editing would silently replace what
            # that run reported, and the owner scrolling back later would find a
            # different run in its place.
            log.info("re-rendered the queue for a callback from digest %r "
                     "(current %r)", intent.digest_id, self._digest_id)
            self._ack(answer, "That digest is out of date - here is the "
                              "current queue.")
            self._show_queue()
            return
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tg_bot.py tests/test_tg_callbacks.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/bot.py tests/test_tg_bot.py
git commit -m "Answer a stale tap with the queue, and notice when the owner is mid-review"
```

---

### Task 7: `_tick`, and the schedule in the loop

**Files:**
- Modify: `inbox_agent/telegram/bot.py:1227` (`run_polling`)
- Test: `tests/test_tg_bot.py`

**Interfaces:**
- Consumes: `Trigger`, `ScheduleStore`, `scheduled_attempt`, `manual_attempt` (Tasks 1–2); `Bot.run_scheduled`, `Bot.idle_for`, `Bot.on_run` (Tasks 5–6).
- Produces:
  - `QUIET: timedelta` (5 min), `MAX_DEFER: timedelta` (30 min) — module constants.
  - `_tick(bot, transport, offset, trigger, store, *, idle=1.0) -> int | None`
  - `run_polling(bot, transport, *, idle=1.0, trigger=None, store=None) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tg_bot.py`:

```python
from datetime import time as _time

from inbox_agent.schedule import ScheduleStore, Trigger, manual_attempt
from inbox_agent.telegram.bot import _tick


class SilentTransport(FakeTransport):
    """get_updates returns nothing, so _tick only ever runs the schedule."""
    def get_updates(self, offset=None, timeout=50):
        return []


def always_due():
    """A slot on every hour: one is always owed, without freezing the clock."""
    return Trigger(slots=tuple(_time(h, 0) for h in range(24)))


def test_tick_runs_an_owed_slot_when_the_owner_is_quiet(bot, tmp_path):
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    store = ScheduleStore(tmp_path / "schedule.json")
    _tick(b, t, None, always_due(), store, idle=0)
    assert b._runs_started == 1
    assert store.last is not None and store.last.slot is not None


def test_tick_defers_while_the_owner_is_tapping(bot, tmp_path):
    """A scheduled digest must not pull the screen out from under a thumb."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    b._last_touch = datetime.now()               # tapped a moment ago
    store = ScheduleStore(tmp_path / "schedule.json")
    _tick(b, t, None, always_due(), store, idle=0)
    assert b._runs_started == 0
    assert store.last is None


def test_tick_runs_anyway_once_the_defer_cap_is_past(bot, tmp_path):
    """Otherwise an owner who taps something every few minutes for an afternoon
    starves the schedule silently."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    b._last_touch = datetime.now()
    store = ScheduleStore(tmp_path / "schedule.json")
    # A slot 31 minutes old: past MAX_DEFER, still well inside grace.
    stale = (datetime.now() - timedelta(minutes=31)).time().replace(
        second=0, microsecond=0)
    _tick(b, t, None, Trigger(slots=(stale,)), store, idle=0)
    assert b._runs_started == 1


def test_tick_does_nothing_without_a_trigger_or_a_store(bot):
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    _tick(b, t, None, None, None, idle=0)
    assert b._runs_started == 0


def test_a_failed_scheduled_run_is_recorded_so_it_is_not_retried_at_once(
        bot, tmp_path, monkeypatch):
    """The whole point of recording on failure: without it the slot is owed
    again 50 seconds later, about 140 times before grace closes."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    monkeypatch.setattr(b.graph, "invoke", boom)
    store = ScheduleStore(tmp_path / "schedule.json")
    trigger = always_due()
    _tick(b, t, None, trigger, store, idle=0)
    assert store.last.failed is True and store.last.count == 1
    _tick(b, t, None, trigger, store, idle=0)
    assert b._runs_started == 1                  # backoff has not elapsed
    assert store.last.count == 1


def test_a_typed_triage_is_recorded_and_covers_the_slot(bot, tmp_path):
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    store = ScheduleStore(tmp_path / "schedule.json")
    b.on_run = lambda at: store.record(manual_attempt(at))   # what run_polling wires
    b.handle_update(msg("/triage 4"))
    assert store.last is not None and store.last.slot is None
    _tick(b, t, None, always_due(), store, idle=0)
    assert b._runs_started == 1                  # cooldown covered the slot


def test_tick_still_drains_updates(bot):
    b, _old, _ = bot

    class OneUpdate(FakeTransport):
        def get_updates(self, offset=None, timeout=50):
            return [] if offset else [dict(update_id=7, **msg("/status"))]

    t = b.transport = OneUpdate()
    assert _tick(b, t, None, None, None, idle=0) == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tg_bot.py -q`
Expected: FAIL — `ImportError: cannot import name '_tick' from 'inbox_agent.telegram.bot'`

- [ ] **Step 3: Write the implementation**

In `inbox_agent/telegram/bot.py`, add the import at the top:

```python
from ..schedule import ScheduleStore, Trigger, manual_attempt, scheduled_attempt
```

Replace `run_polling` (`bot.py:1227`):

```python
# How long the owner must have been quiet before an owed run may start, and how
# long an owed run will wait for that quiet before going anyway. Module
# constants rather than settings: neither has a second sensible value, and every
# setting is another value resolvable from another place.
QUIET = timedelta(minutes=5)
MAX_DEFER = timedelta(minutes=30)


def _run_due(bot: Bot, trigger: Optional[Trigger],
             store: Optional[ScheduleStore]) -> None:
    """Start the owed run, if there is one and now is a fair moment for it."""
    if trigger is None or store is None:
        return
    now = datetime.now()
    slot = trigger.owed(now, store.last)
    if slot is None:
        return
    if bot.idle_for(now) < QUIET and now - slot <= MAX_DEFER:
        return                      # mid-review, and not yet late enough to win
    previous = store.last
    carried = previous.count if (previous and previous.slot == slot) else 0
    # Worked out BEFORE the run so the failure message can name the retry. Grace
    # outranks backoff: a retry that would land outside the window is not one.
    retry_in = (trigger.backoff
                if carried + 1 < trigger.max_attempts
                and (now + trigger.backoff) - slot <= trigger.grace
                else None)
    ok = bot.run_scheduled(slot, retry_in=retry_in)
    # Recorded on every path. If this were reachable only after a success, a
    # failing slot would be owed again on the next iteration - about 140 times
    # before grace closes, each one re-running fetch and classification to reach
    # the same exception.
    store.record(scheduled_attempt(slot, datetime.now(), previous, failed=not ok))


def _tick(bot: Bot, transport: HttpTransport, offset: Optional[int],
          trigger: Optional[Trigger], store: Optional[ScheduleStore],
          *, idle: float = 1.0) -> Optional[int]:
    """One pass of the loop: drain the updates, then run the schedule.

    Extracted from run_polling so the schedule is testable without a while True.
    """
    try:
        updates = transport.get_updates(offset=offset)
    except Exception as exc:            # network blips must not kill the bot
        log.warning("getUpdates failed: %s", exc)
        time.sleep(idle * 5)
        return offset
    for update in updates:
        offset = update["update_id"] + 1
        try:
            bot.handle_update(update)
        except Exception:
            # One bad update must not take down a long-lived process that a
            # parked run depends on. The checkpoint survives; log and go on.
            log.exception("handler failed for update %s", update.get("update_id"))
    if not updates:
        time.sleep(idle)
    # After the batch, never during one: this is the whole of the serialisation
    # story. The loop is single-threaded, so a run can only begin at a point
    # where no update is being handled.
    _run_due(bot, trigger, store)
    return offset


def run_polling(bot: Bot, transport: HttpTransport, *, idle: float = 1.0,
                trigger: Optional[Trigger] = None,
                store: Optional[ScheduleStore] = None) -> None:
    """Long-poll forever. One update at a time, in order."""
    offset = None
    log.info("polling as chat %s in %s mode", bot.chat_id, bot.mode)
    if store is not None:
        # A run the owner typed marks the slot it covered, and the bot has no
        # business holding the store to do it.
        bot.on_run = lambda at: store.record(manual_attempt(at))
    while True:
        offset = _tick(bot, transport, offset, trigger, store, idle=idle)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/bot.py tests/test_tg_bot.py
git commit -m "Ask the schedule once a loop, between updates and never during one"
```

---

### Task 8: Wiring, and four places that say what the schedule is

A bot that acts unasked has to be more legible, not less.

**Files:**
- Modify: `inbox_agent/telegram/__main__.py:70` (`main` — build, wire, banner)
- Modify: `inbox_agent/doctor.py:107` (`run_checks`)
- Modify: `inbox_agent/telegram/bot.py:939` (`_status`)
- Test: `tests/test_tg_banner.py`, `tests/test_doctor.py`, `tests/test_tg_bot.py`

**Interfaces:**
- Consumes: `Settings.schedule` (Task 3), `Trigger`/`ScheduleStore` (Tasks 1–2), `run_polling(..., trigger=, store=)` (Task 7).
- Produces: `_schedule_banner(settings, now=None) -> str` in `telegram/__main__.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tg_banner.py`:

```python
from datetime import datetime, time

from inbox_agent.telegram.__main__ import _schedule_banner


def test_the_banner_names_the_slots_and_the_next_one(monkeypatch):
    """The banner is the honest source - it is printed by the process that
    holds the config, which is why it is trusted over .env."""
    monkeypatch.setenv("INBOX_SCHEDULE", "09:00,12:00,18:00")
    line = _schedule_banner(load_settings(), now=datetime(2026, 9, 7, 10, 0))
    assert "09:00" in line and "12:00" in line and "18:00" in line
    assert "next 12:00" in line


def test_the_banner_says_off_when_there_is_no_schedule(monkeypatch):
    monkeypatch.delenv("INBOX_SCHEDULE", raising=False)
    assert "off" in _schedule_banner(load_settings())
```

Append to `tests/test_doctor.py`:

```python
def test_doctor_reports_the_schedule(monkeypatch):
    monkeypatch.setenv("INBOX_SCHEDULE", "09:00,18:00")
    from inbox_agent.config import load_settings
    from inbox_agent.doctor import run_checks
    row = next(c for c in run_checks(load_settings()) if c.name == "INBOX_SCHEDULE")
    assert "09:00" in row.value and "18:00" in row.value
    assert row.source


def test_doctor_reports_an_absent_schedule_as_off(monkeypatch):
    monkeypatch.delenv("INBOX_SCHEDULE", raising=False)
    from inbox_agent.config import load_settings
    from inbox_agent.doctor import run_checks
    row = next(c for c in run_checks(load_settings()) if c.name == "INBOX_SCHEDULE")
    assert row.value == "off"
```

Append to `tests/test_tg_bot.py`:

```python
def test_status_names_the_next_scheduled_run(bot):
    b, t, _ = bot
    b.settings = replace(b.settings, schedule=(_time(9, 0), _time(18, 0)))
    b.handle_update(msg("/status"))
    assert "scheduled" in t.sent[-1]["text"].lower()


def test_status_says_when_there_is_no_schedule(bot):
    b, t, _ = bot
    b.settings = replace(b.settings, schedule=())
    b.handle_update(msg("/status"))
    assert "no schedule" in t.sent[-1]["text"].lower()
```

Add `from dataclasses import replace` to that test file's imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tg_banner.py tests/test_doctor.py tests/test_tg_bot.py -q`
Expected: FAIL — `ImportError: cannot import name '_schedule_banner'`, and no `INBOX_SCHEDULE` row

- [ ] **Step 3: Write the implementation**

In `inbox_agent/telegram/__main__.py`, add the imports and the banner helper beside `_tracing_banner`:

```python
from datetime import datetime

from ..schedule import ScheduleStore, Trigger


def _schedule_banner(settings, now: Optional[datetime] = None) -> str:
    """What the schedule is, from the process that holds it.

    A .env edit does not reach a running bot - load_dotenv runs once, at import -
    so the banner is the honest source and doctor can only report the file.
    """
    if not settings.schedule:
        return "schedule  : off   <- runs only when you send /triage"
    slots = ", ".join(t.strftime("%H:%M") for t in settings.schedule)
    nxt = Trigger(slots=settings.schedule).next_slot(now or datetime.now())
    return f"schedule  : {slots} local (next {nxt.strftime('%H:%M')})"
```

In `main`, print it beside the other banner lines (after the `mode` line):

```python
    print(_schedule_banner(settings))
```

and build the two objects, then hand them to the loop:

```python
        trigger = Trigger(slots=settings.schedule)
        store = ScheduleStore(settings.store_dir / "schedule.json")
        run_polling(bot, transport, trigger=trigger, store=store)
```

In `inbox_agent/doctor.py`, add a row to `run_checks` beside `INBOX_STORE_DIR`:

```python
        _setting("INBOX_SCHEDULE",
                 ", ".join(t.strftime("%H:%M") for t in s.schedule) or "off"),
```

In `inbox_agent/telegram/bot.py`, extend `_status` (`:939`). Add the import at the top:

```python
from ..schedule import Trigger
```

(Task 7 already imports from `..schedule`; extend that line rather than adding a second.)

Then, before the `send_message` call in `_status`:

```python
        # Read off Settings rather than a Trigger the bot holds: the loop owns
        # the scheduling decision, and this only reports it.
        if self.settings.schedule:
            nxt = Trigger(slots=self.settings.schedule).next_slot(datetime.now())
            body += f"\n\nNext scheduled run: {nxt.strftime('%H:%M')}"
        else:
            body += "\n\nNo schedule configured; runs happen when you send /triage."
```

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass (848 + roughly 45 new)

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/__main__.py inbox_agent/doctor.py inbox_agent/telegram/bot.py tests/test_tg_banner.py tests/test_doctor.py tests/test_tg_bot.py
git commit -m "Say what the schedule is in the banner, doctor and /status"
```

---

## After the plan

`INBOX_SCHEDULE` stays empty. Spec section 8: the done record is not durable —
`_done_items` reads one in-memory slot, so only the most recent run is
correctable and none are after a restart. Three unattended runs a day makes two
of every three uncorrectable, and corrections are the learning loop. The
schedule is off by default, so this is a decision and not a mechanism: build the
durable done view, then turn this on.

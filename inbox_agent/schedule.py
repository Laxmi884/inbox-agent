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

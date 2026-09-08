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


def test_record_against_an_unwritable_store_dir_does_not_raise(tmp_path, caplog):
    """A raise out of record() would propagate past _run_due's finally straight
    into _tick and kill run_polling's while True - the retry budget must not
    depend on the disk being writable. `self._last` is set before the write is
    even attempted, so an unwritable store_dir still bounds the retry for the
    life of this process; it only loses that bound across a restart, which is
    the honest, cheaper trade.

    The parent is an existing FILE (not a permission bit, which root and CI
    sometimes ignore) so mkdir(parents=True) fails the same way on macOS as
    anywhere else, and nothing needs cleaning up afterwards - tmp_path is
    thrown away by pytest either way."""
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("i am a file, not a directory")
    path = blocked / "schedule.json"
    store = ScheduleStore(path)
    attempt = manual_attempt(at(7, 8, 55))
    with caplog.at_level("WARNING"):
        store.record(attempt)                   # must not raise
    assert store.last == attempt
    assert "schedule.json" in caplog.text
    assert not path.exists()

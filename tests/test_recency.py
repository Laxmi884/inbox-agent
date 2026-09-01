"""Age-based demotion.

`Thread.date` has existed since Stage A and participates in no decision - it is
printed into the prompt and nothing else. But a `needs_reply` from two years ago
is not a needs-reply: whatever was waiting on you happened without you, and
treating it as urgent is actively wrong when clearing a backlog.

Deterministic on purpose. Asking a 12B model to reason about dates is exactly
the kind of thing it does unreliably, and this project's stance is to keep
decidable things out of the model.
"""
from datetime import datetime, timedelta, timezone

import pytest

from inbox_agent.models import Action, Decision, Thread
from inbox_agent.recency import KEEP_IN_INBOX, age_days, demote_stale

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def thread(date="2026-08-30T10:00:00Z", **kw):
    base = dict(id="t1", subject="S", sender="a@b.com", to=[], date=date,
                snippet="s", body="", label_ids=["INBOX"])
    return Thread(**(base | kw))


def decision(category="needs_reply", kind="none", **kw):
    base = dict(thread_id="t1", category=category,
                actions=[Action(kind=kind, thread_id="t1")],
                reason="a person is waiting", confidence=0.9, source="model")
    return Decision(**(base | kw))


# --- age_days ---------------------------------------------------------------

def test_age_days_measures_from_the_thread_date():
    assert age_days(thread("2026-08-30T10:00:00Z"), now=NOW) == pytest.approx(1.58, abs=0.1)
    assert age_days(thread("2024-09-01T00:00:00Z"), now=NOW) == pytest.approx(730, abs=1)


def test_age_days_returns_none_for_an_unparseable_date():
    """Never raise on real-world mail. A missing date means 'unknown age',
    which must not be silently treated as 'ancient'."""
    for bad in ("", "not a date", "0000-00-00", "26/08/2026"):
        assert age_days(thread(bad), now=NOW) is None


def test_age_days_handles_a_naive_timestamp_without_crashing():
    assert age_days(thread("2026-08-30T10:00:00"), now=NOW) is not None


def test_a_future_dated_thread_is_age_zero_not_negative():
    """Clock skew and bad senders produce future dates; they are not fresh
    by a negative amount."""
    assert age_days(thread("2027-01-01T00:00:00Z"), now=NOW) == 0.0


# --- demotion ---------------------------------------------------------------

def test_a_stale_needs_reply_is_archived_instead_of_kept():
    d = demote_stale(decision("needs_reply", "none"), thread("2024-09-01T00:00:00Z"),
                     stale_after_days=90, now=NOW)
    assert [a.kind for a in d.actions] == ["archive"]


def test_the_category_is_preserved_so_the_audit_trail_stays_honest():
    """It WAS a needs_reply. Rewriting the category would destroy the fact and
    leave the audit log claiming the model said something it did not."""
    d = demote_stale(decision("needs_reply", "none"), thread("2024-09-01T00:00:00Z"),
                     stale_after_days=90, now=NOW)
    assert d.category == "needs_reply"
    assert "730" in d.reason or "stale" in d.reason.lower()


def test_a_recent_needs_reply_is_untouched():
    d0 = decision("needs_reply", "none")
    d = demote_stale(d0, thread("2026-08-30T10:00:00Z"), stale_after_days=90, now=NOW)
    assert d == d0


def test_only_keep_in_inbox_categories_are_demoted():
    """Demotion means 'stop holding this in the inbox'. A category that was
    already leaving the inbox has nothing to demote."""
    for category in ("promotion", "newsletter_noise", "receipt"):
        d0 = decision(category, "archive")
        d = demote_stale(d0, thread("2020-01-01T00:00:00Z"), stale_after_days=90, now=NOW)
        assert d == d0, category


def test_keep_in_inbox_set_is_the_thing_being_demoted():
    assert "needs_reply" in KEEP_IN_INBOX
    assert "important_fyi" in KEEP_IN_INBOX
    assert "promotion" not in KEEP_IN_INBOX


def test_an_unknown_age_is_never_demoted():
    """Unknown age must not be treated as ancient - that would silently archive
    mail whose date header we simply failed to parse."""
    d0 = decision("needs_reply", "none")
    d = demote_stale(d0, thread("garbage"), stale_after_days=90, now=NOW)
    assert d == d0


def test_a_rule_decided_thread_is_never_demoted():
    """A rule is the owner's explicit instruction. Age does not override it -
    that is the authority ordering from the spec's 'how it learns'."""
    d0 = decision("needs_reply", "none", source="rule", rule_id="r-1")
    d = demote_stale(d0, thread("2020-01-01T00:00:00Z"), stale_after_days=90, now=NOW)
    assert d == d0


def test_demotion_never_escalates_to_trash():
    """Stale is a reason to stop holding something in the inbox. It is never a
    reason to destroy it, and trash is the one reversible action with a
    deadline on the reversal."""
    d = demote_stale(decision("needs_reply", "none"), thread("2010-01-01T00:00:00Z"),
                     stale_after_days=90, now=NOW)
    assert all(a.kind != "trash" for a in d.actions)


def test_demotion_preserves_a_label_and_adds_the_archive():
    """A stale thread that was going to be labelled still gets its label - it
    just also leaves the inbox."""
    d0 = decision("important_fyi", "label")
    d0.actions[0].params = {"label": "important_fyi"}
    d = demote_stale(d0, thread("2020-01-01T00:00:00Z"), stale_after_days=90, now=NOW)
    assert [a.kind for a in d.actions] == ["label", "archive"]
    assert d.actions[0].params["label"] == "important_fyi"

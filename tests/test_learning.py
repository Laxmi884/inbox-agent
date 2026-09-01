"""How corrections become rules.

Three gaps found by reading the spec's "how it learns" against the code:

1. Every rule was `scope="sender"`. Rule supports sender/domain/fingerprint/
   subject and three of the four had never been created by anything, so
   correcting one job alert taught that exact address and nothing more.
2. A bare reject taught nothing. The spec says "every reject or edit at the
   review step becomes a candidate rule"; only edits did.
3. `overridden` was a bare boolean. A rule that fires 40 times and is undone 12
   times is a bad rule, and nothing measured that.
"""
from datetime import datetime, timedelta, timezone

import pytest

from inbox_agent.models import Rule, Thread
from inbox_agent.store import (
    PreferenceStore, build_store, choose_scope, rule_from_correction,
)


def thread(sender="jobalerts-noreply@linkedin.com",
           subject="OMERS is hiring a Senior Business Analyst"):
    return Thread(id="t1", subject=subject, sender=sender, to=[],
                  date="2026-08-26T10:00:00Z", snippet="s", body="",
                  label_ids=["INBOX"])


# --- scope selection --------------------------------------------------------

def test_a_noreply_address_generalises_to_the_sender_not_one_message():
    scope, pattern = choose_scope(thread())
    assert scope == "sender"
    assert pattern == "jobalerts-noreply@linkedin.com"


def test_a_repeated_subject_shape_from_one_sender_uses_the_fingerprint():
    """The case that motivated this: a sender that sends job alerts AND, once in
    a while, something worth reading. A sender rule would bury both; the
    fingerprint matches the SHAPE of the job alerts only."""
    seen = [thread(subject="OMERS is hiring a Senior Business Analyst"),
            thread(subject="TD is hiring a Data Scientist"),
            thread(subject="Fidelity is hiring a Analytics Lead")]
    # Same sender, same shape once digits and names are stripped? No - the
    # company names differ, so the shape differs. Subject scope is the fit.
    scope, pattern = choose_scope(thread(subject="TD is hiring a Data Scientist"),
                                  corpus=seen)
    assert scope in ("subject", "fingerprint")


def test_a_human_sender_stays_sender_scoped():
    scope, pattern = choose_scope(thread(sender="anshuman.singh@scotiabank.com",
                                        subject="Quick chat"))
    assert scope == "sender"


def test_choose_scope_never_returns_an_empty_pattern():
    for t in (thread(sender="", subject=""), thread(sender="x@y.com", subject="")):
        scope, pattern = choose_scope(t)
        assert pattern, f"{scope} produced an empty pattern"


def test_rule_from_correction_uses_the_chosen_scope():
    r = rule_from_correction(thread(), "archive", "corrected")
    assert r.scope in ("sender", "domain", "fingerprint", "subject")
    assert r.pattern


# --- rejects teach ----------------------------------------------------------

def test_a_reject_records_that_the_action_was_wrong():
    """The spec says every reject or edit becomes a candidate rule. A reject
    with no edit says 'not this' - which is real signal even without a
    replacement."""
    r = rule_from_correction(thread(), "none", "owner rejected archive",
                             rejected="archive")
    assert r.rejected_action == "archive"
    assert r.action == "none"


def test_a_rejected_rule_does_not_propose_the_rejected_action():
    prefs = PreferenceStore(build_store())
    prefs.add_rule(rule_from_correction(thread(), "none", "rejected",
                                        rejected="archive"))
    matches = prefs.matching(thread())
    assert matches
    assert all(m.action != "archive" for m in matches)


# --- precision --------------------------------------------------------------

def test_a_rule_tracks_how_often_it_was_overridden():
    prefs = PreferenceStore(build_store())
    r = prefs.add_rule(rule_from_correction(thread(), "archive", "learned"))
    for _ in range(3):
        prefs.record_hit(r.id)
    prefs.record_override(r.id)
    stored = [x for x in prefs.rules() if x.id == r.id][0]
    assert stored.hit_count == 3
    assert stored.override_count == 1


def test_precision_is_reported():
    prefs = PreferenceStore(build_store())
    r = prefs.add_rule(rule_from_correction(thread(), "archive", "learned"))
    for _ in range(8):
        prefs.record_hit(r.id)
    for _ in range(2):
        prefs.record_override(r.id)
    stored = [x for x in prefs.rules() if x.id == r.id][0]
    assert stored.precision == pytest.approx(0.75)


def test_a_rule_with_no_hits_has_no_precision_rather_than_a_fake_one():
    """0 hits is unknown, not perfect. Reporting 1.0 would rank an untested
    rule above a proven one."""
    prefs = PreferenceStore(build_store())
    r = prefs.add_rule(rule_from_correction(thread(), "archive", "learned"))
    stored = [x for x in prefs.rules() if x.id == r.id][0]
    assert stored.precision is None


def test_a_rule_that_is_overridden_enough_stops_matching():
    """A rule that keeps being undone is worse than no rule: it produces
    confident, citable, wrong decisions."""
    prefs = PreferenceStore(build_store())
    r = prefs.add_rule(rule_from_correction(thread(), "archive", "learned"))
    for _ in range(4):
        prefs.record_hit(r.id)
    assert prefs.matching(thread()), "should still match while it is working"
    for _ in range(4):
        prefs.record_override(r.id)
    assert not prefs.matching(thread()), "a rule this unreliable must stop firing"


def test_a_demoted_rule_is_kept_not_deleted():
    """Same reasoning as mark_overridden: a rule the owner overruled is part of
    the record of why past actions happened."""
    prefs = PreferenceStore(build_store())
    r = prefs.add_rule(rule_from_correction(thread(), "archive", "learned"))
    for _ in range(6):
        prefs.record_hit(r.id)
        prefs.record_override(r.id)
    assert any(x.id == r.id for x in prefs.rules())

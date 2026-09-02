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

from inbox_agent.models import ActionTemplate, Rule, Thread
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
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "corrected")
    assert r.scope in ("sender", "domain", "fingerprint", "subject")
    assert r.pattern


# --- rejects teach ----------------------------------------------------------

def test_a_reject_records_that_the_action_was_wrong():
    """The spec says every reject or edit becomes a candidate rule. A reject
    with no edit says 'not this' - which is real signal even without a
    replacement."""
    r = rule_from_correction(thread(), [ActionTemplate(kind="none")], "owner rejected archive",
                             rejected="archive")
    assert r.rejected_action == "archive"
    assert [a.kind for a in r.actions] == ["none"]


def test_a_rejected_rule_does_not_propose_the_rejected_action():
    prefs = PreferenceStore(build_store())
    prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="none")], "rejected",
                                        rejected="archive"))
    matches = prefs.matching(thread())
    assert matches
    assert all("archive" not in [a.kind for a in m.actions] for m in matches)


# --- precision --------------------------------------------------------------

def test_a_rule_tracks_how_often_it_was_overridden():
    prefs = PreferenceStore(build_store())
    r = prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "learned"))
    for _ in range(3):
        prefs.record_hit(r.id)
    prefs.record_override(r.id)
    stored = [x for x in prefs.rules() if x.id == r.id][0]
    assert stored.hit_count == 3
    assert stored.override_count == 1


def test_precision_is_reported():
    prefs = PreferenceStore(build_store())
    r = prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "learned"))
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
    r = prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "learned"))
    stored = [x for x in prefs.rules() if x.id == r.id][0]
    assert stored.precision is None


def test_a_rule_that_is_overridden_enough_stops_matching():
    """A rule that keeps being undone is worse than no rule: it produces
    confident, citable, wrong decisions."""
    prefs = PreferenceStore(build_store())
    r = prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "learned"))
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
    r = prefs.add_rule(rule_from_correction(thread(), [ActionTemplate(kind="archive")], "learned"))
    for _ in range(6):
        prefs.record_hit(r.id)
        prefs.record_override(r.id)
    assert any(x.id == r.id for x in prefs.rules())


# --- what a correction overrode --------------------------------------------
# `record_override` bumps an integer and nothing records WHAT the owner wanted
# instead, so a rule wrong the same way four times is indistinguishable from a
# rule wrong four different ways. The first should converge on the corrected
# action; the second should die. Both currently die, at MIN_HITS_BEFORE_DEMOTION.
#
# The replacement action IS captured - on the new rule - but nothing joins the
# new rule back to the one it replaced, so the pairing is unrecoverable after
# the fact. `supersedes` records the join at the only moment it is known.

def test_a_correction_records_which_rule_it_overrode():
    r = rule_from_correction(thread(), [ActionTemplate(kind="label", params={"label": "job_alerts"})],
                             "corrected", supersedes="r-old1234")
    assert r.supersedes == "r-old1234"


def test_a_correction_of_a_model_decision_supersedes_nothing():
    """Only a rule can be overridden. A correction of the model's own judgement
    has no prior rule to point at, and must not invent one."""
    r = rule_from_correction(thread(), [ActionTemplate(kind="archive")], "corrected")
    assert r.supersedes is None


def test_rules_stored_before_supersedes_existed_still_load():
    """Real rules are on disk. A new required field would make them unloadable."""
    legacy = {"id": "r-old", "scope": "sender", "pattern": "a@b.com",
              "actions": [{"kind": "archive", "params": {}}], "provenance": "note",
              "created_at": "2026-09-01T00:00:00Z"}
    assert Rule.model_validate(legacy).supersedes is None


def test_learn_from_response_links_the_new_rule_to_the_one_it_overrode():
    """The interrupt path learns too, and has the overriding rule id in the
    proposal it is correcting."""
    from inbox_agent.graph import learn_from_response
    from inbox_agent.models import ReviewResponse

    prefs = PreferenceStore(build_store())
    response = ReviewResponse(decisions={"t1": "reject"})
    proposals = [{"thread_id": "t1", "actions": [{"kind": "archive"}],
                  "rule_id": "r-old1234", "source": "rule"}]
    learned, _ = learn_from_response(response, [thread()], prefs, proposals)
    assert learned
    stored = [r for r in prefs.rules() if r.id == learned[0]][0]
    assert stored.supersedes == "r-old1234"

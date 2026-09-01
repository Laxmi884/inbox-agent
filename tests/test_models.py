import json
from datetime import datetime, timezone

from inbox_agent.models import (
    Action, ActionTemplate, Thread, Decision, Rule, ReviewItem, ReviewRequest,
    ReviewResponse, AuditRecord, REVERSIBLE_ACTIONS,
)


def test_thread_fingerprint_is_stable_and_sender_scoped():
    # Test (a): Digit-stripped subjects from same sender produce same fingerprint
    t1 = Thread(id="a", subject="Invoice 8821", sender="vendor@acme.com", to=["me@z.com"],
                date="2026-08-26T00:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    t2 = Thread(id="b", subject="Invoice 8822", sender="vendor@acme.com", to=["me@z.com"],
                date="2026-08-27T00:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    assert t1.fingerprint == t2.fingerprint  # digits stripped, same sender

    # Test (b): Same subject from different senders produces different fingerprints
    t3 = Thread(id="c", subject="Invoice 8821", sender="vendor@acme.com", to=["me@z.com"],
                date="2026-08-26T00:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    t4 = Thread(id="d", subject="Invoice 8821", sender="vendor@other.com", to=["me@z.com"],
                date="2026-08-26T00:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    assert t3.fingerprint != t4.fingerprint  # different sender


def test_action_defaults_to_empty_params():
    a = Action(kind="archive", thread_id="t1")
    assert a.params == {}


def test_action_accepts_unsafe_kinds_for_deny_list_testing():
    # Finding 1: Action.kind is str (not ActionKind) to allow testing deny-list at chokepoint
    a = Action(kind="send_message", thread_id="t1")
    assert a.kind == "send_message"
    assert "send_message" not in REVERSIBLE_ACTIONS


def test_fingerprint_strips_prefixes_and_collapses_whitespace():
    # Test Re:/Fwd:/Fw: prefix stripping
    t1 = Thread(id="a", subject="Budget Review", sender="mgr@acme.com", to=["me@z.com"],
                date="2026-08-26T00:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    t2 = Thread(id="b", subject="Re: Budget Review", sender="mgr@acme.com", to=["me@z.com"],
                date="2026-08-26T00:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    t3 = Thread(id="c", subject="Re: Re: Budget Review", sender="mgr@acme.com", to=["me@z.com"],
                date="2026-08-26T00:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    t4 = Thread(id="d", subject="FWD:  Budget   Review", sender="mgr@acme.com", to=["me@z.com"],
                date="2026-08-26T00:00:00Z", snippet="s", body="b", label_ids=["INBOX"])
    # All should have same fingerprint after normalization
    assert t1.fingerprint == t2.fingerprint == t3.fingerprint == t4.fingerprint


def test_review_request_round_trips_through_json():
    """The interrupt payload crosses a process boundary and must survive JSON."""
    req = ReviewRequest(
        run_id="run-1",
        policy_version="local:abc123",
        items=[ReviewItem(
            thread_id="t1", subject="Sale", sender="deals@shop.com",
            snippet="50% off", proposed=[Action(kind="archive", thread_id="t1")],
            reason="promotional", confidence=0.9, source="model", rule_id=None,
        )],
    )
    restored = ReviewRequest.model_validate(json.loads(req.model_dump_json()))
    assert restored == req


def test_review_response_records_per_thread_verdicts():
    resp = ReviewResponse(
        decisions={"t1": "approve", "t2": "reject"},
        edits={"t2": [Action(kind="label", thread_id="t2", params={"label": "Later"})]},
        instructions=["always keep anything from my manager in the inbox"],
    )
    assert resp.decisions["t2"] == "reject"
    assert resp.edits["t2"][0].params["label"] == "Later"


def test_rule_carries_provenance_and_starts_unfired():
    r = Rule(id="r1", scope="sender", pattern="deals@shop.com", action="archive",
             provenance="correction on thread t1 at 2026-08-26",
             created_at=datetime.now(timezone.utc))
    assert r.hit_count == 0
    assert r.overridden is False
    assert "correction" in r.provenance


def test_audit_record_serialises_every_spec_field():
    rec = AuditRecord(
        ts=datetime.now(timezone.utc), thread_id="t1", action="archive", params={},
        actor="rule:r1", rule_provenance="learned from correction", model="gemma4:12b-mlx",
        backend="ollama", langsmith_run_id=None, checkpoint_id="ckpt-1",
        policy_version="local:abc123", dry_run=True, result="simulated",
        reversible=True, undo_token={"restore_labels": ["INBOX"]},
    )
    d = json.loads(rec.model_dump_json())
    for field in ("ts", "thread_id", "action", "params", "actor", "rule_provenance",
                  "model", "backend", "langsmith_run_id", "checkpoint_id",
                  "policy_version", "dry_run", "result", "reversible", "undo_token"):
        assert field in d


# --- rules carry action sequences -------------------------------------------
# A single ActionKind could not say WHICH label, so a learned label rule
# produced Action(kind="label", params={}) and _dispatch raised KeyError against
# a live mailbox while reporting "simulated" under dry-run.

RULE_NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def a_rule(**kw):
    base = dict(id="r-1", scope="sender", pattern="a@b.com",
                actions=[ActionTemplate(kind="archive")],
                provenance="user archived it", created_at=RULE_NOW)
    return Rule(**(base | kw))


def test_a_rule_carries_a_sequence_of_actions():
    """label-then-archive is the sequence the model produces most often, and a
    rule that cannot say it cannot teach the correction the owner makes."""
    r = a_rule(actions=[ActionTemplate(kind="label", params={"label": "recruiter"}),
                        ActionTemplate(kind="archive")])
    assert [a.kind for a in r.actions] == ["label", "archive"]
    assert r.actions[0].params["label"] == "recruiter"


def test_a_label_template_carries_its_label():
    r = a_rule(actions=[ActionTemplate(kind="label", params={"label": "receipt"})])
    assert r.actions[0].params == {"label": "receipt"}


def test_a_legacy_action_rule_still_loads():
    """Rules are on disk as of this branch. One written before this change must
    not become unreadable, or the store fails to open at all."""
    r = Rule.model_validate({"id": "r-old", "scope": "sender", "pattern": "a@b.com",
                             "action": "archive", "provenance": "p",
                             "created_at": RULE_NOW})
    assert [a.kind for a in r.actions] == ["archive"]


def test_a_legacy_label_rule_loads_without_a_label():
    """It converts, and it is broken - which is the bug being fixed. Visible
    here, refused at bind time; never a KeyError halfway through a run."""
    r = Rule.model_validate({"id": "r-old", "scope": "sender", "pattern": "a@b.com",
                             "action": "label", "provenance": "p",
                             "created_at": RULE_NOW})
    assert r.actions == [ActionTemplate(kind="label", params={})]


def test_summary_reads_like_the_digest():
    r = a_rule(actions=[ActionTemplate(kind="label", params={"label": "recruiter"}),
                        ActionTemplate(kind="archive")])
    assert r.summary == "label(recruiter), archive"


def test_category_is_a_scope():
    assert a_rule(scope="category", pattern="newsletter_valuable").scope == "category"

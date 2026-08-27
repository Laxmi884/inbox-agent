# tests/test_render.py
from inbox_agent.models import Action, ReviewItem, ReviewRequest
from inbox_agent.render import (
    NO_REASON, approve_all, render_review, respond, review_table,
)


def request() -> ReviewRequest:
    return ReviewRequest(
        run_id="r1", policy_version="local:test",
        items=[
            ReviewItem(thread_id="t1", subject="Sale", sender="deals@shop.com",
                       snippet="50% off", proposed=[Action(kind="archive", thread_id="t1")],
                       reason="promotional", confidence=0.9, source="model", rule_id=None),
            ReviewItem(thread_id="t2", subject="Re: budget", sender="boss@work.com",
                       snippet="thoughts?", proposed=[Action(kind="none", thread_id="t2")],
                       reason="needs a reply", confidence=0.4, source="model", rule_id=None),
        ])


def test_review_table_has_one_row_per_item():
    rows = review_table(request())
    assert len(rows) == 2
    for col in ("thread_id", "sender", "subject", "proposed", "confidence", "why"):
        assert col in rows[0]


def test_render_review_shows_subject_and_action():
    text = render_review(request())
    assert "Sale" in text and "archive" in text


def test_low_confidence_is_flagged_for_the_human():
    """The point of confidence is to draw the eye. It must be visible."""
    assert "!" in review_table(request())[1]["confidence"]


def test_approve_all_approves_every_item():
    resp = approve_all(request())
    assert resp.decisions == {"t1": "approve", "t2": "approve"}


def test_respond_defaults_to_approve_and_marks_rejections():
    resp = respond(request(), reject=["t2"])
    assert resp.decisions["t1"] == "approve"
    assert resp.decisions["t2"] == "reject"


def test_respond_records_edits_as_edit_verdicts():
    resp = respond(request(), edit={"t1": [Action(kind="label", thread_id="t1",
                                                  params={"label": "Deals"})]})
    assert resp.decisions["t1"] == "edit"
    assert resp.edits["t1"][0].params["label"] == "Deals"


def test_respond_carries_free_text_instructions():
    resp = respond(request(), instructions=["always keep mail from my boss"])
    assert resp.instructions == ["always keep mail from my boss"]


def test_empty_reason_renders_a_visible_marker():
    """gemma4:12b-mlx reliably omits reason; an empty why-cell must not read
    as a rendering bug."""
    req = ReviewRequest(
        run_id="r1", policy_version="local:test",
        items=[ReviewItem(thread_id="t1", subject="Sale", sender="deals@shop.com",
                           snippet="50% off", proposed=[Action(kind="archive", thread_id="t1")],
                           reason="", confidence=0.9, source="model", rule_id=None)])
    assert review_table(req)[0]["why"] == NO_REASON


def test_whitespace_only_reason_renders_the_marker_too():
    req = ReviewRequest(
        run_id="r1", policy_version="local:test",
        items=[ReviewItem(thread_id="t1", subject="Sale", sender="deals@shop.com",
                           snippet="50% off", proposed=[Action(kind="archive", thread_id="t1")],
                           reason="   ", confidence=0.9, source="model", rule_id=None)])
    assert review_table(req)[0]["why"] == NO_REASON


def test_real_reason_renders_unchanged():
    assert review_table(request())[0]["why"] == "promotional"

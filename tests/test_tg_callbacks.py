"""Callback encoding and index resolution.

This is the module the trust boundary rests on. Telegram callback_data is
attacker-reachable in the sense that matters: it crosses a persistence boundary
and comes back from outside the process that created it. The design rule from
the spec is that a callback carries a POSITION, never a thread id, and the
position is resolved against the review request in the checkpoint - so there is
no callback string that can name a thread the human was not shown.
"""
import pytest

from inbox_agent.models import Action, ReviewItem, ReviewRequest
from inbox_agent.telegram.callbacks import (
    CB_MAX_BYTES, Intent, decode, encode, resolve_thread_id, to_response,
)


def request(n=3) -> ReviewRequest:
    return ReviewRequest(
        run_id="run1", policy_version="local:test",
        items=[ReviewItem(thread_id=f"t{i}", subject=f"S{i}", sender=f"a{i}@b.com",
                          snippet="x", proposed=[Action(kind="archive", thread_id=f"t{i}")],
                          reason="r", confidence=0.9, source="model")
               for i in range(n)])


def test_encode_decode_round_trips():
    for kind, idx, lbl in [("approve", 3, None), ("reject", 0, None),
                           ("label", 12, 7), ("next", None, None),
                           ("prev", None, None), ("approve_all", None, None)]:
        i = decode(encode(kind, idx, lbl))
        assert (i.kind, i.index, i.label_index) == (kind, idx, lbl)


def test_every_callback_fits_telegram_64_byte_cap():
    """A hard protocol limit, not a guideline. Indices keep us far under it."""
    for data in (encode("approve", 999), encode("label", 999, 99),
                 encode("approve_all"), encode("next"), encode("prev")):
        assert len(data.encode()) <= CB_MAX_BYTES, data


def test_thread_ids_never_appear_in_callback_data():
    """The structural property: a thread id cannot travel in a callback."""
    req = request()
    for i in range(len(req.items)):
        for data in (encode("approve", i), encode("reject", i), encode("label", i, 0)):
            for item in req.items:
                assert item.thread_id not in data


def test_decode_never_raises_on_hostile_input():
    for junk in ("", "a", "a:", "a:x", "l:1", "l:x:y", "zzz", "a:1:2:3",
                 "../../etc/passwd", "a:-1", "a:" + "9" * 400, "\x00", "a:1\n"):
        got = decode(junk)
        assert isinstance(got, Intent)
        assert got.kind in {"approve", "reject", "label", "prev", "next",
                            "approve_all", "noop"}


def test_out_of_range_index_resolves_to_nothing():
    """The whole point. An index outside the batch names no thread."""
    req = request(3)
    assert resolve_thread_id(Intent("approve", 0), req) == "t0"
    assert resolve_thread_id(Intent("approve", 2), req) == "t2"
    assert resolve_thread_id(Intent("approve", 3), req) is None
    assert resolve_thread_id(Intent("approve", 999), req) is None
    assert resolve_thread_id(Intent("approve", -1), req) is None
    assert resolve_thread_id(Intent("approve", None), req) is None


def test_negative_index_cannot_wrap_around_to_a_real_thread():
    """Python would happily let -1 index the last item. It must not."""
    req = request(3)
    assert resolve_thread_id(Intent("reject", -1), req) is None
    assert resolve_thread_id(Intent("reject", -3), req) is None


def test_to_response_defaults_everything_to_approve():
    """Reviewing 50 proposals must not require 50 decisions - matching
    render.respond()'s stance that the common case is one keystroke."""
    req = request(3)
    r = to_response(req, {})
    assert r.decisions == {"t0": "approve", "t1": "approve", "t2": "approve"}


def test_to_response_applies_named_verdicts_by_index():
    req = request(3)
    r = to_response(req, {1: Intent("reject", 1)})
    assert r.decisions == {"t0": "approve", "t1": "reject", "t2": "approve"}


def test_to_response_turns_a_label_intent_into_an_edit():
    req = request(3)
    r = to_response(req, {0: Intent("label", 0, 2)}, categories=["a", "b", "recruiter"])
    assert r.decisions["t0"] == "edit"
    assert r.edits["t0"][0].kind == "label"
    assert r.edits["t0"][0].params["label"] == "recruiter"
    assert r.edits["t0"][0].thread_id == "t0"


def test_to_response_ignores_out_of_range_indices_entirely():
    """A forged index contributes nothing - it does not crash, and it does not
    silently land on a neighbouring thread."""
    req = request(3)
    r = to_response(req, {99: Intent("reject", 99), -1: Intent("reject", -1)})
    assert r.decisions == {"t0": "approve", "t1": "approve", "t2": "approve"}
    assert r.edits == {}


def test_label_intent_with_out_of_range_category_is_dropped():
    req = request(3)
    r = to_response(req, {0: Intent("label", 0, 99)}, categories=["a", "b"])
    assert r.decisions["t0"] == "approve"
    assert "t0" not in r.edits

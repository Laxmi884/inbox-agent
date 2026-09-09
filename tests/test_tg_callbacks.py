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
                 "../../etc/passwd", "a:-1", "a:" + "9" * 400, "\x00", "a:1\n",
                 "a:²", "a:٣", "l:1:²", "o:³"):
        got = decode(junk)
        assert isinstance(got, Intent)
        assert got.kind in {"approve", "reject", "label", "prev", "next",
                            "approve_all", "open", "list", "done", "approve_attention", "noop"}


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


# --- opening one item from the digest ---------------------------------------
# The digest shipped with only "Approve all" and "Next", which made it a
# read-only screen: there was no way to correct anything without leaving it.

def test_open_intent_round_trips_and_carries_an_index():
    i = decode(encode("open", 4))
    assert i.kind == "open" and i.index == 4


def test_open_intent_resolves_like_any_other_index():
    req = request(3)
    assert resolve_thread_id(Intent("open", 1), req) == "t1"
    assert resolve_thread_id(Intent("open", 99), req) is None


def test_open_contributes_no_verdict():
    """Opening an item is navigation, not a decision."""
    req = request(3)
    r = to_response(req, {1: Intent("open", 1)})
    assert r.decisions["t1"] == "approve"
    assert r.edits == {}


# --- digest id round-trip and staleness check --------------------------------

def test_digest_id_round_trips_for_every_kind():
    for kind, args in (("open", (3,)), ("approve", (3,)), ("reject", (3,)),
                       ("label", (3, 2)), ("prev", ()), ("next", ()),
                       ("done", ()), ("approve_attention", ()), ("list", ())):
        data = encode(kind, *args, digest_id="7f2a")
        got = decode(data)
        assert got.kind == kind, data
        assert got.digest_id == "7f2a", data


def test_index_and_label_index_survive_alongside_the_digest_id():
    got = decode(encode("label", 3, 2, digest_id="7f2a"))
    assert (got.index, got.label_index) == (3, 2)


def test_encoding_stays_inside_the_64_byte_cap():
    for kind, args in (("label", (9999, 99)), ("open", (9999,))):
        assert len(encode(kind, *args, digest_id="7f2a").encode()) <= CB_MAX_BYTES


def test_a_callback_without_a_digest_id_decodes_with_an_empty_one():
    """Backwards compatible: a message sent before this change still decodes,
    and the bot's staleness check treats an empty id as not-current."""
    assert decode("a:3").digest_id == ""


def test_garbage_is_still_a_noop():
    for data in ("", "zzz", "a:", "a:x:7f2a", "l:1:7f2a", "a:1:2:3:4"):
        assert decode(data).kind == "noop"


def test_a_digest_id_that_is_not_hex_is_rejected():
    """Bounds what we will parse at all, the same reasoning as _MAX_PARSED_INDEX."""
    assert decode("a:3:zz//").kind == "noop"


# --- correction verdicts ----------------------------------------------------
# A verdict names a position, like everything else here. The thread id never
# travels, so no callback string can name a thread the owner was not shown.

def test_the_new_verdict_kinds_round_trip():
    for kind in ("keep", "relabel", "teach_trash", "keep_inbox", "file_away",
                 "scope_narrow", "scope_wide"):
        intent = decode(encode(kind, 3, digest_id="7f2a"))
        assert intent.kind == kind
        assert intent.index == 3
        assert intent.digest_id == "7f2a"


def test_every_new_kind_fits_the_byte_cap():
    """64 bytes is protocol. An index reaches three digits on a backlog."""
    for kind in ("keep", "relabel", "teach_trash", "keep_inbox", "file_away",
                 "scope_narrow", "scope_wide"):
        assert len(encode(kind, 999, digest_id="7f2a").encode()) <= CB_MAX_BYTES


def test_a_relabel_carries_the_category_it_names():
    """Label as … is two taps: pick the item, then pick the label. The second
    has to carry which label without a thread id ever travelling."""
    intent = decode(encode("relabel", 3, 2, digest_id="7f2a"))
    assert intent.kind == "relabel"
    assert intent.index == 3
    assert intent.label_index == 2


def test_a_verdict_without_a_digest_id_is_refused():
    """Same staleness rule as every other kind: positions shift between
    digests, so a verdict from an older message must not resolve."""
    assert decode(encode("keep", 3)).digest_id == ""


# --- bulk trash, and its confirmation ---------------------------------------

def test_trash_all_and_its_confirmation_round_trip():
    for kind in ("trash_all", "trash_all_go"):
        i = decode(encode(kind, digest_id="7f2a"))
        assert i.kind == kind and i.digest_id == "7f2a"


def test_the_confirm_step_is_a_separate_kind_from_the_offer():
    """Two codes, not one with a flag. A single kind carrying 'are you sure'
    state would let a replayed callback skip the confirmation entirely."""
    assert encode("trash_all") != encode("trash_all_go")


def test_bulk_trash_codes_stay_inside_the_64_byte_callback_limit():
    assert len(encode("trash_all_go", digest_id="7f2a").encode()) <= 64


# --- the run list -------------------------------------------------------
# Index-based like every other position here: an 8-character run_id plus a
# digest id plus a kind does not reliably fit in 64 bytes.

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


def test_a_refused_callback_says_so_in_the_log(caplog):
    """A tap the bot declines to parse must leave a trace.

    The refusal is a toast, and on a phone a toast is a banner that vanishes -
    so from the outside it is indistinguishable from a dead button, and from
    the log it was indistinguishable from nothing at all. On 2026-09-08 that
    cost an entire investigation: the only paths writing no log line were
    "worked" and "refused", which is exactly the pair you need to tell apart.
    """
    import logging
    from inbox_agent.telegram.bot import Bot

    calls = []

    class T:
        def answer_callback(self, callback_id, text=""):
            calls.append(text)

    bot = Bot.__new__(Bot)
    bot.transport = T()
    with caplog.at_level(logging.INFO):
        bot._ack_refusal("cb-1", "not parseable", "That button came from an "
                                                  "older message.")
    assert calls == ["That button came from an older message."]
    assert "not parseable" in caplog.text

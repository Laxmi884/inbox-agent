"""Both renderers. Pure functions over ReviewRequest, like render.py."""
import pytest

from inbox_agent.models import Action, ReviewItem, ReviewRequest
from inbox_agent.telegram.callbacks import decode
from inbox_agent.telegram.render_tg import (
    TG_MAX_TEXT, digest, header, paged, rule_decided_count,
)


def item(i, source="model", conf=0.9, kind="archive", label=None):
    params = {"label": label} if label else {}
    return ReviewItem(
        thread_id=f"t{i}", subject=f"Subject number {i}", sender=f"sender{i}@example.com",
        snippet="snip", proposed=[Action(kind=kind, thread_id=f"t{i}", params=params)],
        reason="because", confidence=conf, source=source,
        rule_id=("r-1" if source == "rule" else None))


def request(n=5, n_rule=0):
    items = [item(i, source="rule" if i < n_rule else "model") for i in range(n)]
    return ReviewRequest(run_id="run1", policy_version="local:test", items=items)


# --- the learning-visibility header -----------------------------------------
# The store has learned since Stage A; nothing ever told the owner. The whole
# value of the loop is that review gets shorter and more citable over time, and
# that is invisible to the person doing the reviewing.

def test_rule_decided_count_matches_reality():
    assert rule_decided_count(request(10, n_rule=4)) == 4
    assert rule_decided_count(request(10, n_rule=0)) == 0


def test_header_reports_threads_and_what_rules_decided():
    h = header(request(10, n_rule=4))
    assert "10" in h
    assert "4" in h
    assert "rule" in h.lower()


def test_header_omits_the_rule_clause_when_nothing_was_rule_decided():
    """Do not print '0 decided by rules you taught me' on a first run - it
    reads as a failure rather than as a not-yet."""
    h = header(request(10, n_rule=0))
    assert "10" in h
    assert "taught" not in h.lower()


# --- digest -----------------------------------------------------------------

def test_digest_lists_every_item_when_it_fits():
    text, kb = digest(request(5))
    for i in range(5):
        assert f"Subject number {i}"[:20] in text


def test_digest_never_exceeds_the_telegram_message_cap():
    """4096 chars is a hard protocol limit. 50 threads is right at the edge,
    so this must page rather than truncate silently."""
    text, kb = digest(request(50))
    assert len(text) <= TG_MAX_TEXT


def test_digest_offers_approve_all():
    _, kb = digest(request(5))
    datas = [d for row in kb for (_, d) in row]
    assert any(decode(d).kind == "approve_all" for d in datas)


def test_digest_paginates_and_the_next_button_advances():
    text0, kb0 = digest(request(60), page=0)
    datas = [d for row in kb0 for (_, d) in row]
    assert any(decode(d).kind == "next" for d in datas), "no way to reach page 2"
    text1, _ = digest(request(60), page=1)
    assert text0 != text1


def test_digest_flags_low_confidence_items():
    items = [item(0, conf=0.2), item(1, conf=0.95)]
    req = ReviewRequest(run_id="r", policy_version="v", items=items)
    text, _ = digest(req)
    assert "!" in text


# --- paged ------------------------------------------------------------------

def test_paged_shows_one_item_with_position():
    text, kb = paged(request(10), 3)
    assert "Subject number 3" in text
    assert "4" in text and "10" in text          # "4/10"
    assert "Subject number 4" not in text


def test_paged_buttons_carry_the_right_index():
    _, kb = paged(request(10), 3)
    datas = [d for row in kb for (_, d) in row]
    approves = [decode(d) for d in datas if decode(d).kind == "approve"]
    assert approves and all(a.index == 3 for a in approves)


def test_paged_offers_label_buttons_for_policy_categories():
    _, kb = paged(request(3), 0, categories=["recruiter", "promotion"])
    labels = [decode(d) for row in kb for (_, d) in row]
    label_intents = [x for x in labels if x.kind == "label"]
    assert len(label_intents) == 2
    assert {x.label_index for x in label_intents} == {0, 1}


def test_paged_clamps_out_of_range_index_rather_than_crashing():
    text, _ = paged(request(3), 99)
    assert "Subject number 2" in text          # clamped to the last item
    text, _ = paged(request(3), -5)
    assert "Subject number 0" in text


def test_paged_hides_prev_on_first_and_next_on_last():
    _, kb_first = paged(request(3), 0)
    kinds_first = {decode(d).kind for row in kb_first for (_, d) in row}
    assert "prev" not in kinds_first
    _, kb_last = paged(request(3), 2)
    kinds_last = {decode(d).kind for row in kb_last for (_, d) in row}
    assert "next" not in kinds_last


def test_every_rendered_callback_is_within_the_byte_cap():
    for _, kb in (digest(request(60)), paged(request(60), 30,
                                             categories=["a"] * 10)):
        for row in kb:
            for _, data in row:
                assert len(data.encode()) <= 64, data


# --- digest must be correctable and readable --------------------------------

def test_digest_offers_a_button_per_item_so_it_is_not_read_only():
    from inbox_agent.telegram.callbacks import decode as d
    _, kb = digest(request(6))
    opens = [d(data) for row in kb for (_, data) in row if d(data).kind == "open"]
    assert len(opens) == 6, "no way to correct an individual item"
    assert {o.index for o in opens} == set(range(6))


def test_digest_offers_a_shortcut_to_the_flagged_items():
    from inbox_agent.telegram.callbacks import decode as d
    items = [item(0, conf=0.2), item(1, conf=0.95), item(2, conf=0.1)]
    req = ReviewRequest(run_id="r", policy_version="v", items=items)
    text, kb = digest(req)
    labels = [t for row in kb for (t, _) in row]
    assert any("2" in t and "flag" in t.lower() for t in labels), labels


def test_digest_has_no_flagged_shortcut_when_nothing_is_flagged():
    _, kb = digest(request(4))
    labels = [t for row in kb for (t, _) in row]
    assert not any("flag" in t.lower() for t in labels)


def test_digest_does_not_fake_monospace_columns():
    """Telegram renders proportional text and wraps it, so padded columns
    collapse into a wall. Measured on a real phone before this test existed."""
    text, _ = digest(request(6))
    assert "   " not in text.replace("\n", ""), "still padding columns with spaces"


def test_digest_per_item_buttons_are_capped_on_large_batches():
    """A 50-item keyboard is unusable; the flagged shortcut carries those."""
    _, kb = digest(request(50))
    from inbox_agent.telegram.callbacks import decode as d
    opens = [x for row in kb for (_, data) in row if (x := d(data)).kind == "open"]
    assert len(opens) <= 10

"""Both renderers. Pure functions over their view object, like render.py.

The digest renders a DigestView - this run's counts plus the queue that outlives
runs. `paged` still renders a ReviewRequest: it is the single-item view the
interrupt path uses, and Plan 3's /backlog is what brings it back into the bot.
"""
from datetime import datetime, timedelta, timezone

from inbox_agent.models import Action, HeldItem, ReviewItem, ReviewRequest
from inbox_agent.telegram.callbacks import decode
from inbox_agent.telegram.render_tg import (
    DigestView, HELD_PAGE_SIZE, TG_MAX_TEXT, digest, header, paged,
    rule_decided_count,
)

NOW = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


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


def held(tid, reason, *, conf=0.9, subject=None, held_at=NOW, reason_text="because"):
    return HeldItem(
        thread_id=tid, run_id="r1", first_held_at=held_at, hold_reason=reason,
        item=ReviewItem(
            thread_id=tid, category="promotion", subject=subject or f"Subject {tid}",
            sender=f"{tid}@example.com", snippet="s",
            proposed=[Action(kind="trash", thread_id=tid)],
            reason=reason_text, confidence=conf, source="model"))


def view(held_items=(), *, total=22, done=None, rule_decided=12):
    return DigestView(
        run_at=NOW, total=total,
        done_by_kind=done if done is not None else {"archive": 9, "label": 6, "draft": 3},
        rule_decided=rule_decided, held=list(held_items), digest_id="7f2a")


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

def test_the_stat_line_reports_what_was_done_and_what_waits():
    text, _ = digest(view([held("t1", "trash")]))
    assert "22" in text
    assert "9" in text and "6" in text and "3" in text
    assert "1 waiting" in text


def test_held_items_are_grouped_under_their_reason():
    text, _ = digest(view([held("t1", "trash"), held("t2", "needs_reply"),
                           held("t3", "security_alert"), held("t4", "low_confidence")]))
    assert "TRASH" in text
    assert "NEEDS REPLY" in text
    assert "SECURITY" in text
    assert "NOT SURE" in text


def test_an_empty_section_is_omitted_entirely():
    text, _ = digest(view([held("t1", "trash")]))
    assert "NEEDS REPLY" not in text
    assert "SECURITY" not in text


def test_each_held_item_shows_subject_sender_and_the_agents_reason():
    text, _ = digest(view([held("t1", "trash", subject="Quartz or Mechanical?",
                                reason_text="Marketing mail, no order reference.")]))
    assert "Quartz or Mechanical?" in text
    assert "t1@example.com" in text
    assert "Marketing mail, no order reference." in text


def test_low_confidence_is_shown_numerically():
    text, _ = digest(view([held("t1", "low_confidence", conf=0.41)]))
    assert "0.41" in text


def test_a_carried_over_item_shows_how_long_it_has_waited():
    old = held("t1", "trash", held_at=NOW - timedelta(hours=10))
    text, _ = digest(view([old]))
    assert "waiting since" in text.lower()


def test_an_item_held_in_this_run_shows_no_waiting_since():
    text, _ = digest(view([held("t1", "trash", held_at=NOW)]))
    assert "waiting since" not in text.lower()


def test_carried_items_sort_above_fresh_ones_within_a_section():
    fresh = held("fresh", "trash", held_at=NOW)
    old = held("old", "trash", held_at=NOW - timedelta(hours=10))
    text, _ = digest(view([fresh, old]))
    assert text.index("Subject old") < text.index("Subject fresh")


def test_done_items_are_counts_not_a_list():
    text, _ = digest(view([held("t1", "trash")]))
    assert "DONE" in text
    assert "18" in text
    assert "12" in text and "taught" in text.lower()


def test_the_rule_clause_is_omitted_when_nothing_was_rule_decided():
    text, _ = digest(view([held("t1", "trash")], rule_decided=0))
    assert "taught" not in text.lower()


def test_never_pads_columns():
    """Telegram wraps proportional text; padding produces a wall, not a table.
    Verified on a real phone once already - do not reintroduce it."""
    text, _ = digest(view([held(f"t{i}", "trash") for i in range(4)]))
    assert "   " not in text


def test_a_button_per_held_item_on_this_page():
    _, kb = digest(view([held(f"t{i}", "trash") for i in range(4)]))
    flat = [label for row in kb for (label, _) in row]
    assert [l for l in flat if l in {"1", "2", "3", "4"}] == ["1", "2", "3", "4"]


def test_the_one_tap_button_names_the_attention_tier_only():
    _, kb = digest(view([held("t1", "trash"), held("t2", "needs_reply")]))
    labels = [label for row in kb for (label, _) in row]
    assert any("Approve" in l for l in labels)


def test_there_is_no_one_tap_button_when_nothing_is_held_for_attention():
    """A blanket approve must never be able to reach trash or a guess."""
    _, kb = digest(view([held("t1", "trash"), held("t2", "low_confidence")]))
    labels = [label for row in kb for (label, _) in row]
    assert not any("Approve" in l for l in labels)


def test_held_items_page_at_the_limit():
    items = [held(f"t{i:02d}", "trash") for i in range(HELD_PAGE_SIZE + 3)]
    text, kb = digest(view(items), page=0)
    assert "Subject t00" in text
    assert f"Subject t{HELD_PAGE_SIZE:02d}" not in text
    labels = [label for row in kb for (label, _) in row]
    assert any("Next" in l for l in labels)


def test_page_two_shows_the_remainder_and_keeps_absolute_numbering():
    items = [held(f"t{i:02d}", "trash") for i in range(HELD_PAGE_SIZE + 3)]
    text, _ = digest(view(items), page=1)
    assert f"Subject t{HELD_PAGE_SIZE:02d}" in text
    assert f"{HELD_PAGE_SIZE + 1}." in text


def test_an_out_of_range_page_clamps_rather_than_raising():
    """Reached from a callback. A stale one must land somewhere sane."""
    text, _ = digest(view([held("t1", "trash")]), page=99)
    assert "Subject t1" in text


def test_an_empty_queue_still_renders_the_report():
    text, kb = digest(view([]))
    assert "DONE" in text
    assert "0 waiting" in text or "waiting" not in text


def test_never_exceeds_the_telegram_cap():
    items = [held(f"t{i:02d}", "trash", subject="x" * 200, reason_text="y" * 400)
             for i in range(HELD_PAGE_SIZE)]
    text, _ = digest(view(items))
    assert len(text) <= TG_MAX_TEXT


def test_the_digest_carries_its_id_into_every_callback():
    """A tap has to identify the list it was drawn against, or a stale one
    lands on whatever occupies that position today."""
    _, kb = digest(view([held(f"t{i:02d}", "needs_reply")
                         for i in range(HELD_PAGE_SIZE + 3)]))
    intents = [decode(data) for row in kb for (_, data) in row]
    assert intents
    assert all(i.digest_id == "7f2a" for i in intents), intents


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
    big_digest = digest(view([held(f"t{i:03d}", "needs_reply")
                              for i in range(60)]))
    for _, kb in (big_digest, paged(request(60), 30, categories=["a"] * 10)):
        for row in kb:
            for _, data in row:
                assert len(data.encode()) <= 64, data

"""Both renderers. Pure functions over their view object, like render.py.

The digest renders a DigestView - this run's counts plus the queue that outlives
runs. `paged` still renders a ReviewRequest: it is the single-item view the
interrupt path uses, and Plan 3's /backlog is what brings it back into the bot.
"""
from datetime import datetime, timedelta, timezone

from inbox_agent.models import Action, HeldItem, ReviewItem, ReviewRequest
from inbox_agent.telegram import render_tg
from inbox_agent.telegram.callbacks import decode
from inbox_agent.telegram.render_tg import (
    DigestView, DONE_PAGE_SIZE, DoneItem, HELD_PAGE_SIZE, TG_MAX_TEXT, digest,
    done_panel, header, item_view, paged, rule_decided_count,
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


def held(tid, reason, *, conf=0.9, subject=None, held_at=NOW, reason_text="because",
         sender=None):
    return HeldItem(
        thread_id=tid, run_id="r1", first_held_at=held_at, hold_reason=reason,
        item=ReviewItem(
            thread_id=tid, category="promotion", subject=subject or f"Subject {tid}",
            sender=sender or f"{tid}@example.com", snippet="s",
            proposed=[Action(kind="trash", thread_id=tid)],
            reason=reason_text, confidence=conf, source="model"))


def view(held_items=(), *, total=22, done=None, rule_decided=12,
         dry_run=False, run_report=True):
    return DigestView(
        run_at=NOW, total=total,
        done_by_kind=done if done is not None else {"archive": 9, "label": 6, "draft": 3},
        rule_decided=rule_decided, held=list(held_items), digest_id="7f2a",
        dry_run=dry_run, run_report=run_report)


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


def test_a_section_count_is_the_whole_section_not_just_this_page():
    """A page-scoped count is the same falsehood the redesign set out to remove.

    Held items are ordered by when they were first held, so a section is spread
    across pages whenever reasons interleave in arrival order - which is the
    normal case, not an edge one. Counting only what fits told a reader of page
    one that five things were waiting to be trashed when eight were, and then
    repeated the heading on page two with a different number.
    """
    order = ["trash", "security_alert", "trash", "trash", "trash", "trash",
             "security_alert", "security_alert", "trash", "trash",
             "security_alert", "security_alert", "trash"]   # 8 trash, 5 security
    items = [held(f"t{i:02d}", reason, held_at=NOW + timedelta(seconds=i))
             for i, reason in enumerate(order)]
    first, _ = digest(view(items), page=0)
    second, _ = digest(view(items), page=1)

    assert "🗑 TRASH — approve one at a time (5 of 8)" in first
    assert "🔒 SECURITY (3 of 5)" in first
    assert "🗑 TRASH — approve one at a time (3 of 8)" in second
    assert "🔒 SECURITY (2 of 5)" in second


def test_a_section_shown_whole_reports_a_plain_count():
    """"(2 of 2)" would be noise on the common case - one page, nothing hidden."""
    text, _ = digest(view([held("t1", "trash"), held("t2", "trash")]))
    assert "🗑 TRASH — approve one at a time (2)" in text


def test_an_out_of_range_page_clamps_rather_than_raising():
    """Reached from a callback. A stale one must land somewhere sane."""
    text, _ = digest(view([held("t1", "trash")]), page=99)
    assert "Subject t1" in text


def test_an_empty_queue_still_renders_the_report():
    text, kb = digest(view([]))
    assert "DONE" in text
    assert "0 waiting" in text or "waiting" not in text


def test_a_field_cut_short_says_it_was_cut_short():
    """Subjects are capped at 70 and marketing subjects run long, so on a real
    inbox most trash items are cut. A hard slice ends them mid-word - "Make it
    happen sooner with 30% mo" - which reads as a corrupted message rather than
    as a subject that continues.
    """
    text, _ = digest(view([held("t1", "trash", subject="Make it happen " * 20)]))
    subject_line = [l for l in text.splitlines() if l.startswith("1. ")][0]
    assert subject_line.endswith("…")
    assert len(subject_line) <= len("1. ") + render_tg._SUBJECT_CAP


def test_a_field_that_fits_is_left_alone():
    text, _ = digest(view([held("t1", "trash", subject="Short one")]))
    assert "1. Short one" in text
    assert "…" not in text


def test_never_exceeds_the_telegram_cap():
    items = [held(f"t{i:02d}", "trash", subject="x" * 200, reason_text="y" * 400,
                  sender="s" * 300 + "@example.com")
             for i in range(HELD_PAGE_SIZE)]
    text, kb = digest(view(items))
    assert len(text) <= TG_MAX_TEXT
    # Every field is capped, so a full page of maximal items still fits and
    # nothing is dropped. The keyboard must therefore offer all eight.
    numbers = [label for row in kb for (label, _) in row if label.isdigit()]
    assert numbers == [str(i) for i in range(1, HELD_PAGE_SIZE + 1)]


def test_a_truncated_digest_keeps_its_keyboard_honest(monkeypatch):
    """The cap is unreachable now that every field is capped, so this lowers it.

    What is being pinned is the invariant, not the arithmetic: text and keyboard
    must describe the same list. A slice at the cap broke both halves of that -
    it dropped the DONE block while leaving its button, and cut an item in half
    while leaving the numbered button that claims to open it.
    """
    monkeypatch.setattr(render_tg, "TG_MAX_TEXT", 600)
    items = [held(f"t{i:02d}", "trash", reason_text="y" * 150)
             for i in range(HELD_PAGE_SIZE)]
    text, kb = digest(view(items))

    assert len(text) <= 600
    assert text.endswith("came from rules you taught me"), \
        "the run report was truncated away"
    assert "DONE" in text
    assert "did not fit" in text, "items vanished with nothing saying so"
    assert not text.endswith(" "), "truncated mid-line"

    shown = {label for row in kb for (label, _) in row if label.isdigit()}
    for n in range(1, HELD_PAGE_SIZE + 1):
        in_text = f"\n{n}. " in text
        assert (str(n) in shown) == in_text, \
            f"item {n}: button={str(n) in shown} but shown in text={in_text}"


def test_a_dry_run_never_claims_the_work_was_done():
    """Under INBOX_DRY_RUN nothing reached Gmail. The startup banner that says
    so is on a terminal; this message is on a phone."""
    text, _ = digest(view([held("t1", "trash")], dry_run=True))
    assert "✓ WOULD HAVE DONE (18)" in text
    assert "✓ DONE" not in text


def test_a_live_run_says_done_plainly():
    text, _ = digest(view([held("t1", "trash")], dry_run=False))
    assert "✓ DONE (18)" in text
    assert "WOULD HAVE" not in text


def test_a_queue_only_view_reports_no_run_at_all():
    """/held ran nothing, so there is no run to report - and the last run's
    counts under the current clock would be a report of a run that did not
    happen."""
    text, kb = digest(view([held("t1", "trash")], run_report=False))
    assert "DONE" not in text
    assert "threads" not in text
    assert "taught" not in text
    assert "1 waiting" in text
    labels = [label for row in kb for (label, _) in row]
    assert not any("done" in l.lower() for l in labels)


def test_a_newline_in_a_model_written_reason_cannot_break_the_grid():
    """`reason` comes from an LLM and models routinely emit newlines. One of
    them turns the three-line block into four and shifts everything below."""
    text, _ = digest(view([held("t1", "trash", subject="A\nB",
                                sender="a\nb@example.com",
                                reason_text="line one\n\nline two")]))
    # Everything after the section header, up to the blank line before DONE.
    block = text.split("TRASH")[1].split("\n\n")[0].split("\n")[1:]
    assert block == ["1. A B", "a b@example.com", "line one line two"], block


def test_the_digest_carries_its_id_into_every_callback():
    """A tap has to identify the list it was drawn against, or a stale one
    lands on whatever occupies that position today."""
    _, kb = digest(view([held(f"t{i:02d}", "needs_reply")
                         for i in range(HELD_PAGE_SIZE + 3)]))
    intents = [decode(data) for row in kb for (_, data) in row]
    assert intents
    assert all(i.digest_id == "7f2a" for i in intents), intents


# --- the done panel ---------------------------------------------------------
# "3 label" tells the owner a label happened and refuses to say which one. The
# digest reports the run; a report you cannot expand is a claim, not a report.

def done(tid="t1", subject=None, sender=None, actions=(("archive", None),),
         from_rule=False):
    return DoneItem(thread_id=tid, subject=subject or f"Subject {tid}",
                    sender=sender or f"{tid}@example.com",
                    actions=list(actions), from_rule=from_rule)


def done_view(items, *, dry_run=False):
    return DigestView(run_at=NOW, total=len(items),
                      done_by_kind={"archive": len(items)}, rule_decided=0,
                      held=[], digest_id="7f2a", dry_run=dry_run,
                      done=list(items))


def test_the_panel_names_the_label_rather_than_counting_it():
    """The question this panel exists to answer. "3 label" is not an answer."""
    text, _ = done_panel(done_view([
        done("t1", subject="You are Invited! Senior Data Analyst",
             actions=[("label", "recruiter"), ("archive", None)])]))
    assert "label(recruiter)" in text
    assert "You are Invited! Senior Data Analyst" in text
    assert "t1@example.com" in text


def test_the_panel_uses_the_digest_block_shape():
    """Two dense lines read fine in a terminal and wrap into a wall on a phone.

    The mistake the layout rules at the top of this file exist to prevent, made
    again in a new renderer: subject, sender and what was done each get their
    own line, and a blank line separates one entry from the next, exactly as the
    held items above them do. Checked on a phone before this was written down.
    """
    text, _ = done_panel(done_view([
        done("t1", subject="First one", actions=[("label", "recruiter"),
                                                 ("archive", None)]),
        done("t2", subject="Second one")]))
    body = text.split("labels: recruiter 1", 1)[1]
    assert body.startswith("\n\n1. First one\n"), body[:60]
    assert "\n1. First one\nt1@example.com\n→ label(recruiter), archive\n\n2. Second one" in text


def test_the_panel_summarises_which_labels_were_used():
    """Scanning twelve lines to learn that everything went to one label is the
    work the summary line does instead."""
    items = ([done(f"r{i}", actions=[("label", "recruiter"), ("archive", None)])
              for i in range(3)]
             + [done("n1", actions=[("label", "newsletter_valuable")])])
    text, _ = done_panel(done_view(items))
    assert "recruiter 3" in text
    assert "newsletter_valuable 1" in text


def test_the_panel_headline_matches_the_number_on_the_button(bot_view=None):
    """The button promises "Show the 21 done" and the panel opened on "(13)".

    Both are true and they count different things - 21 actions across 13
    threads, because a label-then-archive is one thread and two actions - which
    is precisely why the message has to say which it means. Found by tapping the
    button on a real run, not by the suite.
    """
    items = [done(f"t{i}", actions=[("label", "recruiter"), ("archive", None)])
             for i in range(3)] + [done("t9", actions=[("archive", None)])]
    view = DigestView(run_at=NOW, total=4, done_by_kind={"archive": 4, "label": 3},
                      rule_decided=0, held=[held("h1", "trash")], digest_id="7f2a",
                      done=items)
    digest_text, kb = digest(view)
    panel_text, _ = done_panel(view)

    button = [label for row in kb for (label, _) in row if "done" in label]
    assert button == ["📋 Show the 7 done"]
    assert "(7)" in panel_text, "the panel headline disagreed with its own button"
    assert "4 threads" in panel_text, "7 of what, across how many threads"


def test_the_panel_says_would_have_under_dry_run():
    """Same rule as the digest: never the word done for something that did not
    reach Gmail."""
    text, _ = done_panel(done_view([done()], dry_run=True))
    assert "WOULD HAVE" in text
    plain, _ = done_panel(done_view([done()]))
    assert "WOULD HAVE" not in plain


def test_the_panel_marks_what_a_rule_decided():
    """The learning is the point; a run the owner taught should look taught."""
    text, _ = done_panel(done_view([done("t1", from_rule=True)]))
    assert "rule" in text.lower()


def test_the_panel_offers_a_way_back_to_the_digest():
    """A screen with no exit is a trap on a phone, where there is no Escape."""
    _, kb = done_panel(done_view([done()]))
    kinds = [decode(data).kind for row in kb for (_, data) in row]
    assert "list" in kinds


def test_the_panel_pages_rather_than_truncating():
    items = [done(f"t{i:02d}") for i in range(DONE_PAGE_SIZE + 3)]
    first, kb = done_panel(done_view(items))
    assert "Subject t00" in first
    assert f"Subject t{DONE_PAGE_SIZE:02d}" not in first
    assert any(decode(data).kind == "next" for row in kb for (_, data) in row)

    second, _ = done_panel(done_view(items), page=1)
    assert f"Subject t{DONE_PAGE_SIZE:02d}" in second


def test_an_out_of_range_panel_page_clamps_rather_than_raising():
    """Reached from a callback, like every other page index here."""
    text, _ = done_panel(done_view([done()]), page=99)
    assert "Subject t1" in text


def test_the_panel_never_exceeds_the_telegram_cap():
    items = [done(f"t{i:02d}", subject="x" * 300, sender="s" * 300,
                  actions=[("label", "y" * 100), ("archive", None)])
             for i in range(DONE_PAGE_SIZE)]
    text, _ = done_panel(done_view(items))
    assert len(text) <= TG_MAX_TEXT


def test_an_empty_run_says_so_rather_than_rendering_an_empty_screen():
    text, kb = done_panel(done_view([]))
    assert "nothing" in text.lower()
    kinds = [decode(data).kind for row in kb for (_, data) in row]
    assert "list" in kinds


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
    """The LAST page matters, not the first: page 0 only ever encodes indices
    0-7, so measuring it alone never sees a multi-digit index at all."""
    items = [held(f"t{i:03d}", "needs_reply") for i in range(60)]
    last_page = (len(items) - 1) // HELD_PAGE_SIZE
    first, last = digest(view(items), 0), digest(view(items), last_page)

    widest = max(len(label) for row in last[1] for (label, _) in row
                 if label.isdigit())
    assert widest == 2, "the last page did not encode a multi-digit index"

    for _, kb in (first, last, paged(request(60), 30, categories=["a"] * 10)):
        for row in kb:
            for _, data in row:
                assert len(data.encode()) <= 64, data


# --- the item view ----------------------------------------------------------
# What the numbered buttons have always implied and never did. One screen for
# both lists, with different verbs, because the two differ in tense.

def view_args(**kw):
    base = dict(subject="Data Scientist, Fraud at Stripe",
                sender="jobalerts-noreply@linkedin.com",
                actions_text="label(recruiter), archive",
                why="job alert from a no-reply address",
                digest_id="7f2a", index=2, kind="done",
                categories=["recruiter", "promotion"])
    return base | kw


def test_the_item_view_shows_what_happened_and_why():
    text, _ = item_view(**view_args())
    assert "Data Scientist, Fraud at Stripe" in text
    assert "jobalerts-noreply@linkedin.com" in text
    assert "label(recruiter), archive" in text
    assert "job alert from a no-reply address" in text


def test_a_done_item_offers_the_correction_verdicts():
    _, kb = item_view(**view_args(kind="done"))
    kinds = [decode(d).kind for row in kb for (_, d) in row]
    assert {"keep", "relabel", "teach_trash"} <= set(kinds)
    assert "approve" not in kinds, "a done action is not pending approval"


def test_a_held_item_offers_corrections_as_well_as_verdicts():
    """A held item is the one the agent STOPPED to ask about, so it is the last
    place the owner should be unable to answer. It shipped with approve and
    reject alone, which made "this is learning, keep it in the inbox"
    unsayable - the correction vocabulary existed and was wired only to the
    done list. Reject stays, because a refusal with no replacement is still
    signal; it is no longer the only thing that can be said."""
    _, kb = item_view(**view_args(kind="held"))
    kinds = [decode(d).kind for row in kb for (_, d) in row]
    assert {"approve", "reject", "keep", "relabel", "teach_trash"} <= set(kinds)


def test_a_held_item_still_leads_with_approve():
    """The common answer stays the first button. Corrections are the exception,
    and a keyboard that buries approve makes the ordinary case the slow one."""
    _, kb = item_view(**view_args(kind="held"))
    assert decode(kb[0][0][1]).kind == "approve"


def test_every_item_view_callback_is_within_the_byte_cap():
    """The cap test above measures digest and paged and never reached this
    screen, so the held keyboard grew three buttons outside anything that
    checks them. Telegram rejects callback_data over 64 bytes at send time,
    which on a phone is a keyboard that simply does not appear."""
    for kind in ("done", "held"):
        _, kb = item_view(**view_args(kind=kind, index=59,
                                      categories=["newsletter_valuable"] * 11))
        for row in kb:
            for _label, data in row:
                assert len(data.encode()) <= 64, data


def test_every_button_carries_the_digest_id():
    _, kb = item_view(**view_args())
    for row in kb:
        for _label, data in row:
            assert decode(data).digest_id == "7f2a"


def test_the_view_always_offers_a_way_back():
    """A screen with no exit is a trap on a phone, where there is no Escape."""
    for kind in ("done", "held"):
        _, kb = item_view(**view_args(kind=kind))
        assert "list" in [decode(d).kind for row in kb for (_, d) in row]


def test_the_item_view_never_exceeds_the_telegram_cap():
    text, _ = item_view(**view_args(subject="x" * 500, sender="s" * 500,
                                    why="y" * 3000, actions_text="a" * 500))
    assert len(text) <= TG_MAX_TEXT


def test_a_missing_reason_is_omitted_rather_than_invented():
    text, _ = item_view(**view_args(why=""))
    assert "Why:" not in text


def test_the_number_shown_matches_the_number_tapped():
    """The owner tapped "3" on a list; the screen that opens has to say 3."""
    text, _ = item_view(**view_args(index=2))
    assert text.startswith("3. ")


def test_the_done_panel_offers_a_button_per_item():
    """Without these the correction verdicts are unreachable: the item view is
    the only place they live, and this panel is the only route to it. Shipped
    without them once, because the flow was verified by calling encode()
    directly rather than by pressing what is on the screen."""
    items = [done(f"t{i}") for i in range(3)]
    _, kb = done_panel(done_view(items))
    opens = [(l, decode(d)) for row in kb for (l, d) in row
             if decode(d).kind == "open"]
    assert [l for l, _ in opens] == ["1", "2", "3"]
    assert [i.index for _, i in opens] == [0, 1, 2]


def test_the_done_panel_numbers_buttons_by_absolute_position():
    """Page two's first item is item 9, and its button has to say 9 and open 9."""
    items = [done(f"t{i:02d}") for i in range(DONE_PAGE_SIZE + 2)]
    _, kb = done_panel(done_view(items), page=1)
    opens = [(l, decode(d)) for row in kb for (l, d) in row
             if decode(d).kind == "open"]
    assert [l for l, _ in opens] == [str(DONE_PAGE_SIZE + 1), str(DONE_PAGE_SIZE + 2)]
    assert [i.index for _, i in opens] == [DONE_PAGE_SIZE, DONE_PAGE_SIZE + 1]


def test_the_panel_names_the_rule_that_decided_a_thread():
    """"→ trash · rule" says a rule decided it and refuses to say which - the
    same shape of half-answer as "3 label". For a category rule the pattern
    appears nowhere else on the screen at all."""
    item = done("t1", actions=[("trash", None)], from_rule=True)
    item.rule_note = "sender no-reply@p.simplywall.st → trash"
    text, _ = done_panel(done_view([item]))
    assert "sender no-reply@p.simplywall.st → trash" in text


def test_the_panel_says_nothing_extra_when_no_rule_decided_it():
    text, _ = done_panel(done_view([done("t1")]))
    assert "your rule" not in text


def test_the_item_view_shows_the_rule_in_full():
    text, _ = item_view(**view_args(
        rule_detail="Rule: sender no-reply@p.simplywall.st → trash\n"
                    "Taught 1 Sep · 1 hit, 0 overrides"))
    assert "sender no-reply@p.simplywall.st → trash" in text
    assert "1 hit, 0 overrides" in text


def test_the_rule_replaces_the_why_rather_than_repeating_it():
    """For a rule-decided thread the reason IS the rule - prefilter writes
    "matched sender rule 'x' -> trash" - so printing both puts the same
    sentence on the screen twice in different words."""
    text, _ = item_view(**view_args(
        why="matched sender rule 'no-reply@p.simplywall.st' -> trash",
        rule_detail="Rule: sender no-reply@p.simplywall.st → trash\n"
                    "Taught 1 Sep · 1 hit, 0 overrides"))
    assert "Why:" not in text
    assert "Rule:" in text


def test_the_item_view_omits_the_rule_line_when_the_model_decided():
    text, _ = item_view(**view_args())
    assert "Rule:" not in text


# --- a snippet under each done entry ----------------------------------------
# Asked for on a phone: "in done items can i get short summary in 2 or 3 lines".
# Taken from the snippet rather than a model-written summary, because the
# snippet is already on ReviewItem and already fetched - it costs no tokens, no
# latency, and does not move a single figure in the model registry.

def _done_view(**kw):
    from inbox_agent.telegram.render_tg import DoneItem
    item = DoneItem(thread_id="t1", subject="Lyft is hiring a Data Analyst",
                    sender="LinkedIn Job Alerts <jobalerts@linkedin.com>",
                    actions=[("archive", None)], **kw)
    return done_view([item])


def test_the_done_panel_shows_the_snippet():
    text, _ = done_panel(_done_view(snippet="Your job alert for data analyst "
                                            "in North York. New jobs match."))
    assert "Your job alert for data analyst" in text


def test_a_done_entry_without_a_snippet_renders_unchanged():
    """Older runs, and anything the join could not resolve, must not render a
    blank line where the summary would go."""
    text, _ = done_panel(_done_view(snippet=""))
    assert "\n\n\n" not in text
    assert "Lyft is hiring" in text


def test_the_snippet_is_capped_to_about_two_lines():
    text, _ = done_panel(_done_view(snippet="x" * 400))
    body = [l for l in text.splitlines() if l.startswith("x")]
    assert body and len(body[0]) <= 120, len(body[0]) if body else 0


def test_the_snippet_never_pushes_the_panel_past_telegrams_limit():
    """The byte budget already shrinks the page rather than truncating mid-word.
    Adding a snippet per entry must go through that same budget."""
    from inbox_agent.telegram.render_tg import DoneItem
    items = [DoneItem(thread_id=f"t{i}", subject="S" * 70, sender="a@b.com" * 6,
                      actions=[("label", "recruiter"), ("archive", None)],
                      snippet="y" * 200) for i in range(40)]
    text, _ = done_panel(done_view(items))
    assert len(text) <= 4096


# --- saying which items the one-tap button cannot reach ----------------------
# Asked on a phone, holding a digest with 8 held items and a button reading
# "Approve 2 replies & alerts": "is this expected?" It was - the button covers
# ATTENTION_REASONS and the other 6 were trash, which needs individual
# authorisation. But nothing on the screen said so, and the owner had to ask.
# A correct design the owner cannot read is not yet a finished design.

def test_the_trash_heading_says_it_needs_approving_one_at_a_time():
    text, _ = digest(view(held_items=[
        held("t1", "trash"), held("t2", "trash")]))
    trash_heading = [l for l in text.splitlines() if "TRASH" in l][0]
    assert "one at a time" in trash_heading or "individually" in trash_heading


def test_the_digest_says_how_many_the_one_tap_button_cannot_cover():
    """The exact question asked from the phone."""
    text, _ = digest(view(held_items=[
        held("t1", "security_alert"), held("t2", "needs_reply"),
        held("t3", "trash"), held("t4", "trash"), held("t5", "trash")]))
    assert "3 more" in text
    assert "one at a time" in text.lower() or "individually" in text.lower()


def test_no_footnote_when_the_button_covers_everything():
    """Nothing to explain, so nothing is said. A line that always appears stops
    being read."""
    text, _ = digest(view(held_items=[
        held("t1", "security_alert"), held("t2", "needs_reply")]))
    assert "more need" not in text.lower()


def test_no_footnote_when_there_is_no_one_tap_button_at_all():
    """With nothing in the attention tier the button is not drawn, so there is
    no split to explain - every item is individual and the numbered buttons
    already say that."""
    text, _ = digest(view(held_items=[held("t1", "trash"), held("t2", "trash")]))
    assert "more need" not in text.lower()


# --- the one-tap button names what is actually there -------------------------
# Read on a phone: "it says approve 2 replies & alerts but i only see security
# alerts and no replies". The label was a fixed string covering both
# ATTENTION_REASONS, so it claimed replies that were not there. A button that
# describes a queue the owner can see is not describing is a button they cannot
# trust, and this one authorises action on their mailbox.

def test_the_button_says_alerts_when_there_are_only_alerts():
    _, kb = digest(view(held_items=[
        held("t1", "security_alert"), held("t2", "security_alert")]))
    label = [l for row in kb for l, _ in row if l.startswith("✅")][0]
    assert label == "✅ Approve 2 alerts", label


def test_the_button_says_replies_when_there_are_only_replies():
    _, kb = digest(view(held_items=[
        held("t1", "needs_reply"), held("t2", "needs_reply")]))
    label = [l for row in kb for l, _ in row if l.startswith("✅")][0]
    assert label == "✅ Approve 2 replies", label


def test_the_button_names_both_when_both_are_present():
    _, kb = digest(view(held_items=[
        held("t1", "needs_reply"), held("t2", "security_alert"),
        held("t3", "security_alert")]))
    label = [l for row in kb for l, _ in row if l.startswith("✅")][0]
    assert label == "✅ Approve 1 reply & 2 alerts", label


def test_the_button_is_singular_for_one():
    _, kb = digest(view(held_items=[held("t1", "security_alert")]))
    label = [l for row in kb for l, _ in row if l.startswith("✅")][0]
    assert label == "✅ Approve 1 alert", label


def test_the_footnote_uses_the_same_words_as_the_button():
    """Two different names for the same set, one line apart, is the confusion
    this whole fix is about."""
    text, kb = digest(view(held_items=[
        held("t1", "security_alert"), held("t2", "trash"), held("t3", "trash")]))
    label = [l for row in kb for l, _ in row if l.startswith("✅")][0]
    assert "1 alert" in label
    assert "1 alert" in text


def test_the_footnote_agrees_in_number_for_a_single_leftover():
    text, _ = digest(view(held_items=[
        held("a1", "security_alert"), held("t1", "trash")]))
    note = [l for l in text.splitlines() if "button below" in l][0]
    assert "1 more needs approving" in note, note
    assert "1 more need approving" not in note


def test_the_header_reports_what_the_cap_left_behind():
    """Scanning 50 of 50 and scanning the first 50 of 80 must not look the
    same. Only one of them means the owner is caught up."""
    view = DigestView(run_at=NOW, total=50, done_by_kind={"archive": 50},
                      rule_decided=0, held=[], digest_id="ab12cd34",
                      dry_run=False, run_report=True, remaining=30)
    text, _ = digest(view)
    assert "50 threads" in text
    assert "30 more waiting" in text


def test_the_header_is_silent_when_the_cap_did_not_bind():
    """A permanently present '0 more waiting' is read for a week and then never
    again. A line that appears only when it means something keeps its meaning."""
    view = DigestView(run_at=NOW, total=12, done_by_kind={"archive": 12},
                      rule_decided=0, held=[], digest_id="ab12cd34",
                      dry_run=False, run_report=True, remaining=0)
    text, _ = digest(view)
    assert "more waiting" not in text


def test_held_reports_no_remainder_because_it_ran_nothing():
    view = DigestView(run_at=NOW, total=0, done_by_kind={}, rule_decided=0,
                      held=[], digest_id="ab12cd34", dry_run=False,
                      run_report=False, remaining=30)
    text, _ = digest(view)
    assert "more waiting" not in text

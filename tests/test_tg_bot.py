"""The bot, tested without Telegram.

A fake transport records outbound calls and lets tests inject updates, so the
whole path - run, queue, digest render, callback parse - runs at the speed of
the rest of the suite.

/triage is INCREMENTAL: it acts and reports. The interrupt path still exists for
Plan 3's /backlog sweep, but nothing here drives it any more, and its guarantees
are covered where they now live - at the graph level in tests/test_graph.py and
at the boundary in tests/test_tg_callbacks.py.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from inbox_agent.audit import AuditLog
from inbox_agent.classify import ThreadJudgment
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.graph import build_graph
from inbox_agent.models import ActionTemplate, Action, ReviewItem, Rule
from inbox_agent.policy import Policy
from inbox_agent.store import HeldQueue, PreferenceStore, build_store
from inbox_agent.telegram.bot import Bot
from inbox_agent.telegram.callbacks import decode, encode

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


class FakeLLM:
    def with_structured_output(self, schema): return self
    def invoke(self, messages):
        return ThreadJudgment(category="promotion", action="archive",
                              reason="a sale", confidence=0.9)


class FakeTransport:
    """Records what the bot would have sent."""
    def __init__(self):
        self.sent, self.edited, self.answered = [], [], []
    def send_message(self, chat_id, text, keyboard=None):
        self.sent.append({"chat_id": chat_id, "text": text, "keyboard": keyboard})
        return {"message_id": 100 + len(self.sent)}
    def edit_message(self, chat_id, message_id, text, keyboard=None):
        self.edited.append({"chat_id": chat_id, "message_id": message_id,
                            "text": text, "keyboard": keyboard})
    def answer_callback(self, callback_id, text=""):
        self.answered.append({"id": callback_id, "text": text})


@pytest.fixture
def snapshot_file(tmp_path):
    # UNREAD matters: settings.inbox_query fetches only unread untriaged mail,
    # so a snapshot without it produces an empty run and every assertion below
    # would pass vacuously.
    data = [{"id": f"t{i}", "subject": f"Sale {i}", "sender": f"deals{i}@shop.com",
             "to": [], "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "",
             "label_ids": ["INBOX", "UNREAD"]} for i in range(4)]
    p = tmp_path / "threads.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def bot(tmp_path, snapshot_file):
    settings = Settings(backend="offline", dry_run=True, snapshot_dir=tmp_path,
                        snapshot_size=50, audit_log=tmp_path / "audit.jsonl",
                        forbidden_actions=ALWAYS_FORBIDDEN,
                        context_hub_skill="s", context_hub_tag="dev",
                        # Under tmp_path on purpose. The bot now reads the OAuth
                        # consent sidecar next to this file, and the default
                        # points at the developer's real secrets/ - which would
                        # make these tests pass or fail according to when
                        # somebody last clicked through Google's consent screen.
                        google_token=tmp_path / "token.json",
                        tg_token="tok", tg_chat_id="42", tg_mode="digest")
    log = AuditLog(settings.audit_log)
    # One queue, shared by the graph that fills it and the bot that renders it.
    # Two instances over two stores would let the bot show an empty queue while
    # the graph quietly filled another one.
    held = HeldQueue(build_store())
    # One store, shared by the graph that reads rules and the bot that writes
    # them - two instances would let a correction land where nothing reads it.
    prefs = PreferenceStore(build_store())
    client = SnapshotGmailClient(snapshot_file)
    graph = build_graph(client=client,
                        prefs=prefs,
                        policy=Policy(text="P", version="local:t", source="local"),
                        llm=FakeLLM(), settings=settings, log=log, held=held,
                        checkpointer=InMemorySaver())
    t = FakeTransport()
    return Bot(transport=t, graph=graph, settings=settings, held=held,
               prefs=prefs, client=client, log=log,
               categories=["recruiter", "promotion"]), t, log


def msg(text, chat_id=42):
    return {"message": {"chat": {"id": chat_id}, "from": {"id": chat_id}, "text": text}}


def cb(data, chat_id=42, message_id=101):
    return {"callback_query": {"id": "cb1", "data": data,
                               "from": {"id": chat_id},
                               "message": {"chat": {"id": chat_id},
                                           "message_id": message_id}}}


def review_item(tid="held-1", category="promotion", action="trash"):
    return ReviewItem(thread_id=tid, category=category,
                      subject=f"Subject {tid}", sender=f"{tid}@example.com",
                      snippet="s",
                      proposed=[Action(kind=action, thread_id=tid)],
                      reason="looks like junk", confidence=0.9, source="model")


# --- authorisation ----------------------------------------------------------

def test_update_from_an_unauthorised_chat_is_dropped(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage", chat_id=999))
    assert t.sent == [], "the bot replied to an unauthorised chat"


def test_unauthorised_callback_is_dropped(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    t.sent.clear()
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id), chat_id=999))
    assert t.sent == [] and t.edited == []


def test_unauthorised_attempts_are_recorded_not_silent(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage", chat_id=999))
    assert b.rejected_updates == 1


def test_an_unauthorised_update_is_dropped_before_anything_runs(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage", chat_id=999999))
    assert b.rejected_updates == 1
    assert b._runs_started == 0


# --- /triage acts, it does not preview --------------------------------------

def test_triage_does_not_park_the_run(bot):
    """The central decision of the whole plan, asserted directly.

    mode="backlog" was a deliberate transitional hack while the bot rendered
    from a parked checkpoint. Left in place it inverts /triage from acting into
    previewing, silently, with a green suite - so the mode is pinned by a test
    rather than by a comment.
    """
    b, t, log = bot
    b.handle_update(msg("/triage 4"))
    assert "__interrupt__" not in b._last_run, "/triage parked instead of acting"
    assert b._last_run["executed"], "the run parked or executed nothing"
    assert [r for r in log.records() if r.action == "archive"], \
        "nothing reached the audit log"


def test_triage_sends_a_digest_built_from_the_queue(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    assert t.sent, "the digest was never sent"
    assert "DONE" in t.sent[-1]["text"]


def test_the_digest_shows_what_the_run_held(bot):
    """The wiring under test: graph -> queue -> bot -> digest, one queue."""
    b, t, _ = bot
    b.held.add(review_item("held-1"), run_id="r0", reason="trash")
    b.handle_update(msg("/triage 4"))
    assert "TRASH" in t.sent[-1]["text"]
    assert "Subject held-1" in t.sent[-1]["text"]


def test_the_done_counts_leave_out_the_triaged_label(bot):
    """The triaged label is bookkeeping, not work the owner cares about.

    Driven against _view() directly rather than through a run. mark_triaged
    currently writes its records to the audit log without surfacing them into
    state's `executed`, so an end-to-end version of this test passes just as
    happily with the filter deleted - it pins nothing. A real label the owner
    asked for is in here too, so the filter cannot be "drop all labels".
    """
    b, t, _ = bot
    b._last_run = {
        "thread_ids": ["t0", "t1", "t2"],
        "executed": [
            {"action": "archive", "params": {}, "actor": "agent"},
            {"action": "label", "params": {"label": "work"}, "actor": "agent"},
            {"action": "label", "params": {"label": b.settings.triaged_label},
             "actor": "agent"},
            {"action": "label", "params": {"label": b.settings.triaged_label},
             "actor": "rule:r-1"},
        ],
    }
    view = b._view()
    assert view.done_by_kind == {"archive": 1, "label": 1}, view.done_by_kind
    assert view.total == 3
    # The rule-decided count must not credit the bookkeeping record either.
    assert view.rule_decided == 0


def test_a_rule_decided_action_is_counted_as_learning(bot):
    b, t, _ = bot
    b._last_run = {"thread_ids": ["t0"], "executed": [
        {"action": "archive", "params": {}, "actor": "rule:r-1"},
        {"action": "archive", "params": {}, "actor": "agent"}]}
    assert b._view().rule_decided == 1


def test_a_malformed_audit_record_does_not_take_the_digest_down(bot):
    """This runs AFTER the graph executed. A crash here costs the owner the
    report for work that already reached Gmail."""
    b, t, _ = bot
    b._last_run = {"thread_ids": ["t0"], "executed": [
        {"actor": "agent"},                        # no action
        {"action": "label", "params": None, "actor": "agent"},   # null params
        {"action": "archive"}]}                    # no params, no actor
    assert b._view().done_by_kind == {"label": 1, "archive": 1}


def test_a_dry_run_digest_does_not_claim_the_work_was_done(bot):
    """The fixture is dry_run=True, which is how this project dogfoods next."""
    b, t, _ = bot
    assert b.settings.dry_run
    b.handle_update(msg("/triage 4"))
    assert "WOULD HAVE DONE" in t.sent[-1]["text"]


def test_a_failing_triage_says_so_rather_than_going_quiet(bot, monkeypatch):
    """Silence is indistinguishable from an empty inbox.

    A model that is down must not look like a morning with no mail - that is
    the failure the owner would trust for days without noticing.
    """
    b, t, _ = bot

    def boom(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(b.graph, "invoke", boom)
    b.handle_update(msg("/triage 4"))
    text = t.sent[-1]["text"]
    assert "failed" in text.lower()
    assert "RuntimeError" in text, "the owner cannot act on an unnamed failure"
    # The run can raise after auto_execute already acted, so it must not claim
    # a clean slate any more than /cancel may.
    assert "nothing was executed" not in text.lower()


# --- commands ---------------------------------------------------------------

def test_held_command_shows_the_queue_without_running_a_triage(bot):
    """The queue outlives runs, so looking at it must not produce more work."""
    b, t, _ = bot
    before = len(t.sent)
    b.handle_update(msg("/held"))
    assert len(t.sent) == before + 1
    assert b._runs_started == 0


def test_held_does_not_re_report_the_previous_runs_work(bot):
    """_last_run survives, so a naive /held stamps 08:00's counts with the
    current clock - and does it again on every later /held."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    assert "DONE" in t.sent[-1]["text"], "the run digest should report the run"

    b.handle_update(msg("/held"))
    text = t.sent[-1]["text"]
    assert "DONE" not in text, "/held re-reported a run it did not perform"
    assert "threads" not in text
    assert "taught" not in text
    labels = [l for row in (t.sent[-1]["keyboard"] or []) for (l, _) in row]
    assert not any("done" in l.lower() for l in labels)


def test_paging_a_held_digest_does_not_bring_the_run_report_back(bot):
    """_run_report has to be state: paging re-renders the same message."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(msg("/held"))
    b.handle_update(cb(encode("next", digest_id=b._digest_id)))
    assert t.edited, "paging did not re-render"
    assert "DONE" not in t.edited[-1]["text"]


# --- the done panel ---------------------------------------------------------
# The digest says "3 label" and cannot say which label. The button that claims
# to expand that was wired to a re-render of the same message, which Telegram
# rejects as unmodified - so on a real phone it did nothing at all.

def test_tapping_show_the_done_opens_a_panel_listing_the_threads(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    assert "Sale 0" not in t.sent[-1]["text"], "the digest already listed them"

    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    assert t.edited, "the done button did not render anything"
    assert "Sale 0" in t.edited[-1]["text"]


def test_the_panel_names_the_label_a_thread_was_given(bot):
    """The question the panel exists to answer, end to end through the bot: an
    audit record knows the thread by id, and the owner does not."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b._last_run = {
        "thread_ids": ["t0"],
        "auto": [review_item("t0", action="archive").model_dump(mode="json")],
        "executed": [
            {"thread_id": "t0", "action": "label",
             "params": {"label": "recruiter"}, "actor": "agent"},
            {"thread_id": "t0", "action": "archive", "params": {},
             "actor": "agent"},
        ],
    }
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    text = t.edited[-1]["text"]
    assert "label(recruiter)" in text
    assert "Subject t0" in text, "the panel showed an id instead of a subject"


def test_the_panel_credits_a_rule_that_decided_a_thread(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b._last_run = {
        "thread_ids": ["t0"],
        "auto": [review_item("t0").model_dump(mode="json")],
        "executed": [{"thread_id": "t0", "action": "archive", "params": {},
                      "actor": "rule:r-123"}],
    }
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    assert "rule" in t.edited[-1]["text"].lower()


def test_the_triaged_label_is_bookkeeping_and_stays_out_of_the_panel(bot):
    """Same exclusion the counts already make. Every thread gets this label;
    listing it would bury the work the owner actually cares about."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b._last_run = {
        "thread_ids": ["t0"],
        "auto": [review_item("t0").model_dump(mode="json")],
        "executed": [{"thread_id": "t0", "action": "label",
                      "params": {"label": "agent/triaged"}, "actor": "agent"},
                     {"thread_id": "t0", "action": "archive", "params": {},
                      "actor": "agent"}],
    }
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    assert "agent/triaged" not in t.edited[-1]["text"]


def test_back_from_the_panel_returns_to_the_digest(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    digest_text = t.sent[-1]["text"]
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("list", digest_id=b._digest_id)))
    assert t.edited[-1]["text"] == digest_text


def test_the_digest_page_survives_a_trip_through_the_panel(bot):
    """Paging is shared state. Opening the panel and coming back must not
    silently move the owner to page one of a queue they were reading."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    for item in [review_item(f"q{i}") for i in range(12)]:
        b.held.add(item, run_id="r1", reason="trash")
    b.handle_update(cb(encode("next", digest_id=b._digest_id)))
    assert b._page == 1
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("list", digest_id=b._digest_id)))
    assert b._page == 1


def test_a_new_triage_leaves_the_panel(bot):
    """Otherwise the next run's digest renders into a screen the owner opened
    for the last one."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(msg("/triage 4"))
    assert "waiting" in t.sent[-1]["text"], "a digest did not come back"


def test_a_stale_tap_says_so_instead_of_doing_nothing(bot):
    """The guard is right to drop it. Dropping it in silence is what makes the
    owner tap three more times and then ask what the button is for."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id="dead")))
    assert t.edited == [], "a stale tap was acted on"
    assert "out of date" in t.answered[-1]["text"]


def test_every_button_on_the_digest_now_acts(bot):
    """Was test_a_button_with_no_behaviour_yet_says_that_too, asserted on
    approve_attention as the last unbuilt button. It is built, so there is no
    inert button left to assert against and the old test asserted the opposite
    of what the code now does.

    The invariant it protected survives and is what is asserted here: a tap is
    never answered with silence. That defect - a button that renders, does
    nothing, and says nothing - is the one this UI has shipped three times, and
    the reason it kept shipping is that verification called encode() instead of
    pressing what was on the screen.
    """
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    for item in [review_item(f"a{i}", category="needs_reply") for i in range(2)]:
        b.held.add(item, run_id="r1", reason="needs_reply")
    before = len(b.held.all())
    b.handle_update(cb(encode("approve_attention", digest_id=b._digest_id)))
    # It acted...
    assert len(b.held.all()) < before, "the button still does nothing"
    # ...and it said so on the screen, not only in a toast that vanishes.
    assert t.edited, "acted but left the screen unchanged"
    assert "Approved" in t.edited[-1]["text"]


def test_held_is_offered_in_the_help_text(bot):
    b, t, _ = bot
    b.handle_update(msg("/nonsense"))
    assert "/held" in t.sent[-1]["text"]


def test_status_reports_no_parked_run_after_an_incremental_triage(bot):
    """/triage completes, so nothing is waiting. Saying otherwise would offer
    the owner a resume that cannot happen."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    t.sent.clear()
    b.handle_update(msg("/status"))
    assert "no run is waiting" in t.sent[-1]["text"].lower()


def test_cancel_invalidates_the_current_digest(bot):
    """A cancelled conversation must not still be tappable."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    stale = encode("open", 0, digest_id=b._digest_id)
    b.handle_update(msg("/cancel"))
    before = len(t.edited)
    b.handle_update(cb(stale))
    assert len(t.edited) == before, "a cancelled digest still accepted a tap"


def test_cancel_does_not_claim_nothing_was_executed(bot):
    """/triage is incremental: by the time /cancel can be typed the agent has
    already acted. Telling the owner their mail is untouched would be false."""
    b, t, log = bot
    b.handle_update(msg("/triage 4"))
    assert log.records(), "the run should have executed before /cancel"
    b.handle_update(msg("/cancel"))
    assert "nothing was executed" not in t.sent[-1]["text"].lower()
    assert "cancelled" in t.sent[-1]["text"].lower()


# --- callbacks --------------------------------------------------------------

def test_a_callback_from_a_previous_digest_is_ignored(bot):
    """The whole point of the digest id."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    before = len(t.edited)
    b.handle_update(cb(encode("open", 0, digest_id="dead")))
    assert len(t.edited) == before


def test_a_callback_from_the_current_digest_is_honoured(bot):
    """The other half of the digest id: a current tap must get through.

    Asserted on `done`, which now renders the panel, rather than on `open`,
    which is still waiting on Plan 2 and proves acceptance only by the toast it
    answers with - see test_a_button_with_no_behaviour_yet_says_that_too.
    """
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    assert t.edited


def test_a_callback_carrying_no_digest_id_at_all_is_ignored(bot):
    """An empty id is what decode() gives an id-less callback, and it is also
    the bot's own starting state - so it must never be allowed to match. Like
    any other stale tap it now re-renders the queue rather than acting.

    Checking only `_panel == "digest"` plus a "waiting" line does NOT
    discriminate here: with the held queue empty and no digest ever shown,
    there is no item 0 to open, so `_item_screen()`'s own "list moved under
    the callback" fallback lands on `_panel == "digest"` with a "waiting"
    line too - even when a BROKEN guard let the id-less `open` fall through.
    What that fallback cannot reproduce is what `_show_queue()` itself does:
    mint a fresh, non-empty digest id and set `run_report = False`, so the
    render carries no DONE section at all. Those are what this asserts on -
    verified by temporarily deleting the guard's `not self._digest_id or`
    clause and confirming this test then fails (see task-6-report.md).
    """
    b, t, _ = bot
    b.handle_update(cb(encode("open", 0)))
    assert t.edited == []
    assert b._digest_id != "", "a stale tap should mint a fresh digest via _show_queue"
    assert "DONE" not in t.sent[-1]["text"], "a run report leaked in from the open-item fallback"


def test_hostile_callback_data_is_answered_and_ignored(bot):
    b, t, log = bot
    b.handle_update(msg("/triage 4"))
    before, answered = len(log.records()), len(t.answered)
    junk = ("", "zzz", "a:-1", "../../x", "l:1")
    for data in junk:
        b.handle_update(cb(data))
    assert len(log.records()) == before, "junk callback data reached Gmail"
    # ANSWERED, not just ignored. Telegram spins the button until the query is
    # answered, so silently dropping one leaves the owner staring at a spinner.
    assert len(t.answered) == answered + len(junk)


def test_a_forged_index_from_the_digest_executes_nothing(bot):
    """An index outside the queue names no thread. Structural, not validated."""
    b, t, log = bot
    b.handle_update(msg("/triage 4"))
    before = len(log.records())
    b.handle_update(cb(encode("open", 99, digest_id=b._digest_id)))
    assert len(log.records()) == before


def test_replaying_a_digest_callback_executes_nothing(bot):
    """The digest's buttons are navigation. Nothing on it commits an action,
    so a replayed tap cannot double-execute anything."""
    b, t, log = bot
    b.handle_update(msg("/triage 4"))
    before = len(log.records())
    data = encode("approve_attention", digest_id=b._digest_id)
    b.handle_update(cb(data))
    b.handle_update(cb(data))
    assert len(log.records()) == before


def test_paging_the_digest_edits_one_message_instead_of_sending_many(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    # A run now sends an acknowledgement before the digest, so the count after
    # the run is the baseline rather than a literal 1. The guarantee under test
    # was never "one message ever" - it is that paging adds none.
    after_run = len(t.sent)
    b.handle_update(cb(encode("next", digest_id=b._digest_id)))
    b.handle_update(cb(encode("prev", digest_id=b._digest_id)))
    assert len(t.sent) == after_run, "paging sent extra messages"
    assert len(t.edited) == 2, "paging did not edit in place"


# --- corrections write rules -------------------------------------------------
# The loop the design named and never closed: the agent acts alone on the
# majority of mail, and until now nothing it did alone could teach it anything.

def _done_run(b, thread_id="t0", category="promotion", actions=(("archive", None),),
              actor="agent"):
    """Put one known done item in front of the bot, so a verdict has something
    to correct that the test controls."""
    b._last_run = {
        "thread_ids": [thread_id],
        "auto": [review_item(thread_id, category=category).model_dump(mode="json")],
        "executed": [{"thread_id": thread_id, "action": kind,
                      "params": {"label": label} if label else {}, "actor": actor}
                     for kind, label in actions],
    }


def test_tapping_a_number_opens_the_item(bot):
    """The button that did nothing for two sessions."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    kinds = [decode(d).kind for row in t.edited[-1]["keyboard"] for (_, d) in row]
    assert "keep" in kinds
    assert "Subject t0" in t.edited[-1]["text"]


def test_a_verdict_asks_how_wide_before_writing_anything(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    assert b.prefs.rules() == [], "a rule was written before the scope was chosen"
    kinds = [decode(d).kind for row in t.edited[-1]["keyboard"] for (_, d) in row]
    assert {"scope_narrow", "scope_wide"} <= set(kinds)


def test_the_narrow_answer_writes_a_rule_about_the_sender(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("label", "promotion"), ("archive", None)))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert rule.scope in ("sender", "domain", "fingerprint", "subject")
    assert "archive" not in [a.kind for a in rule.actions], \
        "keep in inbox taught a rule that still archives"


def test_the_wide_answer_writes_a_rule_about_the_category(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("label", "promotion"), ("archive", None)))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_wide", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert rule.scope == "category"
    assert rule.pattern == "promotion"
    assert [a.kind for a in rule.actions] == ["label"]


def test_the_confirmation_says_the_rule_in_words(bot):
    """A rule that cannot be read cannot be corrected."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("label", "promotion"), ("archive", None)))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_wide", 0, digest_id=b._digest_id)))
    assert "promotion" in t.edited[-1]["text"]
    assert "label(promotion)" in t.edited[-1]["text"]


def test_a_correction_claims_no_reversal(bot):
    """Nothing was reversed - under dry-run nothing happened at all - and
    saying otherwise is the most expensive thing this message could get wrong."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    blob = t.edited[-1]["text"].lower()
    assert "undo" not in blob and "reversed" not in blob and "undone" not in blob


def test_correcting_a_rule_decided_action_records_an_override(bot):
    """record_override has had no caller since it was written. This is it: a
    correction of what a rule proposed IS an override of that rule."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.prefs.add_rule(Rule(id="r-1", scope="sender", pattern="deals0@shop.com",
                          actions=[ActionTemplate(kind="archive")],
                          provenance="p", created_at=NOW))
    _done_run(b, actor="rule:r-1")
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    assert {r.id: r for r in b.prefs.rules()}["r-1"].override_count == 1


def test_teaching_trash_says_it_will_not_ask_again(bot):
    """Plan 1 decided a rule-proposed trash auto-executes. That makes this the
    one verdict that widens what happens without a second question."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("teach_trash", 0, digest_id=b._digest_id)))
    assert "without asking" in t.edited[-1]["text"].lower()
    rule = b.prefs.rules()[0]
    assert [a.kind for a in rule.actions] == ["trash"]
    assert rule.scope != "category", "a category-wide trash rule is too wide to teach"


def test_relabel_offers_the_policys_categories(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    labels = [l for row in t.edited[-1]["keyboard"] for (l, _) in row]
    assert any("recruiter" in l for l in labels)


# --- relabel writes the label it names ---------------------------------------
# 6 of 10 threads in the run that produced this bug were actioned as bare
# `archive, unlabel(UNREAD)` - no `label` action to substitute the chosen
# category into. The old code silently dropped the correction and taught a
# rule that filed mail away instead of labelling it, at rule authority, so
# it auto-executed the opposite of what the owner asked.

def test_relabel_over_actions_with_no_label_produces_label_chosen(bot):
    """The reported bug, reproduced exactly: a run that only archived and
    cleared UNREAD has no `label` action to substitute into. This must
    fail against the code that substitutes into an existing label action,
    because there is nothing here to substitute into."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("archive", None), ("unlabel", "UNREAD")))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, 0, digest_id=b._digest_id)))  # -> recruiter
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert any(a.kind == "label" and a.params.get("label") == "recruiter"
               for a in rule.actions), \
        f"the chosen label never made it into the rule: {rule.actions!r}"


def test_relabel_over_actions_with_a_label_replaces_it(bot):
    """A label was already there - the owner just picked the wrong one -
    so the fix must swap it, not add a second label action alongside it."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("label", "recruiter"), ("archive", None),
                          ("unlabel", "UNREAD")))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, 1, digest_id=b._digest_id)))  # -> promotion
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    kinds = [a.kind for a in rule.actions]
    assert kinds.count("label") == 1, f"expected exactly one label, got {rule.actions!r}"
    assert rule.actions[kinds.index("label")].params.get("label") == "promotion"


def test_unlabel_is_never_mistaken_for_a_label_action(bot):
    """unlabel(UNREAD) carries a `label` param exactly like `label` does.
    Only kind == "label" counts as a label action - unlabel must pass
    through untouched and must not be counted as, or replaced like, one."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("unlabel", "UNREAD"),))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, 0, digest_id=b._digest_id)))  # -> recruiter
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    kinds = [a.kind for a in rule.actions]
    assert kinds == ["label", "unlabel"]
    assert kinds.count("label") == 1


# --- relabel also asks about filing -------------------------------------
# `learning` stays in the inbox per policy, but relabel used to preserve
# whatever filing the wrong category produced. The owner's decision: ask
# every time, right after the category tap and before the scope question.

def test_choosing_a_category_asks_about_filing_before_scope(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, 0, digest_id=b._digest_id)))
    assert b.prefs.rules() == [], "a rule was written before filing/scope were chosen"
    kinds = [decode(d).kind for row in t.edited[-1]["keyboard"] for (_, d) in row]
    assert {"keep_inbox", "file_away"} <= set(kinds)


def test_keep_in_inbox_removes_archive_and_trash(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    # Both archive and trash present, so the trash half of the assertion
    # below is not vacuous - a filter that dropped "trash" from its
    # exclusion set would still pass a fixture with no trash in it.
    _done_run(b, actions=(("archive", None), ("trash", None), ("unlabel", "UNREAD")))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep_inbox", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    kinds = [a.kind for a in rule.actions]
    assert "archive" not in kinds and "trash" not in kinds


def test_file_it_away_also_strips_a_coexisting_trash(bot):
    """The path a review found reachable: a taught teach_trash rule always
    teaches exactly [trash()]. When that rule later fires, the done item's
    only action is trash - render_tg offers "Label as..." on any done item
    with no gate on what it did, so relabel can be tapped on a trashed
    item. Filing it away must not let that trash ride along: "keep this,
    file it under X" would otherwise still re-trash future matching mail,
    the opposite of what was asked."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("trash", None),), actor="rule:x")
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("file_away", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    kinds = [a.kind for a in rule.actions]
    assert "trash" not in kinds, f"trash rode along into the taught rule: {rule.actions!r}"
    assert kinds.count("archive") == 1


def test_file_it_away_adds_archive_when_absent(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("label", "recruiter"), ("unlabel", "UNREAD")))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, 1, digest_id=b._digest_id)))
    b.handle_update(cb(encode("file_away", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    kinds = [a.kind for a in rule.actions]
    assert kinds.count("archive") == 1


def test_file_it_away_does_not_duplicate_an_existing_archive(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("archive", None), ("unlabel", "UNREAD")))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("file_away", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    kinds = [a.kind for a in rule.actions]
    assert kinds.count("archive") == 1


def test_the_full_relabel_sequence_writes_exactly_what_the_taps_said(bot):
    """relabel -> category -> filing -> scope, end to end."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b)  # default: category "promotion", actions (("archive", None),)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("relabel", 0, digest_id=b._digest_id)))       # tap 1
    b.handle_update(cb(encode("relabel", 0, 0, digest_id=b._digest_id)))    # tap 2: recruiter
    b.handle_update(cb(encode("keep_inbox", 0, digest_id=b._digest_id)))    # tap 3
    b.handle_update(cb(encode("scope_wide", 0, digest_id=b._digest_id)))    # tap 4
    rule = b.prefs.rules()[0]
    assert rule.scope == "category"
    assert rule.pattern == "recruiter"
    assert [(a.kind, a.params) for a in rule.actions] == \
        [("label", {"label": "recruiter"})]


# --- held verdicts do something ---------------------------------------------
# The item view shipped with Approve and Not this and no handler behind either.
# A button that renders and does nothing is the defect this whole branch has
# been removing; it got through because the tests asserted the keyboard and
# never pressed it.

def _open_held(b, index=0):
    b.handle_update(msg("/triage 4"))
    for item in [review_item(f"h{i}") for i in range(2)]:
        b.held.add(item, run_id="r1", reason="trash")
    b.handle_update(cb(encode("open", index, digest_id=b._digest_id)))


def test_approving_a_held_item_executes_it(bot):
    b, t, log = bot
    _open_held(b)
    before = len(log.records())
    b.handle_update(cb(encode("approve", 0, digest_id=b._digest_id)))
    after = [r for r in log.records()[before:] if r.action == "trash"]
    assert after, "approve did not push the action through the chokepoint"
    assert after[0].actor == "human"


def test_approving_a_held_item_drains_it_from_the_queue(bot):
    """Otherwise the owner authorises the same trash every morning forever."""
    b, t, _ = bot
    _open_held(b)
    assert len(b.held.all()) == 2
    b.handle_update(cb(encode("approve", 0, digest_id=b._digest_id)))
    assert [h.thread_id for h in b.held.all()] == ["h1"]


def test_not_this_drains_the_queue_without_executing(bot):
    b, t, log = bot
    _open_held(b)
    before = len(log.records())
    b.handle_update(cb(encode("reject", 0, digest_id=b._digest_id)))
    assert not [r for r in log.records()[before:] if r.action == "trash"]
    assert [h.thread_id for h in b.held.all()] == ["h1"]


def test_not_this_teaches_that_the_action_was_wrong(bot):
    """A bare reject is signal. The digest design counts every reject as a
    candidate rule, and requiring an edit is why skipping never taught
    anything."""
    b, t, _ = bot
    _open_held(b)
    b.handle_update(cb(encode("reject", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert rule.rejected_action == "trash"
    assert [a.kind for a in rule.actions] == ["none"]


def test_a_held_verdict_says_what_it_did(bot):
    b, t, _ = bot
    _open_held(b)
    b.handle_update(cb(encode("approve", 0, digest_id=b._digest_id)))
    assert t.edited[-1]["text"], "approve answered with an empty screen"
    assert "1 left" in t.edited[-1]["text"] or "waiting" in t.edited[-1]["text"].lower()


def test_a_dry_run_approval_does_not_claim_it_reached_gmail(bot):
    """settings.dry_run is True in this fixture. Same rule as the digest: never
    the word done for something that did not happen."""
    b, t, _ = bot
    _open_held(b)
    b.handle_update(cb(encode("approve", 0, digest_id=b._digest_id)))
    assert "would have" in t.edited[-1]["text"].lower()


# --- acknowledging a tap must never cost the tap ------------------------------
# answerCallbackQuery clears the spinner and nothing else. Telegram expires a
# query id in seconds, so any tap that queued while the bot was down comes back
# as "query is too old" - and that 400 used to abort the handler before the work
# ran. Seen live: eleven taps, eleven tracebacks, nothing acted on.

class FailingAckTransport(FakeTransport):
    def answer_callback(self, callback_id, text=""):
        super().answer_callback(callback_id, text)
        raise RuntimeError("Bad Request: query is too old")


def test_a_failed_acknowledgement_does_not_cost_the_action(bot):
    b, t, _ = bot
    b.transport = FailingAckTransport()
    b.handle_update(msg("/triage 4"))
    b.transport.edited.clear()
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    assert b.transport.edited, "the ack failing stopped the panel from rendering"


def test_a_failed_acknowledgement_on_a_stale_tap_does_not_raise(bot):
    """The stale branch answers too, and its answer expires for exactly the
    same reason - the tap has been sitting in Telegram's backlog."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.transport = FailingAckTransport()
    b.handle_update(cb(encode("done", digest_id="dead")))   # must not raise


def test_a_failed_acknowledgement_on_a_verdict_still_writes_the_rule(bot):
    """The expensive case: the owner taught something and the ack expired."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("label", "promotion"), ("archive", None)))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.transport = FailingAckTransport()
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_wide", 0, digest_id=b._digest_id)))
    assert b.prefs.rules(), "a correction was lost because the spinner failed"


# --- which rule decided it --------------------------------------------------

def test_the_panel_names_the_rule_that_decided_a_thread(bot):
    """The audit record has carried rule_provenance since Stage A and nothing
    ever showed it. "· rule" told the owner a rule decided this and refused to
    say which one."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.prefs.add_rule(Rule(id="r-1", scope="sender", pattern="deals0@shop.com",
                          actions=[ActionTemplate(kind="trash")],
                          provenance="owner corrected it", created_at=NOW))
    _done_run(b, actions=(("trash", None),), actor="rule:r-1")
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    assert "sender deals0@shop.com" in t.edited[-1]["text"]
    assert "trash" in t.edited[-1]["text"]


def test_opening_a_rule_decided_item_shows_the_rule_and_its_record(bot):
    """The owner is being asked to judge the rule, so they get what it says,
    when they taught it, and how it has performed."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    rule = b.prefs.add_rule(Rule(id="r-1", scope="sender", pattern="deals0@shop.com",
                                 actions=[ActionTemplate(kind="trash")],
                                 provenance="owner corrected it", created_at=NOW,
                                 hit_count=3, override_count=1))
    _done_run(b, actions=(("trash", None),), actor="rule:r-1")
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    text = t.edited[-1]["text"]
    assert "sender deals0@shop.com" in text
    assert "3 hit" in text and "1 override" in text


def test_a_model_decided_item_shows_no_rule_line(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    assert "Rule:" not in t.edited[-1]["text"]


def test_a_deleted_rule_does_not_break_the_item_view(bot):
    """The rule can be gone by the time the owner opens the item - demoted,
    replaced, or the store rebuilt. The record of what happened survives it."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actions=(("trash", None),), actor="rule:r-vanished")
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    assert t.edited[-1]["text"]


def test_correcting_a_rule_decision_records_which_rule_it_overrode(bot):
    """`record_override` already bumps the old rule's counter. What was missing
    is the join: which correction produced which replacement. Without it a
    future vote-distribution over corrections cannot be reconstructed from the
    rules the owner is teaching now - the pairing is only knowable here."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _done_run(b, actor="rule:r-old1234")
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("teach_trash", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert rule.supersedes == "r-old1234"


# --- the attention-tier approve --------------------------------------------
# Reported from a phone as "approve button is not working". It was not broken:
# it was the one button knowingly left unbuilt, answering with a Telegram toast
# that is trivially missed on a phone. The safety objection recorded against it
# is already satisfied on the render side - render_tg.py filters the button's
# set to ATTENTION_REASONS before it is ever offered - so it structurally cannot
# reach trash or a low-confidence guess. Only the wiring was missing.

def _held_of(b, reason, thread_id, kind="draft", label=None):
    from inbox_agent.models import Action, HeldItem, ReviewItem
    from datetime import datetime, timezone
    b.held.add(ReviewItem(thread_id=thread_id, category=reason,
                          subject=f"Subject {thread_id}",
                          sender=f"{thread_id}@x.com", snippet="s",
                          proposed=[Action(kind=kind, thread_id=thread_id,
                                           params={"label": label} if label else {})],
                          reason="r", confidence=0.9, source="model"),
               run_id="r1", reason=reason)


def test_approve_attention_executes_the_attention_tier(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    # Seeded AFTER the run: a held entry for a thread the run itself processed
    # is now retired as superseded, which is the point of
    # test_a_rerun_that_stops_holding_a_thread_clears_the_stale_held_entry.
    _held_of(b, "needs_reply", "h1")
    _held_of(b, "security_alert", "h2")
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    before = len(b.held.all())
    b.handle_update(cb(encode("approve_attention", digest_id=b._digest_id)))
    assert len(b.held.all()) == before - 2, "attention items not drained"
    assert "not built" not in t.edited[-1]["text"].lower()


def test_approve_attention_leaves_the_authorisation_tier_alone(bot):
    """The whole point of the two tiers. A blanket button that also approved a
    trash or a low-confidence guess would rubber-stamp exactly the set the
    partition exists to isolate."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _held_of(b, "needs_reply", "h1")
    _held_of(b, "trash", "h2", kind="trash")
    _held_of(b, "low_confidence", "h3")
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("approve_attention", digest_id=b._digest_id)))
    left = {h.hold_reason for h in b.held.all()}
    assert left == {"trash", "low_confidence"}, left


def test_approve_attention_says_what_it_would_have_done(bot):
    """Under dry-run it must never say 'done' for work that did not reach
    Gmail - the same rule the digest's block title follows."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _held_of(b, "needs_reply", "h1")
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("approve_attention", digest_id=b._digest_id)))
    text = t.edited[-1]["text"]
    assert "Would have run" in text
    assert "draft" in text


def test_approve_attention_with_nothing_to_approve_says_so(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _held_of(b, "trash", "h2", kind="trash")
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("approve_attention", digest_id=b._digest_id)))
    assert "nothing" in t.edited[-1]["text"].lower()
    assert len(b.held.all()) == 1


# --- bulk trash -------------------------------------------------------------
# Asked for after approving six trash items one at a time: "can we have both
# individual approve for trash as well as bulk approve for trash if i agree all
# belong to trash". The original objection to a blanket button was that it
# would rubber-stamp the authorisation tier. That conflated two things: a
# button that sweeps trash in WITH harmless actions, so it is approved
# unnoticed, and a button that trashes only trash after the owner has read the
# list. This is the second, and it is gated by a confirmation naming every
# sender.

def _trash_queue(b, n=3):
    b.handle_update(msg("/triage 4"))
    for i in range(n):
        _held_of(b, "trash", f"x{i}", kind="trash")
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))


def test_trash_all_asks_before_it_acts(bot):
    b, t, _ = bot
    _trash_queue(b, 3)
    before = len(b.held.all())
    b.handle_update(cb(encode("trash_all", digest_id=b._digest_id)))
    assert len(b.held.all()) == before, "acted without confirming"
    text = t.edited[-1]["text"]
    assert "3" in text
    kinds = [decode(d).kind for row in t.edited[-1]["keyboard"] for (_, d) in row]
    assert "trash_all_go" in kinds, kinds


def test_the_confirmation_names_every_sender_it_will_act_on(bot):
    """The point of the screen. A count alone is not something you can check."""
    b, t, _ = bot
    _trash_queue(b, 3)
    b.handle_update(cb(encode("trash_all", digest_id=b._digest_id)))
    text = t.edited[-1]["text"]
    for i in range(3):
        assert f"x{i}@x.com" in text, text


def test_confirming_trashes_every_one_and_drains_the_queue(bot):
    b, t, _ = bot
    _trash_queue(b, 3)
    b.handle_update(cb(encode("trash_all", digest_id=b._digest_id)))
    b.handle_update(cb(encode("trash_all_go", digest_id=b._digest_id)))
    assert [h.hold_reason for h in b.held.all()] == []
    assert "Would have run" in t.edited[-1]["text"]


def test_cancelling_the_confirmation_trashes_nothing(bot):
    """The half of a confirm screen that actually matters."""
    b, t, _ = bot
    _trash_queue(b, 3)
    before = {h.thread_id for h in b.held.all()}
    b.handle_update(cb(encode("trash_all", digest_id=b._digest_id)))
    b.handle_update(cb(encode("list", digest_id=b._digest_id)))
    assert {h.thread_id for h in b.held.all()} == before


def test_bulk_trash_never_touches_the_attention_tier(bot):
    """It is a TRASH button. An alert caught by it would be the exact
    rubber-stamping the two-tier split exists to prevent, in the other
    direction."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _held_of(b, "trash", "x1", kind="trash")
    _held_of(b, "security_alert", "h1")
    _held_of(b, "needs_reply", "h2")
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("trash_all", digest_id=b._digest_id)))
    b.handle_update(cb(encode("trash_all_go", digest_id=b._digest_id)))
    left = sorted(h.hold_reason for h in b.held.all())
    assert left == ["needs_reply", "security_alert"], left


def test_a_stale_confirmation_cannot_fire(bot):
    """The confirm screen can sit on a phone for hours. A digest_id that has
    moved on must not still authorise six threads into the bin."""
    b, t, _ = bot
    _trash_queue(b, 3)
    before = {h.thread_id for h in b.held.all()}
    b.handle_update(cb(encode("trash_all_go", digest_id="dead")))
    assert {h.thread_id for h in b.held.all()} == before


def test_no_bulk_button_when_there_is_nothing_to_bulk(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    _held_of(b, "security_alert", "h1")
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("list", digest_id=b._digest_id)))
    kinds = [decode(d).kind for row in t.edited[-1]["keyboard"] for (_, d) in row]
    assert "trash_all" not in kinds


# --- approving a no-op --------------------------------------------------------
# The Binance case, 2026-09-02. A KYC notice was classified security_alert at
# confidence 1.0 with proposed action `none` - the agent correctly declining to
# touch a security mail - and the owner approved that recommendation live.
#
# Nothing was recorded anywhere. _execute_held skipped the `none` before it
# reached the chokepoint, then held.remove() hard-deleted the item, so the one
# adjudication in the system that most deserved a record left none. The reject
# path leaves a trace because it writes a learned rule; approve-of-nothing left
# the thread indistinguishable from one never triaged at all.
#
# audit.py's docstring already promises "every attempt - permitted, refused, or
# simulated - leaves a durable record", and _dispatch has handled kind "none"
# since it was written. The chokepoint was never the problem; the caller was
# routing around it.

def _open_held_none(b):
    """A held item proposing `none`, the shape the owner actually approved."""
    b.handle_update(msg("/triage 4"))
    b.held.add(ReviewItem(thread_id="sec-1", category="security_alert",
                          subject="[Reminder] Update Your KYC Information",
                          sender="do-not-reply@ses.binance.com", snippet="s",
                          proposed=[Action(kind="none", thread_id="sec-1")],
                          reason="A security and account verification notice.",
                          confidence=1.0, source="model"),
               run_id="r1", reason="security_alert")
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))


def test_approving_a_no_op_still_leaves_an_audit_record(bot):
    """Who decided this, and when, must be answerable afterwards."""
    b, t, log = bot
    _open_held_none(b)
    before = len(log.records())
    b.handle_update(cb(encode("approve", 0, digest_id=b._digest_id)))
    new = [r for r in log.records()[before:] if r.thread_id == "sec-1"]
    assert new, "the owner's approval of a no-op left no trace at all"
    assert new[0].action == "none"
    assert new[0].actor == "human"


def test_approving_a_no_op_still_drains_the_queue(bot):
    b, t, _ = bot
    _open_held_none(b)
    b.handle_update(cb(encode("approve", 0, digest_id=b._digest_id)))
    assert b.held.all() == []


def test_approving_a_no_op_still_says_there_was_nothing_to_do(bot):
    """The record is for the log, not the screen. Reporting `none` as work done
    would be the digest's 'never say done for what did not happen' rule broken
    from the other side."""
    b, t, _ = bot
    _open_held_none(b)
    b.handle_update(cb(encode("approve", 0, digest_id=b._digest_id)))
    text = t.edited[-1]["text"]
    assert "Nothing to do" in text
    assert "none" not in text.lower().split("nothing")[0]


def test_a_no_op_record_is_never_offered_as_an_undo_candidate(bot):
    """It has no undo token, so it must not reach the undo list - approving a
    no-op is not something that can be reversed."""
    b, t, log = bot
    _open_held_none(b)
    b.handle_update(cb(encode("approve", 0, digest_id=b._digest_id)))
    assert all(r.action != "none" for r in log.undo_candidates())


# --- correcting a held item, not only approving or refusing it ---------------
# Held items are the ones the agent stopped to ask about, and were the one place
# the owner could not answer. The item view offered Approve and Not this and
# nothing else, so "this is learning, keep it in the inbox" was unsayable - and
# a bare reject taught `none`, which says what NOT to do and never what to do.
# Reported from the dev bot: "I have 2 options approve and none, there is no
# option to change it or add a rule". The teaching flow already existed; it was
# only ever wired to the done list.


def _open_held_for_correction(b, action="trash"):
    """One held item, opened. Its proposal is what the correction replaces."""
    b.handle_update(msg("/triage 4"))
    b.held.add(review_item("h0", action=action), run_id="r1", reason="trash")
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))


def _relabel_held(b, label_index=0, *, keep_inbox=True, scope_wide=False):
    """The whole correction: category, then filing, then scope."""
    b.handle_update(cb(encode("relabel", 0, label_index, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep_inbox" if keep_inbox else "file_away", 0,
                              digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_wide" if scope_wide else "scope_narrow", 0,
                              digest_id=b._digest_id)))


def test_a_held_item_can_be_corrected_not_just_approved(bot):
    """The whole point. The proposal was trash; the owner says recruiter."""
    b, t, _ = bot
    _open_held_for_correction(b)
    _relabel_held(b)
    rule = b.prefs.rules()[0]
    assert [(a.kind, (a.params or {}).get("label")) for a in rule.actions] == \
           [("label", "recruiter")], "the correction was not what got taught"


def test_correcting_a_held_item_applies_it_to_that_thread_now(bot):
    """A held thread is work in flight, not history. The owner has just said
    what the work is, so it happens - through the same chokepoint, with the
    actor recorded as human, because it was."""
    b, t, log = bot
    _open_held_for_correction(b)
    before = len(log.records())
    _relabel_held(b)
    written = log.records()[before:]
    labels = [r for r in written if r.action == "label"]
    assert labels, "the correction taught a rule but never touched the thread"
    assert labels[0].actor == "human"
    assert (labels[0].params or {}).get("label") == "recruiter"


def test_correcting_a_held_item_does_not_also_run_what_was_rejected(bot):
    """The proposal was trash and the owner said no to it. Executing the
    correction must not execute the thing the correction replaced."""
    b, t, log = bot
    _open_held_for_correction(b)
    before = len(log.records())
    _relabel_held(b)
    assert not [r for r in log.records()[before:] if r.action == "trash"]


def test_correcting_a_held_item_drains_it_from_the_queue(bot):
    """Same reason approving drains it: an item the owner has ruled on is no
    longer in flight, and leaving it asks them the same question tomorrow."""
    b, t, _ = bot
    _open_held_for_correction(b)
    assert len(b.held.all()) == 1
    _relabel_held(b)
    assert b.held.all() == []


def test_correcting_a_held_item_does_not_claim_the_run_is_over(bot):
    """The done path signs off "this run is already done", which is true there
    and false here: this thread was still waiting a moment ago and has just
    been acted on."""
    b, t, _ = bot
    _open_held_for_correction(b)
    _relabel_held(b)
    text = t.edited[-1]["text"].lower()
    assert "already done" not in text
    assert "learned" in text


def test_a_dry_run_correction_does_not_claim_it_reached_gmail(bot):
    """settings.dry_run is True in this fixture. Same rule as everywhere else:
    never the word done for something that did not happen."""
    b, t, _ = bot
    _open_held_for_correction(b)
    _relabel_held(b)
    text = t.edited[-1]["text"].lower()
    assert "would have" in text, text


def test_keeping_a_held_item_in_the_inbox_strips_the_filing(bot):
    """The proposal archives it; Keep in inbox must teach a rule that does
    not, and must not archive the thread on the way past."""
    b, t, log = bot
    _open_held_for_correction(b, action="archive")
    before = len(log.records())
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert "archive" not in [a.kind for a in rule.actions]
    assert not [r for r in log.records()[before:] if r.action == "archive"]


def test_filing_a_corrected_held_item_away_archives_it(bot):
    """The inverse answer, and it must reach the thread as well as the rule."""
    b, t, log = bot
    _open_held_for_correction(b)
    before = len(log.records())
    _relabel_held(b, keep_inbox=False)
    rule = b.prefs.rules()[0]
    assert [a.kind for a in rule.actions] == ["label", "archive"]
    assert [r.action for r in log.records()[before:]] == ["label", "archive"]


def test_correcting_a_rule_held_item_overrides_that_rule(bot):
    """A correction of what a rule proposed IS an override of that rule. The
    held branch of _open_item dropped the rule id entirely, so a rule could be
    corrected from the queue every morning and never lose precision."""
    b, t, _ = bot
    rule = Rule(id="r-old", scope="sender", pattern="h0@example.com",
                actions=[ActionTemplate(kind="trash")],
                provenance="taught earlier", created_at=NOW)
    b.prefs.add_rule(rule)
    b.handle_update(msg("/triage 4"))
    item = review_item("h0")
    item.rule_id = "r-old"
    b.held.add(item, run_id="r1", reason="trash")
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    _relabel_held(b)
    assert b._rule("r-old").override_count == 1, "the rule was never demoted"
    taught = [r for r in b.prefs.rules() if r.id != "r-old"][0]
    assert taught.supersedes == "r-old"


def test_a_bare_reject_still_teaches_that_the_action_was_wrong(bot):
    """Unchanged, and deliberately kept: a reject with no replacement is still
    signal, and requiring an edit is why skipping never taught anything. It is
    no longer the ONLY thing a held item can say."""
    b, t, log = bot
    _open_held_for_correction(b)
    before = len(log.records())
    b.handle_update(cb(encode("reject", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert rule.rejected_action == "trash"
    assert [a.kind for a in rule.actions] == ["none"]
    assert not [r for r in log.records()[before:] if r.action == "trash"]
    assert b.held.all() == []


# --- health alerts reach the phone, not a terminal --------------------------
#
# The banner already stated the OAuth countdown and the policy drift, on a
# terminal, correctly, for weeks - and every incident this project has had was
# still found by the owner noticing something rather than by the bot saying it.
# These pin the delivery, which is the part that was missing.

from datetime import timedelta

from inbox_agent.google_auth import record_consent


def _consent(bot_tuple, *, days_ago):
    b, _, _ = bot_tuple
    record_consent(b.settings.google_token,
                   now=datetime.now(timezone.utc) - timedelta(days=days_ago))


def test_a_run_warns_when_the_refresh_token_is_nearly_dead(bot):
    b, t, _ = bot
    _consent(bot, days_ago=5)
    b.handle_update(msg("/triage 4"))
    assert any("oauth consent" in m["text"] for m in t.sent), \
        "the run finished without mentioning a token that dies in two days"


def test_a_healthy_run_says_nothing_about_health(bot):
    """The digest, and only the digest. An alert on a good day is an alert the
    owner learns to swipe away."""
    b, t, _ = bot
    _consent(bot, days_ago=1)
    b.handle_update(msg("/triage 4"))
    assert not any("oauth consent" in m["text"] for m in t.sent)


def test_an_unrecorded_consent_date_does_not_nag_after_every_run(bot):
    """Permanent state for any token predating the sidecar - see health_alerts."""
    b, t, _ = bot                       # the fixture writes no consent sidecar
    b.handle_update(msg("/triage 4"))
    assert not any("oauth" in m["text"] for m in t.sent)


def test_the_digest_still_arrives_when_the_health_notice_fails(bot, monkeypatch):
    """A diagnostic that cannot be delivered must not turn a good run into a
    failed one - the same rule doctor follows about never being the thing that
    breaks."""
    b, t, _ = bot
    _consent(bot, days_ago=5)
    import inbox_agent.telegram.bot as bot_mod
    monkeypatch.setattr(bot_mod, "health_alerts",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    b.handle_update(msg("/triage 4"))
    assert any("DONE" in m["text"] for m in t.sent), "the digest was lost"


def test_status_reports_the_countdown_even_when_it_is_comfortable(bot):
    """/status is the owner asking. An answer that omits the deadline because
    it is not urgent yet is the terminal banner's failure all over again."""
    b, t, _ = bot
    _consent(bot, days_ago=1)
    b.handle_update(msg("/status"))
    assert "oauth" in t.sent[-1]["text"]


def test_status_still_answers_when_the_countdown_cannot_be_read(bot, monkeypatch):
    b, t, _ = bot
    import inbox_agent.telegram.bot as bot_mod
    monkeypatch.setattr(bot_mod, "oauth_check",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    b.handle_update(msg("/status"))
    assert "No run is waiting" in t.sent[-1]["text"]


# --- traces you can tell apart ----------------------------------------------
#
# P4's first problem is not that runs are untraced, it is that a trace nobody
# can identify answers no question. A run against the real mailbox and one
# against the frozen snapshot look identical until something says which.

def test_the_run_is_tagged_with_the_mailbox_it_touched(bot):
    b, _, _ = bot
    config = b._trace_config(limit=4)
    assert f"gmail:{b.settings.gmail}" in config["tags"]
    assert config["run_name"] == f"triage-{b.settings.gmail}"


def test_the_run_is_tagged_with_whether_it_could_write(bot):
    """dry_run is the difference between a rehearsal and the real thing."""
    b, _, _ = bot
    assert "dry_run:true" in b._trace_config(limit=4)["tags"]


def test_the_trace_carries_the_policy_that_judged(bot):
    b, _, _ = bot
    b.policy_version = "hub:abc123"
    assert b._trace_config(4)["metadata"]["policy_version"] == "hub:abc123"


def test_an_untold_policy_version_is_unknown_not_a_guess(bot):
    b, _, _ = bot                       # the fixture passes none
    assert b._trace_config(4)["metadata"]["policy_version"] == "unknown"


def test_the_trace_does_not_carry_the_owners_chat_id(bot):
    """LangSmith is an external service and the chat id identifies the owner.
    It has already been redacted once from a file in this repo."""
    b, _, _ = bot
    config = b._trace_config(limit=4)
    blob = f"{config['run_name']}{config['tags']}{config['metadata']}"
    assert b.chat_id not in blob


def test_the_trace_config_still_carries_the_checkpointer_identity(bot):
    """Losing thread_id here would silently detach every run from its
    checkpoint - the resume path and /cancel both key off it."""
    b, _, _ = bot
    assert b._trace_config(4)["configurable"]["thread_id"] == \
        b._config["configurable"]["thread_id"]


def test_a_real_run_goes_through_the_traced_config(bot):
    """The config is only worth building if invoke actually receives it."""
    b, t, _ = bot
    seen = {}
    original = b.graph.invoke

    def spy(state, config, *a, **k):
        seen.update(config)
        return original(state, config, *a, **k)

    b.graph.invoke = spy
    b.handle_update(msg("/triage 4"))
    assert seen.get("run_name", "").startswith("triage-")
    assert seen["configurable"]["thread_id"]


# --- a run says it started, not only that it finished ------------------------
#
# graph.invoke is one blocking call and a real run is minutes long: ~5s per
# thread to classify, then two silent Gmail phases at ~2s per action. On
# 2026-09-04 a 19-thread run took 247s and sent nothing at all for the first
# 235 of them, and the owner asked whether the bot was dead. It was not.

def test_a_run_acknowledges_before_it_starts_working(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    assert "Triaging" in t.sent[0]["text"], \
        "the first thing the owner saw was still the digest, minutes later"


def test_the_acknowledgement_says_how_much_was_asked_for(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 7"))
    assert "7" in t.sent[0]["text"]


def test_the_digest_still_follows_the_acknowledgement(bot):
    """The ack must be an addition, not a replacement."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    assert any("DONE" in m["text"] for m in t.sent[1:])


def test_a_failed_run_still_reports_the_failure_after_acknowledging(bot):
    """The ack must not leave a dead run looking merely slow."""
    b, t, _ = bot
    b.graph.invoke = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    b.handle_update(msg("/triage 4"))
    assert "Triaging" in t.sent[0]["text"]
    assert "Triage failed" in t.sent[-1]["text"]


def boom(*a, **kw):
    """A run that dies partway, the way an expired token makes it."""
    raise RuntimeError("connection refused")


SLOT = datetime(2026, 9, 7, 9, 0)


def test_a_scheduled_run_sends_no_pre_notice(bot):
    """The 'this takes a few minutes' line exists because the owner typed
    something and was watching. Nobody is watching a scheduled run, and a second
    unprompted ping per slot is noise."""
    b, t, _ = bot
    b.run_scheduled(SLOT)
    assert not any("takes a few minutes" in m["text"] for m in t.sent)


def test_a_scheduled_run_still_sends_the_digest(bot):
    b, t, _ = bot
    assert b.run_scheduled(SLOT) is True
    assert any("Inbox ·" in m["text"] for m in t.sent)


def test_a_typed_triage_still_announces_itself(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    assert any("takes a few minutes" in m["text"] for m in t.sent)


def test_a_failed_scheduled_run_returns_false_and_names_the_retry(
        bot, monkeypatch):
    b, t, _ = bot
    monkeypatch.setattr(b.graph, "invoke", boom)
    assert b.run_scheduled(SLOT, retry_in=timedelta(minutes=5)) is False
    assert any("Retrying in 5 minutes" in m["text"] for m in t.sent)


def test_a_failed_scheduled_run_with_no_retry_left_says_so(bot, monkeypatch):
    """The two failure messages have to differ, or a repeat reads as a stutter
    rather than as the slot being abandoned."""
    b, t, _ = bot
    monkeypatch.setattr(b.graph, "invoke", boom)
    assert b.run_scheduled(SLOT, retry_in=None) is False
    text = " ".join(m["text"] for m in t.sent)
    assert "Retrying" not in text
    assert "next scheduled run" in text


def test_a_typed_run_notifies_on_run_and_a_scheduled_one_does_not(bot):
    """A typed /triage marks the slot too - it swept the same backlog. The loop
    records scheduled runs itself, so doing it here as well would reset the
    retry count on every attempt."""
    b, t, _ = bot
    marks = []
    b.on_run = marks.append
    b.handle_update(msg("/triage 4"))
    assert len(marks) == 1
    b.run_scheduled(SLOT)
    assert len(marks) == 1


def test_on_run_is_called_even_when_the_run_raised(bot, monkeypatch):
    """Recording the attempt is what bounds the retry. If it only happened on
    success, a failing slot would be owed again 50 seconds later, forever."""
    b, t, _ = bot
    monkeypatch.setattr(b.graph, "invoke", boom)
    marks = []
    b.on_run = marks.append
    b.handle_update(msg("/triage 4"))
    assert len(marks) == 1


def test_idle_for_is_unbounded_before_any_update(bot):
    """A bot nobody has touched is not mid-review, so an owed run should not
    wait five minutes for a conversation that never started."""
    b, _t, _ = bot
    assert b.idle_for(datetime(2026, 9, 7, 9, 0)) > timedelta(days=365)


def test_handling_an_update_stamps_the_touch(bot):
    b, _t, _ = bot
    b.handle_update(msg("/status"))
    assert b.idle_for(datetime.now()) < timedelta(seconds=5)
    assert b.idle_for(datetime.now() + timedelta(minutes=5)) >= timedelta(minutes=5)


def test_an_unauthorised_update_does_not_count_as_the_owner_reviewing(bot):
    b, _t, _ = bot
    b.handle_update(msg("/status", chat_id=999))
    assert b.idle_for(datetime(2026, 9, 7, 9, 0)) > timedelta(days=365)


def test_a_stale_tap_sends_the_current_queue_instead_of_asking_for_a_command(bot):
    """With scheduled runs every digest but the newest is stale, so this stops
    being an edge case and becomes how an absent owner comes back to the phone.
    A toast is a banner that vanishes; the queue is on the screen."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    before = len(t.sent)
    b.handle_update(cb(encode("open", 1, digest_id="dead")))
    assert len(t.sent) > before                  # a new message, not a toast
    assert "waiting" in t.sent[-1]["text"]


def test_a_stale_tap_does_not_edit_the_message_it_came_from(bot):
    """Editing would silently replace what that run reported, and the owner
    scrolling back later would find a different run in its place."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    edits = len(t.edited)
    b.handle_update(cb(encode("open", 1, digest_id="dead")))
    assert len(t.edited) == edits


def test_a_stale_tap_starts_no_run(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    runs = b._runs_started
    b.handle_update(cb(encode("open", 1, digest_id="dead")))
    assert b._runs_started == runs
    assert "DONE" not in t.sent[-1]["text"]      # no run report without a run


def test_a_noop_intent_only_answers_and_sends_nothing(bot):
    """Data too old or malformed to decode has no digest to be stale relative
    to, so there is nothing to re-render."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    before = len(t.sent)
    b.handle_update(cb("a:1234"))
    assert len(t.sent) == before
    assert t.answered[-1]["text"]


from datetime import time as _time

from inbox_agent.schedule import Attempt, ScheduleStore, Trigger, manual_attempt
from inbox_agent.telegram.bot import _tick


class SilentTransport(FakeTransport):
    """get_updates returns nothing, so _tick only ever runs the schedule."""
    def get_updates(self, offset=None, timeout=50):
        return []


def always_due():
    """A slot on every hour: one is always owed, without freezing the clock."""
    return Trigger(slots=tuple(_time(h, 0) for h in range(24)))


def test_tick_runs_an_owed_slot_when_the_owner_is_quiet(bot, tmp_path):
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    store = ScheduleStore(tmp_path / "schedule.json")
    _tick(b, t, None, always_due(), store, idle=0)
    assert b._runs_started == 1
    assert store.last is not None and store.last.slot is not None


def test_tick_defers_while_the_owner_is_tapping(bot, tmp_path):
    """A scheduled digest must not pull the screen out from under a thumb."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    b._last_touch = datetime.now()               # tapped a moment ago
    store = ScheduleStore(tmp_path / "schedule.json")
    # Built relative to now, not always_due(): an hourly slot's age equals the
    # current wall-clock minute, so past :30 past the hour it is already older
    # than MAX_DEFER and the loop would correctly run it anyway - making the
    # assertion below wall-clock flaky. A slot ~1-2 minutes old is always well
    # inside MAX_DEFER, so the defer happens on any clock.
    recent = (datetime.now() - timedelta(minutes=1)).time().replace(
        second=0, microsecond=0)
    _tick(b, t, None, Trigger(slots=(recent,)), store, idle=0)
    assert b._runs_started == 0
    assert store.last is None


def test_tick_runs_anyway_once_the_defer_cap_is_past(bot, tmp_path):
    """Otherwise an owner who taps something every few minutes for an afternoon
    starves the schedule silently."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    b._last_touch = datetime.now()
    store = ScheduleStore(tmp_path / "schedule.json")
    # A slot 31 minutes old: past MAX_DEFER, still well inside grace.
    stale = (datetime.now() - timedelta(minutes=31)).time().replace(
        second=0, microsecond=0)
    _tick(b, t, None, Trigger(slots=(stale,)), store, idle=0)
    assert b._runs_started == 1


def test_tick_does_nothing_without_a_trigger_or_a_store(bot):
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    _tick(b, t, None, None, None, idle=0)
    assert b._runs_started == 0


def test_a_failed_scheduled_run_is_recorded_so_it_is_not_retried_at_once(
        bot, tmp_path, monkeypatch):
    """The whole point of recording on failure: without it the slot is owed
    again 50 seconds later, about 140 times before grace closes."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    monkeypatch.setattr(b.graph, "invoke", boom)
    store = ScheduleStore(tmp_path / "schedule.json")
    trigger = always_due()
    _tick(b, t, None, trigger, store, idle=0)
    assert store.last.failed is True and store.last.count == 1
    _tick(b, t, None, trigger, store, idle=0)
    assert b._runs_started == 1                  # backoff has not elapsed
    assert store.last.count == 1


def test_a_typed_triage_is_recorded_and_covers_the_slot(bot, tmp_path):
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    store = ScheduleStore(tmp_path / "schedule.json")
    b.on_run = lambda at: store.record(manual_attempt(at))   # what run_polling wires
    b.handle_update(msg("/triage 4"))
    assert store.last is not None and store.last.slot is None
    _tick(b, t, None, always_due(), store, idle=0)
    assert b._runs_started == 1                  # cooldown covered the slot


def test_tick_still_drains_updates(bot):
    b, _old, _ = bot

    class OneUpdate(FakeTransport):
        def get_updates(self, offset=None, timeout=50):
            return [] if offset else [dict(update_id=7, **msg("/status"))]

    t = b.transport = OneUpdate()
    assert _tick(b, t, None, None, None, idle=0) == 8


# --- pinning the retry_in arithmetic in _run_due ----------------------------
#
# None of the tests above observe retry_in or the failure message it drives -
# they only check b._runs_started and store.last.count/failed. That leaves the
# grace-vs-backoff expression unpinned: a regression to it can invert the
# comparison and every existing test still passes. These three watch the
# message run_scheduled sends, which is where retry_in becomes visible.

def test_a_scheduled_failure_names_the_retry(bot, tmp_path, monkeypatch):
    """A fresh failure, with budget and grace both to spare, must retry."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    monkeypatch.setattr(b.graph, "invoke", boom)
    store = ScheduleStore(tmp_path / "schedule.json")
    # Built relative to now, not always_due(): see the comment on
    # test_tick_defers_while_the_owner_is_tapping for why.
    recent = (datetime.now() - timedelta(minutes=1)).time().replace(
        second=0, microsecond=0)
    _tick(b, t, None, Trigger(slots=(recent,)), store, idle=0)
    assert any("Retrying in 5 minutes" in m["text"] for m in t.sent)


def test_a_second_failed_attempt_runs_but_does_not_retry(bot, tmp_path,
                                                          monkeypatch):
    """Attempt 2 of max_attempts=2 has no budget left for a further retry, but
    the attempt itself must still run - the retry budget bounds retries, not
    the run count - and the store must show it as attempt 2."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    monkeypatch.setattr(b.graph, "invoke", boom)
    store = ScheduleStore(tmp_path / "schedule.json")
    slot_time = (datetime.now() - timedelta(minutes=10)).time().replace(
        second=0, microsecond=0)
    # The instant Trigger.latest_slot will compute for slot_time today - the
    # store records a slot as a datetime instant, not a time of day, and the
    # two must match or owed() will not see this as a retry of the same slot.
    slot = datetime.combine(datetime.now().date(), slot_time)
    store.record(Attempt(at=datetime.now() - timedelta(minutes=6),  # backoff
                         slot=slot, count=1, failed=True))          # elapsed
    _tick(b, t, None, Trigger(slots=(slot_time,)), store, idle=0)
    text = " ".join(m["text"] for m in t.sent)
    assert "next scheduled run" in text
    assert "Retrying" not in text
    assert store.last.count == 2


def test_grace_outranks_backoff_in_the_retry_decision(bot, tmp_path,
                                                       monkeypatch):
    """The test that catches an inverted grace-vs-backoff comparison. A slot
    117 minutes old is still owed (grace is 120 minutes), but a retry 5
    minutes from now would land at 122 minutes - past grace - so retry_in
    must be None even though this is only attempt 1. Flip `<=` to `>=` (or
    otherwise invert the comparison) in the retry_in expression and this is
    the test that fails."""
    b, _old, _ = bot
    t = b.transport = SilentTransport()
    monkeypatch.setattr(b.graph, "invoke", boom)
    store = ScheduleStore(tmp_path / "schedule.json")
    old = (datetime.now() - timedelta(minutes=117)).time().replace(
        second=0, microsecond=0)
    _tick(b, t, None, Trigger(slots=(old,)), store, idle=0)
    text = " ".join(m["text"] for m in t.sent)
    assert "Not retrying" in text
    assert "Retrying in" not in text


from dataclasses import replace


def test_status_names_the_next_scheduled_run(bot):
    b, t, _ = bot
    b.settings = replace(b.settings, schedule=(_time(9, 0), _time(18, 0)))
    b.handle_update(msg("/status"))
    assert "scheduled" in t.sent[-1]["text"].lower()


def test_status_says_when_there_is_no_schedule(bot):
    b, t, _ = bot
    b.settings = replace(b.settings, schedule=())
    b.handle_update(msg("/status"))
    assert "no schedule" in t.sent[-1]["text"].lower()

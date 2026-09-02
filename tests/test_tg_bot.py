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
from datetime import datetime, timezone

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
    assert "/triage" in t.answered[-1]["text"]


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
    the bot's own starting state - so it must never be allowed to match."""
    b, t, _ = bot
    b.handle_update(cb(encode("open", 0)))
    assert t.sent == [] and t.edited == []


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
    assert len(t.sent) == 1
    b.handle_update(cb(encode("next", digest_id=b._digest_id)))
    b.handle_update(cb(encode("prev", digest_id=b._digest_id)))
    assert len(t.sent) == 1, "paging sent extra messages"
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
    _held_of(b, "needs_reply", "t1")
    _held_of(b, "security_alert", "t2")
    b.handle_update(msg("/triage 4"))
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
    _held_of(b, "needs_reply", "t1")
    _held_of(b, "trash", "t2", kind="trash")
    _held_of(b, "low_confidence", "t3")
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("approve_attention", digest_id=b._digest_id)))
    left = {h.hold_reason for h in b.held.all()}
    assert left == {"trash", "low_confidence"}, left


def test_approve_attention_says_what_it_would_have_done(bot):
    """Under dry-run it must never say 'done' for work that did not reach
    Gmail - the same rule the digest's block title follows."""
    b, t, _ = bot
    _held_of(b, "needs_reply", "t1")
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("approve_attention", digest_id=b._digest_id)))
    text = t.edited[-1]["text"]
    assert "Would have run" in text
    assert "draft" in text


def test_approve_attention_with_nothing_to_approve_says_so(bot):
    b, t, _ = bot
    _held_of(b, "trash", "t2", kind="trash")
    b.handle_update(msg("/triage 4"))
    _done_run(b)
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("approve_attention", digest_id=b._digest_id)))
    assert "nothing" in t.edited[-1]["text"].lower()
    assert len(b.held.all()) == 1

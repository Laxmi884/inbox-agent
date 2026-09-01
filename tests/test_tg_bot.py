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
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from inbox_agent.audit import AuditLog
from inbox_agent.classify import ThreadJudgment
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.graph import build_graph
from inbox_agent.models import Action, ReviewItem
from inbox_agent.policy import Policy
from inbox_agent.store import HeldQueue, PreferenceStore, build_store
from inbox_agent.telegram.bot import Bot
from inbox_agent.telegram.callbacks import encode


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
    graph = build_graph(client=SnapshotGmailClient(snapshot_file),
                        prefs=PreferenceStore(build_store()),
                        policy=Policy(text="P", version="local:t", source="local"),
                        llm=FakeLLM(), settings=settings, log=log, held=held,
                        checkpointer=InMemorySaver())
    t = FakeTransport()
    return Bot(transport=t, graph=graph, settings=settings, held=held,
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


def test_a_button_with_no_behaviour_yet_says_that_too(bot):
    """`open` lands in Plan 2. Until then it must not look broken."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    assert t.answered[-1]["text"], "an inert button answered with silence"


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

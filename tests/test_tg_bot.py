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

    Every processed thread gets one, so leaving it in would make the report say
    the agent did twice as much as it did.
    """
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    counts = b._view().done_by_kind
    assert counts == {"archive": 4}, counts
    assert "label" not in counts


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
    assert "failed" in t.sent[-1]["text"].lower()


# --- commands ---------------------------------------------------------------

def test_held_command_shows_the_queue_without_running_a_triage(bot):
    """The queue outlives runs, so looking at it must not produce more work."""
    b, t, _ = bot
    before = len(t.sent)
    b.handle_update(msg("/held"))
    assert len(t.sent) == before + 1
    assert b._runs_started == 0


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


# --- callbacks --------------------------------------------------------------

def test_a_callback_from_a_previous_digest_is_ignored(bot):
    """The whole point of the digest id."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    before = len(t.edited)
    b.handle_update(cb(encode("open", 0, digest_id="dead")))
    assert len(t.edited) == before


def test_a_callback_from_the_current_digest_is_honoured(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
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
    before = len(log.records())
    for junk in ("", "zzz", "a:-1", "../../x", "l:1"):
        b.handle_update(cb(junk))
    assert len(log.records()) == before, "junk callback data reached Gmail"


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

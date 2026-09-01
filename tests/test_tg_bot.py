"""The bot, tested without Telegram.

A fake transport records outbound calls and lets tests inject updates, so the
whole path - render, callback parse, index resolution, Command(resume=...),
execute - runs at the speed of the rest of the suite.
"""
import json
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from inbox_agent.audit import AuditLog
from inbox_agent.classify import ThreadJudgment
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.graph import build_graph
from inbox_agent.policy import Policy
from inbox_agent.store import PreferenceStore, build_store
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
    data = [{"id": f"t{i}", "subject": f"Sale {i}", "sender": f"deals{i}@shop.com",
             "to": [], "date": "2026-08-26T10:00:00Z", "snippet": "s", "body": "",
             "label_ids": ["INBOX"]} for i in range(4)]
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
    graph = build_graph(client=SnapshotGmailClient(snapshot_file),
                        prefs=PreferenceStore(build_store()),
                        policy=Policy(text="P", version="local:t", source="local"),
                        llm=FakeLLM(), settings=settings, log=log,
                        checkpointer=InMemorySaver())
    t = FakeTransport()
    return Bot(transport=t, graph=graph, settings=settings,
               categories=["recruiter", "promotion"]), t, log


def msg(text, chat_id=42):
    return {"message": {"chat": {"id": chat_id}, "from": {"id": chat_id}, "text": text}}


def cb(data, chat_id=42, message_id=101):
    return {"callback_query": {"id": "cb1", "data": data,
                               "from": {"id": chat_id},
                               "message": {"chat": {"id": chat_id},
                                           "message_id": message_id}}}


# --- authorisation ----------------------------------------------------------

def test_update_from_an_unauthorised_chat_is_dropped(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage", chat_id=999))
    assert t.sent == [], "the bot replied to an unauthorised chat"


def test_unauthorised_callback_is_dropped(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage"))
    t.sent.clear()
    b.handle_update(cb(encode("approve_all"), chat_id=999))
    assert t.sent == [] and t.edited == []


def test_unauthorised_attempts_are_recorded_not_silent(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage", chat_id=999))
    assert b.rejected_updates == 1


# --- the review round trip --------------------------------------------------

def test_triage_runs_to_the_gate_and_sends_a_review(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage"))
    assert len(t.sent) == 1
    assert "Inbox review" in t.sent[0]["text"]
    assert t.sent[0]["keyboard"], "no buttons offered"


def test_approve_all_resumes_the_run_and_executes(bot):
    b, t, log = bot
    b.handle_update(msg("/triage"))
    b.handle_update(cb(encode("approve_all")))
    records = log.records()
    assert records, "nothing executed"
    assert all(r.dry_run for r in records), "dry-run must stay on"
    assert len(records) == 4


def test_a_forged_index_executes_nothing_extra(bot):
    """The structural guarantee: an index outside the batch names no thread."""
    b, t, log = bot
    b.handle_update(msg("/triage"))
    b.handle_update(cb(encode("reject", 99)))     # out of range
    b.handle_update(cb(encode("approve_all")))
    assert len(log.records()) == 4                # all four, none extra, none skipped


def test_rejecting_one_item_then_approving_executes_three(bot):
    b, t, log = bot
    b.handle_update(msg("/triage"))
    b.handle_update(cb(encode("reject", 1)))
    b.handle_update(cb(encode("approve_all")))
    assert len(log.records()) == 3


def test_replaying_a_callback_after_resume_does_not_double_execute(bot):
    b, t, log = bot
    b.handle_update(msg("/triage"))
    b.handle_update(cb(encode("approve_all")))
    n = len(log.records())
    b.handle_update(cb(encode("approve_all")))    # replayed
    assert len(log.records()) == n, "the run was resumed twice"


def test_hostile_callback_data_is_answered_and_ignored(bot):
    b, t, log = bot
    b.handle_update(msg("/triage"))
    for junk in ("", "zzz", "a:-1", "../../x", "l:1"):
        b.handle_update(cb(junk))
    assert log.records() == []


# --- commands ---------------------------------------------------------------

def test_status_reports_a_parked_run(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage"))
    t.sent.clear()
    b.handle_update(msg("/status"))
    assert "wait" in t.sent[-1]["text"].lower() or "review" in t.sent[-1]["text"].lower()


def test_cancel_abandons_the_parked_run(bot):
    b, t, log = bot
    b.handle_update(msg("/triage"))
    b.handle_update(msg("/cancel"))
    b.handle_update(cb(encode("approve_all")))
    assert log.records() == [], "a cancelled run still executed"


def test_paged_mode_edits_one_message_instead_of_sending_many(bot, tmp_path,
                                                              snapshot_file):
    b, t, _ = bot
    b.mode = "paged"
    b.handle_update(msg("/triage"))
    assert len(t.sent) == 1
    b.handle_update(cb(encode("next")))
    b.handle_update(cb(encode("next")))
    assert len(t.sent) == 1, "paged mode sent extra messages"
    assert len(t.edited) == 2, "paged mode did not edit in place"


def test_tapping_an_item_number_opens_it_in_paged_view(bot):
    """The digest was read-only: approve-all or nothing."""
    b, t, _ = bot
    b.handle_update(msg("/triage"))
    assert "1." in t.sent[0]["text"]              # digest listing
    b.handle_update(cb(encode("open", 2)))
    assert t.edited, "opening an item did not update the message"
    assert "3/4" in t.edited[-1]["text"], "not showing item 3 of 4"


def test_list_button_returns_to_the_digest(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage"))
    b.handle_update(cb(encode("open", 1)))
    b.handle_update(cb(encode("list")))
    assert "1." in t.edited[-1]["text"] and "2." in t.edited[-1]["text"]


def test_deciding_an_item_in_paged_view_advances_to_the_next(bot):
    """Reviewing is a flow; stopping on the item you just handled feels stuck."""
    b, t, _ = bot
    b.handle_update(msg("/triage"))
    b.handle_update(cb(encode("open", 0)))
    b.handle_update(cb(encode("reject", 0)))
    assert "2/4" in t.edited[-1]["text"]


def test_a_verdict_given_in_paged_view_survives_to_the_resume(bot):
    b, t, log = bot
    b.handle_update(msg("/triage"))
    b.handle_update(cb(encode("open", 1)))
    b.handle_update(cb(encode("reject", 1)))
    b.handle_update(cb(encode("approve_all")))
    assert len(log.records()) == 3, "the rejection was lost between views"

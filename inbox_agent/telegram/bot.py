"""The bot: polling loop, authorisation, and update dispatch (spec section 4).

Long polling, not webhook. The obvious reading is convenience - no public
endpoint, no TLS, runs on a laptop. The better reason is that it shrinks the
trust boundary: a resume payload is untrusted input crossing a persistence
boundary, and a webhook would make that boundary reachable by anyone who learns
the URL. Polling means the only route to the resume path is to be Telegram,
holding the bot token, and to pass the chat-id check below.

The graph is not touched by any of this. That was the claim in spec section 7 of
the Stage A design - the interrupt payload is UI-agnostic - and this module is
the test of it.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from langgraph.types import Command

from ..config import Settings
from ..models import ReviewRequest
from ..store import HeldQueue
from .callbacks import DIGEST_ID_LEN, Intent, decode, to_response
from .render_tg import DigestView, digest

log = logging.getLogger("inbox_agent.telegram")


class Bot:
    """Owns the graph, the checkpointer and the conversation with one human."""

    def __init__(self, *, transport, graph, settings: Settings,
                 held: HeldQueue, categories: Sequence[str] = (),
                 mode: Optional[str] = None):
        self.transport = transport
        self.graph = graph
        self.settings = settings
        # The queue the graph fills. Injected rather than built here so both
        # sides are looking at the same one - two instances over two stores
        # would let the bot show an empty queue while the graph filled another.
        self.held = held
        self.categories = list(categories)
        self.mode = (mode or settings.tg_mode or "digest").lower()
        self.chat_id = str(settings.tg_chat_id)

        # Per-digest UI state. Deliberately NOT the source of truth for what is
        # outstanding - that is the queue, which outlives every run.
        self._message_id: Optional[int] = None
        self._page = 0
        self._digest_id = ""
        # Counts runs actually started, so a test can assert that /held ran none.
        self._runs_started = 0
        self._last_run: Optional[dict] = None   # the graph result for the digest
        self._intents: dict[int, Intent] = {}
        self._run = 0
        self.rejected_updates = 0

    # --- identity -----------------------------------------------------------

    @property
    def _config(self) -> dict:
        # One LangGraph thread per triage run. A new run gets a new id so a
        # replayed callback against a finished run cannot resume it.
        return {"configurable": {"thread_id": f"tg-{self.chat_id}-{self._run}"}}

    def _authorised(self, update: dict) -> bool:
        """Exactly one Telegram user may drive this bot.

        Without this the bot is open to anyone who finds it, and "anyone" would
        be able to approve actions against the owner's mailbox. Checked before
        anything else touches the update.
        """
        payload = update.get("message") or update.get("callback_query") or {}
        sender = str((payload.get("from") or {}).get("id", ""))
        if sender and sender == self.chat_id:
            return True
        self.rejected_updates += 1
        log.warning("dropped update from unauthorised id %r", sender)
        return False

    # --- state --------------------------------------------------------------

    def _request(self) -> Optional[ReviewRequest]:
        """The review payload as persisted in the checkpoint.

        Read from graph state rather than remembered in this object: the whole
        point of resolving callbacks by index is that the index is resolved
        against what the human was ACTUALLY shown. Keeping a second copy here
        would reintroduce the drift the index scheme exists to prevent.
        """
        try:
            snap = self.graph.get_state(self._config)
        except Exception:
            return None
        raw = (snap.values or {}).get("review")
        if not raw:
            return None
        # Only parked runs are reviewable. Once `review` has been consumed and
        # the run has finished, `next` is empty and a replayed callback must
        # not resume anything.
        if not snap.next:
            return None
        return ReviewRequest.model_validate(raw)

    # --- rendering ----------------------------------------------------------

    def _new_digest_id(self) -> str:
        return uuid.uuid4().hex[:DIGEST_ID_LEN]

    def _view(self) -> DigestView:
        """Assemble what the digest renders: this run's work plus the queue.

        The done counts come from the audit records the run wrote, not from
        graph state: the audit log is the durable record of what actually
        reached Gmail, and it is the same source the undo path will read.
        """
        result = self._last_run or {}
        done_by_kind: dict[str, int] = {}
        rule_decided = 0
        for record in result.get("executed", []):
            if record.get("action") == "label" and \
                    record.get("params", {}).get("label") == self.settings.triaged_label:
                continue  # bookkeeping, not work the owner cares about
            done_by_kind[record["action"]] = done_by_kind.get(record["action"], 0) + 1
            if str(record.get("actor", "")).startswith("rule:"):
                rule_decided += 1
        return DigestView(
            run_at=datetime.now(timezone.utc),
            total=len(result.get("thread_ids", [])),
            done_by_kind=done_by_kind,
            rule_decided=rule_decided,
            held=self.held.all(),
            digest_id=self._digest_id,
        )

    def _show(self, *, edit: bool) -> None:
        text, keyboard = digest(self._view(), self._page)
        if edit and self._message_id is not None:
            self.transport.edit_message(self.chat_id, self._message_id, text, keyboard)
        else:
            sent = self.transport.send_message(self.chat_id, text, keyboard)
            self._message_id = (sent or {}).get("message_id")

    # --- dispatch -----------------------------------------------------------

    def handle_update(self, update: dict) -> None:
        if not self._authorised(update):
            return
        if "callback_query" in update:
            self._on_callback(update["callback_query"])
        elif "message" in update:
            self._on_message(update["message"])

    def _on_message(self, message: dict) -> None:
        text = (message.get("text") or "").strip()
        command, _, arg = text.partition(" ")
        # Telegram appends @botname when a command is sent in a group, or when
        # the client disambiguates. Strip it so /triage@my_bot 6 still parses.
        command = command.split("@", 1)[0]

        if command == "/triage":
            limit = int(arg) if arg.strip().isdigit() else self.settings.snapshot_size
            self._start(limit)
        elif command == "/held":
            # Shows the queue without running anything: the queue outlives runs,
            # so looking at it must not require producing more work.
            self._page = 0
            self._message_id = None
            self._digest_id = self._new_digest_id()
            self._show(edit=False)
        elif command == "/status":
            self._status()
        elif command == "/cancel":
            self._cancel()
        else:
            self.transport.send_message(
                self.chat_id,
                "Commands: /triage [n] · /held · /status · /cancel")

    def _start(self, limit: int) -> None:
        self._run += 1
        self._runs_started += 1
        self._page = 0
        self._intents = {}
        self._message_id = None
        self._digest_id = self._new_digest_id()

        log.info("triage start: limit=%s run=%s", limit, self._run)
        started = time.monotonic()
        # mode="incremental": /triage ACTS. The confident, reversible majority
        # is executed and only what genuinely needs the owner goes to the queue,
        # which is what this whole design is for. It ran in backlog mode while
        # the bot rendered from a parked checkpoint - that transitional hack is
        # what the queue and the digest replace.
        try:
            self._last_run = self.graph.invoke(
                {"limit": limit, "mode": "incremental"}, self._config)
        except Exception as exc:
            # Silence is indistinguishable from an empty inbox, which is a
            # failure the owner would trust for days without noticing. Say so.
            log.exception("triage failed")
            self.transport.send_message(
                self.chat_id, f"Triage failed: {type(exc).__name__}. "
                              f"Nothing was executed. /triage to retry.")
            return
        elapsed = time.monotonic() - started
        log.info("triage done in %.1fs: %s executed, %s held", elapsed,
                 len(self._last_run.get("executed", [])), len(self.held.all()))
        self._show(edit=False)

    def _status(self) -> None:
        request = self._request()
        if request is None:
            self.transport.send_message(self.chat_id, "No run is waiting. /triage to start.")
        else:
            self.transport.send_message(
                self.chat_id,
                f"A run is waiting for review: {len(request.items)} threads, "
                f"policy {request.policy_version}.")

    def _cancel(self) -> None:
        """Abandon the parked run by moving to a fresh graph thread id.

        The checkpoint is left intact rather than deleted - an abandoned run is
        still part of the record, the same reasoning as PreferenceStore keeping
        overridden rules instead of removing them.
        """
        self._run += 1
        self._page = 0
        self._intents = {}
        self._message_id = None
        # Drop the digest id too, so the message still on the owner's screen
        # stops being tappable. Cancelling the conversation has to cancel the
        # buttons it drew, or a stale tap re-renders a run that was abandoned.
        self._digest_id = ""
        log.info("run cancelled; moved to run=%s", self._run)
        self.transport.send_message(self.chat_id, "Cancelled. Nothing was executed.")

    def _on_callback(self, query: dict) -> None:
        intent = decode(query.get("data", ""))
        self.transport.answer_callback(query.get("id", ""))
        if intent.kind == "noop":
            return
        if not self._digest_id or intent.digest_id != self._digest_id:
            # A tap on a superseded digest. Positions have shifted since that
            # message was drawn, so acting on it would act on the wrong thread.
            #
            # The empty-id check is not redundant: decode() reports an id-less
            # callback as digest_id="", which is also this object's state before
            # the first digest and after /cancel. Comparing alone would let ""
            # match "" and make an id-less callback valid in exactly the two
            # moments when no digest exists.
            log.info("ignored a callback from digest %r (current %r)",
                     intent.digest_id, self._digest_id)
            return

        if intent.kind in ("next", "prev"):
            self._page = max(0, self._page + (1 if intent.kind == "next" else -1))
            self._show(edit=True)
            return
        if intent.kind in ("open", "done", "approve_attention"):
            # Plan 2 gives these their real behaviour. Re-rendering keeps the
            # message live rather than silently doing nothing.
            self._show(edit=True)
            return

    # --- the interrupt path -------------------------------------------------
    # Unreached from /triage, which no longer parks. Kept whole for Plan 3's
    # /backlog sweep - the one job that genuinely has to be previewed before it
    # commits - and covered meanwhile at the graph level in tests/test_graph.py.

    def _resume(self, request: ReviewRequest) -> None:
        response = to_response(request, self._intents, self.categories)
        final = self.graph.invoke(
            Command(resume=response.model_dump(mode="json")), self._config)

        log.info("resuming with %s explicit verdict(s)", len(self._intents))
        executed = len(final.get("executed", []))
        refused = len(final.get("refused", []))
        skipped = len(final.get("skipped", []))
        learned = len(final.get("learned", []))

        lines = [f"Done. {executed} action{'s' if executed != 1 else ''} executed"
                 + (" (dry-run)" if self.settings.dry_run else "") + "."]
        if learned:
            lines.append(f"Learned {learned} new rule{'s' if learned != 1 else ''}.")
        if refused:
            lines.append(f"{refused} refused by the deny-list.")
        if skipped:
            # Surfaced, never swallowed: this is where forged or stale ids land.
            lines.append(f"{skipped} skipped (not part of the reviewed batch).")
        self.transport.send_message(self.chat_id, "\n".join(lines))

        log.info("run complete: %s executed, %s refused, %s skipped, %s learned",
                 executed, refused, skipped, learned)
        self._intents = {}
        self._message_id = None


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

class TelegramError(RuntimeError):
    """A Bot API failure, carrying the API's own description and NOT the URL.

    The URL contains the bot token. httpx.HTTPStatusError puts the URL in its
    message, so letting that propagate writes a live credential into every
    traceback and log line.
    """

    def __init__(self, method: str, status: int, description: str):
        self.method, self.status, self.description = method, status, description
        super().__init__(f"{method} failed [{status}]: {description}")


class HttpTransport:
    """Raw Bot API over httpx. No telegram library.

    Roughly a hundred lines against a well-documented HTTP API, versus a
    dependency whose surface we would have to trust in the one component that
    holds a credential with authority over a mailbox.
    """

    def __init__(self, token: str, timeout: float = 65.0):
        import httpx
        self._token = token
        self._base = f"https://api.telegram.org/bot{token}"
        self._client = httpx.Client(timeout=timeout)
        # httpx logs the full request URL at INFO, and the token is IN the URL.
        # Left alone this writes the credential into the log on every single
        # poll. Nothing here needs httpx's request log.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    @staticmethod
    def _clean(payload: dict) -> dict:
        """Drop None values.

        Telegram rejects an explicit null where it expects an object:
        {"reply_markup": null} answers `400 object expected as reply markup`.
        The key has to be ABSENT, not null - which is exactly the difference a
        fake transport does not model, and why this went unnoticed until the
        first real message.
        """
        return {k: v for k, v in payload.items() if v is not None}

    @staticmethod
    def _redact(text: str, token: str) -> str:
        return text.replace(token, "<token>") if token else text

    def _post(self, method: str, **payload) -> dict:
        try:
            r = self._client.post(f"{self._base}/{method}",
                                  json=self._clean(payload))
        except Exception as exc:
            raise TelegramError(method, 0, self._redact(str(exc), self._token)) from None
        if r.status_code >= 400:
            try:
                description = r.json().get("description", r.text[:200])
            except Exception:
                description = r.text[:200]
            # `from None` so the httpx exception - which carries the tokenised
            # URL - never appears in the chained traceback either.
            raise TelegramError(method, r.status_code,
                                self._redact(str(description), self._token)) from None
        return r.json().get("result", {})

    @staticmethod
    def _markup(keyboard) -> Optional[dict]:
        if not keyboard:
            return None
        return {"inline_keyboard": [
            [{"text": t, "callback_data": d} for (t, d) in row] for row in keyboard]}

    def send_message(self, chat_id, text, keyboard=None) -> dict:
        return self._post("sendMessage", chat_id=chat_id, text=text,
                          reply_markup=self._markup(keyboard))

    def edit_message(self, chat_id, message_id, text, keyboard=None) -> dict:
        return self._post("editMessageText", chat_id=chat_id, message_id=message_id,
                          text=text, reply_markup=self._markup(keyboard))

    def answer_callback(self, callback_id, text="") -> dict:
        return self._post("answerCallbackQuery", callback_query_id=callback_id, text=text)

    def get_updates(self, offset: Optional[int] = None, timeout: int = 50) -> list[dict]:
        return self._post("getUpdates", offset=offset, timeout=timeout) or []


def run_polling(bot: Bot, transport: HttpTransport, *, idle: float = 1.0) -> None:
    """Long-poll forever. One update at a time, in order."""
    offset = None
    log.info("polling as chat %s in %s mode", bot.chat_id, bot.mode)
    while True:
        try:
            updates = transport.get_updates(offset=offset)
        except Exception as exc:            # network blips must not kill the bot
            log.warning("getUpdates failed: %s", exc)
            time.sleep(idle * 5)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            try:
                bot.handle_update(update)
            except Exception:
                # One bad update must not take down a long-lived process that a
                # parked run depends on. The checkpoint survives; log and go on.
                log.exception("handler failed for update %s", update.get("update_id"))
        if not updates:
            time.sleep(idle)

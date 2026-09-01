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
from typing import Optional, Sequence

from langgraph.types import Command

from ..config import Settings
from ..models import ReviewRequest
from .callbacks import Intent, decode, to_response
from .render_tg import digest, paged

log = logging.getLogger("inbox_agent.telegram")


class Bot:
    """Owns the graph, the checkpointer and the conversation with one human."""

    def __init__(self, *, transport, graph, settings: Settings,
                 categories: Sequence[str] = (), mode: Optional[str] = None):
        self.transport = transport
        self.graph = graph
        self.settings = settings
        self.categories = list(categories)
        self.mode = (mode or settings.tg_mode or "digest").lower()
        self.chat_id = str(settings.tg_chat_id)

        # Per-review UI state. Deliberately NOT the source of truth for what was
        # proposed - that is read back from the checkpoint (see _request).
        self._message_id: Optional[int] = None
        self._page = 0            # digest page, or item index in paged view
        self._view = self.mode    # "digest" | "paged"; `open` switches at runtime
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

    def _render(self, request: ReviewRequest) -> tuple[str, list]:
        if self._view == "paged":
            return paged(request, self._page, self.categories)
        return digest(request, self._page, self.categories)

    def _show(self, request: ReviewRequest, *, edit: bool) -> None:
        text, keyboard = self._render(request)
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
        elif command == "/status":
            self._status()
        elif command == "/cancel":
            self._cancel()
        else:
            self.transport.send_message(
                self.chat_id,
                "Commands: /triage [n] · /status · /cancel")

    def _start(self, limit: int) -> None:
        self._run += 1
        self._page = 0
        self._view = self.mode
        self._intents = {}
        self._message_id = None

        log.info("triage start: limit=%s run=%s", limit, self._run)
        started = time.monotonic()
        # mode="backlog": the bot's whole review flow (digest/paged rendering,
        # approve_all, per-item reject/edit) is built on the graph parking at
        # the interrupt and being resumed later. The auto-execute/held-queue
        # split (task 8) is not wired into this UI yet, so /triage must still
        # get everything in front of the human rather than have some of it
        # silently act and vanish before _request() ever reads the checkpoint.
        self.graph.invoke({"limit": limit, "mode": "backlog"}, self._config)
        elapsed = time.monotonic() - started

        request = self._request()
        if request is None:
            log.warning("triage produced nothing to review (%.1fs)", elapsed)
            self.transport.send_message(self.chat_id, "Nothing to review.")
            return
        n = len(request.items)
        log.info("triage done: %s threads in %.1fs (%.2fs/thread), %s rule-decided",
                 n, elapsed, elapsed / max(n, 1),
                 sum(1 for i in request.items if i.source == "rule"))
        self._show(request, edit=False)

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
        self._view = self.mode
        self._intents = {}
        self._message_id = None
        log.info("run cancelled; moved to run=%s", self._run)
        self.transport.send_message(self.chat_id, "Cancelled. Nothing was executed.")

    def _on_callback(self, query: dict) -> None:
        intent = decode(query.get("data", ""))
        self.transport.answer_callback(query.get("id", ""))

        request = self._request()
        if request is None:
            # Finished, cancelled, or never started. A replayed callback lands
            # here and must do nothing at all.
            return

        if intent.kind == "noop":
            return

        if intent.kind == "open":
            if intent.index is not None:
                self._view = "paged"
                self._page = intent.index
                self._show(request, edit=True)
            return

        if intent.kind == "list":
            self._view = "digest"
            self._page = 0
            self._show(request, edit=True)
            return

        if intent.kind in ("next", "prev"):
            step = 1 if intent.kind == "next" else -1
            self._page = max(0, self._page + step)
            self._show(request, edit=True)
            return

        if intent.kind in ("approve", "reject", "label"):
            if intent.index is not None:
                self._intents[intent.index] = intent
                log.info("verdict: item %s -> %s", intent.index, intent.kind)
                # After deciding one item, advance - reviewing is a flow, and
                # stopping on the item you just handled makes it feel stuck.
                if self._view == "paged" and intent.index < len(request.items) - 1:
                    self._page = intent.index + 1
            self._show(request, edit=True)
            return

        if intent.kind == "approve_all":
            self._resume(request)

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

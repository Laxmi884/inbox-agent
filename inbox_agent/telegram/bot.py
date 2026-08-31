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
        self._page = 0
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
        if self.mode == "paged":
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
        self._intents = {}
        self._message_id = None
        self.graph.invoke({"limit": limit}, self._config)
        request = self._request()
        if request is None:
            self.transport.send_message(self.chat_id, "Nothing to review.")
            return
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
        self._intents = {}
        self._message_id = None
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

        if intent.kind in ("next", "prev"):
            step = 1 if intent.kind == "next" else -1
            self._page = max(0, self._page + step)
            self._show(request, edit=True)
            return

        if intent.kind in ("approve", "reject", "label"):
            if intent.index is not None:
                self._intents[intent.index] = intent
            self._show(request, edit=True)
            return

        if intent.kind == "approve_all":
            self._resume(request)

    def _resume(self, request: ReviewRequest) -> None:
        response = to_response(request, self._intents, self.categories)
        final = self.graph.invoke(
            Command(resume=response.model_dump(mode="json")), self._config)

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

        self._intents = {}
        self._message_id = None


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

class HttpTransport:
    """Raw Bot API over httpx. No telegram library.

    Roughly a hundred lines against a well-documented HTTP API, versus a
    dependency whose surface we would have to trust in the one component that
    holds a credential with authority over a mailbox.
    """

    def __init__(self, token: str, timeout: float = 65.0):
        import httpx
        self._base = f"https://api.telegram.org/bot{token}"
        self._client = httpx.Client(timeout=timeout)

    def _post(self, method: str, **payload) -> dict:
        r = self._client.post(f"{self._base}/{method}", json=payload)
        r.raise_for_status()
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

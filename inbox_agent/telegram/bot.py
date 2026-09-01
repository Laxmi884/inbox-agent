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
from .render_tg import DigestView, DoneItem, digest, done_panel

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
        # Whether the CURRENT message reports a run. False for a /held digest,
        # and it has to be state rather than an argument to _show: paging that
        # message re-renders it, and the DONE block must not reappear on page 2
        # of a message that never claimed a run in the first place.
        self._run_report = True
        # Which screen the message is currently showing: the digest, or the
        # done panel behind its button. State rather than an argument because
        # paging re-renders whatever is on screen, and page 2 of the panel must
        # not come back as page 2 of the queue.
        self._panel = "digest"
        # The panel's own page, so opening the panel and coming back does not
        # move the owner to page one of a queue they were part way through.
        self._done_page = 0
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

    def _view(self, *, run_report: bool = True) -> DigestView:
        """Assemble what the digest renders: this run's work plus the queue.

        The done counts come from the audit records the run wrote, not from
        graph state: the audit log is the durable record of what actually
        reached Gmail, and it is the same source the undo path will read.

        `run_report=False` is /held, which ran nothing. `_last_run` still holds
        the LAST run's result, and reporting it would stamp 08:00's counts with
        the current clock - and again on every later /held. So /held reports no
        run at all rather than someone else's.
        """
        result = self._last_run if run_report else None
        done_by_kind: dict[str, int] = {}
        rule_decided = 0
        for record in (result or {}).get("executed", []):
            # .get() throughout: a malformed record must not raise here. This
            # runs AFTER the graph executed, so an exception costs the owner the
            # digest for work that already reached Gmail - the one moment a
            # crash is most expensive and least recoverable.
            kind = record.get("action")
            if not kind:
                continue
            if kind == "label" and (record.get("params") or {}).get(
                    "label") == self.settings.triaged_label:
                continue  # bookkeeping, not work the owner cares about
            done_by_kind[kind] = done_by_kind.get(kind, 0) + 1
            if str(record.get("actor", "")).startswith("rule:"):
                rule_decided += 1
        return DigestView(
            run_at=datetime.now(timezone.utc),
            total=len((result or {}).get("thread_ids", [])),
            done_by_kind=done_by_kind,
            rule_decided=rule_decided,
            held=self.held.all(),
            digest_id=self._digest_id,
            # Dry-run actions were audited but never reached Gmail. The banner
            # that says so is on the terminal; the digest is on the phone.
            dry_run=bool(self.settings.dry_run),
            run_report=run_report,
        )

    def _done_items(self) -> list[DoneItem]:
        """What the run did, per thread, for the panel behind the button.

        Two sources, deliberately. The audit records say what actually went
        through the chokepoint - the only honest answer to "what did you do" -
        but they know a thread by id, which is not something the owner can read.
        The proposals in `auto` carry the subject and the sender. A record whose
        thread is missing from `auto` still gets a row, named by its id: an
        action with no visible subject is strange, and hiding it would be worse.

        .get() throughout, like _view, and for the same reason: this runs after
        the graph executed, so an exception here costs the owner the report for
        work that already happened.
        """
        result = self._last_run or {}
        known: dict[str, dict] = {}
        for raw in result.get("auto", []):
            if isinstance(raw, dict) and raw.get("thread_id"):
                known[raw["thread_id"]] = raw

        rows: dict[str, DoneItem] = {}
        for record in result.get("executed", []):
            kind = record.get("action")
            thread_id = record.get("thread_id")
            if not kind or not thread_id:
                continue
            label = (record.get("params") or {}).get("label")
            if kind == "label" and label == self.settings.triaged_label:
                continue    # bookkeeping on every thread, not work to report
            item = rows.get(thread_id)
            if item is None:
                proposal = known.get(thread_id, {})
                item = DoneItem(thread_id=thread_id,
                                subject=proposal.get("subject") or thread_id,
                                sender=proposal.get("sender") or "")
                rows[thread_id] = item
            item.actions.append((kind, label))
            if str(record.get("actor", "")).startswith("rule:"):
                item.from_rule = True
        return list(rows.values())

    def _show(self, *, edit: bool) -> None:
        view = self._view(run_report=self._run_report)
        if self._panel == "done":
            view.done = self._done_items()
            text, keyboard = done_panel(view, self._done_page)
        else:
            text, keyboard = digest(view, self._page)
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
            # so looking at it must not require producing more work. And with no
            # run, no run report - see _view's run_report.
            self._page = 0
            self._message_id = None
            self._digest_id = self._new_digest_id()
            self._run_report = False
            self._panel = "digest"
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
        self._run_report = True
        self._panel = "digest"
        self._done_page = 0

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
            # NOT "nothing was executed" - the same falsehood /cancel used to
            # tell. The run can raise anywhere, including after auto_execute
            # has already pushed actions through the chokepoint, so the honest
            # claim is that it did not finish. /held shows what survived.
            self.transport.send_message(
                self.chat_id, f"Triage failed: {type(exc).__name__}. "
                              f"The run did not finish; some actions may already "
                              f"have run. /held to see the queue, /triage to retry.")
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
        # NOT "nothing was executed": /triage is incremental now and has already
        # acted by the time this can be typed. Saying otherwise would tell the
        # owner their mail is untouched when the agent has archived half of it.
        # What cancelling actually does is retire the buttons, so that is what
        # it says. /cancel's real subject - abandoning a parked run - is Plan 3.
        self.transport.send_message(
            self.chat_id, "Cancelled. Buttons on the last digest are no longer active.")

    def _on_callback(self, query: dict) -> None:
        """Every path answers the callback, and says something when it refuses.

        Telegram spins the button until answerCallbackQuery arrives, and an
        empty answer clears the spinner without saying anything. Both refusals
        below - a stale digest, and a button whose behaviour is not built yet -
        used to be silent, which is indistinguishable from a broken bot: the
        owner taps again, and again, and then asks what the button is for.
        """
        intent = decode(query.get("data", ""))
        answer = query.get("id", "")
        if intent.kind == "noop":
            self.transport.answer_callback(
                answer, "That button came from an older message.")
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
            self.transport.answer_callback(
                answer, "That digest is out of date - send /triage or /held "
                        "for a current one.")
            return

        self.transport.answer_callback(answer)

        if intent.kind in ("next", "prev"):
            step = 1 if intent.kind == "next" else -1
            if self._panel == "done":
                self._done_page = max(0, self._done_page + step)
            else:
                self._page = max(0, self._page + step)
            self._show(edit=True)
            return
        if intent.kind == "done":
            # The report half of act-then-report. The counts say a label
            # happened; this says which one, which is the part a correction
            # would be about.
            self._panel = "done"
            self._done_page = 0
            self._show(edit=True)
            return
        if intent.kind == "list":
            self._panel = "digest"
            self._show(edit=True)
            return
        if intent.kind in ("open", "approve_attention"):
            # Plan 2 gives these their real behaviour. Say so rather than
            # re-rendering an unchanged message, which Telegram rejects as
            # unmodified and which therefore looks like nothing at all.
            self.transport.answer_callback(
                answer, "Not built yet - opening an item and approving the "
                        "attention tier land in the next step.")
            return

    # --- the interrupt path -------------------------------------------------
    #
    # WARNING TO WHOEVER WIRES /backlog: _resume IS NOT READY TO BE CALLED.
    #
    # It is unreached today - /triage no longer parks - and it is not intact.
    # The half that collected verdicts went with the interrupt UI: the
    # approve/reject/label callback branch that wrote into `self._intents` was
    # deleted, because no digest button emits those kinds any more. `_intents`
    # now has three writers, every one of them `= {}`, and a single reader here.
    #
    # to_response() defaults every thread the intents do not name to "approve".
    # So calling _resume as it stands resumes with an empty mapping, which is a
    # blanket approval of the entire parked batch with no route to reject a
    # single item - on /backlog, a 500-thread historical sweep, which is the
    # precise outcome previewing that sweep exists to prevent.
    #
    # Plan 3 must rebuild verdict collection (a detail view, its buttons, and
    # the callback branch that records them) BEFORE connecting anything to this.
    # Restoring the deleted branch now would be untriggerable code pinned by a
    # synthetic test, against a contract Plan 3 is going to redesign anyway.
    #
    # Execution of a resume payload is covered meanwhile at the graph level in
    # tests/test_graph.py; nothing covers this method.

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

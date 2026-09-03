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

from ..audit import AuditLog, ExecutionContext, ForbiddenActionError, execute_action
from ..config import Settings
from ..models import ActionTemplate, ReviewItem, ReviewRequest, Rule, Thread
from ..store import HeldQueue, PreferenceStore, rule_from_correction
from .callbacks import DIGEST_ID_LEN, Intent, decode, encode, to_response
from .render_tg import (ATTENTION_REASONS, DigestView, DoneItem,
                        confirm_trash_all, digest, done_panel, item_view)

log = logging.getLogger("inbox_agent.telegram")


def _short(text: str, cap: int = 60) -> str:
    """Cap a subject for a confirmation line, saying so when it is cut.

    Same rule as the renderer: a hard slice ends mid-word and reads as a
    corrupted message rather than as a subject that continues.
    """
    flat = " ".join((text or "").split())
    return flat if len(flat) <= cap else flat[:cap - 1].rstrip() + "…"


class Bot:
    """Owns the graph, the checkpointer and the conversation with one human."""

    def __init__(self, *, transport, graph, settings: Settings,
                 held: HeldQueue, prefs: PreferenceStore, client=None,
                 log: Optional[AuditLog] = None,
                 categories: Sequence[str] = (), mode: Optional[str] = None):
        self.transport = transport
        self.graph = graph
        self.settings = settings
        # The queue the graph fills. Injected rather than built here so both
        # sides are looking at the same one - two instances over two stores
        # would let the bot show an empty queue while the graph filled another.
        self.held = held
        # The same store the graph reads rules from. A correction is a store
        # write, not a message through graph state, which is what makes the
        # learning loop independent of whether any run is parked.
        self.prefs = prefs
        # Needed to act on a held item the owner approves. The action still goes
        # through execute_action - the one chokepoint, with its deny-list, its
        # dry-run skip and its audit record - so this adds a caller, not a
        # second route to Gmail.
        self.client = client
        self.log = log
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
        # Which item is open, and the verdict waiting on a scope answer. The
        # verdict is held rather than applied because writing on the first tap
        # would pick a blast radius the owner never chose.
        self._open_index = 0
        self._pending: Optional[dict] = None
        # Which list the item was opened from, so Back returns there rather
        # than to whichever screen happens to be default.
        self._panel_before_item = "digest"
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
                                sender=proposal.get("sender") or "",
                                snippet=proposal.get("snippet") or "")
                rows[thread_id] = item
            item.actions.append((kind, label))
            actor = str(record.get("actor", ""))
            if actor.startswith("rule:"):
                item.from_rule = True
                item.rule_id = actor.split(":", 1)[1]
                # In the rule's own terms rather than the audit sentence: the
                # rule may have been corrected since, and what the owner needs
                # to judge is what it says NOW.
                rule = self._rule(item.rule_id)
                item.rule_note = (f"{rule.scope} {rule.pattern} → {rule.summary}"
                                  if rule else "")
        return list(rows.values())

    def _show(self, *, edit: bool) -> None:
        view = self._view(run_report=self._run_report)
        if self._panel == "item":
            text, keyboard = self._item_screen()
        elif self._panel == "done":
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

    def _open_item(self) -> Optional[tuple[str, DoneItem]]:
        """The item the owner tapped, and which list it came from.

        Resolved against the list that was rendered, never from the callback -
        the callback carries a position precisely so a thread id cannot travel
        in it.
        """
        if self._panel_before_item == "done":
            items = self._done_items()
            if 0 <= self._open_index < len(items):
                return "done", items[self._open_index]
            return None
        queue = sorted(self.held.all(), key=lambda h: h.first_held_at)
        if 0 <= self._open_index < len(queue):
            held = queue[self._open_index]
            return "held", DoneItem(
                thread_id=held.thread_id, subject=held.item.subject,
                sender=held.item.sender,
                actions=[(a.kind, (a.params or {}).get("label"))
                         for a in held.item.proposed])
        return None

    def _item_screen(self) -> tuple[str, list]:
        opened = self._open_item()
        if opened is None:
            # The list moved under the callback. Falling back to the list is
            # the honest answer; guessing at a neighbouring item is not.
            self._panel = self._panel_before_item
            return digest(self._view(run_report=self._run_report), self._page)
        kind, item = opened
        why = ""
        for raw in (self._last_run or {}).get("auto", []):
            if isinstance(raw, dict) and raw.get("thread_id") == item.thread_id:
                why = raw.get("reason") or ""
        actions_text = ", ".join(f"{k}({v})" if v else k for k, v in item.actions)
        return item_view(item.subject, item.sender, actions_text, why,
                         digest_id=self._digest_id, index=self._open_index,
                         kind=kind, rule_detail=self._rule_detail(item),
                         categories=self.categories)

    def _ask_scope(self, verdict: str, item: DoneItem, category: str) -> None:
        """Verdict first, scope second.

        "Never archive this sender" and "never archive any valuable newsletter"
        are different instructions behind the same tap, and only the owner knows
        which was meant. Asking costs one tap; guessing costs a rule that
        reaches mail they never meant to include.
        """
        text = (f"{item.subject}\n\nTeach this for…")
        keyboard = [
            [(f"Just {item.sender[:28]}",
              encode("scope_narrow", self._open_index, digest_id=self._digest_id))],
            [(f"Every {category}",
              encode("scope_wide", self._open_index, digest_id=self._digest_id))],
            [("↩ Back", encode("list", digest_id=self._digest_id))],
        ]
        self.transport.edit_message(self.chat_id, self._message_id, text, keyboard)

    def _ask_filing(self, item: DoneItem) -> None:
        """Category first, filing second - asked unconditionally on relabel.

        A `learning` item stays in the inbox per policy, but the archive the
        run produced was a consequence of the category being wrong, not an
        independent decision to preserve. Asking every time is uniform,
        predictable, and needs no reading of the policy to get right; the two
        answers are exact inverses of each other so a reader can predict
        either from the other.
        """
        text = f"{item.subject}\n\nKeep it in the inbox, or file it away?"
        keyboard = [
            [("Keep in inbox",
              encode("keep_inbox", self._open_index, digest_id=self._digest_id))],
            [("File it away",
              encode("file_away", self._open_index, digest_id=self._digest_id))],
            [("↩ Back", encode("list", digest_id=self._digest_id))],
        ]
        self.transport.edit_message(self.chat_id, self._message_id, text, keyboard)

    def _set_filing(self, *, file_away: bool) -> None:
        """Apply the filing answer to the pending actions, then ask scope.

        Both branches start by stripping every archive AND trash - not just
        archive - because a done item's actions are not guaranteed to be
        the label/archive shape relabel usually sees. A fired teach_trash
        rule's done item is a bare trash(), and render_tg offers "Label
        as..." on any done item with no gate on what it did, so relabel can
        reach a trash action here. Filtering only archive would let that
        trash survive file_away untouched and ride along into the taught
        rule - "keep this, file it under X" would still re-trash future
        mail, the opposite of what was asked. Keep stops there; file adds
        back exactly one archive. That makes the two genuinely exact
        inverses of the same starting point, not just of each other's name.
        """
        pending = self._pending
        if pending is None:
            return
        opened = self._open_item()
        if opened is None:
            self._panel = self._panel_before_item
            self._show(edit=True)
            return
        _kind, item = opened
        actions = [a for a in pending["actions"] if a.kind not in ("archive", "trash")]
        if file_away:
            actions = actions + [ActionTemplate(kind="archive")]
        pending["actions"] = actions
        self._ask_scope(pending["verdict"], item, pending["category"])

    def _held_verdict(self, intent) -> None:
        """Approve or refuse one held item, and drain it from the queue.

        Both drain. The queue is work in flight, and an item the owner has
        ruled on is no longer in flight: leaving it would ask them to authorise
        the same trash tomorrow morning, and the morning after that.

        Approve executes through execute_action - the same chokepoint, the same
        deny-list, the same dry-run skip, the same audit record - with the actor
        recorded as `human`, because it was. Not this executes nothing and
        teaches instead: a bare reject is signal, and the digest design counts
        every reject as a candidate rule. Requiring an edit is exactly why
        skipping something never taught this agent anything.
        """
        queue = sorted(self.held.all(), key=lambda h: h.first_held_at)
        index = intent.index or 0
        if not (0 <= index < len(queue)):
            self._panel = self._panel_before_item
            self._show(edit=True)
            return
        item = queue[index]

        if intent.kind == "approve":
            done = self._execute_held(item)
            # Never the word "done" for something that did not reach Gmail -
            # the same rule the digest's block title follows.
            verb = "Would have run" if self.settings.dry_run else "Ran"
            summary = f"{verb}: {done}." if done else "Nothing to do."
        else:
            proposed = item.item.proposed[0].kind if item.item.proposed else "none"
            rule = rule_from_correction(
                self._thread_for(DoneItem(thread_id=item.thread_id,
                                          subject=item.item.subject,
                                          sender=item.item.sender)),
                [ActionTemplate(kind="none")],
                f"owner refused {proposed} on {item.item.subject[:50]!r}",
                rejected=proposed, supersedes=item.item.rule_id)
            self.prefs.add_rule(rule)
            summary = (f"Left alone. Learned: {rule.scope} {rule.pattern} "
                       f"→ {rule.summary}.")

        self.held.remove(item.thread_id)
        self._panel = self._panel_before_item
        remaining = len(self.held.all())
        self.transport.edit_message(
            self.chat_id, self._message_id,
            f"{_short(item.item.subject)}\n{summary}\n\n{remaining} left waiting.",
            [[("↩ Back to the digest", encode("list", digest_id=self._digest_id))]])

    def _approve_attention(self) -> None:
        """Approve every held item in the attention tier, in one tap.

        Reported from a phone as "the approve button is not working". It was not
        broken - it was unbuilt, and answered with a Telegram toast, which on a
        phone is a banner that vanishes. A button that looks dead and a button
        that is dead are the same button to the person pressing it.

        The objection recorded against building it was that a blanket approve
        could rubber-stamp a trash or a low-confidence guess - exactly the set
        the two-tier partition exists to isolate. That objection is already
        answered on the render side: render_tg only offers this button over
        items whose hold_reason is in ATTENTION_REASONS. This filters by the
        same constant rather than trusting the caller, because a callback id is
        attacker-reachable in principle and the deny-list is not the only thing
        worth enforcing twice.

        Everything goes through _execute_held, so it is the same chokepoint, the
        same deny-list, the same dry-run skip and the same audit record as
        approving items one at a time - with the actor recorded as `human`,
        because it was.
        """
        queue = sorted(self.held.all(), key=lambda h: h.first_held_at)
        attention = [h for h in queue if h.hold_reason in ATTENTION_REASONS]
        if not attention:
            self._panel = self._panel_before_item or self._panel
            self.transport.edit_message(
                self.chat_id, self._message_id,
                "Nothing in the attention tier to approve.\n\n"
                f"{len(queue)} still waiting, all of them needing a decision "
                "one at a time.",
                [[("↩ Back to the digest",
                   encode("list", digest_id=self._digest_id))]])
            return

        did: list[str] = []
        for item in attention:
            done = self._execute_held(item)
            self.held.remove(item.thread_id)
            if done:
                did.append(f"{_short(item.item.subject)} → {done}")

        # Never the word "done" for something that did not reach Gmail - the
        # same rule the digest's block title follows, and the same reason: the
        # banner saying dry_run is on is on a terminal, and this is on a phone.
        verb = "Would have run" if self.settings.dry_run else "Ran"
        remaining = len(self.held.all())
        lines = [f"Approved {len(attention)}.", ""]
        lines += [f"{verb}: {line}" for line in did] or ["Nothing to do."]
        lines += ["", f"{remaining} left waiting."]
        self.transport.edit_message(
            self.chat_id, self._message_id, "\n".join(lines),
            [[("↩ Back to the digest",
               encode("list", digest_id=self._digest_id))]])

    def _trash_held(self) -> list:
        """Every held item proposed for trash, oldest first.

        One definition, used by the offer, the confirmation and the execution,
        so the count on the button, the list on the confirm screen and the
        threads actually trashed cannot disagree.
        """
        return [h for h in sorted(self.held.all(), key=lambda x: x.first_held_at)
                if h.hold_reason == "trash"]

    def _offer_trash_all(self) -> None:
        """Show what would go, and ask. Nothing is executed on this path."""
        items = self._trash_held()
        if not items:
            self._show(edit=True)
            return
        text, keyboard = confirm_trash_all(
            items, digest_id=self._digest_id, dry_run=self.settings.dry_run)
        self.transport.edit_message(self.chat_id, self._message_id, text, keyboard)

    def _trash_all(self) -> None:
        """The confirmed bulk trash.

        Re-reads the queue rather than trusting anything carried on the
        callback: the confirm screen can sit on a phone for hours, and the set
        that is trashed must be the set that is held NOW, not the set that was
        held when the button was drawn.

        Trash only - an alert or a reply caught here would be exactly the
        rubber-stamping the two-tier split exists to prevent, in the other
        direction.
        """
        items = self._trash_held()
        if not items:
            self._show(edit=True)
            return
        did = []
        for item in items:
            done = self._execute_held(item)
            self.held.remove(item.thread_id)
            if done:
                did.append(f"{_short(item.item.subject)} → {done}")
        verb = "Would have run" if self.settings.dry_run else "Ran"
        remaining = len(self.held.all())
        lines = [f"Trashed {len(items)}.", ""]
        lines += [f"{verb}: {line}" for line in did] or ["Nothing to do."]
        lines += ["", f"{remaining} left waiting."]
        self.transport.edit_message(
            self.chat_id, self._message_id, "\n".join(lines),
            [[("↩ Back to the digest",
               encode("list", digest_id=self._digest_id))]])

    def _execute_held(self, item) -> str:
        """Push one held item's proposed actions through the chokepoint.

        Returns what it did, in the digest's vocabulary, for the confirmation.
        A refusal is reported rather than raised: the deny-list saying no is an
        answer the owner needs to see, not a crash.
        """
        if self.client is None or self.log is None:
            return ""
        context = ExecutionContext(policy_version=None, model=None, backend=None)
        did = []
        for action in item.item.proposed:
            # `none` goes through the chokepoint like everything else. It used
            # to be skipped here, before execute_action ever saw it, which
            # silently exempted the one decision most worth recording: the
            # owner approving the agent's recommendation to leave a
            # security_alert alone. That approval landed in no log, and
            # held.remove() then destroyed the queue entry, leaving the thread
            # indistinguishable from one nobody had ever looked at. audit.py
            # promises a durable record of every attempt and _dispatch has
            # always handled "none"; only this caller broke the promise.
            try:
                execute_action(action, client=self.client, settings=self.settings,
                               log=self.log, actor="human", context=context)
            except ForbiddenActionError as exc:
                did.append(f"refused {action.kind} ({exc})")
                continue
            # Recorded, but never reported as work done. An empty summary is
            # what the caller turns into "Nothing to do.", and saying anything
            # else would break the same rule the digest follows: never claim
            # something happened when nothing did.
            if action.kind == "none":
                continue
            label = (action.params or {}).get("label")
            did.append(f"{action.kind}({label})" if label else action.kind)
        return ", ".join(did)

    def _verdict(self, intent) -> None:
        """Turn a tapped verdict into the action sequence it stands for.

        Each verdict maps to exactly one sequence, so what gets taught is what
        the button said - the reason the vocabulary is buttons and not free
        text. Nothing is written here: the scope question comes first, except
        for trash, which is sender-only by design.
        """
        opened = self._open_item()
        if opened is None:
            self._panel = self._panel_before_item
            self._show(edit=True)
            return
        _kind, item = opened
        category = self._category_of(item.thread_id)

        if intent.kind == "relabel" and intent.label_index is None:
            # First tap: which label? Second tap arrives as relabel with one.
            rows, row = [], []
            for i, name in enumerate(self.categories):
                row.append((name, encode("relabel", self._open_index, i,
                                         digest_id=self._digest_id)))
                if len(row) == 3:
                    rows.append(row); row = []
            if row:
                rows.append(row)
            rows.append([("↩ Back", encode("list", digest_id=self._digest_id))])
            self.transport.edit_message(self.chat_id, self._message_id,
                                        f"{item.subject}\n\nLabel it as…", rows)
            return

        if intent.kind == "keep":
            # Keep the label, drop everything that removes it from the inbox.
            # Built from what actually happened rather than from a template, so
            # a thread that was only labelled teaches only a label.
            actions = [ActionTemplate(kind=k, params={"label": v} if v else {})
                       for k, v in item.actions if k not in ("archive", "trash")]
            if not actions:
                actions = [ActionTemplate(kind="label", params={"label": category})]
        elif intent.kind == "relabel":
            chosen = self.categories[intent.label_index] \
                if 0 <= (intent.label_index or 0) < len(self.categories) else category
            # One label(chosen), always, then everything else the run did
            # that isn't a label, in order. NOT built by substituting into an
            # existing label action: 6 of 10 threads in the run that found
            # this bug were bare `archive, unlabel(UNREAD)` with no label
            # action to substitute into, so the substitution silently
            # dropped the correction. unlabel is not a label action - only
            # kind == "label" is - so unlabel(UNREAD) passes through here
            # untouched rather than being mistaken for the thing to replace.
            actions = [ActionTemplate(kind="label", params={"label": chosen})] + \
                      [ActionTemplate(kind=k, params={"label": v} if v else {})
                       for k, v in item.actions if k != "label"]
            category = chosen
        else:
            actions = [ActionTemplate(kind="trash")]

        self._pending = {"verdict": intent.kind, "actions": actions,
                         "category": category, "rule_id": self._rule_id_of(item)}
        if intent.kind == "teach_trash":
            # Sender-only, and not asked: a category-wide trash rule would
            # auto-execute trash across a whole class of future mail on one tap
            # (Plan 1's partition decision), which is a blast radius no single
            # correction should be able to reach.
            self._teach(wide=False)
            return
        if intent.kind == "relabel":
            # Asked every time, unconditionally: the filing the run chose may
            # have been a consequence of the wrong category, and there is no
            # policy-reading shortcut that is both simple and predictable.
            self._ask_filing(item)
            return
        self._ask_scope(intent.kind, item, category)

    def _category_of(self, thread_id: str) -> str:
        for raw in (self._last_run or {}).get("auto", []):
            if isinstance(raw, dict) and raw.get("thread_id") == thread_id:
                return raw.get("category") or "other"
        held = self.held.get(thread_id)
        return (held.item.category if held else "other") or "other"

    def _rule(self, rule_id: str):
        """One rule by id, or None. Absent is not an error: a rule can be
        demoted, replaced or lost between acting and being asked about, and the
        record of what happened has to survive its own rule."""
        return {r.id: r for r in self.prefs.rules()}.get(rule_id)

    def _rule_detail(self, item: DoneItem) -> str:
        """What decided this, when it was taught, and how it has done since.

        The owner opening a rule-decided item is being asked to judge the rule,
        not just this thread, so precision belongs here: a rule they have
        overridden twice out of three firings is one they should be replacing.
        """
        rule = self._rule(item.rule_id) if item.rule_id else None
        if rule is None:
            return ""
        taught = rule.created_at.astimezone().strftime("%-d %b")
        hits = f"{rule.hit_count} hit" + ("" if rule.hit_count == 1 else "s")
        overs = (f"{rule.override_count} override"
                 + ("" if rule.override_count == 1 else "s"))
        return (f"Rule: {rule.scope} {rule.pattern} → {rule.summary}\n"
                f"Taught {taught} · {hits}, {overs}")

    def _rule_id_of(self, item: DoneItem) -> Optional[str]:
        for record in (self._last_run or {}).get("executed", []):
            if record.get("thread_id") == item.thread_id:
                actor = str(record.get("actor", ""))
                if actor.startswith("rule:"):
                    return actor.split(":", 1)[1]
        return None

    def _teach(self, *, wide: bool) -> None:
        """Write the rule the pending verdict describes, and say what it says.

        The confirmation names the rule in the digest's own vocabulary, because
        a rule the owner cannot read is one they cannot correct. It never
        mentions undo: nothing here reverses anything, and under dry-run there
        was nothing to reverse in the first place.
        """
        pending, self._pending = self._pending, None
        if pending is None:
            return
        opened = self._open_item()
        if opened is None:
            self._panel = self._panel_before_item
            self._show(edit=True)
            return
        _kind, item = opened
        category = pending["category"]
        actions = pending["actions"]

        # Guarded exactly as record_override is below, and for the same reason:
        # only a rule can be superseded, so a correction of the model's own
        # judgement must not name one.
        superseded = pending.get("rule_id") if item.from_rule else None

        if wide:
            rule = Rule(id=f"r-{uuid.uuid4().hex[:8]}", scope="category",
                        pattern=category, actions=actions,
                        provenance=f"owner corrected {item.subject[:60]!r}",
                        created_at=datetime.now(timezone.utc),
                        supersedes=superseded)
            self.prefs.add_rule(rule)
            reach = f"every {category}"
        else:
            thread = self._thread_for(item)
            rule = rule_from_correction(
                thread, actions, f"owner corrected {item.subject[:60]!r}",
                supersedes=superseded)
            self.prefs.add_rule(rule)
            reach = f"{rule.scope} {rule.pattern}"

        # A correction of what a rule proposed IS an override of that rule.
        # This is the caller record_override has been waiting for since it was
        # written; without it precision never moves and a bad rule is never
        # demoted, however often it is corrected.
        if superseded:
            self.prefs.record_override(superseded)

        extra = (" I will do that without asking again, because a rule you "
                 "taught is your own instruction." if pending["verdict"] == "teach_trash"
                 else "")
        self._panel = self._panel_before_item
        self.transport.edit_message(
            self.chat_id, self._message_id,
            f"Learned: {reach} → {rule.summary}.{extra}\n\n"
            f"That is for next time; this run is already done.",
            [[("↩ Back to the digest", encode("list", digest_id=self._digest_id))]])

    def _thread_for(self, item: DoneItem) -> Thread:
        """The Thread choose_scope picks a scope from.

        Built from what the item already carries rather than re-fetched: the
        only fields choose_scope reads are sender and subject, and both are on
        screen in front of the owner at the moment they press the button. The
        rest of a Thread is not part of the decision.
        """
        return Thread(id=item.thread_id, subject=item.subject, sender=item.sender,
                      to=[], date="", snippet="", body="", label_ids=[])

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

    def _ack(self, callback_id: str, text: str = "") -> None:
        """Clear the spinner. Never let failing to do so cost the tap.

        answerCallbackQuery is cosmetic - it stops Telegram spinning the button
        and optionally shows a toast. The query id expires in seconds, so any
        tap that queued while the bot was down comes back as
        "query is too old and response timeout expired or query ID is invalid",
        and this used to be called BEFORE the work, unguarded: the 400 aborted
        the handler and the action was silently lost. Seen live as eleven
        tracebacks and nothing acted on.

        Swallowed at info, not warning: an expired ack is the normal
        consequence of a restart, not a fault to investigate.
        """
        try:
            self.transport.answer_callback(callback_id, text)
        except Exception as exc:
            log.info("could not acknowledge callback %s: %s", callback_id, exc)

    def _on_callback(self, query: dict) -> None:
        """Every path answers the callback, and says something when it refuses.

        Telegram spins the button until answerCallbackQuery arrives, and an
        empty answer clears the spinner without saying anything. A stale digest
        used to refuse silently, which is indistinguishable from a broken bot:
        the owner taps again, and again, and then asks what the button is for.

        approve_attention was the other case and is no longer one - it acts now.
        Its toast was reported from a phone as "the approve button is not
        working", which is the lesson: on a phone a toast is a banner that
        vanishes, so anything worth telling the owner belongs on the screen.
        """
        intent = decode(query.get("data", ""))
        answer = query.get("id", "")
        if intent.kind == "noop":
            self._ack(answer, "That button came from an older message.")
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
            self._ack(answer, "That digest is out of date - send /triage or "
                             "/held for a current one.")
            return

        self._ack(answer)

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
        if intent.kind == "open":
            # The screen the numbered buttons have always implied.
            self._panel_before_item = self._panel
            self._panel = "item"
            self._open_index = intent.index or 0
            self._show(edit=True)
            return
        if intent.kind in ("approve", "reject"):
            self._held_verdict(intent)
            return
        if intent.kind in ("keep", "relabel", "teach_trash"):
            self._verdict(intent)
            return
        if intent.kind in ("keep_inbox", "file_away"):
            self._set_filing(file_away=intent.kind == "file_away")
            return
        if intent.kind in ("scope_narrow", "scope_wide"):
            self._teach(wide=intent.kind == "scope_wide")
            return
        if intent.kind in ("approve_attention",):
            self._approve_attention()
            return
        if intent.kind == "trash_all":
            self._offer_trash_all()
            return
        if intent.kind == "trash_all_go":
            self._trash_all()
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

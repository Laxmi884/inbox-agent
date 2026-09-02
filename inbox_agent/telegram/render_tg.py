"""Telegram renderers (spec section 4.4). Pure functions over their view object.

Two renderers, doing different jobs. `digest` is the report the owner reads
after a run: what the agent did, and what it held back. `paged` is the
single-item view - it renders a ReviewRequest, and Plan 3's /backlog sweep is
what puts it back in front of a person.

They no longer share an input type, and that is the point. The digest stopped
reporting one run the moment the queue started outliving runs, so it renders a
DigestView instead: this run's counts alongside everything still waiting from
any run.

Nothing here knows about the graph, the transport, or LangGraph. Same contract
as render.py, which is why the notebook renderer keeps working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

from ..models import HeldItem, ReviewRequest
from ..render import LOW_CONFIDENCE, NO_REASON
from .callbacks import encode

# Telegram hard limits. Protocol, not preference.
TG_MAX_TEXT = 4096

# A phone screen, roughly. Beyond this the queue is scrolling, not scanning.
HELD_PAGE_SIZE = 8

# Same as the queue above it. Done entries were packed two to a line and 12 to a
# page on the theory that a list to scan tolerates more density than a queue to
# work through; on a phone that theory produced a wall, because Telegram wraps
# every one of those lines. Same block shape, same page size, one layout to
# learn.
DONE_PAGE_SIZE = HELD_PAGE_SIZE

# Section order is hold-reason precedence order, so the most consequential
# things are nearest the top of the message where they are read first.
SECTIONS: tuple[tuple[str, str], ...] = (
    # "one at a time" rather than just "needs your OK": the one-tap button
    # below covers ATTENTION_REASONS only, and an owner holding a digest with
    # two alerts and six trash proposals read "Approve 2" as the button being
    # broken. The heading is where that is answered while the items are being
    # read; the footnote under the button answers it again while deciding.
    ("trash", "🗑 TRASH — approve one at a time"),
    ("needs_reply", "✉️ NEEDS REPLY"),
    ("security_alert", "🔒 SECURITY"),
    ("low_confidence", "❓ NOT SURE"),
)

# Held for attention rather than authorisation: the proposed action is harmless
# and the thread simply wants a person. Only these are one-tap approvable.
ATTENTION_REASONS = frozenset({"needs_reply", "security_alert"})


def _attention_phrase(items) -> str:
    """"2 alerts", "1 reply & 2 alerts" - what is ACTUALLY in the tier.

    The button used to read "N replies & alerts" whatever it contained, so a
    queue of two security alerts and no replies was described as both. Read on
    a phone as: "it says approve 2 replies & alerts but i only see security
    alerts". A button that describes a queue the owner can see is not there is
    a button they cannot trust, and this one authorises action on a mailbox.

    Shared by the button and the footnote above it so the two cannot drift into
    naming the same set differently one line apart.
    """
    replies = sum(1 for h in items if h.hold_reason == "needs_reply")
    alerts = sum(1 for h in items if h.hold_reason == "security_alert")
    parts = []
    if replies:
        parts.append(f"{replies} repl" + ("y" if replies == 1 else "ies"))
    if alerts:
        parts.append(f"{alerts} alert" + ("" if alerts == 1 else "s"))
    return " & ".join(parts)

_SUBJECT_CAP = 70
# Capped like the others. Uncapped it was the one unbounded field in the message
# and the only route to the 4096-char wall, where the budget below would start
# dropping items off the bottom of a page that has room for them.
_SENDER_CAP = 60
_REASON_CAP = 160

# Room reserved per section heading when budgeting: the blank line before it,
# plus the widest count it can carry - " (nnn of nnn)". Reserved rather than
# measured because the count is not known until the budget has decided what
# fits, and over-reserving costs at most one item on a page that is already at
# the 4096 wall.
_SECTION_HEADER_SLACK = 16

# Printed when a page's items do not all fit. Reserved for up front, so the
# note itself cannot be the thing that overflows.
_OVERFLOW = "… %d more did not fit in one message"


@dataclass
class DoneItem:
    """One thread the run acted on, and what it did to it.

    `actions` is (kind, label) pairs rather than pre-rendered text so the panel
    can both print "label(recruiter)" and count how many went to `recruiter`
    without parsing a string it just formatted.

    Built from the audit records - what actually went through the chokepoint -
    joined to the proposals for a subject and a sender, because an audit record
    knows a thread by id and the owner does not.
    """
    thread_id: str
    subject: str
    sender: str
    actions: list[tuple[str, Optional[str]]] = field(default_factory=list)
    # The digest's header counts these; here they are marked individually, so
    # "the rules you taught me" can be checked rather than taken on trust.
    # Gmail's own preview of the thread, so the report says what the mail WAS
    # and not only what happened to it. The snippet rather than a model-written
    # summary on purpose: it is already on ReviewItem and already fetched, so it
    # costs no tokens, no latency, and moves no figure in the model registry.
    # It is also the literal opening of the mail, so it cannot be hallucinated.
    snippet: str = ""
    from_rule: bool = False
    # WHICH rule, in the rule's own terms: "sender no-reply@x.com → trash".
    # `from_rule` alone said a rule decided it and refused to say which, and for
    # a category rule the pattern appears nowhere else on the screen.
    rule_note: str = ""
    rule_id: str = ""


@dataclass
class DigestView:
    """Everything the digest renders, assembled by the caller.

    A dataclass rather than a ReviewRequest because the digest no longer reports
    one run: it reports what this run DID (audit records) alongside what is
    still waiting from any run (the queue). Keeping the renderer pure over this
    view is what kept render.py's contract worth having.
    """
    run_at: datetime
    total: int
    done_by_kind: dict[str, int]
    rule_decided: int
    held: list[HeldItem] = field(default_factory=list)
    digest_id: str = ""
    # Under dry-run the actions were audited but never reached Gmail. The old
    # digest listed proposals, so it could not mislead; this one reports work as
    # DONE, and the startup banner that says dry_run is on does not reach the
    # phone. "Done" for something that did not happen is the single most
    # expensive thing this message could get wrong.
    dry_run: bool = False
    # False for /held, which shows the queue without having run anything. With
    # it true the previous run's counts get stamped with the current clock -
    # "16:00 · 22 threads · DONE (18)" for work done at 08:00, again on every
    # later /held. A report of a run that did not happen.
    run_report: bool = True
    # What the run did, expanded. The digest carries the counts; this carries
    # the list behind the "Show the N done" button, which said a label happened
    # and could not say which one.
    done: list[DoneItem] = field(default_factory=list)


def rule_decided_count(request: ReviewRequest) -> int:
    return sum(1 for i in request.items if i.source == "rule")


def header(request: ReviewRequest) -> str:
    """Make the learning visible.

    `ReviewItem.source` has existed since Stage A and has never been shown to
    anyone. The loop's whole value proposition is that review gets shorter and
    more citable as the owner corrects it; today that improvement is invisible
    to the person doing the reviewing. As rules accumulate this number climbs
    and the list below it shrinks.

    Suppressed at zero: "0 decided by rules you taught me" reads as a failure
    rather than as a not-yet.
    """
    n = len(request.items)
    line = f"Inbox review · {n} thread{'s' if n != 1 else ''}"
    decided = rule_decided_count(request)
    if decided:
        line += f" · {decided} decided by rules you taught me"
    return line


def _flag(item) -> str:
    return " !" if item.confidence < LOW_CONFIDENCE else ""


def _actions(item) -> str:
    return ", ".join(
        f"{a.kind}({a.params.get('label')})" if a.params.get("label") else a.kind
        for a in item.proposed) or "none"


def _age(item: HeldItem, now: datetime) -> str:
    """'waiting since Tue 8:00', or empty for something held in this run.

    Only carried-over items say it. Printing it on everything would make the
    phrase meaningless, and its whole job is to make an ignored queue look
    ignored.
    """
    held_at = item.first_held_at
    if held_at.tzinfo is None:
        held_at = held_at.replace(tzinfo=timezone.utc)
    if (now - held_at).total_seconds() < 3600:
        return ""
    return f" · waiting since {held_at.astimezone().strftime('%a %H:%M')}"


def _oneline(text: str, cap: int) -> str:
    """Collapse to a single line, then cap.

    `reason` is model-generated and models routinely emit newlines; a single one
    breaks the three-line grid the whole layout depends on, and .strip() only
    trims the ends. Subject and sender get the same treatment - a folded header
    can carry a newline too - so the grid is guaranteed rather than usual.

    A cut field ends in an ellipsis. Marketing subjects routinely run past the
    70-character cap, and a hard slice ends them mid-word, which reads as a
    corrupted message rather than as a subject that continues. The ellipsis is
    counted inside the cap, so every field stays exactly as wide as the budget
    below was told it would be.
    """
    flat = " ".join(text.split())
    if len(flat) <= cap:
        return flat
    return flat[:cap - 1].rstrip() + "…"


def _held_line(number: int, item: HeldItem, now: datetime) -> str:
    """Three lines: what it is, who sent it, and why the agent wants this.

    `reason` is here because these are the items the agent deliberately would
    not decide alone - it is what turns a rejection into training data rather
    than a shrug. The one-line form that was right for a fifty-item list is
    wrong for a list of four.
    """
    conf = (f" · {item.item.confidence:.2f}"
            if item.item.confidence < LOW_CONFIDENCE else "")
    why = _oneline(item.item.reason or NO_REASON, _REASON_CAP) or NO_REASON
    return (f"{number}. {_oneline(item.item.subject, _SUBJECT_CAP)}\n"
            f"{_oneline(item.item.sender, _SENDER_CAP)}{conf}{_age(item, now)}\n"
            f"{why}")


def digest(view: DigestView, page: int = 0) -> tuple[str, list]:
    """Counts on top, held items in full, done items behind a button.

    Grouped by why an item is held rather than by what the agent proposes: the
    sections ARE the queue, and the stat line is the report.
    """
    # Carried-over items first within each section: an ignored queue should read
    # as one. all() is already oldest-first, so this is stable.
    ordered = sorted(view.held, key=lambda h: h.first_held_at)
    pages = max(1, (len(ordered) + HELD_PAGE_SIZE - 1) // HELD_PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    start = page * HELD_PAGE_SIZE
    window = ordered[start:start + HELD_PAGE_SIZE]

    # Absolute numbering, computed once over the whole queue: item 9 is item 9
    # on page 2, so a number the owner reads means the same thing on every page.
    numbers = {h.thread_id: n for n, h in enumerate(ordered, start=1)}

    clock = view.run_at.astimezone().strftime("%H:%M")
    if view.run_report:
        stat = " · ".join(f"{count} {kind}"
                          for kind, count in sorted(view.done_by_kind.items())
                          if count)
        head = [f"Inbox · {clock} · {view.total} threads",
                f"{stat} · {len(ordered)} waiting" if stat
                else f"{len(ordered)} waiting"]
    else:
        # /held. No run happened, so there is no run to report - not even a
        # thread count, which would be a count of nothing dressed as a result.
        head = [f"Held · {clock}", f"{len(ordered)} waiting"]
    if pages > 1:
        head[0] += f"  (page {page + 1}/{pages})"

    done_total = sum(view.done_by_kind.values())
    tail: list[str] = []
    if view.run_report:
        # Never the word "done" for something that did not reach Gmail.
        done_title = "✓ WOULD HAVE DONE" if view.dry_run else "✓ DONE"
        tail = ["", f"{done_title} ({done_total})"]
        if view.rule_decided:
            # Make the learning visible. As rules accumulate this climbs and the
            # sections above shrink. Suppressed at zero: "0 came from rules you
            # taught me" reads as a failure rather than as a not-yet.
            tail.append(f"{view.rule_decided} came from rules you taught me")

    # Lay the window out in section order before budgeting, so what falls off
    # the cap is whole items from the end rather than half of one.
    laid_out = [(title, h) for reason, title in SECTIONS
                for h in window if h.hold_reason == reason]

    # Budget the item blocks instead of slicing the finished message. The DONE
    # block is appended last, so a slice at the cap would drop the entire report
    # while leaving its "Show the N done" button on the keyboard; and a slice
    # lands mid-line, leaving half an item on screen under a numbered button
    # that still claims to open it. Dropping whole items keeps the text and the
    # keyboard describing the same list, which is what every callback index
    # assumes. With every field capped this is unreachable in practice - it is
    # the guarantee, not the common path.
    budget = TG_MAX_TEXT - len("\n".join(head + tail)) - len(_OVERFLOW % 999) - 2
    used = 0
    shown: list[HeldItem] = []
    charged: set[str] = set()
    for title, held_item in laid_out:
        cost = len(_held_line(numbers[held_item.thread_id], held_item,
                              view.run_at)) + 1
        if title not in charged:
            cost += len(title) + _SECTION_HEADER_SLACK
        if used + cost > budget:
            break
        used += cost
        charged.add(title)
        shown.append(held_item)

    # Counted over the whole queue, not over `shown`. A section spans pages
    # whenever hold reasons interleave in arrival order, which is the normal
    # case; counting only what fits told the reader of page one that five things
    # were waiting to be trashed when eight were, and then repeated the same
    # heading on page two with a different number. The queue size in the header
    # was right the whole time, which is what made the section counts read as
    # authoritative rather than as a subtotal.
    totals: dict[str, int] = {}
    for h in ordered:
        totals[h.hold_reason] = totals.get(h.hold_reason, 0) + 1

    lines = list(head)
    for reason, title in SECTIONS:
        section = [h for h in shown if h.hold_reason == reason]
        if not section:
            continue
        total = totals[reason]
        # "(2 of 2)" would be noise on the common case - one page, nothing
        # hidden - so the plain count survives wherever it is the whole truth.
        count = (f"{len(section)} of {total}" if len(section) != total
                 else str(total))
        lines.append("")
        lines.append(f"{title} ({count})")
        for held_item in section:
            lines.append(_held_line(numbers[held_item.thread_id],
                                    held_item, view.run_at))
    if len(shown) < len(window):
        lines.append("")
        lines.append(_OVERFLOW % (len(window) - len(shown)))

    # Said only when there IS a split to explain: with nothing in the attention
    # tier no one-tap button is drawn, and with nothing outside it the button
    # covers everything. A line that always appears stops being read.
    attention_held = [h for h in ordered if h.hold_reason in ATTENTION_REASONS]
    individual_total = len(ordered) - len(attention_held)
    if attention_held and individual_total:
        lines.append("")
        rest = (f"{individual_total} more needs approving on its own — tap its "
                f"number." if individual_total == 1 else
                f"{individual_total} more need approving one at a time — tap a "
                f"number.")
        lines.append(f"The button below covers the "
                     f"{_attention_phrase(attention_held)}. {rest}")
    lines += tail

    # Belt and braces on a hard protocol limit. The budget above should already
    # have kept this under; a message Telegram rejects outright is worse than a
    # message with one truncated line.
    text = "\n".join(lines)[:TG_MAX_TEXT]

    # Built from `shown`, never `window`: a button for an item the text does not
    # show is a button whose number the owner cannot read.
    keyboard: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    for held_item in shown:
        row.append((str(numbers[held_item.thread_id]),
                    encode("open", numbers[held_item.thread_id] - 1,
                           digest_id=view.digest_id)))
        if len(row) == 4:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    if view.run_report and done_total:
        keyboard.append([(f"📋 Show the {done_total} done",
                          encode("done", digest_id=view.digest_id))])

    attention = [h for h in ordered if h.hold_reason in ATTENTION_REASONS]
    individual = len(ordered) - len(attention)
    if attention:
        # Attention tier only. A blanket button that ALSO reached trash would
        # sweep the authorisation tier in with the harmless actions, so it would
        # be approved unnoticed - which is the rubber-stamping this design
        # isolated the tiers to prevent.
        keyboard.append([(f"✅ Approve {_attention_phrase(attention)}",
                          encode("approve_attention", digest_id=view.digest_id))])

    # Trash gets its own bulk button, and it is a different thing: it acts on
    # trash ONLY, and only after a confirmation naming every sender. Approving
    # six trash proposals the owner has just read one at a time is tedium, not
    # safety. What the tiers exist to stop is trash being approved WITHOUT being
    # noticed - which a separate, confirmed, trash-only button does not do.
    trash_held = [h for h in ordered if h.hold_reason == "trash"]
    if len(trash_held) > 1:
        keyboard.append([(f"🗑 Trash all {len(trash_held)}",
                          encode("trash_all", digest_id=view.digest_id))])

    nav: list[tuple[str, str]] = []
    if page > 0:
        nav.append(("◀ Prev", encode("prev", digest_id=view.digest_id)))
    if page < pages - 1:
        nav.append(("Next ▶", encode("next", digest_id=view.digest_id)))
    if nav:
        keyboard.append(nav)

    return text, keyboard


_ACTION_CAP = 60


def _done_actions(item: DoneItem) -> str:
    """"label(recruiter), archive" - the same shape the review UI has always
    used for a proposal, so a report of what happened reads like the proposal
    it came from."""
    rendered = ", ".join(f"{kind}({label})" if label else kind
                         for kind, label in item.actions) or "none"
    return _oneline(rendered, _ACTION_CAP)


def done_panel(view: DigestView, page: int = 0) -> tuple[str, list]:
    """The list behind the digest's "Show the N done" button.

    The digest says "33 archive · 17 label". That tells the owner a label
    happened and refuses to say which one, which is the opposite of a report -
    and the label is the part a correction would be about. This is the same
    data expanded: one thread per entry, with what was done to it named.

    Read-only on purpose. Undo belongs here and is not here yet: undo_action()
    refuses on dry-run records, so its buttons cannot be exercised until the
    agent is live against a real mailbox. A panel that shows what happened is
    useful today; a row of buttons that all refuse would not be.
    """
    items = view.done
    pages = max(1, (len(items) + DONE_PAGE_SIZE - 1) // DONE_PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    window = items[page * DONE_PAGE_SIZE:(page + 1) * DONE_PAGE_SIZE]

    # Never the word "done" for something that did not reach Gmail. Same rule as
    # the digest, and the same reason: the banner that says dry_run is on is on
    # a terminal, and this message is on a phone.
    title = "✓ WOULD HAVE DONE" if view.dry_run else "✓ DONE"
    # Actions, not threads, because that is what the button the owner just
    # pressed promised: the digest counts "13 archive · 8 label" and its button
    # says "Show the 21 done". Opening that on "(13)" - the thread count, since
    # a label-then-archive is one thread and two actions - is two true numbers
    # disagreeing in public, which is the same defect as a section heading
    # counting only its page. So the headline matches the button, and then says
    # how many threads that was.
    actions = sum(len(item.actions) for item in items)
    head = [f"{title} ({actions})"
            + (f" · {len(items)} threads" if len(items) != actions else "")]
    if pages > 1:
        head[0] += f"  (page {page + 1}/{pages})"

    # Counted over every item, not this page: the point of a summary is to save
    # the owner from paging through to find out that everything went to one
    # label, and a per-page count would be the same subtotal-as-total mistake
    # the section headings were just fixed for.
    labels: dict[str, int] = {}
    for item in items:
        for kind, label in item.actions:
            if kind == "label" and label:
                labels[label] = labels.get(label, 0) + 1
    if labels:
        head.append("labels: " + " · ".join(
            f"{name} {count}" for name, count in sorted(labels.items())))

    if not items:
        head.append("")
        head.append("This run executed nothing.")

    lines = list(head)
    if window:
        lines.append("")
    shown = 0
    budget = TG_MAX_TEXT - len("\n".join(head)) - len(_OVERFLOW % 999) - 2
    used = 0
    for offset, item in enumerate(window, start=page * DONE_PAGE_SIZE + 1):
        # The held item's three-line block, with what was done where the reason
        # goes: what it was, who sent it, what happened to it. The rule mark
        # goes on the action line because that is the claim it qualifies.
        block = (f"{offset}. {_oneline(item.subject, _SUBJECT_CAP)}\n"
                 f"{_oneline(item.sender, _SENDER_CAP)}\n"
                 f"→ {_done_actions(item)}")
        if item.snippet.strip():
            # Appended before the budget check below, so a snippet can never
            # push the message past Telegram's limit - it shrinks the page
            # instead, which is what the budget has always done.
            block += f"\n{_oneline(item.snippet, _SNIPPET_CAP)}"
        if item.rule_note:
            block += f"\n↳ your rule: {_oneline(item.rule_note, _ACTION_CAP + 40)}"
        elif item.from_rule:
            block += "  · rule"
        if used + len(block) + 2 > budget:
            break
        used += len(block) + 2
        shown += 1
        lines.append(block)
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()     # the separator after the last block has nothing to separate
    if shown < len(window):
        lines.append("")
        lines.append(_OVERFLOW % (len(window) - shown))

    text = "\n".join(lines)[:TG_MAX_TEXT]

    # A button per item shown, numbered absolutely, exactly as the digest does
    # it. Without these the correction verdicts are unreachable - the item view
    # is the only place they live and this panel is the only route to it. They
    # were missing on the first cut because the flow was checked by calling
    # encode() directly instead of by pressing what is on the screen.
    keyboard: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    for offset, _item in enumerate(window, start=page * DONE_PAGE_SIZE):
        row.append((str(offset + 1),
                    encode("open", offset, digest_id=view.digest_id)))
        if len(row) == 4:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    nav: list[tuple[str, str]] = []
    if page > 0:
        nav.append(("◀ Prev", encode("prev", digest_id=view.digest_id)))
    if page < pages - 1:
        nav.append(("Next ▶", encode("next", digest_id=view.digest_id)))
    if nav:
        keyboard.append(nav)
    # Always last, always present: a screen with no way out is a trap on a
    # phone, where there is no Escape key.
    keyboard.append([("↩ Back to the digest",
                      encode("list", digest_id=view.digest_id))])
    return text, keyboard


_WHY_CAP = 300
# Two lines on a phone. Gmail's snippets run to ~200 chars, which is three or
# four lines and starts to bury the entries either side of it.
_SNIPPET_CAP = 110


def confirm_trash_all(items, *, digest_id: str, dry_run: bool
                      ) -> tuple[str, list]:
    """The screen between "Trash all 6" and six threads in the bin.

    Names every sender, not just a count: a count is not something the owner can
    check, and the whole reason this screen exists is to make the second tap a
    decision rather than a reflex. Truncated to the message limit like every
    other list here, with the remainder counted rather than silently dropped.

    Cancel returns to the digest and does nothing at all - which is the half of
    a confirmation that actually matters.
    """
    verb = "WOULD trash" if dry_run else "Trash"
    head = [f"{verb} all {len(items)}?", ""]
    lines = list(head)
    shown = 0
    budget = TG_MAX_TEXT - len("\n".join(head)) - len(_OVERFLOW % 999) - 80
    used = 0
    for item in items:
        block = (f"• {_oneline(item.item.subject, _SUBJECT_CAP)}\n"
                 f"  {_oneline(item.item.sender, _SENDER_CAP)}")
        if used + len(block) + 1 > budget:
            break
        used += len(block) + 1
        shown += 1
        lines.append(block)
    if shown < len(items):
        lines.append(_OVERFLOW % (len(items) - shown))
    lines.append("")
    # Trash is the reversible end of the ladder and saying so is what makes this
    # a proportionate confirmation rather than a scary one.
    lines.append("Gmail keeps trashed mail for 30 days.")
    text = "\n".join(lines)[:TG_MAX_TEXT]
    keyboard = [
        [(f"🗑 Yes, trash {len(items)}",
          encode("trash_all_go", digest_id=digest_id))],
        [("↩ Cancel", encode("list", digest_id=digest_id))],
    ]
    return text, keyboard


def item_view(subject: str, sender: str, actions_text: str, why: str, *,
              digest_id: str, index: int, kind: str = "done",
              rule_detail: str = "",
              categories: Sequence[str] = ()) -> tuple[str, list]:
    """One item, opened. The screen the numbered buttons have always implied.

    Serves both lists with different verbs, because the two differ in tense. A
    held item has not happened and asks approve-or-not; a done item has happened
    and asks was-that-right. Offering "Keep in inbox" on a thread that was never
    archived would describe work that does not exist.

    The number is the position the owner tapped, so the screen that opens says
    the number they pressed rather than a different one.
    """
    lines = [f"{index + 1}. {_oneline(subject, _SUBJECT_CAP)}",
             _oneline(sender, _SENDER_CAP),
             f"→ {_oneline(actions_text, _ACTION_CAP)}"]
    if why.strip() and not rule_detail.strip():
        # Omitted rather than filled with a placeholder: the reason is the only
        # record of a judgement the owner ever sees, and inventing one here
        # would be inventing evidence. Suppressed entirely when a rule decided
        # this, because then the reason IS the rule - prefilter writes "matched
        # sender rule 'x' -> trash" - and the block below says it better.
        lines += ["", f"Why: {_oneline(why, _WHY_CAP)}"]
    if rule_detail.strip():
        # Not collapsed to one line: this is the rule, when it was taught, and
        # how it has performed, and the owner is being asked to judge it.
        lines += ["", rule_detail.strip()]
    text = "\n".join(lines)[:TG_MAX_TEXT]

    if kind == "held":
        keyboard: list[list[tuple[str, str]]] = [[
            ("✅ Approve", encode("approve", index, digest_id=digest_id)),
            ("✖ Not this", encode("reject", index, digest_id=digest_id))]]
    else:
        keyboard = [
            [("📥 Keep in inbox", encode("keep", index, digest_id=digest_id)),
             ("🏷 Label as …", encode("relabel", index, digest_id=digest_id))],
            [("🗑 Trash these instead",
              encode("teach_trash", index, digest_id=digest_id))],
        ]
    keyboard.append([("↩ Back", encode("list", digest_id=digest_id))])
    return text, keyboard


def paged(request: ReviewRequest, index: int,
          categories: Sequence[str] = ()) -> tuple[str, list]:
    """One item at a time, in a message that edits itself in place.

    Clamps rather than raises on an out-of-range index: this is reached from a
    callback, and a stale or replayed one must land somewhere sane rather than
    take the bot down.
    """
    items = request.items
    if not items:
        return header(request) + "\n\n(nothing to review)", []
    i = max(0, min(index, len(items) - 1))
    item = items[i]

    why = item.reason.strip() or NO_REASON
    lines = [
        header(request),
        f"— {i + 1}/{len(items)} —",
        "",
        f"From:    {item.sender}",
        f"Subject: {item.subject}",
        "",
        item.snippet[:300],
        "",
        f"Proposed: {_actions(item)}",
        f"Why:      {why[:200]}",
        f"Conf:     {item.confidence:.2f}{_flag(item)}   source: {item.source}",
    ]
    if item.rule_id:
        lines.append(f"Rule:     {item.rule_id}  (you taught this)")
    text = "\n".join(lines)[:TG_MAX_TEXT]

    keyboard: list[list[tuple[str, str]]] = [[
        ("✅ OK", encode("approve", i)),
        ("✖ Skip", encode("reject", i)),
    ]]
    # Label buttons, three to a row, so the keyboard stays readable on a phone.
    row: list[tuple[str, str]] = []
    for li, cat in enumerate(categories):
        row.append((f"🏷 {cat[:14]}", encode("label", i, li)))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    nav: list[tuple[str, str]] = []
    if i > 0:
        nav.append(("◀", encode("prev")))
    nav.append(("☰ List", encode("list")))
    if i < len(items) - 1:
        nav.append(("▶", encode("next")))
    keyboard.append(nav)
    keyboard.append([("✅ Approve all", encode("approve_all"))])

    return text, keyboard

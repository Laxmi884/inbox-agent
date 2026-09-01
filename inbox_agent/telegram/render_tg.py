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
    ("trash", "🗑 TRASH — needs your OK"),
    ("needs_reply", "✉️ NEEDS REPLY"),
    ("security_alert", "🔒 SECURITY"),
    ("low_confidence", "❓ NOT SURE"),
)

# Held for attention rather than authorisation: the proposed action is harmless
# and the thread simply wants a person. Only these are one-tap approvable.
ATTENTION_REASONS = frozenset({"needs_reply", "security_alert"})

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
    from_rule: bool = False


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
    if attention:
        # Attention tier only. A blanket button that could reach trash or a
        # low-confidence guess would rubber-stamp exactly the set this design
        # isolated to avoid rubber-stamping.
        keyboard.append([(f"✅ Approve {len(attention)} replies & alerts",
                          encode("approve_attention", digest_id=view.digest_id))])

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
                 f"→ {_done_actions(item)}"
                 f"{'  · rule' if item.from_rule else ''}")
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

    keyboard: list[list[tuple[str, str]]] = []
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

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
from typing import Sequence

from ..models import HeldItem, ReviewRequest
from ..render import LOW_CONFIDENCE, NO_REASON
from .callbacks import encode

# Telegram hard limits. Protocol, not preference.
TG_MAX_TEXT = 4096

# A phone screen, roughly. Beyond this the queue is scrolling, not scanning.
HELD_PAGE_SIZE = 8

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
_REASON_CAP = 160


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


def _held_line(number: int, item: HeldItem, now: datetime) -> str:
    """Three lines: what it is, who sent it, and why the agent wants this.

    `reason` is here because these are the items the agent deliberately would
    not decide alone - it is what turns a rejection into training data rather
    than a shrug. The one-line form that was right for a fifty-item list is
    wrong for a list of four.
    """
    conf = (f" · {item.item.confidence:.2f}"
            if item.item.confidence < LOW_CONFIDENCE else "")
    why = (item.item.reason or NO_REASON).strip()[:_REASON_CAP]
    return (f"{number}. {item.item.subject[:_SUBJECT_CAP]}\n"
            f"{item.item.sender}{conf}{_age(item, now)}\n"
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

    stat = " · ".join(f"{count} {kind}"
                      for kind, count in sorted(view.done_by_kind.items())
                      if count)
    lines = [f"Inbox · {view.run_at.astimezone().strftime('%H:%M')} · "
             f"{view.total} threads"]
    lines.append(f"{stat} · {len(ordered)} waiting" if stat
                 else f"{len(ordered)} waiting")
    if pages > 1:
        lines[0] += f"  (page {page + 1}/{pages})"

    for reason, title in SECTIONS:
        section = [h for h in window if h.hold_reason == reason]
        if not section:
            continue
        lines.append("")
        lines.append(f"{title} ({len(section)})")
        for held_item in section:
            lines.append(_held_line(numbers[held_item.thread_id],
                                    held_item, view.run_at))

    done_total = sum(view.done_by_kind.values())
    lines.append("")
    lines.append(f"✓ DONE ({done_total})")
    if view.rule_decided:
        # Make the learning visible. As rules accumulate this climbs and the
        # sections above shrink. Suppressed at zero: "0 came from rules you
        # taught me" reads as a failure rather than as a not-yet.
        lines.append(f"{view.rule_decided} came from rules you taught me")

    text = "\n".join(lines)[:TG_MAX_TEXT]

    keyboard: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    for held_item in window:
        row.append((str(numbers[held_item.thread_id]),
                    encode("open", numbers[held_item.thread_id] - 1,
                           digest_id=view.digest_id)))
        if len(row) == 4:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    if done_total:
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

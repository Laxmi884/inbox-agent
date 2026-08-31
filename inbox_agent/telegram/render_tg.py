"""Telegram renderers (spec section 4.4). Pure functions over ReviewRequest.

Two modes, deliberately both. They do different jobs: the digest clears the
confident majority in one tap, paged handles the handful worth reading. Which
one wins gets settled by using both for a few days rather than by argument -
the method the model registry applies to model choice.

Nothing here knows about the graph, the transport, or LangGraph. Same contract
as render.py, which is why the notebook renderer keeps working unchanged.
"""
from __future__ import annotations

from typing import Sequence

from ..models import ReviewRequest
from ..render import LOW_CONFIDENCE, NO_REASON
from .callbacks import encode

# Telegram hard limits. Protocol, not preference.
TG_MAX_TEXT = 4096

# Leave room for the header and the "... and N more" footer so pagination
# arithmetic never has to be exact-to-the-byte.
_TEXT_BUDGET = TG_MAX_TEXT - 400
DIGEST_PAGE_SIZE = 25


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


def digest(request: ReviewRequest, page: int = 0,
           categories: Sequence[str] = ()) -> tuple[str, list]:
    """One message, all items numbered, approve-all in a tap.

    Pages rather than truncates: 50 rows at ~80 chars is right at Telegram's
    4096 cap, and silently dropping the tail of a review list is exactly the
    kind of invisible failure this project keeps trying to design out.
    """
    items = request.items
    pages = max(1, (len(items) + DIGEST_PAGE_SIZE - 1) // DIGEST_PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    start = page * DIGEST_PAGE_SIZE
    window = items[start:start + DIGEST_PAGE_SIZE]

    lines = [header(request)]
    if pages > 1:
        lines[0] += f"  (page {page + 1}/{pages})"
    lines.append("")

    for offset, item in enumerate(window):
        i = start + offset
        src = "rule" if item.source == "rule" else "    "
        row = (f"{i + 1:>3}. {item.sender[:24]:<24} {item.subject[:34]:<34} "
               f"→ {_actions(item)}{_flag(item)}  {src}")
        # Budget check per row rather than at the end, so the message is never
        # assembled oversized and then chopped.
        if sum(len(x) + 1 for x in lines) + len(row) > _TEXT_BUDGET:
            lines.append(f"     … {len(window) - offset} more on this page")
            break
        lines.append(row)

    keyboard: list[list[tuple[str, str]]] = [
        [("✅ Approve all", encode("approve_all"))]
    ]
    nav: list[tuple[str, str]] = []
    if page > 0:
        nav.append(("◀ Prev", encode("prev")))
    if page < pages - 1:
        nav.append(("Next ▶", encode("next")))
    if nav:
        keyboard.append(nav)

    return "\n".join(lines), keyboard


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
        nav.append(("◀ Prev", encode("prev")))
    if i < len(items) - 1:
        nav.append(("Next ▶", encode("next")))
    nav.append(("✅ Approve all", encode("approve_all")))
    keyboard.append(nav)

    return text, keyboard

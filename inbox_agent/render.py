# inbox_agent/render.py
"""Notebook-side rendering of the UI-agnostic review payload (spec section 7).

Everything here is a pure function over ReviewRequest/ReviewResponse. A Telegram
renderer replaces only this module; the graph does not change.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from .audit import AuditLog
from .models import Action, ReviewRequest, ReviewResponse

LOW_CONFIDENCE = 0.5

# gemma4:12b-mlx reliably omits ThreadJudgment.reason (measured empty on 50 of
# 50 real threads), so an empty reason is the common case, not an anomaly. An
# empty "why" cell is indistinguishable from a rendering bug; this marker
# makes "the model gave no explanation" visible instead of silent.
NO_REASON = "(no reason given)"


def review_table(request: ReviewRequest) -> list[dict]:
    """Rows for pandas.DataFrame, or any other tabular renderer."""
    rows = []
    for item in request.items:
        actions = ", ".join(
            f"{a.kind}({a.params.get('label')})" if a.params.get("label") else a.kind
            for a in item.proposed)
        flag = " !" if item.confidence < LOW_CONFIDENCE else ""
        why = item.reason[:60] if item.reason.strip() else NO_REASON
        rows.append({
            "thread_id": item.thread_id,
            "sender": item.sender[:34],
            "subject": item.subject[:44],
            "proposed": actions,
            "confidence": f"{item.confidence:.2f}{flag}",
            "src": item.source,
            "why": why,
        })
    return rows


def render_review(request: ReviewRequest) -> str:
    lines = [f"Run {request.run_id} · policy {request.policy_version} · "
             f"{len(request.items)} threads", ""]
    for r in review_table(request):
        lines.append(f"  {r['thread_id']:<18} {r['sender']:<34} {r['subject']:<44} "
                     f"-> {r['proposed']:<22} {r['confidence']:<7} {r['why']}")
    return "\n".join(lines)


def approve_all(request: ReviewRequest) -> ReviewResponse:
    return ReviewResponse(decisions={i.thread_id: "approve" for i in request.items})


def respond(
    request: ReviewRequest,
    *,
    reject: Iterable[str] = (),
    edit: Mapping[str, list[Action]] | None = None,
    instructions: Iterable[str] = (),
) -> ReviewResponse:
    """Approve everything except what you name. The common case is one keystroke."""
    edit = dict(edit or {})
    reject = set(reject)
    decisions = {}
    for item in request.items:
        if item.thread_id in edit:
            decisions[item.thread_id] = "edit"
        elif item.thread_id in reject:
            decisions[item.thread_id] = "reject"
        else:
            decisions[item.thread_id] = "approve"
    return ReviewResponse(decisions=decisions, edits=edit,
                          instructions=list(instructions))


def audit_table(log: AuditLog) -> list[dict]:
    return [
        {"ts": r.ts.isoformat(timespec="seconds"), "thread_id": r.thread_id,
         "action": r.action, "actor": r.actor, "dry_run": r.dry_run,
         "result": r.result, "policy": r.policy_version}
        for r in log.records()
    ]

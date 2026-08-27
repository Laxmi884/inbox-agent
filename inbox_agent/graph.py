"""Stage A pipeline (spec section 4.2).

    fetch -> prefilter -> classify -> propose -> <interrupt> -> execute -> learn

The graph owns control flow; the model only judges individual threads. The
interrupt is durable, so a run can be resumed hours later from a different UI.
"""
from __future__ import annotations

import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .audit import AuditLog, ExecutionContext, ForbiddenActionError, execute_action
from .classify import classify_batch
from .config import Settings
from .models import (
    Action, Decision, ReviewItem, ReviewRequest, ReviewResponse, Thread,
)
from .policy import Policy
from .prefilter import prefilter
from .store import PreferenceStore, rule_from_correction


class TriageState(TypedDict, total=False):
    limit: int
    threads: list[dict]
    decisions: list[dict]
    review: dict
    response: dict
    executed: list[dict]
    refused: list[dict]
    skipped: list[dict]
    learned: list[str]


def learn_from_response(
    response: ReviewResponse, threads: list[Thread], prefs: PreferenceStore
) -> list[str]:
    """Turn corrections into durable rules (spec section 2, mechanism 2)."""
    by_id = {t.id: t for t in threads}
    learned: list[str] = []

    for thread_id, verdict in response.decisions.items():
        if verdict == "approve" or thread_id not in by_id:
            continue
        edits = response.edits.get(thread_id, [])
        if not edits:
            continue
        # A "reject" that carries edits still teaches a rule from the edit's
        # action - the owner is saying "not this, but here's what I'd have
        # wanted" - even though execute() will not act on it for this thread.
        # Reject means "do not do this now"; the edit is preference signal for
        # next time. Intended: do not "fix" this into skipping reject+edits.
        rule = rule_from_correction(
            by_id[thread_id], edits[0].kind,
            f"corrected proposal on thread {thread_id}: owner chose {edits[0].kind}",
        )
        prefs.add_rule(rule)
        learned.append(rule.id)

    return learned


def build_graph(
    *,
    client,
    prefs: PreferenceStore,
    policy: Policy,
    llm,
    settings: Settings,
    log: AuditLog,
    checkpointer=None,
):
    context = ExecutionContext(
        model=getattr(llm, "model", None) or getattr(llm, "model_name", None),
        backend=settings.backend,
        policy_version=policy.version,
    )

    def fetch(state: TriageState) -> dict:
        threads = client.list_threads(limit=state.get("limit", settings.snapshot_size))
        return {"threads": [t.model_dump() for t in threads]}

    def triage(state: TriageState) -> dict:
        """Prefilter first, model only on what is left."""
        threads = [Thread.model_validate(d) for d in state["threads"]]
        decided, undecided = prefilter(threads, prefs)
        decided += classify_batch(undecided, llm, policy)
        order = {t.id: i for i, t in enumerate(threads)}
        decided.sort(key=lambda d: order[d.thread_id])
        return {"decisions": [d.model_dump() for d in decided]}

    def propose(state: TriageState) -> dict:
        threads = {d["id"]: Thread.model_validate(d) for d in state["threads"]}
        items = []
        for raw in state["decisions"]:
            d = Decision.model_validate(raw)
            t = threads[d.thread_id]
            items.append(ReviewItem(
                thread_id=d.thread_id, subject=t.subject, sender=t.sender,
                snippet=t.snippet, proposed=d.actions, reason=d.reason,
                confidence=d.confidence, source=d.source, rule_id=d.rule_id,
            ))
        request = ReviewRequest(
            run_id=uuid.uuid4().hex[:8], policy_version=policy.version, items=items)
        return {"review": request.model_dump(mode="json")}

    def review(state: TriageState) -> dict:
        """Suspend for the human. Durable: resume from any UI, any time."""
        answer = interrupt(state["review"])
        return {"response": answer}

    def execute(state: TriageState) -> dict:
        response = ReviewResponse.model_validate(state.get("response") or {})
        decisions = {d["thread_id"]: Decision.model_validate(d)
                     for d in state["decisions"]}
        executed = []
        refused = []
        skipped = []

        for thread_id, verdict in response.decisions.items():
            if verdict == "reject":
                continue

            # The interrupt's whole purpose is a trust boundary: the executed
            # set must be a subset of what the human was actually shown. A
            # resume payload naming a thread that never appeared in this
            # batch - stale, replayed, or forged - must be skipped before any
            # indexing happens, on every verdict branch (approve AND edit),
            # not just guarded where a crash would otherwise be obvious.
            decision = decisions.get(thread_id)
            if decision is None:
                skipped.append({"thread_id": thread_id, "verdict": verdict,
                                 "reason": "not part of the reviewed batch"})
                continue

            actions = (response.edits.get(thread_id)
                       if verdict == "edit" else decision.actions) or []
            for action in actions:
                if action.kind == "none":
                    continue
                try:
                    rec = execute_action(
                        action, client=client, settings=settings, log=log,
                        actor="human" if verdict == "edit" else (
                            f"rule:{decision.rule_id}" if decision.rule_id
                            else "agent"),
                        context=context,
                        rule_provenance=decision.reason if decision.rule_id else None,
                    )
                    executed.append(rec.model_dump(mode="json"))
                except ForbiddenActionError as exc:
                    # execute_action already wrote the durable refusal record;
                    # this makes the refusal visible to whatever UI is holding
                    # the state too (a notebook, a future Telegram bot), since
                    # print() reaches neither and the JSONL file is not
                    # something either renders by default.
                    refused.append({"thread_id": thread_id, "kind": action.kind,
                                     "error": str(exc)})

        return {"executed": executed, "refused": refused, "skipped": skipped}

    def learn(state: TriageState) -> dict:
        response = ReviewResponse.model_validate(state.get("response") or {})
        threads = [Thread.model_validate(d) for d in state["threads"]]
        return {"learned": learn_from_response(response, threads, prefs)}

    builder = StateGraph(TriageState)
    for name, fn in (("fetch", fetch), ("triage", triage), ("propose", propose),
                     ("review", review), ("execute", execute), ("learn", learn)):
        builder.add_node(name, fn)

    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "triage")
    builder.add_edge("triage", "propose")
    builder.add_edge("propose", "review")
    builder.add_edge("review", "execute")
    builder.add_edge("execute", "learn")
    builder.add_edge("learn", END)

    return builder.compile(checkpointer=checkpointer)

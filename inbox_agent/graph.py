"""Stage A pipeline (spec section 4.2).

    fetch -> prefilter -> classify -> propose -> <interrupt> -> execute -> learn

The graph owns control flow; the model only judges individual threads. The
interrupt is durable, so a run can be resumed hours later from a different UI.
"""
from __future__ import annotations

import operator
import uuid
from typing import Annotated, Optional, TypedDict, get_args

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith.run_helpers import get_current_run_tree

from .audit import AuditLog, ExecutionContext, ForbiddenActionError, execute_action
from .classify import classify_batch
from .config import Settings
from .models import (
    Action, ActionKind, Decision, ReviewItem, ReviewRequest, ReviewResponse, Thread,
)
from .policy import Policy
from .prefilter import prefilter
from .recency import demote_stale
from .store import PreferenceStore, rule_from_correction

# Derived from the Literal itself, not a hand-copied list, so this can't
# silently drift from ActionKind if it's ever extended.
VALID_ACTION_KINDS = frozenset(get_args(ActionKind))


class TriageState(TypedDict, total=False):
    limit: int
    # Identifiers, not payloads. LangGraph persists the FULL state after every
    # node, so whatever lives here is duplicated once per checkpoint - measured
    # at 11-16 checkpoints for a single run. Bodies stayed invisible only
    # because the frozen snapshot has none (0 of 50 populated); live threads
    # measure up to 204 KB, which would be written ~11 times per run. The client
    # is the source of truth and is already injected into every node.
    thread_ids: list[str]
    decisions: list[dict]
    review: dict
    response: dict
    executed: list[dict]
    refused: list[dict]
    # execute() and learn() both write here. Without a reducer the second
    # silently discards the first, and this is where refusals are recorded -
    # a forged thread id rejected in execute() would vanish. The manual
    # `state.get(...) + new` merge that used to do this failed in exactly the
    # direction that loses data, so the framework enforces it now.
    skipped: Annotated[list[dict], operator.add]
    learned: list[str]


def learn_from_response(
    response: ReviewResponse, threads: list[Thread], prefs: PreferenceStore,
    proposals: Optional[list[dict]] = None,
) -> tuple[list[str], list[dict]]:
    """Turn corrections into durable rules (spec section 2, mechanism 2).

    Returns (learned_rule_ids, skipped). A correction is skipped rather than
    learned when its edit names a kind outside ActionKind - reachable because
    Action.kind is str, not the Literal, so a human can type e.g.
    "send_message" in an edit. execute() already refuses that kind; there is
    also no valid rule to learn from it here: it names an action the system
    will never perform under any circumstances (spec section 2's deny-list),
    so recording a "preference" for it would record a preference that can
    never be honoured. Validating at this boundary - before
    rule_from_correction, not inside it - keeps a malformed edit from
    crashing the graph's final node after mutations have already run.
    """
    by_id = {t.id: t for t in threads}
    # What was PROPOSED for each thread, so a bare reject can record what was
    # rejected rather than just that something was.
    by_decision = {d.get("thread_id"): (d.get("actions") or [{}])[0].get("kind")
                   for d in (proposals or [])}
    learned: list[str] = []
    skipped: list[dict] = []

    for thread_id, verdict in response.decisions.items():
        if verdict == "approve" or thread_id not in by_id:
            continue
        edits = response.edits.get(thread_id, [])
        # A bare reject teaches too. The spec counts "every reject or edit at
        # the review step" as a candidate rule; requiring an edit meant a Skip
        # taught nothing at all, which is why correcting the agent by skipping
        # never made it better.
        rejected = None
        if not edits:
            proposed = by_decision.get(thread_id)
            if not proposed:
                continue
            rejected = proposed
            kind = "none"
        else:
            kind = edits[0].kind
        if kind not in VALID_ACTION_KINDS:
            skipped.append({
                "thread_id": thread_id, "kind": kind, "stage": "learn",
                "reason": f"{kind!r} is not a learnable action kind; no rule recorded",
            })
            continue

        # A "reject" that carries edits still teaches a rule from the edit's
        # action - the owner is saying "not this, but here's what I'd have
        # wanted" - even though execute() will not act on it for this thread.
        # Reject means "do not do this now"; the edit is preference signal for
        # next time. Intended: do not "fix" this into skipping reject+edits.
        note = (f"owner rejected {rejected} on thread {thread_id}" if rejected
                else f"corrected proposal on thread {thread_id}: owner chose {kind}")
        rule = rule_from_correction(by_id[thread_id], kind, note,
                                    rejected=rejected, corpus=threads)
        prefs.add_rule(rule)
        learned.append(rule.id)

    return learned, skipped


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
    def _threads(state: TriageState) -> list[Thread]:
        """Re-read from the client. State carries ids only (see TriageState)."""
        return [client.get_thread(i) for i in state["thread_ids"]]

    def fetch(state: TriageState) -> dict:
        threads = client.list_threads(limit=state.get("limit", settings.snapshot_size))
        return {"thread_ids": [t.id for t in threads]}

    def triage(state: TriageState) -> dict:
        """Prefilter first, model only on what is left."""
        threads = _threads(state)
        decided, undecided = prefilter(threads, prefs)
        decided += classify_batch(undecided, llm, policy, prefs.instructions())

        # Age is decided here, deterministically, rather than asked of the
        # model. A two-year-old needs_reply is not a needs-reply, and the model
        # is unreliable at date arithmetic. Rule-decided threads are exempt
        # inside demote_stale - an explicit instruction outranks an inference.
        by_id = {t.id: t for t in threads}
        decided = [demote_stale(d, by_id[d.thread_id], settings.stale_after_days)
                   for d in decided]

        order = {t.id: i for i, t in enumerate(threads)}
        decided.sort(key=lambda d: order[d.thread_id])
        return {"decisions": [d.model_dump() for d in decided]}

    def propose(state: TriageState) -> dict:
        threads = {t.id: t for t in _threads(state)}
        items = []
        for raw in state["decisions"]:
            d = Decision.model_validate(raw)
            t = threads[d.thread_id]
            items.append(ReviewItem(
                thread_id=d.thread_id, category=d.category,
                subject=t.subject, sender=t.sender,
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

    def execute(state: TriageState, config: RunnableConfig) -> dict:
        # ExecutionContext is built here, per invocation, rather than once in
        # build_graph: checkpoint_id/langsmith_run_id are run-scoped, not
        # graph-scoped, and only exist once a run is actually underway.
        #
        # checkpoint_id: verified against a real config dict at runtime
        # (langgraph 1.2.11) - config["configurable"]["checkpoint_id"] exists
        # as a key but is None on both the initial and the resumed invocation
        # of a node; it is not populated by ordinary forward execution in
        # this version. thread_id IS reliably present on every invocation, so
        # that is what gets recorded here (field left named checkpoint_id per
        # the audit schema; see the fix report for the runtime evidence).
        configurable = config.get("configurable", {}) if config else {}
        checkpoint_id = configurable.get("thread_id")

        # langsmith_run_id: only present when tracing is actually active.
        # Tracing is OFF by default locally, so this must never raise or add
        # latency when LangSmith isn't configured.
        langsmith_run_id = None
        try:
            run_tree = get_current_run_tree()
            if run_tree is not None:
                langsmith_run_id = str(run_tree.id)
        except Exception:
            langsmith_run_id = None

        context = ExecutionContext(
            model=getattr(llm, "model", None) or getattr(llm, "model_name", None),
            backend=settings.backend,
            policy_version=policy.version,
            checkpoint_id=checkpoint_id,
            langsmith_run_id=langsmith_run_id,
        )

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
        threads = _threads(state)
        # Mechanism 1: the owner speaking directly. Recorded before the
        # derived rules below, because it outranks them.
        for text in response.instructions:
            prefs.add_instruction(text)
        learned, learn_skips = learn_from_response(response, threads, prefs,
                                                   state.get("decisions"))
        # `skipped` carries an operator.add reducer, so returning only this
        # node's skips appends rather than overwrites. The manual merge this
        # replaces was correct but easy to lose in a refactor.
        return {"learned": learned, "skipped": learn_skips}

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

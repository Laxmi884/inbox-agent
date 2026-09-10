"""Stage A pipeline (spec section 4.2).

    fetch -> prefilter -> classify -> apply_rules -> propose -> partition -+-> auto_execute -> enqueue_held -+
                                                             +-> <interrupt> -> execute ------+-> mark_triaged -> learn

partition splits the batch by the autonomy ladder (inbox_agent/partition.py).
Confident, reversible actions act then report (auto_execute); everything else
either queues for later (enqueue_held, incremental mode) or waits on the
human right now (<interrupt>, backlog mode). The interrupt is durable, so a
backlog run can be resumed hours later from a different UI.
"""
from __future__ import annotations

import logging as log_module
import operator
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional, TypedDict, get_args

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith.run_helpers import get_current_run_tree

from .audit import AuditLog, ExecutionContext, ForbiddenActionError, execute_action
from .classify import classify_batch
from .config import Settings
from .models import (
    Action, ActionKind, ActionTemplate, Decision, DoneRecord, ReviewItem,
    ReviewRequest, ReviewResponse, RunReport, Thread,
)
from .partition import partition
from .policy import Policy
from .prefilter import prefilter
from .recency import demote_stale
from .store import DoneStore, HeldQueue, PreferenceStore, rule_from_correction

# Derived from the Literal itself, not a hand-copied list, so this can't
# silently drift from ActionKind if it's ever extended.
VALID_ACTION_KINDS = frozenset(get_args(ActionKind))

# TriageState.mode is a plain str, not a Literal - see route_after_partition
# for why an unrecognised value must raise rather than quietly act.
VALID_MODES = frozenset({"incremental", "backlog"})


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
    # "incremental" (default) or "backlog". Selects the edge out of partition:
    # incremental acts then reports, backlog previews then commits. The two
    # differ in risk, not in classification - a bad rule applied across 500
    # historical threads is not something per-item undo repairs comfortably.
    mode: str
    auto: list[dict]
    held: list[dict]
    # How many more threads matched the query than this run took. Not a count
    # of the mailbox: the probe asks for limit + 1 ids and stops, so this says
    # "the cap bound, and by at least this much".
    remaining: int


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
    # Which rule decided each thread, so a correction can name what it replaces.
    # None for anything the model judged itself - there is no rule to supersede.
    by_rule = {d.get("thread_id"): d.get("rule_id") for d in (proposals or [])}
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
        # params travel with the kind now. An edit that said label(receipt)
        # used to teach a rule that said "label" and nothing else, which is the
        # rule shape that raised KeyError against a live mailbox.
        params = dict(edits[0].params) if edits else {}
        rule = rule_from_correction(by_id[thread_id],
                                    [ActionTemplate(kind=kind, params=params)], note,
                                    rejected=rejected, corpus=threads,
                                    supersedes=by_rule.get(thread_id), seen=prefs)
        prefs.add_rule(rule)
        learned.append(rule.id)

    return learned, skipped


# `log` throughout build_graph is the AuditLog, so the progress logger needs a
# name of its own rather than shadowing it.
_log = log_module.getLogger(__name__)


def _run_actions(actions, *, decision, verdict, client, settings, log, context):
    """Push one item's actions through the chokepoint. Returns (executed, refused).

    Lifted out of execute() so auto_execute and the interrupt path cannot drift
    apart. There must be exactly one place where an Action reaches Gmail.
    """
    executed, refused = [], []
    for action in actions:
        if action.kind == "none":
            continue
        try:
            record = execute_action(
                action, client=client, settings=settings, log=log,
                actor="human" if verdict == "edit" else (
                    f"rule:{decision.rule_id}" if decision.rule_id else "agent"),
                context=context,
                rule_provenance=decision.reason if decision.rule_id else None,
            )
            executed.append(record.model_dump(mode="json"))
        except ForbiddenActionError as exc:
            # execute_action already wrote the durable refusal record; this
            # makes the refusal visible to whatever is holding the state too
            # (a notebook, the interrupt path's reviewer, auto_execute's
            # confident tier, a future Telegram bot), since print() reaches
            # none of them and the JSONL file is not something any of them
            # renders by default.
            refused.append({"thread_id": decision.thread_id,
                            "kind": action.kind, "error": str(exc)})
    return executed, refused


def run_report_from_state(state: TriageState, *, triaged_label: str,
                          now: Optional[datetime] = None) -> RunReport:
    """What this run did, joined into one durable record.

    Two sources, the same two the digest panel has always used. The audit
    records say what actually went through the chokepoint - the only honest
    answer to "what did you do" - but they know a thread by id, which is not
    something the owner can read. The proposals carry the subject, the sender,
    the category and the model's reason. A record whose thread is missing from
    the proposals still gets a row, named by its id: an action with no visible
    subject is strange, and hiding it would be worse.

    Read off `review.items` rather than `auto`: in an incremental run the
    executed set is a subset of `auto` and the two agree, and taking the whole
    proposal list means a record can never lose its subject to a partition
    detail.

    .get() throughout, and never raising: this runs after the actions have
    reached Gmail, so an exception here would cost the owner the report for
    work that already happened.
    """
    request = ReviewRequest.model_validate(state["review"])
    proposals = {i.thread_id: i for i in request.items}

    rows: dict[str, DoneRecord] = {}
    for record in state.get("executed", []):
        if not isinstance(record, dict):
            continue
        kind = record.get("action")
        thread_id = record.get("thread_id")
        if not kind or not thread_id:
            continue
        label = (record.get("params") or {}).get("label")
        if kind == "label" and label == triaged_label:
            continue    # bookkeeping on every thread, not work to report
        row = rows.get(thread_id)
        if row is None:
            item = proposals.get(thread_id) or ReviewItem(
                thread_id=thread_id, subject=thread_id, sender="", snippet="",
                proposed=[], reason="", confidence=0.0, source="model")
            row = DoneRecord(thread_id=thread_id, item=item)
            rows[thread_id] = row
        row.actions.append((kind, label))
        actor = str(record.get("actor", ""))
        if actor.startswith("rule:"):
            row.rule_id = actor.split(":", 1)[1]

    return RunReport(run_id=request.run_id,
                     ran_at=now or datetime.now(timezone.utc),
                     total=len(state.get("thread_ids", [])),
                     remaining=int(state.get("remaining", 0) or 0),
                     done=list(rows.values()))


def build_graph(
    *,
    client,
    prefs: PreferenceStore,
    policy: Policy,
    llm,
    settings: Settings,
    log: AuditLog,
    held: HeldQueue,
    done: DoneStore,
    checkpointer=None,
):
    def _context(config: Optional[RunnableConfig]) -> ExecutionContext:
        """Per-invocation, not per-graph: checkpoint_id and langsmith_run_id are
        run-scoped and only exist once a run is actually underway.

        Accepts None so nodes without a RunnableConfig can still build one.

        checkpoint_id: verified against a real config dict at runtime (langgraph
        1.2.11) - config["configurable"]["checkpoint_id"] exists as a key but is
        None on both the initial and the resumed invocation of a node; it is not
        populated by ordinary forward execution in this version. thread_id IS
        reliably present on every invocation, so that is what gets recorded
        (field left named checkpoint_id per the audit schema).
        """
        configurable = config.get("configurable", {}) if config else {}

        # Only present when tracing is actually active. Tracing is OFF by
        # default locally, so this must never raise or add latency when
        # LangSmith is not configured.
        langsmith_run_id = None
        try:
            run_tree = get_current_run_tree()
            if run_tree is not None:
                langsmith_run_id = str(run_tree.id)
        except Exception:
            langsmith_run_id = None

        return ExecutionContext(
            model=getattr(llm, "model", None) or getattr(llm, "model_name", None),
            backend=settings.backend,
            policy_version=policy.version,
            checkpoint_id=configurable.get("thread_id"),
            langsmith_run_id=langsmith_run_id,
        )

    def _threads(state: TriageState) -> list[Thread]:
        """Re-read from the client. State carries ids only (see TriageState)."""
        return [client.get_thread(i) for i in state["thread_ids"]]

    def fetch(state: TriageState) -> dict:
        limit = state.get("limit", settings.snapshot_size)
        threads = client.list_threads(limit=limit, query=settings.inbox_query)
        # Ids only, one page of up to 500 (list_thread_ids' own page size),
        # nothing hydrated. A page of 500 ids costs 5 quota units against 10
        # to hydrate a single thread, so a real count of how much matched is
        # cheaper than reading one email - not just cheaper than reading what
        # the cap cut. This understates for a mailbox with more than 500
        # untriaged threads (the probe stops at that page), but it is exact
        # below that, which limit + 1 could never be: that only ever answers
        # "did the cap bind" (0 or 1), not by how much. It belongs here rather
        # than in the bot: fetch already owns the query and the limit, and a
        # second caller deciding a run's corpus is a second place for the two
        # to drift.
        ids = client.list_thread_ids(query=settings.inbox_query, max_ids=500)
        return {"thread_ids": [t.id for t in threads],
                "remaining": max(0, len(ids) - len(threads))}

    def triage(state: TriageState) -> dict:
        """Prefilter first, model only on what is left."""
        threads = _threads(state)
        # Every thread the run looks at, before anything is decided about it.
        # This is the sender history choose_scope reads at teach time, and the
        # teaching usually happens in a later run than the sighting that makes
        # the display name look like payload - so it has to be durable, and it
        # has to be written for mail the owner never corrects.
        for seen_thread in threads:
            prefs.note_sender(seen_thread.sender)
        decided, undecided = prefilter(threads, prefs)
        decided += classify_batch(undecided, llm, policy, prefs.instructions(),
                                  body_budget=settings.body_budget)

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

    def apply_rules(state: TriageState) -> dict:
        """Rewrite what the model proposed where the owner has taught otherwise.

        The second of two places rules fire, and the split is forced by the
        data rather than chosen: prefilter matches properties of the raw thread
        and runs before the model, so a rule about a CATEGORY has nowhere to be
        applied there - the category is the model's conclusion, not the
        thread's attribute.

        A rewrite, not a re-judgment. The model's category stands and the
        owner's rule decides what happens to mail of that category, which is
        exactly the correction that motivated it: "you were right that it is a
        valuable newsletter, you were wrong to archive it."

        Attributed to the rule - source, rule_id, a recorded hit - so the
        digest's "came from rules you taught me" counts it and precision can
        move. A rewrite the owner cannot see taught them nothing.
        """
        rewritten = []
        for raw in state.get("decisions", []):
            decision = Decision.model_validate(raw)
            matches = prefs.matching_category(decision.category)
            if matches:
                # Most recently created wins, the same way prefilter resolves a
                # tie: the owner's latest word is the current one.
                rule = max(matches, key=lambda r: r.created_at)
                prefs.record_hit(rule.id)
                decision = decision.model_copy(update={
                    "actions": [Action(kind=t.kind, thread_id=decision.thread_id,
                                       params=dict(t.params))
                                for t in rule.actions],
                    "reason": (f"{decision.reason} (your rule for "
                               f"{decision.category}: {rule.summary})"),
                    "source": "rule",
                    "rule_id": rule.id,
                })
            rewritten.append(decision.model_dump())
        return {"decisions": rewritten}

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

    def partition_node(state: TriageState) -> dict:
        request = ReviewRequest.model_validate(state["review"])
        auto, held_pairs = partition(request.items)
        return {
            "auto": [i.model_dump(mode="json") for i in auto],
            "held": [{"item": i.model_dump(mode="json"), "reason": r}
                     for i, r in held_pairs],
        }

    def route_after_partition(state: TriageState) -> str:
        # Absent means incremental: act then report. But an unrecognised value is
        # a caller bug, and this router chooses between "act on a real mailbox"
        # and "ask first" - so a near miss must stop rather than fall through to
        # acting. Same reasoning as the deny-list normalising "Send_Message" and
        # " send_message" instead of letting a near miss through.
        mode = state.get("mode") or "incremental"
        if mode not in VALID_MODES:
            raise ValueError(
                f"unknown run mode {mode!r}; expected one of {sorted(VALID_MODES)}")
        return "review" if mode == "backlog" else "auto_execute"

    def auto_execute(state: TriageState, config: RunnableConfig) -> dict:
        context = _context(config)
        decisions = {d["thread_id"]: Decision.model_validate(d)
                     for d in state["decisions"]}
        executed, refused = [], []
        auto = state.get("auto", [])
        total = len(auto)
        for index, raw in enumerate(auto, 1):
            item = ReviewItem.model_validate(raw)
            decision = decisions.get(item.thread_id)
            if decision is None:
                continue
            started = time.monotonic()
            ran, refused_here = _run_actions(
                decision.actions, decision=decision, verdict="approve",
                client=client, settings=settings, log=log, context=context)
            executed += ran
            refused += refused_here
            # Progress only, with no slow-action threshold. Gmail already says
            # when it is struggling: _with_backoff logs every rate-limit retry,
            # on a 62s schedule, so a single action can legitimately take over a
            # minute while the quota clears. A second threshold here would fire
            # during exactly that healthy backoff and mean nothing.
            _log.info("execute %d/%d %s %s in %.1fs", index, total,
                      item.thread_id, [a.kind for a in decision.actions],
                      time.monotonic() - started)
        return {"executed": executed, "refused": refused}

    def enqueue_held(state: TriageState) -> dict:
        """Persist this run's held items, and retire what it superseded.

        The queue outliving runs is the point (Plan 1): an item waits until the
        owner rules on it, not until the run ends. But a thread this run
        processed and did NOT hold has been re-judged, and the older verdict is
        dead. Leaving it queued put one thread in the digest twice with
        contradictory proposals - seen live, where a LangChain workshop held as
        `promotion -> trash` was re-classified `learning -> label` and
        auto-executed, while the trash entry stayed on screen. Approving that
        entry would have trashed a thread the agent had just filed as learning
        material: executing a proposal the agent itself had superseded.

        Scoped to `auto`, never to the whole queue. A /triage of 10 must not
        empty a queue holding 40 - threads this run never fetched are exactly
        the carry-over the queue exists for.

        Also writes this run's RunReport: same node, same join, and the
        record outlives the run.
        """
        request = ReviewRequest.model_validate(state["review"])
        run_id = request.run_id
        for raw in state.get("auto", []):
            thread_id = raw.get("thread_id") if isinstance(raw, dict) else None
            if thread_id:
                # Absent is not an error, and most of these were never held.
                held.remove(thread_id)
        for raw in state.get("held", []):
            # The policy travels with the proposal. An item can wait in the
            # queue across a policy edit, so the version loaded when the owner
            # eventually approves it is not the version that proposed it - and
            # the audit record is about the proposal.
            held.add(ReviewItem.model_validate(raw["item"]),
                     run_id=run_id, reason=raw["reason"],
                     policy_version=request.policy_version)
        # The report goes here rather than in the bot because this node already
        # holds `executed`, the proposals and the run id together - and a run
        # gets its record whether or not Telegram drove it. Backlog runs
        # (review → execute → mark_triaged) skip this node and write no report,
        # which is intentional since /backlog is unwired.
        #
        # Wrapped because this node sits between auto_execute and mark_triaged:
        # the actions have already reached Gmail, and an exception here aborts
        # the run BEFORE the triaged label lands, so the next run reclassifies
        # and re-executes the whole batch against the real mailbox. A lost
        # report costs one panel; a re-execution costs the owner twice.
        #
        # run_report_from_state documents itself as never raising, but nothing
        # enforced that - model_validate and the pydantic row construction can -
        # and done.record()'s sqlite write was never covered by the promise at
        # all. The guarantee belongs here, where the cost of breaking it is.
        try:
            done.record(run_report_from_state(
                state, triaged_label=settings.triaged_label))
        except Exception:
            _log.exception("run report failed; the run itself stands")
        return {}

    def mark_triaged(state: TriageState, config: RunnableConfig) -> dict:
        """Label every thread this run processed, both tiers.

        Held items are marked too: they are in the queue and will be shown from
        there, so leaving them unlabelled would re-triage them on every run
        while they wait - which is exactly the flood this label exists to stop.

        Goes through execute_action like anything else, so it is audited, and is
        refused by the deny-list and skipped by dry-run on the same terms.

        Takes `config` like auto_execute and execute do, so `_context` fills in
        the real checkpoint_id instead of None - otherwise every triaged-label
        record in the audit log would be untraceable to the run that wrote it.
        """
        context = _context(config)
        refused_ids = []
        thread_ids = state.get("thread_ids", [])
        total = len(thread_ids)
        # Announced before the loop, not only after it. This is the second of
        # the two silent Gmail phases: one label per thread, both tiers, so on
        # a twenty-thread run it is another twenty round trips after the digest
        # already looks ready.
        _log.info("mark_triaged: labelling %d thread(s)", total)
        for index, thread_id in enumerate(thread_ids, 1):
            action = Action(kind="label", thread_id=thread_id,
                            params={"label": settings.triaged_label})
            started = time.monotonic()
            try:
                execute_action(action, client=client, settings=settings, log=log,
                               actor="agent", context=context)
                _log.info("mark_triaged %d/%d %s in %.1fs", index, total,
                          thread_id, time.monotonic() - started)
            except ForbiddenActionError:
                # Configured out. Not fatal: the run's real work already
                # happened. Surfaced into `skipped`, not `refused` - `refused`
                # has no reducer (see TriageState), so returning it here would
                # silently overwrite whatever execute()/auto_execute() already
                # wrote there. `skipped` DOES have one (operator.add), and
                # this is the same reason the interrupt path's refusals are
                # surfaced into state at all: print() reaches no notebook, no
                # Telegram bot, and the JSONL file is not something either
                # renders by default.
                refused_ids.append(thread_id)

        skipped = []
        if refused_ids:
            log_module.getLogger(__name__).warning(
                "triaged label refused by the deny-list for %d thread(s); "
                "they will be re-triaged", len(refused_ids))
            skipped = [{"thread_id": tid, "stage": "mark_triaged",
                       "reason": "triaged label refused by the deny-list"}
                      for tid in refused_ids]
        return {"skipped": skipped}

    def review(state: TriageState) -> dict:
        """Suspend for the human. Durable: resume from any UI, any time."""
        answer = interrupt(state["review"])
        return {"response": answer}

    def execute(state: TriageState, config: RunnableConfig) -> dict:
        context = _context(config)

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
            ran, refused_here = _run_actions(
                actions, decision=decision, verdict=verdict,
                client=client, settings=settings, log=log, context=context)
            executed += ran
            refused += refused_here

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
    for name, fn in (("fetch", fetch), ("triage", triage),
                     ("apply_rules", apply_rules), ("propose", propose),
                     ("partition", partition_node), ("auto_execute", auto_execute),
                     ("enqueue_held", enqueue_held), ("mark_triaged", mark_triaged),
                     ("review", review), ("execute", execute), ("learn", learn)):
        builder.add_node(name, fn)

    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "triage")
    builder.add_edge("triage", "apply_rules")
    builder.add_edge("apply_rules", "propose")
    builder.add_edge("propose", "partition")
    builder.add_conditional_edges("partition", route_after_partition,
                                  {"auto_execute": "auto_execute", "review": "review"})
    builder.add_edge("auto_execute", "enqueue_held")
    builder.add_edge("enqueue_held", "mark_triaged")
    builder.add_edge("mark_triaged", "learn")
    builder.add_edge("review", "execute")
    builder.add_edge("execute", "mark_triaged")
    builder.add_edge("learn", END)

    return builder.compile(checkpointer=checkpointer)

"""The single chokepoint every mutation passes through (spec section 4.1).

Two invariants this module exists to guarantee:
  1. No action on the deny-list can reach Gmail, whatever the caller believes.
  2. Every attempt - permitted, refused, or simulated - leaves a durable record.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import ALWAYS_FORBIDDEN, Settings
from .models import Action, AuditRecord, REVERSIBLE_ACTIONS


class ForbiddenActionError(RuntimeError):
    """Raised when an action on the deny-list is attempted."""


@dataclass
class ExecutionContext:
    """Everything an audit record needs that isn't part of the action itself."""
    model: Optional[str] = None
    backend: Optional[str] = None
    policy_version: Optional[str] = None
    checkpoint_id: Optional[str] = None
    langsmith_run_id: Optional[str] = None


class AuditLog:
    """Append-only JSONL. Never rewrites or truncates."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def records(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        return [AuditRecord.model_validate(json.loads(line))
                for line in self.path.read_text().splitlines() if line.strip()]

    def undo_candidates(self) -> list[AuditRecord]:
        """Real, reversible actions with something to undo, most recent first.

        Note: "draft" is nominally reversible (deleting the draft would undo it),
        but _undo_token has no draft branch, because SnapshotGmailClient.create_draft
        does not return a draft id an undo could later target. Requiring a non-empty
        undo_token keeps such records out of the candidate list instead of handing
        an undo tool a candidate with nothing to act on.
        """
        records = self.records()
        undone = {r.undoes for r in records if r.undoes}
        return [r for r in reversed(records)
                if r.reversible and not r.dry_run and r.result == "ok"
                and r.undo_token and r.id and r.id not in undone
                and r.undoes is None]


def _undo_token(action: Action, prior_labels: list[str]) -> dict[str, Any]:
    if action.kind in ("archive", "trash"):
        return {"restore_labels": list(prior_labels)}
    if action.kind == "label":
        return {"remove_label": action.params.get("label")}
    if action.kind == "unlabel":
        return {"add_label": action.params.get("label")}
    return {}


def _dispatch(action: Action, client) -> dict[str, Any]:
    if action.kind == "label":
        return client.apply_label(action.thread_id, action.params["label"])
    if action.kind == "unlabel":
        return client.remove_label(action.thread_id, action.params["label"])
    if action.kind == "archive":
        return client.archive(action.thread_id)
    if action.kind == "trash":
        return client.trash(action.thread_id)
    if action.kind == "draft":
        return client.create_draft(action.thread_id, action.params.get("body", ""))
    if action.kind == "none":
        return {"noop": True}
    raise ForbiddenActionError(f"unknown action kind {action.kind!r}")


def execute_action(
    action: Action,
    *,
    client,
    settings: Settings,
    log: AuditLog,
    actor: str,
    context: ExecutionContext,
    rule_provenance: Optional[str] = None,
) -> AuditRecord:
    """Perform one action, or refuse it, and record either way."""
    try:
        prior_labels = list(client.get_thread(action.thread_id).label_ids)
    except KeyError:
        prior_labels = []

    base = dict(
        id=uuid.uuid4().hex[:12],
        ts=datetime.now(timezone.utc), thread_id=action.thread_id, action=action.kind,
        params=action.params, actor=actor, rule_provenance=rule_provenance,
        model=context.model, backend=context.backend,
        langsmith_run_id=context.langsmith_run_id, checkpoint_id=context.checkpoint_id,
        policy_version=context.policy_version, dry_run=settings.dry_run,
        reversible=action.kind in REVERSIBLE_ACTIONS,
        undo_token=_undo_token(action, prior_labels),
    )

    # Normalise before the deny-list check, and OR in the ALWAYS_FORBIDDEN floor
    # so this chokepoint never fully trusts a caller-supplied Settings: an empty
    # or case/whitespace-mangled forbidden_actions must not let a deny-listed
    # kind through. This must stay before the dry-run branch below, or a
    # deny-list evasion in dry-run mode would be filed as ordinary "simulated"
    # activity instead of being refused.
    normalized_kind = action.kind.strip().lower()
    if normalized_kind in (settings.forbidden_actions | ALWAYS_FORBIDDEN):
        log.append(AuditRecord(**base, result=f"refused: {action.kind} is on the deny-list"))
        raise ForbiddenActionError(
            f"{action.kind} is permanently forbidden (spec section 2). "
            "This is enforced here, not in the prompt."
        )

    if settings.dry_run:
        rec = AuditRecord(**base, result="simulated")
        log.append(rec)
        return rec

    try:
        _dispatch(action, client)
        rec = AuditRecord(**base, result="ok")
    except Exception as exc:
        rec = AuditRecord(**base, result=f"error: {type(exc).__name__}: {exc}")
        log.append(rec)
        raise

    log.append(rec)
    return rec


def _reverse(record: AuditRecord, client) -> dict[str, Any]:
    """Apply the inverse recorded in the undo token."""
    token = record.undo_token
    if "restore_labels" in token:
        # archive/trash removed INBOX (and trash added TRASH). Put back exactly
        # what was there before, and clear TRASH if we put it there.
        if record.action == "trash":
            client.remove_label(record.thread_id, "TRASH")
        for label in token["restore_labels"]:
            client.apply_label(record.thread_id, label)
        return {"restored": token["restore_labels"]}
    if token.get("remove_label"):
        client.remove_label(record.thread_id, token["remove_label"])
        return {"removed": token["remove_label"]}
    if token.get("add_label"):
        client.apply_label(record.thread_id, token["add_label"])
        return {"added": token["add_label"]}
    raise ValueError(f"no reversal defined for {record.action!r}")


def undo_action(
    record: AuditRecord,
    *,
    client,
    settings: Settings,
    log: AuditLog,
    actor: str,
    context: ExecutionContext,
) -> AuditRecord:
    """Reverse one executed action, and record the reversal.

    Goes through the same discipline as execute_action: refuse loudly, and
    leave a durable record either way. The original record is NEVER mutated -
    append-only means the evidence that the action happened survives next to
    the evidence that it was taken back.

    An undo is also the strongest correction signal the system can get. At the
    review gate the owner is judging a proposal; here they have seen the actual
    consequence and disliked it. Callers that learn from corrections should
    weight this above a verdict given at the gate.
    """
    if not record.id:
        raise ValueError("record has no id; it predates undo support")
    if record.result != "ok":
        raise ValueError(f"cannot undo an action that did not succeed: {record.result!r}")
    if record.dry_run:
        raise ValueError("cannot undo a simulated action: nothing happened in Gmail")
    if not record.undo_token:
        raise ValueError(f"nothing to undo for {record.action!r}")
    if any(r.undoes == record.id for r in log.records()):
        raise ValueError(f"{record.id} was already undone")

    base = dict(
        id=uuid.uuid4().hex[:12],
        ts=datetime.now(timezone.utc),
        thread_id=record.thread_id,
        action=f"undo:{record.action}",
        params=dict(record.undo_token),
        actor=actor,
        rule_provenance=None,
        model=context.model,
        backend=context.backend,
        langsmith_run_id=context.langsmith_run_id,
        checkpoint_id=context.checkpoint_id,
        policy_version=context.policy_version,
        dry_run=settings.dry_run,
        # An undo is not itself undoable: re-applying the original action is a
        # new decision, not a reversal, and should be made as one.
        reversible=False,
        undo_token={},
        undoes=record.id,
    )

    try:
        _reverse(record, client)
        rec = AuditRecord(**base, result="ok")
    except Exception as exc:
        rec = AuditRecord(**base, result=f"error: {type(exc).__name__}: {exc}")
        log.append(rec)
        raise

    log.append(rec)
    return rec

"""The single chokepoint every mutation passes through (spec section 4.1).

Two invariants this module exists to guarantee:
  1. No action outside the allow-list can reach Gmail, whatever the caller believes.
  2. Every attempt - permitted, refused, or simulated - leaves a durable record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import Settings
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
        """Real, reversible actions, most recent first."""
        return [r for r in reversed(self.records())
                if r.reversible and not r.dry_run and r.result == "ok"]


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
        ts=datetime.now(timezone.utc), thread_id=action.thread_id, action=action.kind,
        params=action.params, actor=actor, rule_provenance=rule_provenance,
        model=context.model, backend=context.backend,
        langsmith_run_id=context.langsmith_run_id, checkpoint_id=context.checkpoint_id,
        policy_version=context.policy_version, dry_run=settings.dry_run,
        reversible=action.kind in REVERSIBLE_ACTIONS,
        undo_token=_undo_token(action, prior_labels),
    )

    if action.kind in settings.forbidden_actions:
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

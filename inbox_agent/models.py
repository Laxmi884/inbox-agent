"""Domain types. Everything crossing the interrupt boundary must be JSON-safe:
the notebook renders it today, a Telegram bot renders it tomorrow (spec section 7).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ActionKind = Literal["label", "unlabel", "archive", "trash", "draft", "none"]
Verdict = Literal["approve", "reject", "edit"]

# Actions whose effects can be undone. Drives AuditRecord.reversible.
# Deliberately excludes irreversible actions like "send_message" and "delete_forever"
# so that AuditRecord.reversible can be False for them.
REVERSIBLE_ACTIONS = frozenset({"label", "unlabel", "archive", "trash", "draft", "none"})


class Action(BaseModel):
    kind: str  # Widened from ActionKind to allow testing deny-list at chokepoint
    thread_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class Thread(BaseModel):
    id: str
    subject: str
    sender: str
    to: list[str] = Field(default_factory=list)
    date: str
    snippet: str
    body: str = ""
    label_ids: list[str] = Field(default_factory=list)

    @property
    def sender_domain(self) -> str:
        m = re.search(r"@([\w.\-]+)", self.sender)
        return m.group(1).lower() if m else ""

    @property
    def fingerprint(self) -> str:
        """Stable identity for 'mail like this one': sender plus a digit-stripped
        subject, so 'Invoice 8821' and 'Invoice 8822' collapse together. Strips
        leading Re:/Fwd:/Fw: prefixes and collapses internal whitespace."""
        # Strip leading Re:, Fwd:, Fw: (case-insensitive, possibly repeated)
        shape = self.subject.lower()
        shape = re.sub(r"^(\s*(re|fwd|fw):\s*)+", "", shape, flags=re.IGNORECASE)
        # Collapse internal whitespace to single space
        shape = re.sub(r"\s+", " ", shape).strip()
        # Strip digits
        shape = re.sub(r"\d+", "#", shape).strip()
        return hashlib.sha256(f"{self.sender.lower()}|{shape}".encode()).hexdigest()[:16]


class Decision(BaseModel):
    thread_id: str
    category: str
    actions: list[Action] = Field(default_factory=list)
    reason: str
    confidence: float = 0.0
    source: Literal["rule", "model"] = "model"
    rule_id: Optional[str] = None


class Rule(BaseModel):
    id: str
    scope: Literal["sender", "domain", "fingerprint", "subject"]
    pattern: str
    action: ActionKind
    provenance: str
    created_at: datetime
    hit_count: int = 0
    overridden: bool = False


class ReviewItem(BaseModel):
    thread_id: str
    subject: str
    sender: str
    snippet: str
    proposed: list[Action]
    reason: str
    confidence: float
    source: Literal["rule", "model"]
    rule_id: Optional[str] = None


class ReviewRequest(BaseModel):
    """Payload handed to whatever UI is attached. Renderer-agnostic by contract."""
    run_id: str
    policy_version: str
    items: list[ReviewItem]


class ReviewResponse(BaseModel):
    decisions: dict[str, Verdict] = Field(default_factory=dict)
    edits: dict[str, list[Action]] = Field(default_factory=dict)
    instructions: list[str] = Field(default_factory=list)


class AuditRecord(BaseModel):
    ts: datetime
    thread_id: str
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    actor: str
    rule_provenance: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    langsmith_run_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    policy_version: Optional[str] = None
    dry_run: bool = True
    result: str = ""
    reversible: bool = True
    undo_token: dict[str, Any] = Field(default_factory=dict)

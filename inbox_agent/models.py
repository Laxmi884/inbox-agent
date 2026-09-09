"""Domain types. Everything crossing the interrupt boundary must be JSON-safe:
the notebook renders it today, a Telegram bot renders it tomorrow (spec section 7).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

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


class ActionTemplate(BaseModel):
    """An action with no thread attached: what a rule stores.

    `Action` requires a thread_id, correctly - an action is always about one
    thread, and the chokepoint audits it that way. A rule is the shape of an
    action the owner wants taken on mail it has not seen yet, so it carries
    everything except the thread, and `prefilter` binds the two together.
    """
    kind: str
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
    # `category` is not a property of a thread: it is what the model concluded
    # about one. So a category rule cannot be matched in prefilter, which runs
    # before the model - it is applied afterwards, by the apply_rules node.
    scope: Literal["sender", "domain", "fingerprint", "subject", "category"]
    pattern: str
    # A sequence, not a kind. One kind could not say WHICH label, so a learned
    # label rule produced Action(kind="label", params={}) and _dispatch raised
    # KeyError against a live mailbox while reporting "simulated" under dry-run.
    # A list also makes "label it and archive it" and "label it and leave it in
    # the inbox" the same kind of statement, which is what a correction needs.
    actions: list[ActionTemplate]
    provenance: str
    created_at: datetime
    hit_count: int = 0
    overridden: bool = False
    # A bare reject says "not this" without saying what instead. That is real
    # signal - the spec's "how it learns" counts every reject OR edit as a
    # candidate rule - so it is recorded rather than thrown away.
    rejected_action: Optional[ActionKind] = None
    # A rule that fires 40 times and is undone 12 times is a bad rule. The old
    # `overridden` boolean could only say "someone disagreed once", which is not
    # enough to tell a slightly-wrong rule from a broken one.
    override_count: int = 0
    # The rule this one was taught to replace, when the correction that created
    # it was a correction OF a rule. `override_count` says THAT the owner
    # disagreed; this says which rule they disagreed with, and this rule's
    # `actions` say what they wanted instead. The three together are a vote:
    # four corrections all replacing `archive` with `label` mean "keep the
    # scope, change the action", while four corrections replacing it with four
    # different things mean the rule is genuinely unsound. Nothing consumes
    # this yet - demotion is still the scalar count - but the pairing is only
    # knowable at correction time, so not recording it now would make the
    # distinction unrecoverable from rules taught in the meantime.
    supersedes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_action(cls, data):
        """Rules written before `actions` existed are on disk and must load.

        Converted, not repaired: a legacy `label` rule has no label to recover,
        so it becomes an empty-params template and is refused at bind time with
        a message naming the rule. Dropping it silently would delete something
        the owner taught; executing it silently is the KeyError.
        """
        if isinstance(data, dict) and "action" in data and "actions" not in data:
            data = dict(data)
            data["actions"] = [{"kind": data.pop("action"), "params": {}}]
        return data

    @property
    def summary(self) -> str:
        """The digest's own vocabulary: `label(recruiter), archive`."""
        return ", ".join(
            f"{a.kind}({a.params['label']})" if a.params.get("label") else a.kind
            for a in self.actions) or "none"

    @property
    def precision(self) -> Optional[float]:
        """Share of firings that were NOT overridden, or None if untested.

        None rather than 1.0 at zero hits: an untested rule is unknown, and
        reporting perfect precision would rank it above a rule that has actually
        been proven.
        """
        if self.hit_count == 0:
            return None
        return max(0.0, (self.hit_count - self.override_count) / self.hit_count)


class ReviewItem(BaseModel):
    thread_id: str
    # What the model thinks the mail IS, not just what it will do about it.
    # Dropped when Decision became ReviewItem, so the review UI could show the
    # action but never the classification behind it - which is the more useful
    # of the two when you are deciding whether the judgement is right.
    category: str = ""
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


class HeldItem(BaseModel):
    """One proposal waiting on the owner, persisted outside any single run.

    `first_held_at` is what lets the digest say "waiting since Tue 8:00". It is
    set once and never refreshed, so an item held this morning and still held
    this evening reads as ten hours old rather than brand new - the ageing IS
    the pressure to deal with the queue.

    Embeds the whole ReviewItem rather than flattening its fields: ReviewItem is
    already the renderer's contract, and re-declaring it here would give the
    digest two shapes to render instead of one.
    """
    thread_id: str
    run_id: str
    first_held_at: datetime
    hold_reason: str
    item: ReviewItem


class DoneRecord(BaseModel):
    """One thread a run acted on, and what it did to it.

    Embeds the whole ReviewItem for the reason HeldItem does: ReviewItem is
    already the renderer's contract, and re-declaring subject and sender here
    would give the report two shapes to render instead of one. It also happens
    to carry every field the correction path reads - `category` and `reason`
    come free with it, and `rule_id` says which rule to demote when the owner
    overrules it.

    `actions` is (kind, label) pairs, the shape DoneItem already renders and
    counts, so the panel does not parse a string it just formatted.
    """
    thread_id: str
    item: ReviewItem
    actions: list[tuple[str, Optional[str]]] = Field(default_factory=list)
    rule_id: Optional[str] = None


class RunReport(BaseModel):
    """What one run did, addressed by the run's own id.

    `total` and `remaining` live here rather than being left in graph state so
    a past run's digest header renders identically to a live one, with no
    special case for "this run is not the current one".
    """
    run_id: str
    ran_at: datetime
    total: int = 0
    remaining: int = 0
    done: list[DoneRecord] = Field(default_factory=list)

    @field_validator("ran_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        """Naive in, UTC out. Reports are sorted against each other by this
        field, and comparing a naive datetime to an aware one raises - a crash
        in the one place whose whole job is to still be there after a restart.
        """
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class ReviewResponse(BaseModel):
    decisions: dict[str, Verdict] = Field(default_factory=dict)
    edits: dict[str, list[Action]] = Field(default_factory=dict)
    instructions: list[str] = Field(default_factory=list)


class AuditRecord(BaseModel):
    # Identity, so a record can be referenced later - by an undo, or by a UI
    # offering one. Defaults to "" rather than a generated uuid: records written
    # before this field existed have no id, and minting one on every read would
    # invent an identity that changes each time the log is parsed. No id means
    # "cannot be undone", which is honest.
    id: str = ""
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
    # Set on an undo record, naming the record it reversed. The original is
    # never mutated - append-only means the evidence that the action happened
    # survives alongside the evidence that it was taken back.
    undoes: Optional[str] = None

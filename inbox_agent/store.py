# inbox_agent/store.py
"""Preference memory over a LangGraph BaseStore (spec section 5.1).

The schema is owned here rather than inherited from a memory SDK so that every
rule carries its own provenance - which correction produced it, how often it has
fired, whether it was ever overridden. That is what makes 'why did it do that?'
answerable months later, and it is why BaseStore beats a managed service for
Stage A. The interface is deliberately narrow so Mem0 can sit behind it later.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from langgraph.store.memory import InMemoryStore

from .models import ActionKind, Rule, Thread

RULES_NS = ("prefs", "rules")

# BaseStore.search() defaults to limit=10. rules() pages through with an
# explicit limit and offset until a page comes back short, so the rule set is
# read in full regardless of how many rules exist - not just up to whatever
# magic number happens to be passed. A truncated read here would silently
# drop rules 11+ with no error, which is exactly the failure this system's
# audit design exists to prevent.
_SEARCH_PAGE_SIZE = 1000


def build_store(embeddings=None, dims: int = 768) -> InMemoryStore:
    """A store with optional semantic search over rule text.

    Embeddings are optional so the test suite runs without Ollama. With them,
    `matching()` can be extended to fuzzy retrieval; exact scope matching is the
    Stage A path and needs no vectors.
    """
    if embeddings is None:
        return InMemoryStore()
    return InMemoryStore(index={"embed": embeddings, "dims": dims, "fields": ["text"]})


def rule_from_correction(thread: Thread, action: ActionKind, note: str) -> Rule:
    """Turn one human correction into a durable, attributable rule."""
    return Rule(
        id=f"r-{uuid.uuid4().hex[:8]}",
        scope="sender",
        pattern=thread.sender.lower(),
        action=action,
        provenance=note,
        created_at=datetime.now(timezone.utc),
    )


class PreferenceStore:
    def __init__(self, store: InMemoryStore):
        self._store = store

    def add_rule(self, rule: Rule) -> Rule:
        self._store.put(
            RULES_NS, rule.id,
            {"rule": rule.model_dump(mode="json"),
             "text": f"{rule.scope} {rule.pattern} -> {rule.action}. {rule.provenance}"},
        )
        return rule

    def rules(self) -> list[Rule]:
        """All stored rules, regardless of how many there are.

        `InMemoryStore.search()` defaults to limit=10, which would otherwise
        silently truncate the rule set as it grows past that default. Paginate
        with an explicit limit/offset until a page comes back short of a full
        page, which is the correct end-of-results signal for any rule count.
        """
        out: list[Rule] = []
        offset = 0
        while True:
            page = self._store.search(RULES_NS, limit=_SEARCH_PAGE_SIZE, offset=offset)
            out.extend(Rule.model_validate(item.value["rule"]) for item in page)
            if len(page) < _SEARCH_PAGE_SIZE:
                break
            offset += _SEARCH_PAGE_SIZE
        return out

    def _put(self, rule: Rule) -> None:
        self._store.put(
            RULES_NS, rule.id,
            {"rule": rule.model_dump(mode="json"),
             "text": f"{rule.scope} {rule.pattern} -> {rule.action}. {rule.provenance}"},
        )

    def _get(self, rule_id: str) -> Optional[Rule]:
        item = self._store.get(RULES_NS, rule_id)
        return Rule.model_validate(item.value["rule"]) if item else None

    def matching(self, thread: Thread) -> list[Rule]:
        """Active rules that apply to this thread. Overridden rules never match."""
        out = []
        for rule in self.rules():
            if rule.overridden:
                continue
            if rule.scope == "sender" and rule.pattern == thread.sender.lower():
                out.append(rule)
            elif rule.scope == "domain" and rule.pattern == thread.sender_domain:
                out.append(rule)
            elif rule.scope == "fingerprint" and rule.pattern == thread.fingerprint:
                out.append(rule)
            elif rule.scope == "subject" and rule.pattern.lower() in thread.subject.lower():
                out.append(rule)
        return out

    def record_hit(self, rule_id: str) -> None:
        rule = self._get(rule_id)
        if rule:
            rule.hit_count += 1
            self._put(rule)

    def mark_overridden(self, rule_id: str) -> None:
        """Kept, not deleted: a rule the owner overruled is part of the record."""
        rule = self._get(rule_id)
        if rule:
            rule.overridden = True
            self._put(rule)

    def delete_rule(self, rule_id: str) -> None:
        self._store.delete(RULES_NS, rule_id)

    def as_table(self) -> list[dict]:
        return [
            {"id": r.id, "scope": r.scope, "pattern": r.pattern, "action": r.action,
             "hit_count": r.hit_count, "overridden": r.overridden,
             "provenance": r.provenance}
            for r in sorted(self.rules(), key=lambda r: r.created_at)
        ]

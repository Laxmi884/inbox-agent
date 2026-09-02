# inbox_agent/store.py
"""Preference memory over a LangGraph BaseStore (spec section 5.1).

The schema is owned here rather than inherited from a memory SDK so that every
rule carries its own provenance - which correction produced it, how often it has
fired, whether it was ever overridden. That is what makes 'why did it do that?'
answerable months later, and it is why BaseStore beats a managed service for
Stage A. The interface is deliberately narrow so Mem0 can sit behind it later.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

from .models import ActionKind, ActionTemplate, HeldItem, ReviewItem, Rule, Thread

RULES_NS = ("prefs", "rules")
INSTRUCTIONS_NS = ("prefs", "instructions")
HELD_NS = ("held", "items")

# BaseStore.search() defaults to limit=10. rules() pages through with an
# explicit limit and offset until a page comes back short, so the rule set is
# read in full regardless of how many rules exist - not just up to whatever
# magic number happens to be passed. A truncated read here would silently
# drop rules 11+ with no error, which is exactly the failure this system's
# audit design exists to prevent.
_SEARCH_PAGE_SIZE = 1000


def _search_all(store, namespace: tuple[str, ...]) -> list:
    """Every entry in a namespace, not just the first page.

    BaseStore.search() defaults to limit=10. Both stores in this module read
    collections that grow past that, and a silently truncated read would drop
    rules 11+ or hide queued work the owner is waiting on - exactly the
    invisible failure this system's audit design exists to prevent. Paginate
    with an explicit limit/offset until a page comes back short, which is the
    correct end-of-results signal for any count.
    """
    out: list = []
    offset = 0
    while True:
        page = store.search(namespace, limit=_SEARCH_PAGE_SIZE, offset=offset)
        out.extend(page)
        if len(page) < _SEARCH_PAGE_SIZE:
            break
        offset += _SEARCH_PAGE_SIZE
    return out


def build_store(embeddings=None, dims: int = 768) -> InMemoryStore:
    """A store with optional semantic search over rule text.

    Embeddings are optional so the test suite runs without Ollama. With them,
    `matching()` can be extended to fuzzy retrieval; exact scope matching is the
    Stage A path and needs no vectors.
    """
    if embeddings is None:
        return InMemoryStore()
    return InMemoryStore(index=_index(embeddings, dims))


def _index(embeddings, dims: int) -> Optional[dict]:
    """Index config, or None when there is nothing to embed with.

    Shared by both builders so an in-memory store and a SQLite one are indexed
    on the same terms - otherwise a rule found by semantic search in the
    notebook could be missed by the bot, and the difference would be invisible.
    """
    if embeddings is None:
        return None
    return {"embed": embeddings, "dims": dims, "fields": ["text"]}


def open_store(path: Path | str, embeddings=None, dims: int = 768) -> SqliteStore:
    """A store that outlives the process, at `path`.

    `build_store` is memory: right for the suite, for the notebook, and for
    anything whose lifetime is one run. It is wrong for the bot, and became
    dangerous rather than merely lossy when `mark_triaged` landed. Every thread
    a run processes leaves the fetch query, held ones included, and `fetch` uses
    `settings.inbox_query` in both modes - so an item lost from the queue is not
    re-fetched by `/triage`, and `/backlog` will not find it either. On a
    restart the owner would not see a shorter digest; they would have threads
    that no longer exist as far as the agent is concerned, recoverable only by
    searching `label:agent/triaged` in Gmail by hand.

    SQLite rather than a service for the same reason the audit log is a file:
    it is one path, it is inspectable with tools the owner already has, and it
    has no operational story to get wrong. Same BaseStore interface either way,
    so `PreferenceStore` and `HeldQueue` are untouched by which one they get -
    which is the whole reason they were written against the interface.

    Autocommit (`isolation_level=None`) because SqliteStore opens its own
    transactions; the default would nest them and every write would raise
    `cannot start a transaction within a transaction`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn, index=_index(embeddings, dims))
    store.setup()   # migrations, idempotent
    return store


# A rule that is wrong this often is worse than no rule: it produces confident,
# citable, wrong decisions, and confidence is exactly what makes them hard to
# catch. Demoted rules are kept, never deleted - they are part of the record of
# why past actions happened.
MIN_PRECISION = 0.5
MIN_HITS_BEFORE_DEMOTION = 4

# Addresses that are structurally incapable of holding a conversation. Mail from
# these is bulk by construction, so a sender-scoped rule is safe: there is no
# "but sometimes this person writes to me personally" case to worry about.
_NOREPLY_MARKERS = ("noreply", "no-reply", "no_reply", "donotreply",
                    "do-not-reply", "notifications", "mailer-daemon")


def choose_scope(thread: Thread, corpus: Optional[list[Thread]] = None
                 ) -> tuple[str, str]:
    """Pick the narrowest scope that still generalises past this one message.

    Everything used to be sender-scoped, which meant correcting one job alert
    taught that exact address and nothing else - and, worse, would bury a
    genuinely useful mail from a sender that mostly sends noise.

    The rule of thumb: a no-reply address cannot hold a conversation, so
    everything it sends is bulk and the SENDER is the right unit. A human
    address can send anything, so the correction is about this KIND of mail from
    them, not about them - which is what fingerprint (sender + digit-stripped
    subject shape) and subject scopes capture.
    """
    sender = (thread.sender or "").lower().strip()
    subject = (thread.subject or "").strip()

    # EVIDENCE BEFORE HEURISTIC. The no-reply shortcut below is a guess; the
    # corpus is a fact. Checking noreply first was wrong in exactly the case
    # that motivated this: a no-reply address that sends job alerts AND, now and
    # then, something worth reading. Sender-scoping that buries both.
    if corpus and subject:
        shape_matches = sum(1 for t in corpus
                            if t.sender.lower() == sender
                            and t.fingerprint == thread.fingerprint)
        if shape_matches >= 2:
            return "fingerprint", thread.fingerprint
        same_sender = sum(1 for t in corpus if t.sender.lower() == sender)
        if same_sender >= 2:
            # Several mails, different shapes - the subject line is the
            # distinguishing feature, so scope to it rather than to the person.
            return "subject", subject[:60].lower()

    # No corpus evidence. A no-reply address cannot hold a conversation, so
    # everything it sends is bulk and the sender is a safe unit; a human address
    # gets the same treatment only because one correction is not enough to infer
    # anything narrower.
    if sender:
        return "sender", sender
    if subject:
        return "subject", subject[:60].lower()
    return "sender", "unknown"


def rule_from_correction(thread: Thread, actions: list[ActionTemplate], note: str,
                         *, rejected: Optional[ActionKind] = None,
                         corpus: Optional[list[Thread]] = None,
                         supersedes: Optional[str] = None) -> Rule:
    """Turn one human correction into a durable, attributable rule.

    `actions` is the sequence to take next time, not a single kind: the
    correction the owner most wants to teach - "label it, but leave it in the
    inbox" - is a statement about a sequence, and a single kind could not even
    say which label to apply.

    `rejected` records a bare "not this" - a reject with no replacement. The
    spec counts every reject OR edit as a candidate rule; only edits used to
    produce one, so a Skip taught nothing at all.

    `supersedes` names the rule being corrected, when there is one. Keyword-only
    and defaulted: a correction of the model's own judgement overrides no rule,
    and every existing caller passes three positional arguments.
    """
    scope, pattern = choose_scope(thread, corpus)
    return Rule(
        id=f"r-{uuid.uuid4().hex[:8]}",
        scope=scope,
        pattern=pattern,
        actions=list(actions),
        rejected_action=rejected,
        supersedes=supersedes,
        provenance=note,
        created_at=datetime.now(timezone.utc),
    )


class PreferenceStore:
    def __init__(self, store: InMemoryStore):
        self._store = store

    def add_rule(self, rule: Rule) -> Rule:
        """Store a rule, retiring any live rule it replaces.

        Same scope AND same pattern means the two decide the same mail, so a
        newer one is a replacement rather than a second opinion. Without this
        the older rule stayed live and merely lost the `max(created_at)`
        tie-break in `matching` - which hid the problem rather than avoiding it.
        The moment the newer rule is demoted, `_live_rules` drops it and the
        older takes over, resurrecting the correction the owner threw away. And
        it does so permanently: a rule that never fires has `hit_count` 0, so it
        can never meet MIN_HITS_BEFORE_DEMOTION and can never itself be demoted.

        Retired, not penalised. `override_count` stays untouched because it
        means "this rule fired and the owner undid its decision", and a
        superseded rule may never have fired at all - counting an override there
        would record a disagreement with a decision that was never made, into a
        `precision` that divides by `hit_count`.

        Retired, not deleted, for the reason `mark_overridden` already gives: a
        rule the owner overruled is part of the record. It stops deciding; it
        does not stop existing.

        Scope is part of the test, not just pattern: a `category` rule for
        `receipt` decides different mail from a `sender` rule that happens to
        apply the receipt label, so neither may retire the other.
        """
        replaced = [r for r in self._live_rules()
                    if r.id != rule.id
                    and r.scope == rule.scope
                    and r.pattern == rule.pattern]
        for existing in replaced:
            self.mark_overridden(existing.id)
        # A caller that already named what it supersedes knows better than this
        # does: it saw the rule that actually produced the item on screen, which
        # may be a different scope from the one being written now.
        if replaced and not rule.supersedes:
            rule = rule.model_copy(
                update={"supersedes": max(replaced, key=lambda r: r.created_at).id})
        self._put(rule)
        return rule

    def rules(self) -> list[Rule]:
        """All stored rules, regardless of how many there are."""
        page = _search_all(self._store, RULES_NS)
        return [Rule.model_validate(item.value["rule"]) for item in page]

    def _put(self, rule: Rule) -> None:
        self._store.put(
            RULES_NS, rule.id,
            {"rule": rule.model_dump(mode="json"),
             "text": f"{rule.scope} {rule.pattern} -> {rule.summary}. {rule.provenance}"},
        )

    def _get(self, rule_id: str) -> Optional[Rule]:
        item = self._store.get(RULES_NS, rule_id)
        return Rule.model_validate(item.value["rule"]) if item else None

    def _live_rules(self) -> list[Rule]:
        """Rules still entitled to decide anything.

        Overridden rules never match, and neither do rules proven unreliable: a
        rule below MIN_PRECISION after enough firings is worse than no rule,
        because it produces confident and citable wrong decisions. Factored out
        so the thread lookup and the category lookup cannot drift apart on the
        one question that has nothing to do with which kind of rule it is.
        """
        return [r for r in self.rules()
                if not r.overridden
                and not (r.hit_count >= MIN_HITS_BEFORE_DEMOTION
                         and (r.precision or 0.0) < MIN_PRECISION)]

    def matching(self, thread: Thread) -> list[Rule]:
        """Active rules that apply to this thread.

        Category-scoped rules are excluded structurally, not by omission: a
        category is the model's conclusion, and this runs before the model. See
        matching_category.
        """
        out = []
        for rule in self._live_rules():
            if rule.scope == "category":
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

    def matching_category(self, category: str) -> list[Rule]:
        """Rules about a conclusion rather than about a thread.

        Separate from matching() because the input differs in kind: a category
        is what the model decided, so this cannot run until it has. Same
        demotion filter - a category rule reaches every thread of that category,
        so a rule that is wrong half the time does more damage here than a
        sender rule ever could.
        """
        return [r for r in self._live_rules()
                if r.scope == "category" and r.pattern == category]

    def record_hit(self, rule_id: str) -> None:
        rule = self._get(rule_id)
        if rule:
            rule.hit_count += 1
            self._put(rule)

    def record_override(self, rule_id: str) -> None:
        """The owner undid or corrected something this rule decided.

        Counted rather than latched, so a rule that is right 90% of the time is
        distinguishable from one that is simply broken.
        """
        rule = self._get(rule_id)
        if rule:
            rule.override_count += 1
            self._put(rule)

    def mark_overridden(self, rule_id: str) -> None:
        """Kept, not deleted: a rule the owner overruled is part of the record."""
        rule = self._get(rule_id)
        if rule:
            rule.overridden = True
            self._put(rule)

    def delete_rule(self, rule_id: str) -> None:
        self._store.delete(RULES_NS, rule_id)

    # --- explicit instructions ---------------------------------------------
    # Mechanism 1 of the spec's "how it learns", and the highest authority of
    # the three: "Always archive these." Stored separately from rules because a
    # rule is a derived, pattern-matched inference while an instruction is the
    # owner speaking directly - different provenance, different authority.

    def add_instruction(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if any(text.lower() == existing.lower() for existing in self.instructions()):
            return
        key = f"i-{uuid.uuid4().hex[:8]}"
        self._store.put(INSTRUCTIONS_NS, key,
                        {"text": text,
                         "created_at": datetime.now(timezone.utc).isoformat()})

    def instructions(self) -> list[str]:
        """Oldest first, so the prompt reads in the order they were given."""
        items = self._store.search(INSTRUCTIONS_NS, limit=_SEARCH_PAGE_SIZE)
        rows = [(i.value.get("created_at", ""), i.value.get("text", ""))
                for i in items]
        return [text for _, text in sorted(rows) if text]

    def as_table(self) -> list[dict]:
        return [
            {"id": r.id, "scope": r.scope, "pattern": r.pattern, "action": r.summary,
             "hit_count": r.hit_count, "overrides": r.override_count,
             "precision": ("-" if r.precision is None else f"{r.precision:.2f}"),
             "overridden": r.overridden,
             "provenance": r.provenance,
             "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")}
            for r in sorted(self.rules(), key=lambda r: r.created_at)
        ]


class HeldQueue:
    """Proposals waiting on the owner, across runs.

    Separate from PreferenceStore because the lifetimes differ: a rule is
    permanent knowledge, a held item is a piece of work in flight. The class
    itself is store-agnostic - any BaseStore, its own namespace - so the two
    could in principle share one store the way they already share one kind of
    backend.

    Callers wire them with two separate store instances instead. Rules go in
    a store built with embeddings (`build_store(get_embeddings())`) so rule
    text is semantically searchable; a held item's payload has no `text`
    field for that index to key off, so sharing that store would spend real
    embedding calls on a payload the index has nothing to do with. One
    interface, one persistence mechanism to reason about later - just not one
    instance in practice.
    """

    def __init__(self, store):
        self._store = store

    def add(self, item: ReviewItem, *, run_id: str, reason: str,
            now: Optional[datetime] = None) -> HeldItem:
        """Hold `item`, preserving the original wait time if already held.

        Idempotent on thread_id: a thread the agent holds twice is one item that
        has been waiting since the first time, not two items. The content and
        the reason ARE refreshed, so a re-classified thread shows its current
        proposal.
        """
        existing = self.get(item.thread_id)
        held = HeldItem(
            thread_id=item.thread_id,
            run_id=run_id,
            first_held_at=existing.first_held_at if existing
            else (now or datetime.now(timezone.utc)),
            hold_reason=reason,
            item=item,
        )
        self._store.put(HELD_NS, held.thread_id,
                        {"held": held.model_dump(mode="json")})
        return held

    def get(self, thread_id: str) -> Optional[HeldItem]:
        entry = self._store.get(HELD_NS, thread_id)
        return HeldItem.model_validate(entry.value["held"]) if entry else None

    def remove(self, thread_id: str) -> None:
        """Absent is not an error: a double-tap must not raise at the transport."""
        self._store.delete(HELD_NS, thread_id)

    def all(self) -> list[HeldItem]:
        """Everything held, oldest first."""
        page = _search_all(self._store, HELD_NS)
        items = [HeldItem.model_validate(entry.value["held"]) for entry in page]
        return sorted(items, key=lambda h: h.first_held_at)

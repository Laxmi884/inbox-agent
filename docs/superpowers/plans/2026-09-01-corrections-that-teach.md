# Corrections That Teach — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the owner correct an action the agent already took, and have that
correction become a durable rule — including rules about a whole category, which
is the case that exposed the gap.

**Architecture:** A rule stops being one action kind and becomes a sequence of
action templates, so it can name its own label. `category` joins the scopes, and
because a category is only known after the model runs, category rules fire in a
new `apply_rules` node between `triage` and `propose` rather than in `prefilter`.
The digest's numbered buttons open an item view whose verdicts write those rules.

**Tech Stack:** Python 3.11+, Pydantic v2, LangGraph (StateGraph, BaseStore),
pytest, raw Telegram Bot API over httpx.

**Spec:** `docs/superpowers/specs/2026-09-01-corrections-that-teach-design.md`

## Global Constraints

- Telegram caps message text at **4096 characters** and `callback_data` at
  **64 bytes**. Protocol, not preference.
- Telegram wraps proportional text. **Never pad columns.** A terminal preview is
  not a fair test of a phone layout — this project has made that mistake twice.
- Every keyboard carries the **digest id**; a callback without a current one is
  refused (`bot.py:_on_callback`).
- `Action.thread_id` is required, so a rule must NOT store `Action`. It stores
  templates and `prefilter` binds them to a thread.
- A correction **teaches and does not reverse**. Nothing in this plan calls
  `undo_action()`.
- The action chokepoint (`execute_action`), the deny-list and `INBOX_DRY_RUN`
  are unchanged. Nothing here grants a new capability.
- Run the **full suite** (`pytest -q`) before every commit, not just the new test.

---

### Task 1: A rule carries action templates

`Rule.action` is a single `ActionKind`, which is why a label rule cannot name its
label. `Action` is the wrong type to store — it requires a `thread_id`, and a
rule is a template that has not met a thread yet.

**Files:**
- Modify: `inbox_agent/models.py:22-26` (add `ActionTemplate`), `:68-96` (`Rule`)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `ActionTemplate(kind: str, params: dict)`;
  `Rule.actions: list[ActionTemplate]`; `Rule.summary: str`;
  a `model_validator(mode="before")` accepting a legacy `action` key.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
from datetime import datetime, timezone

from inbox_agent.models import ActionTemplate, Rule

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def rule(**kw):
    base = dict(id="r-1", scope="sender", pattern="a@b.com",
                actions=[ActionTemplate(kind="archive")],
                provenance="user archived it", created_at=NOW)
    return Rule(**(base | kw))


def test_a_rule_carries_a_sequence_of_actions():
    """label-then-archive is the sequence the model produces most often, and a
    rule that cannot say it cannot teach the correction the owner makes."""
    r = rule(actions=[ActionTemplate(kind="label", params={"label": "recruiter"}),
                      ActionTemplate(kind="archive")])
    assert [a.kind for a in r.actions] == ["label", "archive"]
    assert r.actions[0].params["label"] == "recruiter"


def test_a_label_template_carries_its_label():
    """The whole point. _dispatch reads params['label']; a rule that omits it
    raises KeyError against a live mailbox."""
    r = rule(actions=[ActionTemplate(kind="label", params={"label": "receipt"})])
    assert r.actions[0].params == {"label": "receipt"}


def test_a_legacy_action_rule_still_loads():
    """Rules are on disk as of this branch. A stored rule written before this
    change must not become unreadable, or the store fails to open at all."""
    r = Rule.model_validate({"id": "r-old", "scope": "sender", "pattern": "a@b.com",
                             "action": "archive", "provenance": "p",
                             "created_at": NOW})
    assert [a.kind for a in r.actions] == ["archive"]


def test_a_legacy_label_rule_loads_without_a_label():
    """It converts, and it is broken - that is the bug this plan is fixing, and
    it must be visible rather than raise at execution time. Task 2 refuses it."""
    r = Rule.model_validate({"id": "r-old", "scope": "sender", "pattern": "a@b.com",
                             "action": "label", "provenance": "p",
                             "created_at": NOW})
    assert r.actions == [ActionTemplate(kind="label", params={})]


def test_summary_reads_like_the_digest():
    """Used in the rule table and in the embedding text, so it has to match the
    vocabulary the owner already sees in the digest: label(recruiter), archive."""
    r = rule(actions=[ActionTemplate(kind="label", params={"label": "recruiter"}),
                      ActionTemplate(kind="archive")])
    assert r.summary == "label(recruiter), archive"


def test_category_is_a_scope():
    assert rule(scope="category", pattern="newsletter_valuable").scope == "category"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_models.py -q`
Expected: FAIL — `ImportError: cannot import name 'ActionTemplate'`

- [ ] **Step 3: Implement**

```python
# inbox_agent/models.py — after Action

class ActionTemplate(BaseModel):
    """An action with no thread attached: what a rule stores.

    `Action` requires a thread_id, correctly - an action is always about one
    thread, and the chokepoint audits it that way. A rule is the shape of an
    action the owner wants taken on mail it has not seen yet, so it carries
    everything except the thread. `prefilter` binds the two.
    """
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
```

```python
# inbox_agent/models.py — Rule

class Rule(BaseModel):
    id: str
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
    rejected_action: Optional[ActionKind] = None
    override_count: int = 0

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_action(cls, data):
        """Rules written before actions existed are on disk and must still load.

        Converted, not repaired: a legacy `label` rule has no label to recover,
        so it becomes an empty-params template and is refused at bind time with
        a message naming the rule (Task 2). Silently dropping it would delete
        something the owner taught; silently executing it is the KeyError.
        """
        if isinstance(data, dict) and "action" in data and "actions" not in data:
            data = dict(data)
            data["actions"] = [{"kind": data.pop("action"), "params": {}}]
        return data

    @property
    def summary(self) -> str:
        """The digest's vocabulary: `label(recruiter), archive`."""
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
```

Add `model_validator` to the pydantic import at the top of the file.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_models.py -q`
Expected: PASS. The rest of the suite is expected to FAIL here — `prefilter` and
`store` still read `rule.action`. Task 2 fixes them; do not commit yet.

- [ ] **Step 5: Commit after Task 2.** This task and the next are one green
      suite; committing here would commit a broken tree.

---

### Task 2: Bind templates to threads, and refuse the unbindable

`prefilter.py:32` builds `Action(kind=rule.action, thread_id=thread.id)` with no
params. That line is the bug. Every other reader of `rule.action` moves to
`rule.summary`.

**Files:**
- Modify: `inbox_agent/prefilter.py:29-37`, `inbox_agent/store.py:208`,
  `:290-299` (`as_table`), `:170-188` (`rule_from_correction`)
- Modify: `inbox_agent/graph.py` (`learn_from_response`, wherever it calls
  `rule_from_correction`)
- Test: `tests/test_prefilter.py`, `tests/test_store.py`, `tests/test_learning.py`,
  `tests/test_graph.py`

**Interfaces:**
- Consumes: `ActionTemplate`, `Rule.actions`, `Rule.summary` (Task 1)
- Produces: `rule_from_correction(thread, actions: list[ActionTemplate], note, *, rejected=None, corpus=None)`;
  `UnbindableRuleError`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prefilter.py
import pytest

from inbox_agent.audit import AuditLog, ExecutionContext, execute_action
from inbox_agent.config import ALWAYS_FORBIDDEN, Settings
from inbox_agent.models import ActionTemplate, Rule
from inbox_agent.prefilter import UnbindableRuleError, prefilter


def test_a_label_rule_applies_the_label_it_names(tmp_path):
    """The regression. Before this, a label rule produced params={} and raised
    KeyError inside _dispatch - reported as 'simulated' under dry-run, so the
    only place it could ever be noticed was a live mailbox."""
    s = store()
    s.add_rule(Rule(id="r-1", scope="sender", pattern="deals@shop.com",
                    actions=[ActionTemplate(kind="label",
                                            params={"label": "promotion"})],
                    provenance="p", created_at=NOW))
    decided, _ = prefilter([thread()], s)
    action = decided[0].actions[0]
    assert action.kind == "label"
    assert action.thread_id == "t1"
    assert action.params == {"label": "promotion"}


def test_a_rule_binds_every_action_in_its_sequence():
    s = store()
    s.add_rule(Rule(id="r-1", scope="sender", pattern="deals@shop.com",
                    actions=[ActionTemplate(kind="label", params={"label": "recruiter"}),
                             ActionTemplate(kind="archive")],
                    provenance="p", created_at=NOW))
    decided, _ = prefilter([thread()], s)
    assert [(a.kind, a.thread_id) for a in decided[0].actions] == [
        ("label", "t1"), ("archive", "t1")]


def test_a_legacy_label_rule_is_refused_by_name_not_by_keyerror():
    """A converted legacy rule cannot say which label. Executing it raises
    KeyError halfway through a run, which the bot reports as 'the run did not
    finish'. Refusing at bind time names the rule the owner has to re-teach."""
    s = store()
    s.add_rule(Rule.model_validate(
        {"id": "r-old", "scope": "sender", "pattern": "deals@shop.com",
         "action": "label", "provenance": "p", "created_at": NOW}))
    with pytest.raises(UnbindableRuleError) as exc:
        prefilter([thread()], s)
    assert "r-old" in str(exc.value)


def test_the_reason_names_what_the_rule_does():
    s = store()
    s.add_rule(Rule(id="r-1", scope="sender", pattern="deals@shop.com",
                    actions=[ActionTemplate(kind="label", params={"label": "recruiter"}),
                             ActionTemplate(kind="archive")],
                    provenance="p", created_at=NOW))
    decided, _ = prefilter([thread()], s)
    assert "label(recruiter), archive" in decided[0].reason
```

`thread()`, `store()` and `NOW` already exist in `tests/test_prefilter.py`; reuse
them rather than redefining.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_prefilter.py -q`
Expected: FAIL — `ImportError: cannot import name 'UnbindableRuleError'`

- [ ] **Step 3: Implement**

```python
# inbox_agent/prefilter.py

class UnbindableRuleError(Exception):
    """A stored rule cannot be turned into an action against a thread.

    Raised at bind time rather than at execution time. The only rules that hit
    this are legacy `label` rules converted by Rule's validator, which have no
    label to apply: _dispatch would raise KeyError mid-run, after earlier
    actions had already gone through the chokepoint. Naming the rule here tells
    the owner exactly which one to teach again.
    """


def _bind(rule: Rule, thread: Thread) -> list[Action]:
    out = []
    for template in rule.actions:
        if template.kind in ("label", "unlabel") and not template.params.get("label"):
            raise UnbindableRuleError(
                f"rule {rule.id} ({rule.scope} {rule.pattern!r}) says "
                f"{template.kind!r} but names no label. It predates labelled "
                f"rules; correct one of these threads again to re-teach it.")
        out.append(Action(kind=template.kind, thread_id=thread.id,
                          params=dict(template.params)))
    return out
```

```python
# inbox_agent/prefilter.py — inside prefilter(), replacing lines 29-37
        decided.append(Decision(
            thread_id=thread.id,
            category="rule_match",
            actions=_bind(rule, thread),
            reason=f"matched {rule.scope} rule {rule.pattern!r} -> {rule.summary}",
            confidence=1.0,
            source="rule",
            rule_id=rule.id,
        ))
```

```python
# inbox_agent/store.py:208 — the embedding text
             "text": f"{rule.scope} {rule.pattern} -> {rule.summary}. {rule.provenance}"},
```

```python
# inbox_agent/store.py — as_table
            {"id": r.id, "scope": r.scope, "pattern": r.pattern, "action": r.summary,
```

```python
# inbox_agent/store.py — rule_from_correction
def rule_from_correction(thread: Thread, actions: list[ActionTemplate], note: str,
                         *, rejected: Optional[ActionKind] = None,
                         corpus: Optional[list[Thread]] = None) -> Rule:
    """Turn one human correction into a durable, attributable rule.

    `actions` is the sequence to take next time, not a single kind: the
    correction the owner most wants to teach - "label it but leave it in the
    inbox" - is a statement about a sequence.

    `rejected` records a bare "not this" - a reject with no replacement. The
    spec counts every reject OR edit as a candidate rule; only edits used to
    produce one, so a Skip taught nothing at all.
    """
    scope, pattern = choose_scope(thread, corpus)
    return Rule(
        id=f"r-{uuid.uuid4().hex[:8]}",
        scope=scope,
        pattern=pattern,
        actions=list(actions),
        rejected_action=rejected,
        provenance=note,
        created_at=datetime.now(timezone.utc),
    )
```

Then sweep every caller. `learn_from_response` in `graph.py` passes a kind today;
it passes `[ActionTemplate(kind=..., params=...)]` now, carrying the label from
the edit when the edit names one. Update the assertions in `tests/test_learning.py`
and `tests/test_graph.py` that read `rules[0].action` to read `rules[0].actions`
or `.summary`.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS, every test.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent tests
git commit -m "A rule can say which label, and how many actions"
```

---

### Task 3: Category-scoped rules, and a store that can find them

A category rule cannot be matched against a raw thread — `matching()` compares
sender, domain, fingerprint and subject. It needs its own lookup, because its
input is a `Decision`, not a `Thread`.

**Files:**
- Modify: `inbox_agent/store.py:215-243` (`matching`), add `matching_category`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Rule.actions`, the `category` scope (Task 1)
- Produces: `PreferenceStore.matching_category(category: str) -> list[Rule]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py
from inbox_agent.models import ActionTemplate, Rule


def category_rule(category="newsletter_valuable", rid="r-cat"):
    return Rule(id=rid, scope="category", pattern=category,
                actions=[ActionTemplate(kind="label", params={"label": category})],
                provenance="owner said keep it in the inbox",
                created_at=datetime.now(timezone.utc))


def test_a_category_rule_is_found_by_its_category():
    s = store()
    s.add_rule(category_rule())
    assert [r.id for r in s.matching_category("newsletter_valuable")] == ["r-cat"]


def test_a_category_rule_does_not_match_another_category():
    s = store()
    s.add_rule(category_rule())
    assert s.matching_category("promotion") == []


def test_a_category_rule_never_matches_a_thread():
    """It cannot: the category is not a property of the thread, it is the
    model's conclusion about it. matching() runs before the model."""
    s = store()
    s.add_rule(category_rule())
    assert s.matching(thread()) == []


def test_a_demoted_category_rule_stops_matching():
    """Same precision discipline as every other rule - a category rule reaches
    more mail, so a bad one is worse."""
    s = store()
    r = s.add_rule(category_rule())
    for _ in range(4):
        s.record_hit(r.id)
        s.record_override(r.id)
    assert s.matching_category("newsletter_valuable") == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_store.py -q`
Expected: FAIL — `AttributeError: 'PreferenceStore' object has no attribute 'matching_category'`

- [ ] **Step 3: Implement**

Read `matching()` first: it already filters demoted rules and returns them
sorted. `matching_category` applies the same filter to a different predicate, so
factor the shared filter rather than duplicating it.

```python
# inbox_agent/store.py

    def matching_category(self, category: str) -> list[Rule]:
        """Rules about a conclusion rather than about a thread.

        Separate from matching() because the input is different in kind: a
        category is what the model decided, so this cannot run until it has.
        Same demotion filter - a category rule reaches every thread of that
        category, so a rule that is wrong half the time is worse here than
        anywhere else.
        """
        return [r for r in self._live_rules()
                if r.scope == "category" and r.pattern == category]
```

and `matching()` gains `and r.scope != "category"` so a category rule can never
be matched against a thread by accident.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/store.py tests/test_store.py
git commit -m "Rules can be about a category, and are found by it"
```

---

### Task 4: `apply_rules`, between the model and the proposal

**Files:**
- Modify: `inbox_agent/graph.py:225-241` (after `triage`), and the edge wiring
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `PreferenceStore.matching_category` (Task 3)
- Produces: graph node `apply_rules`; edge `triage -> apply_rules -> propose`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph.py

def test_a_category_rule_rewrites_what_the_model_proposed(wiring):
    """The case that started this: the model is right that it is a valuable
    newsletter and wrong to archive it. The category stands; the actions change."""
    wiring["prefs"].add_rule(Rule(
        id="r-cat", scope="category", pattern="promotion",
        actions=[ActionTemplate(kind="label", params={"label": "promotion"})],
        provenance="keep promotions in the inbox", created_at=NOW))
    out = wiring["graph"].invoke({"limit": 4, "mode": "incremental"}, CONFIG)
    kinds = [r.action for r in wiring["log"].records()
             if r.action != "label" or r.params.get("label") != "agent/triaged"]
    assert "archive" not in kinds, "the category rule did not remove the archive"


def test_a_rewritten_decision_is_credited_to_the_rule(wiring):
    """The digest counts 'came from rules you taught me'. A rewrite is the rule
    deciding, so it has to be attributed or the learning stays invisible."""
    wiring["prefs"].add_rule(Rule(
        id="r-cat", scope="category", pattern="promotion",
        actions=[ActionTemplate(kind="label", params={"label": "promotion"})],
        provenance="p", created_at=NOW))
    wiring["graph"].invoke({"limit": 4, "mode": "incremental"}, CONFIG)
    assert wiring["prefs"].rules()[0].hit_count == 4


def test_a_thread_with_no_category_rule_is_left_alone(wiring):
    out = wiring["graph"].invoke({"limit": 4, "mode": "incremental"}, CONFIG)
    assert any(r.action == "archive" for r in wiring["log"].records())


def test_a_sender_rule_still_short_circuits_the_model(wiring):
    """Precedence, asserted rather than described: a pre-model rule means the
    model never runs, so apply_rules never sees the thread."""
    wiring["prefs"].add_rule(Rule(
        id="r-send", scope="sender", pattern="deals0@shop.com",
        actions=[ActionTemplate(kind="trash")],
        provenance="p", created_at=NOW))
    out = wiring["graph"].invoke({"limit": 4, "mode": "incremental"}, CONFIG)
    decisions = [Decision.model_validate(d) for d in out["decisions"]]
    by_id = {d.thread_id: d for d in decisions}
    assert by_id["t0"].source == "rule"
    assert by_id["t0"].category == "rule_match"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph.py -q`
Expected: FAIL — the archive still happens; `hit_count` is 0.

- [ ] **Step 3: Implement**

```python
# inbox_agent/graph.py — a node, added after triage

    def apply_rules(state: TriageState) -> dict:
        """Rewrite what the model proposed where the owner has taught otherwise.

        This is the second of two places rules fire, and the split is forced by
        the data: prefilter matches properties of the raw thread and runs before
        the model, so a rule about a CATEGORY has nowhere to be applied there -
        the category is the model's conclusion, not the thread's attribute.

        A rewrite, not a re-judgment. The model's category stands and the
        owner's rule decides what happens to mail of that category, which is
        exactly the correction that motivated it: "you were right that it is a
        valuable newsletter, you were wrong to archive it."

        Attributed to the rule (source, rule_id, a recorded hit) so the digest's
        "came from rules you taught me" counts it and precision can move.
        """
        rewritten = []
        for raw in state["decisions"]:
            decision = Decision.model_validate(raw)
            matches = prefs.matching_category(decision.category)
            if matches:
                rule = max(matches, key=lambda r: r.created_at)
                prefs.record_hit(rule.id)
                decision = decision.model_copy(update={
                    "actions": [Action(kind=t.kind, thread_id=decision.thread_id,
                                       params=dict(t.params))
                                for t in rule.actions],
                    "reason": f"{decision.reason} (your rule for "
                              f"{decision.category}: {rule.summary})",
                    "source": "rule",
                    "rule_id": rule.id,
                })
            rewritten.append(decision.model_dump())
        return {"decisions": rewritten}
```

Wire it: `graph.add_node("apply_rules", apply_rules)`, then replace the
`triage -> propose` edge with `triage -> apply_rules` and `apply_rules -> propose`.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/graph.py tests/test_graph.py
git commit -m "Apply category rules after the model, not before it"
```

---

### Task 5: The item view

**Files:**
- Modify: `inbox_agent/telegram/render_tg.py` (add `item_view`)
- Test: `tests/test_tg_render.py`

**Interfaces:**
- Consumes: `DoneItem` (this branch), `HeldItem`, `encode`
- Produces: `item_view(subject, sender, actions_text, why, *, digest_id, index, kind, categories) -> tuple[str, list]`
  where `kind` is `"held"` or `"done"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tg_render.py
from inbox_agent.telegram.render_tg import item_view


def view_args(**kw):
    base = dict(subject="Data Scientist, Fraud at Stripe",
                sender="jobalerts-noreply@linkedin.com",
                actions_text="label(recruiter), archive",
                why="job alert from a no-reply address",
                digest_id="7f2a", index=2, kind="done",
                categories=["recruiter", "promotion"])
    return base | kw


def test_the_item_view_shows_what_happened_and_why():
    text, _ = item_view(**view_args())
    assert "Data Scientist, Fraud at Stripe" in text
    assert "jobalerts-noreply@linkedin.com" in text
    assert "label(recruiter), archive" in text
    assert "job alert from a no-reply address" in text


def test_a_done_item_offers_the_correction_verdicts():
    _, kb = item_view(**view_args(kind="done"))
    labels = [l for row in kb for (l, _) in row]
    assert "Keep in inbox" in labels
    assert "Label as …" in labels
    assert "Trash these instead" in labels
    assert not any("Approve" in l for l in labels), "a done action is not pending"


def test_a_held_item_offers_verdicts_not_corrections():
    """A held item has not happened yet, so the question is approve or not -
    the past tense verdicts would be a lie about work still waiting."""
    _, kb = item_view(**view_args(kind="held"))
    labels = [l for row in kb for (l, _) in row]
    assert "Approve" in labels
    assert "Not this" in labels
    assert "Keep in inbox" not in labels


def test_every_button_carries_the_digest_id():
    """Same staleness rule as everywhere else: positions shift between digests."""
    _, kb = item_view(**view_args())
    for row in kb:
        for _label, data in row:
            assert decode(data).digest_id == "7f2a"


def test_the_view_always_offers_a_way_back():
    _, kb = item_view(**view_args())
    assert "list" in [decode(d).kind for row in kb for (_, d) in row]


def test_the_item_view_never_exceeds_the_telegram_cap():
    text, _ = item_view(**view_args(subject="x" * 500, sender="s" * 500,
                                    why="y" * 3000, actions_text="a" * 500))
    assert len(text) <= TG_MAX_TEXT


def test_a_missing_reason_is_omitted_rather_than_invented():
    text, _ = item_view(**view_args(why=""))
    assert "Why:" not in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_tg_render.py -q`
Expected: FAIL — `ImportError: cannot import name 'item_view'`

- [ ] **Step 3: Implement**

```python
# inbox_agent/telegram/render_tg.py

_WHY_CAP = 300


def item_view(subject: str, sender: str, actions_text: str, why: str, *,
              digest_id: str, index: int, kind: str = "done",
              categories: Sequence[str] = ()) -> tuple[str, list]:
    """One item, opened. The screen the numbered buttons have always implied.

    Serves both lists, with different verbs, because the two states differ in
    tense: a held item has not happened and asks approve-or-not, a done item has
    happened and asks was-that-right. Offering "Keep in inbox" on something not
    yet archived would describe work that does not exist.
    """
    lines = [f"{index + 1}. {_oneline(subject, _SUBJECT_CAP)}",
             _oneline(sender, _SENDER_CAP),
             f"→ {_oneline(actions_text, _ACTION_CAP)}"]
    if why.strip():
        lines += ["", f"Why: {_oneline(why, _WHY_CAP)}"]
    text = "\n".join(lines)[:TG_MAX_TEXT]

    if kind == "held":
        keyboard = [[("Approve", encode("approve", index, digest_id=digest_id)),
                     ("Not this", encode("reject", index, digest_id=digest_id))]]
    else:
        keyboard = [
            [("Keep in inbox", encode("keep", index, digest_id=digest_id)),
             ("Label as …", encode("relabel", index, digest_id=digest_id))],
            [("Trash these instead", encode("teach_trash", index, digest_id=digest_id))],
        ]
    keyboard.append([("↩ Back", encode("list", digest_id=digest_id))])
    return text, keyboard
```

`_ACTION_CAP` already exists (added with the done panel). `keep`, `relabel` and
`teach_trash` are new callback kinds — Task 6 adds them; this task's tests will
fail on `encode` until it does, so write Task 6 first if executing out of order.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/render_tg.py tests/test_tg_render.py
git commit -m "An item view, with verdicts that match the tense"
```

---

### Task 6: Callback kinds for verdicts and scope

**Do this before Task 5's implementation** — `item_view` calls `encode` with
kinds that do not exist yet.

**Files:**
- Modify: `inbox_agent/telegram/callbacks.py:41-56`
- Test: `tests/test_tg_callbacks.py`

**Interfaces:**
- Produces: kinds `keep`, `relabel`, `teach_trash`, `scope_narrow`, `scope_wide`
  with codes `k`, `R`, `X`, `n`… — note `n` is taken by `next`; use `s` and `S`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tg_callbacks.py

def test_the_new_verdict_kinds_round_trip():
    for kind in ("keep", "relabel", "teach_trash", "scope_narrow", "scope_wide"):
        data = encode(kind, 3, digest_id="7f2a")
        intent = decode(data)
        assert intent.kind == kind
        assert intent.index == 3
        assert intent.digest_id == "7f2a"


def test_every_new_kind_fits_the_byte_cap():
    """64 bytes is protocol. An index can reach three digits on a backlog."""
    for kind in ("keep", "relabel", "teach_trash", "scope_narrow", "scope_wide"):
        assert len(encode(kind, 999, digest_id="7f2a").encode()) <= CB_MAX_BYTES


def test_a_relabel_carries_the_category_it_names():
    """Label as … is a two-step: pick the item, then pick the label. The second
    tap has to carry which label without a thread id ever travelling."""
    data = encode("relabel", 3, label_index=2, digest_id="7f2a")
    intent = decode(data)
    assert intent.kind == "relabel"
    assert intent.index == 3
    assert intent.label_index == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_tg_callbacks.py -q`
Expected: FAIL — `decode` returns `noop` for unknown codes.

- [ ] **Step 3: Implement**

Read `callbacks.py` first: `Kind` is a `Literal`, `_CODE_TO_KIND` maps one letter
per kind, and `encode`/`decode` already handle an optional `label_index` for the
existing `label` kind. Extend all three:

```python
Kind = Literal["approve", "reject", "label", "prev", "next", "approve_all",
               "open", "list", "done", "approve_attention",
               # Corrections. Each maps to exactly one action sequence, so what
               # is taught is what the button said.
               "keep", "relabel", "teach_trash",
               # How wide to teach it: this sender, or every thread of this
               # category. Asked because the two are different intentions
               # behind the same tap.
               "scope_narrow", "scope_wide",
               "noop"]

_CODE_TO_KIND: dict[str, Kind] = {
    ...,
    "k": "keep", "R": "relabel", "X": "teach_trash",
    "s": "scope_narrow", "S": "scope_wide",
}
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/telegram/callbacks.py tests/test_tg_callbacks.py
git commit -m "Callback kinds for the correction verdicts"
```

---

### Task 7: The bot writes the rule

**Files:**
- Modify: `inbox_agent/telegram/bot.py` (`_on_callback`, new `_open`,
  `_verdict`, `_write_rule`)
- Test: `tests/test_tg_bot.py`

**Interfaces:**
- Consumes: `item_view` (Task 5), the new kinds (Task 6),
  `rule_from_correction`, `ActionTemplate`, `PreferenceStore.record_override`
- Produces: `Bot._pending: Optional[dict]` — the verdict awaiting a scope answer

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tg_bot.py

def test_tapping_a_number_opens_the_item(bot):
    """The button that did nothing for two sessions."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    assert "Why:" in t.edited[-1]["text"] or "→" in t.edited[-1]["text"]
    labels = [l for row in t.edited[-1]["keyboard"] for (l, _) in row]
    assert "Keep in inbox" in labels


def test_a_verdict_asks_how_wide_before_writing_anything(bot):
    b, t, log = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    assert b.prefs.rules() == [], "a rule was written before the scope was chosen"
    labels = [l for row in t.edited[-1]["keyboard"] for (l, _) in row]
    assert any("Every" in l for l in labels)


def test_the_narrow_answer_writes_a_sender_rule(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert rule.scope in ("sender", "domain", "fingerprint", "subject")
    assert "archive" not in [a.kind for a in rule.actions], \
        "keep in inbox taught a rule that still archives"


def test_the_wide_answer_writes_a_category_rule(bot):
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_wide", 0, digest_id=b._digest_id)))
    rule = b.prefs.rules()[0]
    assert rule.scope == "category"
    assert "archive" not in [a.kind for a in rule.actions]


def test_the_confirmation_says_the_rule_in_words(bot):
    """A rule that cannot be read cannot be corrected."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_wide", 0, digest_id=b._digest_id)))
    text = t.edited[-1]["text"] + " ".join(a["text"] for a in t.answered)
    assert "promotion" in text


def test_a_correction_claims_no_reversal_under_dry_run(bot):
    """Nothing happened, so there is nothing to undo, and saying otherwise
    would be the most expensive thing this message could get wrong."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    blob = t.edited[-1]["text"].lower()
    assert "undo" not in blob and "reversed" not in blob


def test_correcting_a_rule_decided_action_records_an_override(bot):
    """record_override has had no caller since it was written. This is it: a
    correction of what a rule proposed IS an override of that rule."""
    b, t, _ = bot
    rule = b.prefs.add_rule(Rule(
        id="r-1", scope="sender", pattern="deals0@shop.com",
        actions=[ActionTemplate(kind="archive")], provenance="p", created_at=NOW))
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("keep", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("scope_narrow", 0, digest_id=b._digest_id)))
    assert b.prefs.rules_by_id()["r-1"].override_count == 1


def test_teaching_trash_says_it_will_not_ask_again(bot):
    """Plan 1 decided a rule-proposed trash auto-executes. That is the one
    verdict that widens what happens without a second question, so it says so."""
    b, t, _ = bot
    b.handle_update(msg("/triage 4"))
    b.handle_update(cb(encode("done", digest_id=b._digest_id)))
    b.handle_update(cb(encode("open", 0, digest_id=b._digest_id)))
    b.handle_update(cb(encode("teach_trash", 0, digest_id=b._digest_id)))
    assert "without asking" in t.edited[-1]["text"].lower()
```

The `bot` fixture must expose `prefs`; add `prefs=PreferenceStore(...)` to the
`Bot` constructor and the fixture, since the bot now writes rules. Add
`PreferenceStore.rules_by_id()` if it does not exist — a dict comprehension over
`rules()`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_tg_bot.py -q`
Expected: FAIL — `Bot.__init__() got an unexpected keyword argument 'prefs'`

- [ ] **Step 3: Implement**

`Bot.__init__` takes `prefs: PreferenceStore` and stores it. `_on_callback`
gains branches:

```python
        if intent.kind == "open":
            self._panel = "item"
            self._open_index = intent.index
            self._show(edit=True)
            return
        if intent.kind in ("keep", "relabel", "teach_trash"):
            # Verdict first, scope second. Writing on the first tap would pick a
            # blast radius the owner never chose, and "this sender" versus
            # "every valuable newsletter" are different instructions.
            self._pending = {"verdict": intent.kind, "index": intent.index,
                             "label_index": intent.label_index}
            self._ask_scope(intent.kind)
            return
        if intent.kind in ("scope_narrow", "scope_wide"):
            self._write_rule(wide=intent.kind == "scope_wide")
            return
```

`_write_rule` resolves the pending verdict to an action sequence, builds the rule
with `rule_from_correction` (narrow) or a `category`-scoped `Rule` (wide), calls
`record_override` when the corrected decision came from a rule, writes it, and
edits the message to a confirmation naming the rule via `rule.summary`. The
confirmation never mentions undo; under dry-run nothing happened, and under live
this plan still does not reverse anything (spec §2.3).

`teach_trash` is narrow-only: skip the scope question, and say in the
confirmation that a taught trash runs without asking again.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 5: Verify it on the phone before committing**

Run the bot against the snapshot, `/triage 12`, open a done item, correct it, and
confirm the rule appears in `/status` or the next run's "came from rules you
taught me" count. A terminal preview is not a fair test of this.

- [ ] **Step 6: Commit**

```bash
git add inbox_agent/telegram tests
git commit -m "Corrections from the digest write rules"
```

---

## Self-review notes

- **Task ordering:** Task 6 (callback kinds) must land before Task 5's
  implementation, which calls `encode` with those kinds. Task 5 is listed first
  because the view is the thing being built and the codes exist to serve it;
  an executor working strictly in order should do 6 then 5.
- **Tasks 1 and 2 share one commit.** Task 1 leaves the suite red on purpose —
  `prefilter` and `store` still read `rule.action` — and Task 2 makes it green.
  Splitting the commit would commit a broken tree.

## What this plan does not cover

- **Undo.** Blocked on `INBOX_DRY_RUN=false`; the same buttons gain a second
  effect then (spec §2.3).
- **Free-text corrections.** The escape hatch in spec §2.4. The buttons cover
  the cases that motivated this; typing is the next increment and needs a
  model call to turn a sentence into a rule.
- **The schedule** and **`/backlog`**, unchanged from the digest design.

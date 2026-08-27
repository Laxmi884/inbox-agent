# Inbox Agent — Design

**Date:** 2026-08-26
**Status:** Approved, ready for implementation planning
**Deliverable:** `inbox_agent.ipynb` — a comparative prototype, not a product

---

## 1. Purpose

Build a personal Gmail assistant that learns its owner's preferences instead of
following fixed rules. The notebook is a **lab**: its job is to produce defensible
opinions about agent architecture, memory, and local-model viability that carry
forward into a production build. Polish is explicitly not a goal; comparability is.

Two questions the notebook must answer with evidence:

1. Does model-driven control flow (a tool-calling agent) beat graph-driven control
   flow (a deterministic pipeline) on real inbox triage — and where exactly does it
   stop being worth it?
2. Can a 12B local model (Gemma via Ollama) carry this workload, or is a hosted
   model the floor?

## 2. Product shape

The agent reads and understands everything in the inbox, and acts freely on anything
reversible: label, categorise, archive, summarise, draft. It **never sends**.
Destructive actions (trash) start as proposals and become autonomous only once a
learned rule authorises them.

Newsletters are not noise by default. A discount, a release note, or a market digest
may be the most valuable thing in the inbox; only the owner can teach which is which.
This is why the system is a preference model with hands, rather than a classifier.

### Autonomy ladder

| Action | Authority | Reversible |
|---|---|---|
| read, search, summarise | always | n/a |
| label, categorise | always | yes |
| archive | always | yes |
| draft reply | always | yes |
| trash | learned rule, else human approval at interrupt | yes — 30 days, `untrash_thread` |
| send | **never** — human only | no |
| delete forever | **never** | no |

`send_message` and `delete_forever` are enforced as a deny-list in the action
chokepoint (`INBOX_FORBIDDEN_ACTIONS`), not merely omitted from the tool schema. A
model that hallucinates the tool still cannot reach Gmail.

### How it learns

Three mechanisms, in descending authority:

1. **Explicit instruction.** "Always archive these." Highest authority, directly
   auditable, always overrides.
2. **Correction of proposals.** The daily loop. Every reject or edit at the review
   step becomes a candidate rule.
3. **Sent-mail bootstrap** (Stage C). A one-time pass over sent mail to learn tone
   and who actually gets replies.

**Silent observation is deliberately excluded.** Inferring intent from inbox-state
diffs learns wrong lessons without ever surfacing them, which is incompatible with
the audit requirement.

## 3. Constraints

- **Gemma is a design constraint, not a config value.** A 12B local model is reliable
  at one small structured judgment per call and unreliable across long autonomous
  tool-calling loops. The graph therefore owns control flow and hands the model only
  leaf-level judgments with tight context. This is also what makes decisions
  auditable: a decision the graph made is a decision you can point at.
- **Audit-worthiness is a pillar, not a feature.** Every action traceable to a rule,
  a policy version, and a trace ID.
- **Real email lands on disk.** `inbox_agent/snapshot/`, `inbox_agent/store/`, and
  `*.jsonl` are gitignored.
- **Prompt injection is in scope.** Email bodies are attacker-controlled text.

## 4. Architecture

### 4.1 Audit spine

Every mutation passes through a single `execute_action()` chokepoint. Nothing reaches
Gmail otherwise. It appends one JSONL record per action to `INBOX_AUDIT_LOG`:

```
ts · thread_id · action · params · actor(agent|human|rule:<id>)
rule_provenance · model · backend · langsmith_run_id
checkpoint_id · policy_version · dry_run · result · reversible · undo_token
```

Two properties follow. *Why did it archive that?* is answered by joining one log line
to a rule, a policy commit, and a LangSmith trace. And any run can be reversed,
because `undo_token` carries what's needed to undo it (`untrash_thread`, re-apply the
prior label set).

`INBOX_DRY_RUN=true` is the default. Under dry-run the chokepoint logs the intended
action with `dry_run: true` and returns a simulated result without calling Gmail.

### 4.2 Stage A — deterministic pipeline

A LangGraph `StateGraph` with a checkpointer:

```
fetch → prefilter → classify → propose → ⟨interrupt⟩ → execute → learn → END
```

- **fetch** — reads the frozen snapshot (or live Gmail once the shape settles).
- **prefilter** — pure Python, zero LLM. Applies known rules and cached per-sender
  dispositions. On a 200-thread inbox this is what prevents 200 Gemma calls; only
  genuinely novel or ambiguous threads continue. Load-bearing for local-model
  viability.
- **classify** — one thread per call, structured output, tight context. Retrieves only
  the rules relevant to this thread from the store. The Gemma-shaped unit of work.
- **propose** — assembles the action set. No side effects.
- **interrupt** — LangGraph `interrupt()`. The graph suspends, the checkpoint
  persists, the owner approves / rejects / edits, and `Command(resume=...)` continues.
  Suspension is durable: the notebook can be closed and the run resumed later.
- **execute** — routes every approved action through `execute_action()`.
- **learn** — records corrections as candidate rules with full provenance.

### 4.3 Stage B — tool-calling agent

The same tools, the same store, but `create_react_agent` driving: the model decides
what to do next. Run against the identical snapshot, on both Gemma and a hosted model
via OpenRouter.

Expected finding: Gemma degrades visibly (drift, loops, token burn, silent
mislabelling at scale) where a frontier model does not. That finding *is* the
deliverable — it sets the production model floor. Stage A is the baseline that makes
the comparison meaningful.

### 4.4 Stage C — memory and proactivity

Sent-mail bootstrap, correction-to-rule extraction, and consolidation when rules
conflict. Proactivity is defined as **the agent volunteering things it wasn't asked
about** — "you've left this person on read twice", "this discount expires Friday" —
not as a scheduler. A timer is trivial and teaches nothing; deciding what is worth
interrupting a human for is the hard part.

## 5. Memory

The system has two kinds of state, on two substrates. Conflating them was the main
correction made during design.

| | Preference memory | Behaviour memory |
|---|---|---|
| Holds | learned rules, per-sender dispositions, tone profile | classifier prompt, category taxonomy, autonomy policy |
| Changes | constantly, from corrections | rarely, deliberately |
| Needs | provenance, high write volume, cheap retrieval | versioning, review, rollback, promotion |
| Substrate | LangGraph `BaseStore` | LangSmith Context Hub |

### 5.1 Preference memory — `BaseStore`

Namespaces: `("prefs","rules")`, `("prefs","senders")`, `("prefs","tone")`.

Every rule carries: the instruction or correction that produced it, when, how often it
has fired, and whether it has ever been overridden. Semantic search over rules so
`classify` retrieves only what is relevant — a context-budget necessity under Gemma.
The rule set renders as a plain table; the owner must always be able to read what the
agent thinks it knows and delete a line they disagree with.

**Why `BaseStore` over Mem0 / Zep / LangMem / Letta:** audit-worthiness requires owning
the provenance schema rather than inheriting one; Gemma cannot afford a fat memory
context (Zep is measured at >600k tokens per conversation against Mem0's ~1.7k); and
the notebook's purpose is to learn the mechanics, which a managed service hides. The
rule schema is designed so Mem0 can slot in behind the same interface later —
**that swap is itself a Stage C experiment.**

### 5.2 Behaviour memory — Context Hub

The agent policy — classifier prompt, taxonomy, autonomy rules — lives in a Context
Hub skill repo (`CONTEXT_HUB_SKILL`), tagged `dev` / `prod`. Runs pin a commit or tag
(`CONTEXT_HUB_TAG`) so any past run is reproducible, and `policy_version` is stamped
into every audit record.

```python
from langsmith import Client
from langsmith.schemas import FileEntry

client.push_skill(identifier, files, description, tags)
client.pull_skill(identifier, version)   # -> SkillContext
```

The **hot/cold split** is borrowed from Context Hub's Deep Agents memory model: a
compact always-loaded policy file, plus detailed files read only when relevant. This
applies to the rule set too — hot rules inline, cold rules retrieved on demand.

### 5.3 Learning as a reviewable diff

Stage C restructures learning along the LangSmith Engine "sleep-time compute" pattern:
rather than `learn` writing rules inline, corrections accumulate, and a separate
offline pass reads the traces and **proposes a diff** to the policy, which the owner
reviews and merges.

Learning becomes a pull request against the agent's brain — versioned, attributable,
revertable. This is the strongest available answer to the audit requirement.

## 6. Evaluation method

**Frozen snapshot.** `INBOX_SNAPSHOT_SIZE=50` recent inbox threads pulled to disk once.
Every architecture runs against the identical set; otherwise the inbox changes between
runs and every comparison is anecdote. It also means iteration never hammers the Gmail
API and dry-run is genuinely safe.

**LangSmith tracing** on throughout, so Stage A and Stage B can be diffed on a
per-thread basis and Gemma compared against a hosted model on identical input.

Comparison axes: decision quality against the owner's judgment, cost and latency per
thread, rule-set growth and stability, and failure modes under each model.

## 7. Interfaces

**The interrupt payload is UI-agnostic JSON from day one.** The notebook renders it as
a dataframe; Telegram later renders it as a message with inline buttons. Same graph,
same checkpointer, different renderer.

Durable interrupts are what make "agent asks, owner answers four hours later from
their phone" work — so the Telegram path needs no change to Stage A. Telegram is out
of scope for this spec and lands before daily use begins.

## 8. Risks

**Prompt injection.** Email bodies are attacker-controlled. With no send capability
the blast radius is labels and trash, both reversible. Mitigations: body text is
fenced as data in every prompt and never concatenated into instructions; the forbidden
action deny-list is enforced at the chokepoint rather than the prompt; a full audit
trail makes any injection visible after the fact. Hardened guardrails are a
prerequisite for live daily use, not for snapshot iteration.

**Local-model reliability.** Gemma may fail at Stage B. This is an expected finding,
not a project risk — Stage A is designed to remain viable regardless.

**Privacy.** Snapshot, store, checkpoints and audit log are gitignored and never
leave the machine.

**LangSmith tracing sends email content off the machine.** An earlier draft of this
spec claimed "only instructions and traces reach LangSmith"; that was wrong, and the
distinction it drew does not exist. A trace of the `classify` call captures the prompt,
and the prompt contains the sender, the subject, and the body or snippet of the mail
being classified. Enabling `LANGSMITH_TRACING` therefore uploads the content of every
classified thread to LangChain's servers.

That is a legitimate trade — tracing is the only practical way to compare Stage A
against Stage B, or Gemma against a hosted model, on identical input — but it is the
owner's decision to make knowingly, not a default. `LANGSMITH_TRACING` ships as
`false`. Context Hub, which stores only the policy and never mail, is unaffected and
works with tracing off.

**Rule overfitting.** The rule set may overfit the 50-thread snapshot. Mitigation: a
held-out set is available as a Stage C check if rule growth suggests it is needed.

## 9. Out of scope

Sending mail. Permanent deletion. Telegram. Calendar and Drive. Multi-account.
Production deployment. All are downstream of the findings this notebook produces.

## 10. Open items for the implementation plan

- Notebook cell structure, and how the shared substrate is factored so Stage A/B/C
  swap cleanly against it.
- Rule schema fields and the `BaseStore` ↔ Mem0 interface boundary.
- Snapshot fetch script and its refresh policy.
- Concrete review-table rendering at the interrupt.
- The structured-output schema `classify` asks Gemma to fill.

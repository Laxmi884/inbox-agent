# Rules engine, mailbox bootstrap, and unsubscribe

2026-09-01

**These are notes, not a spec.** Nothing here is agreed to build. It is the
parking place for work deliberately deferred until the live Gmail client lands,
so the design conversation that produced it is not lost between sessions.

Depends on `../specs/2026-09-01-live-gmail-design.md` (commits `db3377a`,
`eaef803`). Everything below needs a real mailbox to be worth doing: section 1
is much easier to get right once threads carry real headers rather than a
50-thread snapshot with empty bodies, and section 2 is *blocked outright* -
you cannot mine a mailbox you cannot read.

Resume order: live client (plan 1) -> backlog sweep (plan 2) -> MCP surface
(plan 3) -> this.

## 1. Rules schema

### 1.1 What is true today

`Rule` (`models.py:80`) carries one `scope` and one `pattern`:

```python
scope: Literal["sender", "domain", "fingerprint", "subject", "category"]
pattern: str
```

A single predicate. "Newsletters from this domain, but only the ones older than
a week" is not sayable, and neither is anything else with two conditions.

The pipeline already has two evaluation stages, and `models.py:82` already says
why they are different:

> `category` is not a property of a thread: it is what the model concluded
> about one. So a category rule cannot be matched in prefilter, which runs
> before the model - it is applied afterwards, by the apply_rules node.

- `prefilter` (`prefilter.py:43`) runs BEFORE the model. Zero LLM calls. Sees
  raw thread facts only.
- `apply_rules` (`graph.py:249`) runs AFTER the model. Costs one call. Sees
  model output.

### 1.2 Decision: the schema's primary axis is evaluation stage, not field

The instinct is to ask "which fields can a rule match on". The more useful
question is "at which stage can this predicate be evaluated", because that is
what determines **cost**. A rule made only of raw-thread predicates saves an
inference. A rule mentioning `category` or `intent` cannot, by construction.

Make it explicit rather than incidental:

```python
class Predicate(BaseModel):
    field: str   # sender | domain | subject | age_days | list_id | ...
                 # | category | intent | confidence   <- model-stage only
    op:    str   # eq | contains | matches | in | gt | lt
    value: Any

class Rule(BaseModel):
    when: list[Predicate]          # ALL must hold
    actions: list[ActionTemplate]

    @property
    def stage(self) -> Literal["pre", "post"]:
        return "post" if any(p.field in MODEL_FIELDS for p in self.when) else "pre"
```

`prefilter` evaluates `stage == "pre"`; `apply_rules` evaluates `stage ==
"post"`. Neither has to know the field list - `stage` is derived, so adding a
model-stage field cannot accidentally route a rule into prefilter where its
predicate is unevaluable.

Migration is mechanical: the existing `scope`/`pattern` pair becomes a
one-predicate `when` list. `_accept_legacy_action` (`models.py:108`) is the
precedent - rules already on disk are converted at load time by a
`model_validator(mode="before")`, not repaired and not silently dropped.

### 1.3 Decision: flat AND-list. No nesting, no OR

Rejected: nested boolean trees, and any expression DSL (CEL, JMESPath, a
mini-parser).

Not on implementation cost - on where rules come from. **A correction gives one
example**: one thread, one verdict. From one example a nested boolean cannot be
inferred, because too many hypotheses are consistent with it. `choose_scope()`
already performs this inference for the single-predicate case and it is the
hard part of the learning path.

The rule language should be bounded by what the correction UI can teach it. Any
grammar beyond that is reachable only by hand-editing JSON, which nobody will
do, while making the inference strictly harder. OR is two rules; that is not a
problem worth syntax.

A DSL additionally introduces a parser and an evaluation sandbox over
attacker-influenced input, for a system whose entire value is being predictable.

### 1.4 Decision: `summary` is rejected as a predicate field

`category` works because it is a closed vocabulary the model selects from.
A summary is free generated text, so matching a substring of it means **the same
thread can match or not match depending on sampling**.

That is non-deterministic output used as a deterministic key. It breaks the one
property the digest was built to have - `graph.py` reports which rule decided
each item, and `prefilter.py:63` writes that provenance into
`Decision.reason`. A rule that matches only sometimes makes that line a lie.

`intent` is acceptable **only** as a closed enum added to `ThreadJudgment`. As
free text it has exactly the same defect.

### 1.5 Decision: conflict resolution by specificity, then recency

`prefilter.py:57` currently resolves overlaps with `max(matches, key=lambda r:
r.created_at)` - latest word wins. Correct for single predicates, wrong for
multiple.

Counter-example: a general `sender=linkedin -> archive` and a specific
`sender=linkedin AND category=needs_reply -> keep`. If the specific rule was
taught first, latest-wins archives the thread that was explicitly protected.

Order by **specificity (predicate count) desc, then `created_at` desc**.

`precision` (`models.py:129`) stays out of the ordering and keeps its existing
job: demotion below 0.5. It is a ratio over `hit_count`, so at low hit counts it
is noise, and letting noise reorder rules makes "why did it do that?"
unanswerable. It is also `None` when untested, which would need a sort key
special case that means nothing.

### 1.6 `age_days` collapses `/backlog` into an ordinary rule

If `age_days` is a pre-stage predicate field, the bulk-archive mode specced in
`live-gmail-design.md` section 2.6 stops being a special mode:

```
age_days > 365 AND NOT starred AND NOT i_replied -> archive
```

A rule, plus a bulk executor behind it. This is the general case of the
observation that a strong rule system simplifies the pipeline rather than
complicating it, and it is the strongest argument for doing this work at all.

Note the ordering dependency: section 2.6 ships first, as a mode. This would
retire that mode later, not replace it before it exists.

## 2. Bootstrapping rules from the mailbox

### 2.1 The inversion

The obvious bootstrap is "find senders you never read, write archive rules".
The measurement in `live-gmail-design.md` section 1.1 kills it:

```
INBOX  threadsTotal 21058   threadsUnread 16748
```

**79% of the inbox is unread.** "You never read this sender" is very nearly
universal, so it barely discriminates. The signal that *does* discriminate is
its inverse: the small set that was engaged with.

So mine **positive signals into protect rules**, not absence into archive rules.
All of these are Gmail queries - deterministic, no model, no cost:

| Signal | Source | Rule |
|---|---|---|
| Replied to them | sender appears in sent mail | never auto-archive |
| Labelled by hand | the 11 existing user labels | label, in the owner's own words |
| Starred | `is:starred` | protect |
| Read and kept in inbox | read AND `in:inbox` | protect |

Row two is the valuable one. `Education/AI`, `Job Alerts`, `Market Insights`,
`Property Listings`, `Security Alerts` (`live-gmail-design.md` section 1.3 and
1.5) are rules the owner **already wrote by hand**, in their own taxonomy.
Mining them gives the agent the owner's vocabulary instead of the policy's.

### 2.2 Volume-weighting on the archive side

Absence is not worthless, it is just weak alone. Zero reads out of 412 threads
is evidence; zero reads out of 1 is noise. Any archive rule derived from
non-engagement needs a volume floor, and the floor is the whole reason the rule
is trustworthy.

### 2.3 Scope the bootstrap to senders active in the last ~90 days

`live-gmail-design.md` section 1.8 measured zero sender overlap between today's
mail and two years ago (0 of 75 distinct senders). A bootstrapped rule for a
sender that has stopped writing is dead weight, evaluated by `prefilter` on
every thread of every run, forever.

This also bounds the bootstrap's output size, which matters: a rule store with
thousands of never-matching entries is a performance problem and an
explainability problem at once.

## 3. Unsubscribe

### 3.1 Mechanism

`List-Unsubscribe` (RFC 2369) carries `mailto:` and/or `https:` URIs.
RFC 8058 adds `List-Unsubscribe-Post: List-Unsubscribe=One-Click`, which
signals the HTTPS URI can be actioned by a bare POST - no browser, no page,
no scraping, no model.

Verified against RFC 8058 itself:

- Body is the pair `List-Unsubscribe=One-Click`, sent as `multipart/form-data`
  (SHOULD) or `application/x-www-form-urlencoded` (MAY).
- HTTPS is mandated by the spec for the URI. Our "HTTPS only" is not a local
  safety preference; it is the standard.
- The request **MUST NOT include cookies, HTTP authorization, or any other
  context information**. So the client sends a bare POST from a fresh
  connection, never the session used for anything else.
- The RFC recommends the URI contain an opaque, hard-to-forge component,
  precisely because anyone holding the message can trigger it.

**Correction to the working assumption:** RFC 8058 specifies **no** honouring
deadline. The "within 2 days" figure comes from Google's bulk sender
guidelines, not the standard, and should be attributed there.

**Second correction:** the Gmail/Yahoo bulk sender rules (senders above ~5,000
messages/day) took effect February 2024, but **one-click enforcement was
deferred to June 2024**. The distinction matters only for how much of an
archived backlog is likely to carry the header - older mail, less.

The useful consequence stands either way: the high-volume senders that dominate
this mailbox are exactly the ones obliged to support one-click.

### 3.2 The `mailto:` variant is structurally unavailable

`mailto:` unsubscribe requires sending a message. `send_message` is in
`ALWAYS_FORBIDDEN` (`config.py:28`), and `gmail.modify` - the only scope the
live client requests - does not grant send.

So this is decided by the deny-list and by the OAuth scope boundary, not by
preference. HTTPS one-click or nothing.

### 3.3 Safety lines

- **Header only, never the body.** The `List-Unsubscribe` header is set by
  sending infrastructure; a link in the body is arbitrary untrusted content.
  `classify.py:250` already fences bodies as untrusted data for exactly this
  reason, and following a body link would walk straight through that fence.
- **HTTPS only** (and per 3.1, required by the RFC anyway).
- **Block private, link-local and loopback ranges** after DNS resolution. An
  unsubscribe URI is attacker-supplied input that we make an outbound request
  to; without this it is a plain SSRF primitive. Resolve first, check the
  resolved address, then connect - checking the hostname alone loses to a DNS
  record pointing at 127.0.0.1.
- **Never on SPAM-labelled mail.** Hitting an unsubscribe endpoint confirms a
  live human reads the mailbox, which is precisely the thing a spammer paid to
  learn.

### 3.4 Risk class: the first allowed-but-irreversible action

Today the two sets coincide exactly:

```python
REVERSIBLE_ACTIONS = frozenset({"label", "unlabel", "archive", "trash", "draft", "none"})   # models.py:19
ALWAYS_FORBIDDEN   = frozenset({"send_message", "delete_forever"})                          # config.py:28
```

Everything permitted is reversible; everything irreversible is forbidden. The
system has never had to represent a third case, and `unsubscribe` is one:
permitted in principle, and **not undoable** - re-subscribing means finding the
sender's signup form by hand, if one still exists.

Consequences:

- `unsubscribe` stays **out** of `REVERSIBLE_ACTIONS`, so
  `AuditRecord.reversible` is False for it.
- `undo_action()` refuses it. Note this already happens structurally: an
  unsubscribe produces no `undo_token`, and `audit.py:200` raises
  `nothing to undo for {record.action!r}`. Relying on that alone would be
  accidental correctness, so the exclusion should still be explicit.
- It belongs in the **authorisation tier** of `hold_reason()`
  (`partition.py:27`), which currently gates `trash`. Arguably it should
  precede `trash` in that precedence chain: trash is recoverable for 30 days,
  this is not recoverable at all.
- **Never autonomous**, and never granted by a learned rule the way
  `partition.py:39` lets a taught rule authorise `trash`. The graduation clause
  that makes sense for a reversible action does not transfer to one that is not.
- Adding it means widening `ActionKind` (`models.py:13`) and `HoldReason`
  (`partition.py:19`).

### 3.5 Pairing with the backlog sweep

Bulk archive clears the symptom; unsubscribe stops the source. They are
complementary, and the natural offer is at the end of a sweep: "these 12 senders
account for 3,400 of the threads just archived - unsubscribe from any?"

But per 2.3 and section 1.8 of the spec, **dead senders no longer send**.
Unsubscribing from `legeropinion.com` in 2026 achieves nothing except an
outbound request to a stranger. The offer is only worth making for senders that
are both high-volume and currently active, which is a much shorter list than
the sweep itself.

## 4. What the enterprise RAG notebook contributed

Source: the user's own notebook **"Building Baseline Enterprise RAG Pipelines
brick by brick"** (`b7dd23bd-61bc-42af-b390-208ac30b422b`), 17 Towards Data
Science PDFs. Five were on-point: *Parse the question before you search*,
*Dispatching the Parsed RAG Question: Chunk Strategy, Model Tier, Activations,
Audit*, *Five fields RAG should extract from any question*, *Retrieval Is
Filtering, Not Search*, and *When RAG Users Ask Vague Questions: Clarify Once,
Learn the Default*.

More transferable than expected. The pipeline it describes has a learned-default
subsystem that is structurally the same problem as our learned rules, and it has
solved two things we have not.

### 4.1 Validated: typed parse, then pure functions over its fields

Their dispatcher parses a question once into a typed `ParsedQuestion`, then runs
pure functions keying on its fields:

```python
def decide_activations(parsed: ParsedQuestion, doc_profile: DocumentProfile):
    plan = ExecutionPlan()                      # defaults
    if parsed.decomposition.pattern == "independent":
        plan.decompose_compound = True
    if doc_profile.format == "docx":
        plan.extract_page_numbers = False
    if parsed.answer_shape == "listing":
        plan.iterate_on_feedback = True
    return plan
```

This is exactly the shape of `apply_rules` (`graph.py:249`) running over
`ThreadJudgment` (`classify.py:44`) - a closed typed struct produced by one
model call, then deterministic dispatch on its fields. 1.2's post-model stage is
independently arrived at by a production system.

### 4.2 Validated: every dispatch predicate is flat

The dispatcher is described as accumulating **15-30 rules over a deployment's
lifetime**, and every rule shown across the material is a flat equality or
boolean check on one typed field. No nesting, no boolean trees, anywhere.

That is independent support for 1.3. A system that reached 30 rules in
production without needing nesting is better evidence than our argument from the
learning signal, and it points the same way.

### 4.3 Borrow: a vote distribution, not a scalar override count

The strongest finding. Their `ClarificationDefault` stores `candidate_votes`
(values mapped to weighted votes), `confidence`, `sample_size` and
`last_refreshed`. On disagreement, the update **decrements the wrong value and
credits the alternative**:

```
votes[wrong] = votes.get(wrong, 0) - 1     # and the alternative gains
confidence_new = max(0.0, top) / n_new
```

Ours cannot do this. `record_override` (`store.py:273`) is:

```python
rule.override_count += 1
```

It records *that* someone disagreed and never *what they wanted instead*.
`rejected_action` (`models.py:100`) is a single scalar, not a distribution, so it
cannot accumulate across corrections either.

The consequence is concrete: a rule that is consistently wrong **in the same
way** - `sender=X -> archive` where the owner always wants `label(news)` - dies
by demotion instead of converging on the right answer. The correction signal is
spent proving the rule bad rather than making it good.

Worth taking. It also composes with the richer schema: with `Rule.when` a
predicate list and votes over actions, a rule can move its *action* while keeping
its *match*.

### 4.4 Borrow: losing confidence reverts to "ask", it does not delete

Below `0.60`, their gating "invalidates silent execution and reverts the system
to ask mode". The rule survives; only its *autonomy* is withdrawn.

That maps precisely onto the autonomy ladder. A rule whose precision falls should
drop out of the auto-execute tier and back into the held/authorisation tier -
where the owner sees it again and can correct it, which is the only way it
recovers. Our threshold is 0.5 and `mark_overridden` (`store.py:281`) already
insists a rule is "kept, not deleted"; what needs checking when this is picked up
is whether demotion actually re-gates the action or merely deprioritises the
rule.

### 4.5 Borrow: record what did *not* fire, and why

Their audit contract carries the negative space:

```python
class AuditedRetrievalResult(BaseModel):
    ...
    methods_run: list
    methods_skipped: list
    skipped_reasons: dict
```

`prefilter` (`prefilter.py:43`) records only the winner. A rule that matched and
lost to `max(matches, key=created_at)` (`prefilter.py:57`) leaves no trace at
all, so "why didn't my rule fire?" is unanswerable today. Under 1.5's
specificity ordering there will be *more* losing matches, not fewer, so this
becomes load-bearing rather than nice to have.

`TriageState.skipped` (`graph.py:65`) is already `Annotated[list[dict],
operator.add]`, so the channel exists.

### 4.6 Considered and rejected

- **Their conflict model.** "Conflicts largely avoided by construction because
  each rule targets a specific activation flag", resolved sequentially as a plan
  object is mutated. That works because their rules write *different fields*.
  Ours all write the same thing - an action on a thread - so conflict is
  structural for us and cannot be designed away. 1.5 stands unchanged; the
  material offers nothing here.
- **Stats computed externally rather than stored on the rule.**
  `ClarificationDefault` deliberately stores no precision; it is computed by
  joining clarification tables to a `query_log`. We store `hit_count`
  (`models.py:95`) and `override_count` (`models.py:104`) on the `Rule` itself.
  A real tension - we do have a query log in `audit.jsonl` - but on-rule counters
  are what make `precision` (`models.py:129`) a local read with no join on the
  hot path in `prefilter`. Keep ours.
- **Model-tier cascade by expected answer type** (`gpt-4.1-nano` for amounts and
  dates, larger models for text). Genuinely interesting against the measured
  `MODELS` registry in `config.py`, but it is a routing idea for the
  *classifier*, not the rules engine. Recorded so it is not lost; out of scope
  for this doc.
- Chunking, embeddings, rerankers, TOC anchoring, keyword/BM25 method selection,
  the LLM arbiter's candidate ranking. Retrieval-shaped with no email analogue.

### 4.7 Verdict on the stage split: half validated, half unaddressed

**Validated:** typed-parse-then-pure-functions is their dispatcher and our
`apply_rules`. That half of 1.2 is sound.

**Not validated, and not refuted - simply absent.** They have no analogue of
`prefilter`. `decide_activations(parsed, doc_profile)` mixes model-derived fields
(`parsed.answer_shape`, `parsed.intent`) with deterministic corpus facts
(`doc_profile.format`, `doc_has_toc`) **in one function**. They never split by
stage, because they have no reason to: the question parse always runs, and it is
one cheap call on one short question.

Our pressure is per-item across 50 to 16,748 threads per run, which is the entire
reason `prefilter` exists. So the half of 1.2 that actually saves money - a cheap
tier that skips the model outright - gets no support from this material. It is a
problem they do not have.

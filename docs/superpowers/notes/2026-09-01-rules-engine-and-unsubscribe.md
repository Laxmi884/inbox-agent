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

## 4. Outstanding

**The NotebookLM review did not happen.** The user's "building enterprise rag"
notebook (`b7dd23bd-61bc-42af-b390-208ac30b422b`) was to be mined for
transferable concepts - specifically deterministic pre-retrieval routing as an
analogue to the stage split in 1.2, any structured metadata-predicate layer,
filter precedence, and selection provenance.

Blocked on authentication. `~/.notebooklm/storage_state.json` was rewritten by
a fresh `notebooklm login` at 21:24 and every RPC still redirected to the Google
sign-in page minutes later; `notebooklm auth check` reports cookies present and
`token_fetch: null`. Likely the login completed against an account other than
the notebook's owner, or the session was invalidated server-side.

To resume:

```bash
notebooklm login                                     # must be the user; needs a browser
notebooklm list --json                               # confirm it actually works now
notebooklm source list --notebook b7dd23bd-61bc-42af-b390-208ac30b422b --json
```

Nothing in sections 1-3 depends on it. Treat it as a possible source of
refinements, not a blocker.

# Corrections that teach

2026-09-01

The digest now acts on the confident majority and reports what it did. Dogfooding
that report on a real phone produced one observation that is not a rendering
complaint: *"I don't have a way to pass the rules on the done actions already."*

That is the hole the digest design named and did not close. Its section 1 found
that `learn_from_response` (`graph.py:75`) is keyed on the interrupt's response,
so "anything the agent does alone teaches nothing, by construction". Plan 1 then
moved the *majority* of mail onto exactly that path. The agent got quieter and
learned less, which is the wrong direction for a system whose entire premise is
that review gets shorter as it is corrected.

The specific case that surfaced it: the model labelled two newsletters
`newsletter_valuable` and archived them. The owner wants them labelled and left
in the inbox — "and I guess this could be a rule learned because it can be
different for everyone". Both halves of that sentence turn out to be blocked.

## 1. What is true today

Established by running it, not assumed.

- **A learned `label` rule cannot say which label.** `prefilter`
  (`prefilter.py:32`) builds `Action(kind=rule.action, thread_id=thread.id)` with
  no `params`, and `_dispatch` (`audit.py:80`) reads `action.params["label"]`.
  Reproduced end to end:

  ```
  rule: sender newsletter@towardsdatascience.com -> label
  action produced: label params: {}
  dry_run=True:  result=simulated      <- looks fine
  dry_run=False: RAISED KeyError: 'label'
  ```

  It passes as `simulated` under dry-run because `execute_action` returns before
  `_dispatch` (`audit.py:135`). So this is invisible in the mode the project is
  dogfooding in and fatal in the mode it is about to enter, and the bot catches
  the exception as "the run did not finish" — one taught rule would break a whole
  run against a live mailbox.

- **A rule cannot express a sequence.** `Rule.action` (`models.py:68`) is a
  single `ActionKind`. "Label it and archive it" is what the model already does
  via `also_archive`, and "label it but leave it in the inbox" is the correction
  the owner wants to teach. Neither is sayable.

- **A rule cannot be about a category.** `Rule.scope` is `sender`, `domain`,
  `fingerprint` or `subject` — all properties of the raw thread, because
  `prefilter` runs *before* the model. "Never archive a valuable newsletter" is a
  statement about a conclusion the model reaches, and there is nowhere for it to
  live.

- **`record_override()` (`store.py:245`) still has no caller** outside tests.
  `Rule.precision` therefore never moves and the auto-demotion below 0.5 cannot
  fire. The digest design listed this as a consumer with no producer; Plan 1 did
  not add the producer, because the producer is a correction and corrections had
  nowhere to happen.

- **`open` is inert.** The numbered buttons on the digest answer with "not built
  yet". Pressed on a phone, twice, by the owner.

- **The done panel is real** as of this branch: it lists what the run did, per
  thread, with the label named. It is the screen a correction should start from,
  and it has no verdicts on it.

## 2. Decisions

### 2.1 A rule carries an action sequence

`Rule.action: ActionKind` becomes `Rule.actions: list[Action]`. Whatever the
agent can propose, a rule can teach — one vocabulary, and `params["label"]` is
populated by construction, which is what closes the `KeyError` rather than
patching around it.

*Rejected: one action plus params.* It fixes the crash and still cannot say
"label then archive", which is the single most common sequence the model
produces. Two rules would not help: `prefilter` picks exactly one match
(`prefilter.py:27`, most recently created wins) and nothing sequences them.

*Rejected: a rule teaches a category, and the policy decides what a category
does.* Attractive — it keeps "what happens to a newsletter" as one versioned
decision — but it makes every correction an argument with the policy file, and
the owner's answer to "different for everyone" was explicitly that the preference
is theirs, not the policy's.

**Migration.** The store is on disk as of this branch, so rules written before
this change exist. `Rule` gains a validator that accepts a legacy `action` field
and converts it to a one-element `actions` list. Legacy `label` rules convert to
a label action with no label name, which is the bug above; they are converted and
then flagged, not silently executed. See §6.

### 2.2 `category` becomes a scope, and rules fire in two places

A category is not known until the model has run, so a category rule cannot fire
in `prefilter`. Rules split by when they *can* fire:

- **Pre-model rules** — `sender`, `domain`, `fingerprint`, `subject`. Matched
  against the raw thread in `prefilter`, and the model is never called. Unchanged.
- **Post-model rules** — `category`. Applied by a new node between `triage`
  (`graph.py:225`) and `propose` (`graph.py:243`), rewriting the action sequence a
  decision carries.

The second kind is a rewrite, not a re-judgment: the model's category stands, and
the owner's rule decides what happens to mail of that category. That is the whole
newsletter case — "you were right that it is a valuable newsletter, you were
wrong to archive it".

*Rejected: applying category rules inside `classify`.* It would put a preference
store inside the module whose job is talking to a model, and it would make the
prompt and the rules two competing statements about the same decision with no
visible precedence. A separate node keeps the precedence readable in the graph.

### 2.3 A correction teaches; it does not reverse

`undo_action()` refuses on dry-run records, correctly: under dry-run nothing
happened, so there is nothing to reverse. A correction therefore writes a rule
and says so — "next time, mail I classify `newsletter_valuable` is labelled and
stays in the inbox" — and does not claim to have changed anything.

When `INBOX_DRY_RUN=false`, the same button gains a second effect: reverse this
one, and teach for the future. That is a follow-on plan, not this one.

*Rejected: teach and undo together now.* The undo half cannot be exercised until
the agent is live, so it would ship pinned only by synthetic tests — which is
precisely what Plan 1 refused to do when it left `_resume` unwired rather than
restore untriggerable code.

### 2.4 Verdicts are buttons, with typing as the escape hatch

Three verdicts cover what the owner actually does: `Keep in inbox`,
`Label as …`, `Trash these instead`. Each maps to exactly one action sequence, so
what is taught is what the button said. A free text reply handles everything else
and becomes an instruction or a rule.

A fourth was drafted and removed in review: `Never archive this sender` is
`Keep in inbox` answered with the narrow scope, so shipping both would put the
same instruction on two buttons and make the scope question mean different things
depending on which was pressed.

The design's earlier decision that correction is *modal* — open an item, then act
on it — is unchanged and is what button `1` finally does.

### 2.5 A correction asks how wide before it writes

Two taps: the verdict, then the blast radius — this sender, or every thread of
this category. `choose_scope()` (`store.py:126`) still picks the default for the
sender-shaped option, so the narrow choice stays as narrow as the evidence
supports.

Asking is not ceremony. "Never archive mail from this sender" and "never archive
any valuable newsletter" are different intentions that produce the same tap, and
only the owner knows which was meant. The confirmation names the rule in words,
because a rule that cannot be read cannot be corrected.

### 2.6 A taught `trash` rule auto-executes, and the confirmation says so

Plan 1 decided that trash proposed by a *learned rule* auto-executes while trash
from the model is always held — the rule is the owner's own instruction, and
asking again for something they just taught is the noise this design exists to
remove. `Trash these instead` is therefore the one verdict that widens what
happens without asking again, and its confirmation says exactly that before it
is written.

### 2.7 The attention button stays whole-tier

`✅ Approve N replies & alerts` names the whole attention tier while the page may
show only some of it. Kept: those actions are harmless by construction (a draft
or a label), and a button whose meaning depends on how far the reader has
scrolled is worse than one that names its count honestly.

## 3. Architecture

### 3.1 The graph

```
fetch -> prefilter -> classify -> apply_rules -> propose -> partition -+-> auto_execute -> enqueue_held -+
                                                                       +-> <interrupt> -> execute ------+-> mark_triaged -> learn
```

`apply_rules` is the only new node. It reads category-scoped rules, rewrites
`decisions[].actions` where one matches, sets `source="rule"` and `rule_id` so the
digest's "came from rules you taught me" count includes them, and records a hit.

### 3.2 Data model

**`Rule`** — `action: ActionKind` becomes `actions: list[Action]`; `scope` gains
`category`. Everything else (provenance, `hit_count`, `override_count`,
`precision`, `overridden`) is unchanged: this widens what a rule says, not what
is known about it.

**No new store.** Category rules are rules, in `RULES_NS`, subject to the same
precision tracking and the same auto-demotion.

**`ReviewItem`** already carries `category`, `subject`, `sender`, `proposed` and
`reason`, which is everything the item view needs; the view resolves a done entry
back to its proposal in `state["auto"]` for the `Why:` line, and omits that line
when the proposal is not there rather than inventing one. The done panel's
`DoneItem` carries what was actually executed. A verdict names a position in the list it was rendered
from, resolved the way every other callback is (§3.3 of the digest design).

### 3.3 Where a correction is applied

`learn_from_response` stays where it is and keeps serving the interrupt path. A
verdict from the item view does not travel through graph state at all: the bot
holds the `PreferenceStore`, the correction is a store write plus a confirmation
message, and it is durable the moment it is written. This is what makes the
learning loop independent of whether a run is parked.

`record_override()` gains its caller here: correcting an action that a rule
proposed is, by definition, an override of that rule.

## 4. The item view

```
3. Data Scientist, Fraud at Stripe
jobalerts-noreply@linkedin.com
→ label(recruiter), archive
Why: job alert from a no-reply address.

[Keep in inbox]        [Label as …]
[Trash these instead]  [↩ Back]
```

Reached from a numbered button on either list. A held item shows the proposal and
offers approve / not this / relabel, and acting on it removes it from the queue. A
done item shows what happened and offers the correction verdicts above.

`Label as …` opens the policy's categories, three to a row, as the paged view
already does. Every keyboard carries the digest id, so a stale tap is refused the
way it is everywhere else.

## 5. What each verdict writes

| Verdict | Rule written | Scope offered |
|---|---|---|
| Keep in inbox | `[label(<category>)]` | sender / category |
| Label as X | the original sequence with the label swapped for X | sender / category |
| Trash these instead | `[trash]` | sender only |
| Free text | model-derived rule, or an instruction | asked |

`Trash these instead` is sender-only on purpose: a category-scoped trash rule
would auto-execute trash across a whole class of future mail on one tap, which is
a blast radius no single correction should be able to reach.

## 6. Failure modes

- **A legacy `label` rule with no label name.** Converted by the validator, then
  refused at execution with a message naming the rule, rather than raising
  `KeyError` mid-run. The owner is told which rule to re-teach.
- **A category rule and a sender rule both match.** The sender rule already won
  by construction: it fires in `prefilter` and the model never runs, so
  `apply_rules` never sees the thread. Documented rather than arbitrated.
- **A rule that teaches an action the deny-list forbids.** Refused at write time,
  not at execution time, with the same message `execute_action` would have given.
- **A correction on a thread that is no longer in the list.** Resolved by index
  against the rendered list, and the digest id makes a stale list unusable.

## 7. Testing

- A label rule executes against a live-mode client without raising, and applies
  the label it names. This is the regression the `KeyError` above becomes.
- A legacy `action` rule loads, converts, and is refused loudly rather than
  silently executed.
- A category rule rewrites an action sequence after classification and does not
  fire in `prefilter`.
- A sender rule still short-circuits the model.
- Each verdict writes the rule in §5, at the scope chosen, and the confirmation
  names it.
- Correcting a rule-proposed action calls `record_override`, and enough overrides
  demote the rule below `MIN_PRECISION`.
- A correction under dry-run writes the rule and claims no reversal.

## 8. Sequencing

Three pieces, each of which leaves the system working:

1. **The rule schema.** `actions`, the `category` scope, the legacy validator,
   and the `KeyError` regression test. Nothing new is taught yet; the crash is
   gone and rules can say what they need to say.
2. **`apply_rules`.** The node, its precedence, and the hit recording. Category
   rules now do something — written by hand in a test, since nothing writes them
   from the UI yet.
3. **The item view and its verdicts.** Button `1` opens an item, verdicts write
   rules at a chosen scope, and `record_override` gains its caller.

Piece 1 is a prerequisite for both others. Pieces 2 and 3 are independent of each
other and could land in either order; 2 first means the newsletter case is fixed
for the owner before the UI to teach it exists.

## 9. Not in this plan

- **Undo.** Blocked on `INBOX_DRY_RUN=false`; the same buttons gain the second
  effect then (§2.3).
- **The schedule.** The 08:00/18:00 timer with catch-up on wake stays where the
  digest design put it.
- **`/backlog`.** The interrupt path and its verdict collection are still Plan 3,
  and `_resume` is still not safe to call — see the warning in `bot.py`.

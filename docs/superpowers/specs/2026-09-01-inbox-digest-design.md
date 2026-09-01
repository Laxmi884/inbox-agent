# Inbox digest design

2026-09-01

The Telegram review UI works end to end and its digest has been fixed twice —
made readable, then made correctable — but never designed. This is that design.

It turned out not to be a rendering task. The complaint that started it ("I don't
like how the digest is presented. Information is not in easily graspable format")
has a layout answer, but the follow-up ("how am I supposed to correct something
from the digest?") does not: the digest is ungraspable mostly because it reports
fifty things that all look equally like they need a decision, and it is
uncorrectable because the learning loop cannot reach most of what it shows.

Fixing either one moves the boundary between what the agent does alone and what
it asks about. That boundary is the actual subject of this document.

## 1. What is true today

Established by reading the code, not assumed:

- **There is no gate.** `propose` (`graph.py:158`) puts *every* decision into the
  review list, and `execute` acts only on what comes back. The autonomy ladder in
  the Stage A spec — `label`, `archive`, `draft` all at "always" authority — exists
  on paper and nowhere in the implementation. Gating everything is a deviation
  from our own design, not a conservative reading of it.
- **The learning loop cannot see anything the agent does alone.**
  `learn_from_response` (`graph.py:58`) is a graph node keyed on
  `state["response"]`; it only ever sees `thread_id → approve/reject/edit`
  verdicts that arrived through the interrupt. Any action taken without a
  verdict teaches nothing, by construction.
- **`PreferenceStore.record_override()` (`store.py:190`) is called from nothing
  but tests.** In production its counter never increments, so `Rule.precision` is
  permanently `1.0` or `None` and the auto-demotion below 0.5 can never fire. The
  precision machinery has a consumer and no producer.
- **`undo_action()` (`audit.py:172`) already predicted this.** Its docstring:
  *"An undo is also the strongest correction signal the system can get… Callers
  that learn from corrections should weight this above a verdict given at the
  gate."* No caller does.
- **`demote_stale()` (`recency.py:52`) has never had a job.** It exists to stop
  a two-year-old `needs_reply` being treated as a live one, which only matters
  once something sweeps history.
- **`ReviewItem.category` has never been rendered** by any UI, despite being the
  more useful of the two fields when judging whether a decision is right.
- **`GmailClient.list_threads(limit)` (`gmail.py:18`) cannot express a query.**
  It can only say "the most recent N".
- **The snapshot carries `UNREAD` on 48 of 50 threads**, so unread filtering is
  testable against it today.

Four of these — the missing gate, the unreachable learning path, the
producerless `record_override`, the jobless `demote_stale` — are the same hole
seen from different sides. The system was built to learn from corrections and
then only ever asked for corrections at a gate that fires on everything.

## 2. Decisions

Each of these was chosen against alternatives; the rejected ones are recorded
because they will look attractive again later.

### 2.1 Act, then report

Reversible actions at "always" authority execute during the run, **before** the
digest is sent. The digest is a receipt, not a request.

*Why:* approval fatigue is the dominant documented failure mode of
human-in-the-loop systems — when the gate fires constantly, people rubber-stamp
and review becomes theatre. Gating everything guarantees the gate is worthless.
This is also the only option where the agent is useful while the owner is
asleep, which is the entire point of a morning digest.

*Rejected — hold with a timer:* reversible actions fire N minutes later unless
objected to. Becomes act-then-report whenever the owner is asleep, but adds a
deferred-execution scheduler and an undefined resolution when the morning and
evening digests disagree. All of the cost, none of the honesty.

*Rejected — hold until tapped (today's behaviour):* keeps the failure mode.

### 2.2 Two kinds of held, not one

An item is held for one of two reasons, and they do not deserve the same
treatment:

| Tier | Trigger | Why held |
|---|---|---|
| **Authorisation** | `trash`, `confidence < 0.5` | The agent may not, or is not sure |
| **Attention** | `needs_reply`, `security_alert` | The action is harmless; the *thread* wants a person |

One exception, taken directly from the ladder: **trash authorised by a learned
rule auto-executes.** That clause is what lets trash graduate; without it the
ladder's "learned rule, else human approval" row never does anything.

An item can match more than one trigger. `hold_reason` takes the **first match
in the order `trash → low_confidence → needs_reply → security_alert`**, so the
authorisation tier always wins over the attention tier and a low-confidence
`needs_reply` is never one-tap approvable.

### 2.3 The one-tap button covers the attention tier only

`✅` approves held-for-attention items in one tap. Concretely, approving means
**executing the item's proposed action and removing it from the queue**: a
`needs_reply` gets its draft created; a `security_alert` gets its proposed label
applied and stays in the inbox, since the policy's instruction for these is to
surface rather than act. Approving a `security_alert` is therefore an
acknowledgement — "I have seen this" — and its only side effects are the label
and the queue removal. `trash` and `confidence < 0.5` each require their own tap.

*Why:* the design spends its whole effort isolating the items that deserve a
decision. A button that decides all of them at once undoes that. But creating a
draft is not a decision worth two taps.

*Rejected — one tap clears everything:* recreates the rubber stamp on precisely
the set isolated to avoid it.

*Rejected — no blanket approve:* eight taps on a four-item day, for no safety
gained on the harmless half.

### 2.4 Held items carry forward

Held items form a persistent queue that outlives any run. Each digest is *new
arrivals + the outstanding queue*, with carried items pinned to the top of their
section and aged (`waiting since Tue 8:00`).

*Rejected — re-run held threads with fresh context:* burns model calls
re-deciding what was already decided, and renumbers the item every digest, which
breaks index-based correction.

*Rejected — expire to a safe default after N hours:* silently converts a
non-decision into an action. This is the invisible-failure class the project
keeps designing out.

**This decision is what breaks `interrupt()`** — see §3.1.

### 2.5 Corrections teach, including undos

A correction of something already done runs: `undo_action()` → if
`source == "rule"`, `record_override(rule_id)` → mint a `rejected_action` rule
the same way a bare reject at the gate does. If the owner also types a comment:
`add_instruction(text)` (mechanism 1, highest authority) plus a rule for the
action they named, and `mark_overridden()` on any existing rule it contradicts.

*Why the bare undo still teaches:* "bare rejects now teach" was a deliberate fix
in this repo. An undo is the *stronger* signal of the two — the owner has seen
the actual consequence, not just a proposal — so having it teach less than a
reject would be a regression.

*On "weight it above a gate verdict":* `Rule` has no weight field and this design
does not add one. Precedence is made structural instead of numeric — an
undo-sourced correction calls `mark_overridden()` on a rule it contradicts,
where a gate correction merely adds alongside. Same existing machinery, real
precedence, no new concept.

### 2.6 Correction is modal, not syntactic

Tapping an item's number opens it and sets *awaiting text for item N*; the next
non-command message is that item's correction. It clears on any other button
press, so text can never land on a stale target.

*Rejected — index-prefixed text ("3 archive anything from stripe"):* stateless
and fast, but the numbers shift between digests and a stale number silently
corrects the wrong thread.

*Rejected — Telegram reply-to-message:* requires one message per item, which
destroys the single-message digest and floods the chat.

### 2.7 A triaged label, applied in Gmail

Fetch becomes `in:inbox is:unread -label:agent/triaged`, and the agent applies
`agent/triaged` to anything it has processed.

*Why a marker is needed at all:* archive and trash remove `INBOX`, so those items
leave the query naturally. Threads that were **labelled but left in the inbox**,
and threads decided `none`, stay `INBOX + UNREAD` forever and would be
re-triaged, re-charged and re-reported in every digest until read by hand.

*Why in Gmail rather than a local set:* it is visible, so "why did it ignore
this?" has an answer you can see in the mailbox; it survives losing the local
store; and `label` is already at "always" authority, so it grants no new
capability.

*Rejected — mark them read:* destroys unread as a human signal, and
mark-as-read is not on the ladder — it would be a new capability granted
quietly.

### 2.8 Three job shapes

| | Trigger | Scope | Size | Executes |
|---|---|---|---|---|
| Scheduled digest | 08:00 / 18:00 timer | new unread | ~20 | act-then-report |
| `/triage [n]` | owner, now | new unread | ~20 | act-then-report |
| `/backlog [n]` | owner, now | oldest-first sweep of history | 200–500 | **preview, then commit** |

`/backlog` is the one job where act-then-report is wrong: acting on hundreds of
unfamiliar historical threads before the owner has seen anything is the exact
scenario the design exists to prevent, and one bad rule applied 500 times is not
something per-item undo repairs comfortably. It approves by **proposed-action
group** (`archive 340 · recruiter 80 · trash 60`), never 500 rows.

`/backlog` uses the same `in:inbox is:unread -label:agent/triaged` query as the
incremental runs — the difference is ordering and depth, oldest-first and
unbounded by the recency window, rather than a different corpus. It is what
drains an inbox that has never been triaged; the scheduled runs then only ever
see what arrived since.

`demote_stale()` finally earns its keep here: it stops a backlog sweep flooding
the held queue with years-old `needs_reply`.

## 3. Architecture

### 3.1 The graph, and what happens to `interrupt()`

`interrupt()` parks **one run**. Once held items outlive runs and merge across
them (§2.4), "the review" is no longer run-scoped: a held item from Tuesday
morning and one from Tuesday evening belong to different checkpoints, and no
single resume can answer both.

So the interrupt is **scoped to `/backlog`**, the one job that genuinely waits.

```
fetch → triage → propose → partition ─┬─→ auto_execute → enqueue_held → learn → END
                                      └─→ review (interrupt) → execute → learn → END
```

The conditional edge out of `partition` selects on run mode. Scheduled and
`/triage` runs take the upper path and always complete. `/backlog` takes the
lower path, which is today's graph unchanged.

This is not preservation for its own sake. `interrupt()` was the right primitive
when everything was gated and the run genuinely had to wait; act-then-report
removed that for the incremental path, and kept it for the bulk one. Spec
section 7's claim — the interrupt payload is UI-agnostic, so a renderer consumes
it unchanged — stays true for the path that still pauses.

New nodes:

- **`partition`** — splits `ReviewRequest.items` into auto and held by §2.2, and
  routes.
- **`auto_execute`** — runs held-exempt actions through the existing
  `execute_action()` chokepoint, `actor="agent"` or `f"rule:{rule_id}"`. The
  deny-list and dry-run guard are unchanged; nothing new reaches Gmail.
- **`enqueue_held`** — writes held items to the queue.

### 3.2 Data model

**`HeldItem`** — the `ReviewItem` fields plus `first_held_at`, `run_id`, and
`hold_reason` (`trash` | `low_confidence` | `needs_reply` | `security_alert`).
Persisted in a new namespace in the existing store. Not a new database.

**The done list needs no new storage.** `AuditRecord.checkpoint_id` already
carries the run's thread id, so "what this run did" is a filter over `AuditLog`
— and each record already carries the `id` and `undo_token` that `undo_action()`
requires. This is the only reason a ✓ DONE panel with working undo is cheap.

**Protocol change:** `list_threads(limit, query: str = "")`, mapping onto the
Gmail API `q` parameter. `SnapshotGmailClient` filters in memory. Widening this
before `LiveGmailClient` is written is materially cheaper than after.

### 3.3 Callback identity

Today an index resolves against a parked checkpoint, so it always names the list
the human was shown. With a persistent queue, positions shift between digests
and a stale tap on yesterday's message hits today's item.

Callback data gains a short `digest_id` (`a:3:7f2`) — comfortably inside
Telegram's 64-byte cap. A callback whose `digest_id` is not current becomes a
no-op, exactly as decoded garbage does today. Thread ids still never travel in
callback data, so the structural guarantee is unchanged.

## 4. The digest message

```
Inbox · 8:00 · 22 threads
📥 9 archived · 🏷 6 labelled · ✍️ 3 drafted · ⏳ 4 waiting

🗑 TRASH — needs your OK (2)
1. Quartz or Mechanical?
   orientwatchusa.com · 0.41 unsure
   Marketing mail from a watch retailer; no order reference.
2. Townhomes and Detached in Fort Erie from $499k
   shared1.ccsend.com · waiting since Tue 8:00
   Bulk real-estate promotion for a region with no mail history.

✉️ NEEDS REPLY (1)
3. Adhoxhaja just messaged you
   linkedin.com · draft ready
   A person messaged you directly and is waiting.

🔒 SECURITY (1)
4. You shared some Google Account data with Strava
   google.com
   Account permission grant. Surfaced rather than archived, per policy.

✓ DONE (18)
archived 9 · recruiter 5 · reading 4
12 came from rules you taught me

[1] [2] [3] [4]
[📋 Show the 18 done]
[✅ Approve replies & alerts]
```

Grouped by action with counts on top, held items in full, done items reduced to
counts behind a button. Three lines per held item: subject, then sender with
confidence and age, then the agent's `reason`.

**Why three lines here when a one-liner was right before:** the density argument
applied to a fifty-item list. This list is the handful the agent deliberately
would not decide alone, and for those, `reason` is the whole point — it is what
turns a rejection into training data rather than a shrug.

**Why `category` is not shown:** under a header that already says TRASH or NEEDS
REPLY it says the same thing twice. It earns its place only where the section is
broad, which this layout does not have.

Constraints that bind: Telegram's 4096-character cap; **proportional text that
wraps**, so padded columns produce a wall rather than alignment; and a terminal
preview is not a fair test of a phone layout. These layouts were compared at
phone width in a proportional font before being chosen.

Held items page at 8. The queue can grow; the message cannot.

## 5. Correction paths

**Held item.** Tap `N` → the item opens → approve / not this / relabel buttons,
or type free text. **Approve** executes the proposed action and clears the item
from the queue. **Not this** executes nothing, clears the item, and mints a
`rejected_action` rule — the same bare-reject learning the gate already does.
**Relabel** executes the label the owner chose instead and mints a rule for it.
All three call `execute_action()` and `rule_from_correction()` (`store.py:102`)
directly. No graph resume — the learning primitives are pure functions over a
`Thread` and never needed the graph.

**Done item.** `📋 Show the 18 done` → paged list with undo buttons → tap →
`undo_action()` → `record_override()` if rule-sourced → mint a `rejected_action`
rule → optionally type why, which adds an instruction and a corrected rule
per §2.5.

The second path is what finally gives `record_override()` a producer, which is
what lets `Rule.precision` move, which is what lets the existing auto-demotion
below 0.5 fire. Three built mechanisms come alive because one path exists.

## 6. Scheduling

A timer thread inside the existing polling loop. `INBOX_DIGEST_TIMES=08:00,18:00`
in local time. On wake from sleep a missed digest fires late and says so —
`8:00 digest, delivered 9:20` — rather than being silently skipped; a fired
marker per slot per day prevents a double fire.

*Why in-process:* the bot is already long-lived and owns the graph, the store and
the chat state. Cron would mean two processes sharing the queue while the bot's
in-memory digest state (message id, current page) goes stale behind a digest it
did not send.

## 7. Failure modes

| Failure | Behaviour |
|---|---|
| Telegram unreachable at digest time | Retry with backoff; the run's actions already happened and are in the audit log, so the digest is re-sendable, not lost |
| Model unavailable | Run fails loudly with a "triage failed" message. Silence is indistinguishable from an empty inbox |
| Undo of an already-undone record | `undo_action()` raises; surfaced to the chat rather than swallowed |
| Stale `digest_id` callback | No-op |
| Held queue grows unbounded | Paged at 8, aged in the display; ageing is visible pressure, never auto-resolution |

## 8. Testing

- Renderer golden tests at realistic phone width.
- Table tests for the partition rules, including the rule-authorised-trash
  exception.
- Queue carryover and ageing across consecutive runs.
- **Integration: undo → `record_override` → precision drop → auto-demotion
  fires.** Nothing currently exercises this end to end.
- Fetch query construction, including that archived items leave the query and
  labelled-but-inbox items are excluded by `-label:agent/triaged`.
- Scheduler catch-up on wake, and no double fire.
- Stale `digest_id` callbacks become no-ops.
- `/backlog` executes nothing before approval.

## 9. Sequencing

`undo_action()` refuses on `dry_run` records — correctly, since nothing happened
in Gmail to reverse. The ✓ DONE / undo / learn-from-undo half of this design is
therefore **inert until `INBOX_DRY_RUN=false` against live Gmail**.

This ships in two pieces, matching the existing roadmap (Telegram → live Gmail →
dogfood → Stage B):

1. **Against the snapshot:** partition, auto-execute, held queue and carryover,
   the digest message, held-item corrections, scheduling, the `query` parameter
   and the `agent/triaged` marker, `/backlog`. Labelling is an ordinary action
   that the snapshot client already simulates, so the fetch query is correct and
   testable from the start — it does not wait for live Gmail.
2. **Once Gmail is live:** the done panel, undo, and learning from undo.

Everything in piece 2 is buildable and unit-testable in piece 1; it simply
cannot be *used* until real actions have real consequences.

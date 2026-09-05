# Proactive scheduled runs

2026-09-05

Project 3 of the productionalisation milestone, first half. The agent triages
well and acts on what it is confident about, but only when the owner types
`/triage`. Every run in production so far exists because a human remembered to
ask for one. This spec makes the agent run on its own schedule, and makes an
unattended run legible enough to trust.

Scope was fixed before drafting. P3 as recorded covers two independent
subsystems: scheduled runs, and rebuilding verdict collection so `/backlog` can
be wired. They share only the fact that both touch `Bot`; neither needs the
other. **This spec is scheduled runs only.** `/backlog` keeps its safety on and
gets its own spec — see section 9.

One dependency came out of review and is recorded in section 8: the record of
what a run *did* is not durable, so scheduled runs should not be switched on
until it is. That is also its own spec.

## 1. What is true today

Established by reading the code, not assumed.

### 1.1 The polling loop is single-threaded and blocking

`run_polling` (`telegram/bot.py:1227`) is a `while True` that calls
`transport.get_updates(offset)` — a long poll that blocks up to 50s — and then
handles each returned update in order, synchronously. There is no thread, no
queue and no async anywhere in the process.

This is the single most important fact in this design. It means serialisation
between a scheduled run and a typed one is not a mechanism to be built; it is a
property the loop already has. A run can only ever begin at a point in the loop
where no update is being handled.

It also means that while a run is in progress the loop is not polling, so taps
queue at Telegram and arrive afterwards, some with callback ids that have
expired. That is pre-existing behaviour and `_ack` already absorbs it
(`bot.py:981`) — it is noted here so it is not mistaken for something this spec
introduces.

### 1.2 `Bot` holds one slot of per-digest UI state

`Bot.__init__` (`bot.py:50`) carries `_message_id`, `_page`, `_digest_id`,
`_panel`, `_done_page`, `_open_index`, `_pending` and `_run`. There is exactly
one of each, because there is exactly one conversation.

`_start` (`bot.py:870`) resets all of them and increments `_run`, which changes
the LangGraph thread id and mints a new `_digest_id`. Buttons on the previous
digest are keyed by the old `_digest_id` and stop being tappable. This is the
existing contract — `/cancel` (`bot.py:957`) relies on it deliberately — but it
is defined only for a run the owner typed. A run nobody asked for would retire
the buttons under the owner's thumb, and nothing today decides whether that is
allowed.

### 1.3 There is no notion of when a run last happened

Nothing persists a run timestamp. `_runs_started` counts within a process and
is described in its own comment as existing for a test. `audit.jsonl` records
actions, not runs, and a run that executed nothing writes nothing to it.

### 1.4 A run fetches a capped backlog, not a time window

`fetch` (`graph.py:235`) calls `list_threads(limit=state["limit"],
query=settings.inbox_query)`. `inbox_query` (`config.py:111`) is:

    in:inbox is:unread -label:agent/triaged

and the live client (`gmail.py:666`) passes `maxResults=limit` and takes the
first page, which Gmail returns newest-first. `limit` defaults to
`settings.snapshot_size`, which is 50.

So a run does **not** fetch "everything since the last run". There is no time
term in the query at all. **The schedule decides *when* a run happens; the query
decides *what* it fetches, and the two are independent.** A run has no idea when
the previous one was — the recorded `Attempt` (section 3) exists only to decide
whether a slot is owed, and is never consulted by `fetch`. It is also not
`Bot._last_run`, which is the previous graph result held for the digest; the two
are unrelated despite the names, which is why the schedule's is an `Attempt`. A run that fires late, or after
a weekend with the laptop shut, fetches exactly what it would have fetched on
time: everything untriaged. It cannot miss a window because it never had one. It fetches up to 50 of the untriaged unread inbox,
newest first. Three consequences follow, and they matter to section 5:

- **Nothing is lost.** A thread leaves the query only once it is labelled
  `agent/triaged`. If 80 arrived since the last run, 50 are triaged now and 30
  are still in the query at the next slot. The backlog drains; it does not drop.
- **It drains newest-first.** Sustained volume above the cap would leave the
  oldest untriaged mail at the back indefinitely.
- **"Everything since the last run" and "up to 50" coincide only while volume
  stays under the cap**, and today nothing says which case you are in.

### 1.5 The digest already reports how many threads were scanned

`_view` (`bot.py:196`) sets `DigestView.total` to
`len(result["thread_ids"])`, and `digest` (`render_tg.py:270`) renders:

    Inbox · 13:00 · 22 threads
    3 archive · 1 label · 5 waiting

The scanned count is therefore free for scheduled runs, which reuse the same
path. What the header cannot express is the distinction from 1.4: a run that
scanned 50 of 50 and a run that scanned the first 50 of 80 render identically,
and only one of them means the owner is caught up.

## 2. Approach

Three placements for the schedule were considered.

**A scheduler thread pushing a synthetic update into the loop** was rejected. It
is precise to the second, but it puts a thread and a queue into a codebase that
has neither, and converts the free serialisation of 1.1 into an invariant that
must be maintained. The precision buys nothing at a cadence of hours.

**A second launchd job running a CLI subcommand on `StartCalendarInterval`** was
rejected on the same grounds LangGraph Platform was rejected for hosting. The
graph is handed live `HeldQueue` and `PreferenceStore` objects at build time, so
a second process gets its own queue, its own bot and no route into the running
conversation — and `single_instance`'s flock refuses it anyway with exit 3.

**Chosen: a `Trigger` the polling loop asks, once per iteration.** `Trigger` is
pure logic with no clock and no I/O. The loop asks it after draining each update
batch; when it says a slot is owed, the loop calls the same run path `/triage`
uses. Granularity is bounded by the long-poll timeout, ~50s, which is irrelevant
for slots hours apart.

## 3. The `Trigger` seam

A new module, `inbox_agent/schedule.py`. Frozen dataclasses, no dependencies on
the bot, the graph or Gmail:

```python
@dataclass(frozen=True)
class Attempt:
    """What the schedule remembers about the last run of any kind."""
    at: datetime                  # when it was attempted
    slot: datetime | None = None  # the slot it was for; None for a typed run
    count: int = 0                # attempts against `slot` so far
    failed: bool = False          # did it raise


@dataclass(frozen=True)
class Trigger:
    slots: tuple[time, ...]                       # local times, sorted, unique
    grace: timedelta = timedelta(hours=2)         # how stale a slot may be
    cooldown: timedelta = timedelta(minutes=15)   # a run just before covers it
    backoff: timedelta = timedelta(minutes=5)     # wait before the one retry
    max_attempts: int = 2                         # the run, plus one retry

    def owed(self, now: datetime, last: Attempt | None) -> datetime | None
```

One rule, from which every scheduling behaviour in this spec follows:

1. Find the most recent slot instant at or before `now` — today's if one has
   passed, otherwise yesterday's last. No slots at all, or none yet today or
   yesterday, and nothing is owed.
2. If `now - slot > grace`, nothing is owed. *(08:30's triage does not arrive
   at 23:00)*
3. If `last.at >= slot - cooldown` — a run already covered this slot — then
   nothing is owed **unless** that run was a failed attempt at this same slot
   with retries left, in which case see 3.1.
4. Otherwise that slot is owed.

Because only the *most recent* slot is ever considered, a laptop that slept
through three of them wakes owing exactly one run. Catch-up, collapsing and the
grace window are not three features; they are one rule read three ways.

Empty `slots` never owes anything, which is how the schedule is disabled.

### 3.1 The schedule remembers attempts, not successes

**Any run marks the slot, typed or scheduled.** A `/triage` you type is a real
sweep of the same mailbox the slot would have swept, so letting the slot fire
anyway would run twice over an inbox that was just cleared.

The `cooldown` in rule 3 is what makes this work across the boundary. A run at
08:25 is *before* the 08:30 slot, so comparing timestamps alone would still
leave 08:30 owed and fire a second run six minutes after the first. Fifteen
minutes of cooldown says a run that recent already did this slot's work — which
is true for the reason §1.4 gives: the fetch is an untriaged-backlog query, not
a time window, so a run at 08:25 and a run at 08:30 look for exactly the same
thing.

**A failed run also marks the slot, but buys one retry.** `_start` catches,
reports and returns (`bot.py:896`), so a failure is a normal return as far as
the loop is concerned. If a failure left the slot unmarked, the loop would find
it owed again ~50 seconds later — `get_updates` long-polls with `timeout=50`
(`bot.py:1223`) — and again, and again, for the whole two-hour grace window:
on the order of **140 attempts and 140 failure messages**, each one re-running
fetch and classification from the top to reach the same exception, with the loop
not polling for your taps in between. The failures that actually happen here are
persistent — expired OAuth, Ollama not listening, a Gmail 403 — so every one of
those retries is known-doomed.

Marking on failure alone would cost the opposite case: a network blip at 08:59
loses the whole morning digest when trying again a minute later would have
worked. So the slot gets **one** retry, after `backoff` (5 minutes), and then is
done:

- attempt 1 raises → record `failed=True, count=1`. Not owed again until
  `last.at + backoff`.
- at `+5 min`, still inside `grace` → owed once more. Attempt 2 runs.
- attempt 2 raises → `count=2` reaches `max_attempts`. The slot is finished;
  the next slot is the next chance.
- either attempt succeeding → `failed=False`, nothing more owed for that slot.

Two messages in the bad case instead of 140, and the transient blip still gets
its digest. The retry passes through the same quiet gate as any other run
(§4.1), so it cannot interrupt a review either.

The two failure messages say which is which: the first names the retry and when,
the second says the slot is being abandoned until the next one. A failure that
repeats silently is the thing this project keeps having to design against.

**What this deliberately does not do is retry until it works.** A slot that
fails twice waits for the next slot, hours away. That is affordable only because
of §1.4: the query is a backlog, not a window, so a skipped slot costs latency
and never coverage — whatever 09:00 could not do is still sitting there at 12:00.

### 3.2 Persistence

The `Attempt` is written to `store_dir/schedule.json` — a single JSON object,
rewritten after every run attempt of either kind. launchd restarts the process,
and both catch-up and the retry budget are worthless if they die with it. Losing
`count` across a restart would be the retry storm again, one restart at a time.
`store_dir` is already gitignored and already the home of state that outlives a
run.

A missing or unparseable file is treated as no previous attempt, logged at
warning. The consequence is at worst one extra run; refusing to start over an
unreadable scheduling hint would be a worse trade.

`schedule.py` owns both pieces: the pure `Trigger` plus `Attempt`, and a small
`ScheduleStore` that reads and writes the file. `Trigger` stays I/O-free and
clock-free — `ScheduleStore` is what touches disk — so every rule above is
testable with constructed values and no filesystem. `telegram/__main__.py`
builds both from `Settings` and passes them to `run_polling`; `Bot` does not
hold either, because a scheduled run is something the loop decides on, not
something the conversation owns.

## 4. Where it plugs in

`run_polling`'s body extracts into a `_tick` function so the loop is testable
without a `while True`. `_tick` drains the update batch exactly as today, then:

```python
slot = trigger.owed(now(), store.last)
if slot and (bot.idle_for(now()) >= QUIET or now() - slot > MAX_DEFER):
    store.record(bot.run_scheduled(slot))     # attempt written before anything else
```

### 4.1 Deferring to a review in progress

`QUIET` is 5 minutes since the last update `handle_update` (`bot.py:833`)
processed; that method stamps a timestamp. Taps are the only honest signal of
"mid-review" available — there is no explicit done-reviewing event, and the held
queue cannot serve as one because a queue the owner is deliberately leaving
alone would read as a review that never ends.

`MAX_DEFER` is 30 minutes past the slot, after which the run happens regardless.
Without the cap, an owner who taps something every few minutes for an afternoon
starves the schedule silently — and silence is the failure mode this project
keeps having to design against. With it, the worst case is a digest half an hour
late.

Both are module constants, not settings. Neither has a second sensible value,
and section 7 of the clone-and-run spec is the standing argument against adding
configuration nobody asked for: every setting is another value resolvable from
another place.

### 4.2 The run path

`_start` splits into the run itself and its announcement. `run_scheduled` calls
the same code: same `graph.invoke` with `mode="incremental"`, same limit
(`snapshot_size`), same failure message, same `_send_health_alerts`
(`bot.py:919`), same `_show(edit=False)`.

`run_scheduled` returns the `Attempt` describing what just happened — the slot,
the attempt count and whether it raised — and the loop writes it before doing
anything else. Recording the attempt is what bounds the retry, so it must not be
reachable only on the success path; a crash between running and recording is the
one way back into the storm §3.1 exists to prevent.

It omits only the "Triaging up to N threads…" pre-notice. That line exists
because the owner typed something and was owed a sign of life during a wait they
were watching; nobody is watching a scheduled run, and a second unprompted ping
per slot is noise.

A scheduled run resets `_digest_id` exactly as `/triage` does, retiring the
previous digest's buttons. Having waited for the owner to be quiet (4.1) is what
makes that acceptable.

The log line distinguishes the two, so the operational log can tell an
unattended sweep from one someone watched.

### 4.3 A stale tap re-renders instead of refusing

Taps on a superseded digest are refused (`bot.py:1018`) because buttons address
items by **position**, not by thread id — callback data is capped at 64 bytes,
which is why (`callbacks.py:75`). Positions shift as the queue is re-sorted and
items are actioned, so honouring a tap from an older digest would act on
whichever thread now sits at that index. The `digest_id` check is what stands
between the owner and a mis-archived thread, and this spec does not touch it.

What it does change is the refusal. Today the bot answers with a toast telling
the owner to send `/triage` or `/held`. That was tuned for a rare event — with
runs the owner typed, there was usually only one digest in play. Scheduled runs
invert that: every digest except the newest is stale, so a stale tap goes from
an edge case to the normal way an absent owner comes back to their phone. And
the codebase has already learned what a toast is worth here — `_on_callback`'s
own docstring records `approve_attention` being reported from a phone as "the
approve button is not working", because a toast is a banner that vanishes.

So a stale tap **sends the current queue as a new message**, exactly as `/held`
does, and says why in the toast. One tap instead of a toast plus typing.

It sends a new message rather than editing the stale one in place. Editing would
make an old mid-conversation message suddenly live, and the owner scrolling back
later would find that run's record silently replaced by a different one. Old
digests stay as the record of what that run actually said.

It runs nothing: `_run_report=False`, the same as `/held`, so no thread count and
no DONE block belonging to a run that is not this one. `noop` intents — data too
old or too malformed to decode — keep answering with the existing message, since
there is no digest to be stale relative to.

## 5. Saying what the run did not get to

From 1.4 and 1.5: the header reports what was scanned but cannot say whether the
cap bound. This spec closes that.

Before triaging, the run probes the same `inbox_query` for **ids only** with
`maxResults=limit + 1`. No thread bodies are fetched and nothing is classified,
so the cost is one cheap API call.

The probe belongs in the `fetch` node (`graph.py:235`), not in the bot: fetch
already holds the query and the limit, and a second Gmail call from `Bot` would
be a second place that decides what a run's corpus is. It returns a `remaining`
count into `TriageState`, which `_view` (`bot.py:196`) carries onto
`DigestView` beside `total`. This adds an ids-only listing method to the Gmail
client protocol (`gmail.py:61`), implemented by both the live and snapshot
clients so the suite stays offline. If more threads exist than the run will take,
the digest header carries the remainder:

    Inbox · 13:00 · 50 threads · 30 more waiting

The line appears **only when the cap actually bound**. A permanently present
"· 0 more waiting" would be read for a week and then never again; a line that
shows up only when it means something keeps its meaning. It also gives the
newest-first starvation ceiling of 1.4 an alarm — a nonzero remainder at
consecutive slots is volume outgrowing the cap, visible rather than discovered
months later.

This lands in the shared renderer, so a typed `/triage` gets it too. The
question "did it get everything?" is not unique to scheduled runs; it was simply
never answerable before.

### 5.1 The cap stays

Uncapping a scheduled run was considered and rejected. Classification is seconds
per thread — `SLOW_CLASSIFY_SECONDS` exists because of exactly this — so an
uncapped run on a bad day is an hour or more inside a loop that is not polling
for the owner's taps. Capped-and-honest drains at 150 threads/day across three
slots and always says where it stands. Uncapped is unbounded run time in
exchange for a guarantee that, at this volume, is never needed.

## 6. Configuration and visibility

`INBOX_SCHEDULE`, default `""` — **off unless explicitly turned on.** Format is
comma-separated `HH:MM` in local wall-clock time, sorted and de-duplicated at
parse:

    INBOX_SCHEDULE=08:30,13:00,18:00

Malformed input raises at config load, the way `INBOX_TRIAGED_LABEL` and
`INBOX_BODY_BUDGET` already do (`config.py:115`, `config.py:211`). A schedule
that silently disabled itself on a typo would be a proactive agent that is not
proactive and does not say so.

No separate limit setting; a scheduled run uses `snapshot_size`, the same as a
bare `/triage`.

A bot that acts unasked has to be more legible, not less, so four places report
the schedule:

- **The startup banner** (`telegram/__main__.py`) —
  `schedule  : 08:30, 13:00, 18:00 local (next 13:00)`, or `off`. The banner is
  the honest source: it is printed by the process that holds the config, which
  is the whole reason it is trusted over `.env`.
- **`doctor`** — a `Check` with provenance, alongside the rest (`doctor.py:107`).
- **`/status`** (`bot.py:939`) — the next scheduled run, or that there is none.
- **The operational log** — whether a run was typed or scheduled.

## 7. Testing

`python -m pytest` stays the whole feedback loop: offline, no credentials.

**`tests/test_schedule.py` — `Trigger` as pure functions with an injected `now`:**

- a slot that has passed with no prior attempt is owed
- a slot already attempted is not owed *(no double-run)*
- a slot older than `grace` is not owed
- a run inside `cooldown` *before* the slot covers it, so a typed `/triage` at
  08:25 does not produce a second run at 08:31
- a failed attempt is not owed again before `backoff` elapses
- a failed attempt is owed again once `backoff` has elapsed *(the one retry)*
- a second failure reaches `max_attempts` and the slot is never owed again
- a retry that would fall outside `grace` is not owed — `grace` outranks
  `backoff`
- a successful attempt is never owed again regardless of `backoff`
- three missed slots collapse to one owed run
- before the first slot of the day, yesterday's last slot is the candidate
- empty `slots` never owes anything
- a DST-shift day: the repeated local hour does not run twice (rule 2 absorbs
  it), and the skipped one is absorbed by `grace`

**Bot level:**

- `run_scheduled` takes the same path as `/triage` and sends no pre-notice
- a tap 30 seconds ago defers the owed run
- 30 minutes past the slot runs it despite recent taps
- a run that raised is still recorded, with `failed=True`
- a typed `/triage` records an attempt with `slot=None`, and covers a slot
  inside `cooldown`
- the digest header carries the remainder when the id probe exceeds the limit,
  and omits it when it does not
- a tap carrying a superseded `digest_id` sends a fresh queue as a new message,
  does not edit the stale one, and starts no run
- a `noop` intent still only answers, and sends nothing
- a failed run records `failed=True` with its count, and the two failure
  messages differ: the first names the retry, the second the abandonment

**Config and doctor:**

- `INBOX_SCHEDULE` parses, sorts and de-duplicates
- malformed values raise, naming the setting
- empty means off
- the `doctor` check appears with its source

**Persistence:**

- an `Attempt` survives a simulated restart, `count` included — a restart must
  not hand a failing slot a fresh retry budget
- a missing or corrupt `schedule.json` is treated as no prior attempt and logged

Roughly 20 tests, against 848 today.

## 8. Sequencing: the done view comes first

`_done_items` (`bot.py:238`) reads `self._last_run`, a single in-memory slot. So
the DONE panel — and the correction buttons `keep`, `relabel` and `teach_trash`
that index into it — reach only the most recent run, and nothing at all after a
restart.

That is survivable today because every run is one the owner typed and watched.
Three unattended runs a day makes two of every three uncorrectable the moment
the next one fires, and corrections are the learning loop. Section 4.3 gets the
owner back to the *queue*; it does nothing for the *record*, because the record
is not durable to get back to.

The fix is a separate spec: a per-run report persisted in `store_dir` beside the
held queue, and a view over the last few runs carrying the correction buttons.
`audit.jsonl` already holds every action with `ts`, `actor` and
`rule_provenance`, but it knows threads by id only — deliberately — so the
subject and sender have to be persisted rather than recovered.

**`INBOX_SCHEDULE` should stay empty until that ships.** Nothing in this spec
enforces that, and nothing needs to: the schedule is off by default (section 6),
so the order is a decision, not a mechanism. Turning it on before the done view
exists trades the learning loop for a convenience.

## 9. Not doing

- **The durable done view.** Its own spec — see section 8, which is where the
  reasoning and the sequencing live.
- **Addressing callbacks by thread id instead of position.** It would make old
  buttons safe to honour, but 64 bytes of callback data is the reason indices
  exist, and redesigning that boundary to rescue superseded messages is a much
  larger change than 4.3, which reaches the same outcome in one tap.
- **`/backlog` and verdict collection.** The other half of P3, and a separate
  spec. `Bot._resume` stays unreached and the WARNING above it
  (`bot.py:1086`) stays true: `to_response()` still defaults every unnamed
  thread to approve, and nothing here changes that.
- **Smarter-than-scheduled triggers** — a webhook, a push notification, a
  heuristic about when mail arrives. "Proactive means scheduled now, smarter
  later" is a fixed decision of the milestone.
- **Per-slot limits or per-slot policies.** One cadence, one limit.
- **Quiet hours as a setting.** Fixed slots make them implicit: an hour you do
  not want a digest is an hour you do not list.
- **Making the fetch time-windowed.** The backlog query of 1.4 is correct — it
  is what makes an interrupted or capped run recoverable by simply running
  again. Section 5 makes its behaviour visible instead of changing it.

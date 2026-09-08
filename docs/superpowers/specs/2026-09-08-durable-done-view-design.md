# The durable done view

2026-09-08

Project 3 of the productionalisation milestone, second half. P3a made the agent
run on its own schedule; this makes what a run *did* survive the next one.

The two are sequenced, not independent. `INBOX_SCHEDULE` shipped empty on
2026-09-08 (`9a26e0b`) and was set to five slots the same day, on the owner's
instruction and with the cost stated: the record of what a run did lives in one
in-memory slot, so at five runs a day four of every five become uncorrectable.
Corrections are the learning loop. This spec closes that.

Scope was fixed before drafting: **correcting a past run, not undoing one.** See
section 8.

## 1. What is true today

Established by reading the code, not assumed.

### 1.1 The done record lives in one in-memory slot

`Bot._last_run` (`telegram/bot.py:114`) holds the previous graph result. Six
places read it, and every one of them is part of the report-and-correct path:

| Line | Reader | What it needs |
|---|---|---|
| `bot.py:219` | `_view` | header counts, `total`, `remaining` |
| `bot.py:264` | `_done_items` | the executed records joined to proposals |
| `bot.py:369` | item screen | the model's `reason` for one thread |
| `bot.py:724` | `_category_of` | the thread's category |
| `bot.py:754` | `_rule_id_of` | which rule decided it, from the audit `actor` |
| `bot.py:948` | `_run_triage` | the write |

A later run overwrites it. A restart empties it. So the DONE panel and the
`keep` / `relabel` / `teach_trash` buttons that index into it reach only the
most recent run of the current process — and reach nothing at all after
launchd restarts the bot.

This was survivable while every run was one the owner typed and watched. It is
not survivable at five scheduled runs a day.

### 1.2 The queue already solved this problem

`HeldQueue` (`store.py:408`) persists proposals across runs, in its own
namespace (`HELD_NS`, `store.py:26`) inside a `SqliteStore` under `store_dir`.
Its docstring records why it is a separate *instance* from `PreferenceStore`:
rule text needs an embedding index, and a held payload has no `text` field for
that index to key off, so sharing would spend embedding calls on a payload the
index has nothing to do with. It is explicit that they could otherwise share
one store.

`HeldItem` (`models.py:176`) embeds the whole `ReviewItem` rather than
flattening subject and sender out of it, "because ReviewItem is already the
renderer's contract, and re-declaring it here would give the digest two shapes
to render instead of one."

Both decisions apply unchanged to a run report. This spec copies them.

### 1.3 A stable run identity already exists

`ReviewRequest.run_id` is `uuid.uuid4().hex[:8]`, minted once per run
(`graph.py:328`) and already stamped on every held item (`graph.py:405`). It is
random, not a counter, so it does not collide across restarts.

This matters because the LangGraph checkpoint thread id does. It is
`tg-{chat_id}-{run}` where `_run` is an in-process counter starting at zero
(`bot.py`'s `_config`), so after a restart run ids repeat. A past run is not
addressable by checkpoint id, which is the reason section 2 does not read from
the checkpointer.

### 1.4 The data exists on disk, but not in a usable join

- `inbox_agent/checkpoints.sqlite` is a real `SqliteSaver` (`telegram/__main__.py:131`),
  persisting full graph state after every node — including `auto`, which carries
  subject, sender, category, reason and confidence. Unaddressable per 1.3, and
  coupling the UI to it would tie the report to LangGraph's checkpoint schema
  and to a store holding full thread content, which `TriageState` deliberately
  kept payloads out of state to avoid.
- `inbox_agent/audit.jsonl` holds 1,425 records with `id`, `ts`, `thread_id`,
  `action`, `actor`, `rule_provenance`, `undo_token` and `reversible`
  (`models.py:201`). It is the honest record of what reached Gmail — and
  `_view`'s own docstring already says the done counts come from it. But it
  carries no `run_id`, so grouping by run needs a new field, and it knows
  threads by id only, so subjects must still come from elsewhere.

Both are gitignored (`.gitignore:21`, `:31`, and `*.jsonl` at `:22`), so
nothing here leaks real mail into git.

### 1.5 Undo is not wired anywhere

`undo_action` and `undo_candidates` exist in `audit.py` (`:172`, `:51`) and
nothing calls them. `render_tg.py:443` says so directly: "Undo belongs here and
is not here yet." That remains true after this spec — see section 8.

## 2. The record

A new `RunReport`, persisted per run:

```python
class DoneRecord(BaseModel):
    thread_id: str
    item: ReviewItem                            # subject, sender, snippet, category, reason
    actions: list[tuple[str, Optional[str]]]    # (kind, label), as DoneItem already carries
    rule_id: Optional[str] = None               # set when a rule decided it


class RunReport(BaseModel):
    run_id: str
    ran_at: datetime
    total: int                                  # threads scanned
    remaining: int                              # what the cap left behind (P3a section 5)
    done: list[DoneRecord]
```

`DoneRecord` embeds the whole `ReviewItem` for the reason `HeldItem` does: it is
already the renderer's contract. It also happens to carry every field the six
readers in 1.1 need — `category` and `reason` come free with it, and `rule_id`
covers `_rule_id_of`. Nothing else has to be stored to retire `_last_run` as the
report's source.

`total` and `remaining` are on the report rather than left in graph state so
that a past run's digest header renders identically to a live one, with no
special case for "this run is not the current one".

### 2.1 Where it lives

In the store `HeldQueue` already uses, under its own namespace `DONE_NS` —
**not** a third sqlite file. The reason `PreferenceStore` is a separate instance
(1.2) is the embedding index; a run report needs one no more than a held item
does. Sharing avoids threading a third object through `build_graph` and `Bot`
for no gain.

A new `DoneStore` class in `store.py`, mirroring `HeldQueue`: store-agnostic,
its own namespace, `record()` / `get(run_id)` / `recent()`.

### 2.2 Retention

Keep the **last 10 reports**, pruned oldest-first by `ran_at` when a new one is
written. Two days at five slots a day, and a count window is robust to the bot
being off in a way a time window is not — after a quiet weekend, ten runs are
still ten runs, where "the last 48 hours" is empty.

Pruned on write, never on read: the write already touches the store, and a read
that mutates would make `/done` surprising to reason about.

## 3. Writing it

In the graph, beside where held items are enqueued — that node already holds
`executed`, `auto` and `run_id` together, which is exactly the join
`_done_items` performs today at `bot.py:264`. Writing it there rather than in
the bot means a run gets its report whether or not Telegram drove it.

The join is unchanged from `_done_items`: audit records say what went through
the chokepoint, proposals supply the subject and sender an audit record does not
have, and the `agent/triaged` bookkeeping label is skipped because it is applied
to every thread and is not work the owner cares about.

## 4. `_last_run` stops being the source of truth

This is the part that removes the failure rather than routing around it.

All five *readers* in 1.1 re-point at the current run's `RunReport`, looked up
by a `run_id` the `Bot` remembers in place of the graph result. Consequences:

- The live digest's "Show the N done" button reads the store, so **it survives a
  restart too** — not just past runs.
- `_category_of`'s existing fallback to the held queue stays, since a thread can
  be held rather than done.
- `_last_run` is not deleted. `_run_triage` still needs the graph result for the
  immediate post-run flow and the failure message. It simply stops being what
  the report is built from.

## 5. The view

`/done` lists the last 10 runs, newest first, as numbered buttons:

    Done · last 10 runs
    1. today 15:00 · 22 threads · 18 archive · 2 label
    2. today 12:00 · 9 threads · 7 archive
    3. today 09:00 · 31 threads · 24 archive · 3 label

Tapping a run opens its done panel, drawn by the **existing** `done_panel`
renderer (`render_tg.py:435`), because a `RunReport` rebuilds exactly the
`DoneItem` list it already takes. Tapping an item opens the existing item screen
with the existing `keep` / `relabel` / `teach_trash` buttons. Back returns to the
run list.

The only new screen is the run list. Everything below it is the UI that already
exists, pointed at a different source.

`Bot._panel` gains a `"runs"` state, and the done panel becomes run-addressed.
Callback data stays index-based — a run's position in the list of ten, not its
`run_id` — because callback data is capped at 64 bytes (`callbacks.py:75`) and
an 8-character `run_id` plus a digest id plus a kind does not reliably fit. The
existing `digest_id` guard therefore covers the run list unchanged: a tap from a
superseded list is refused for exactly the reason it always was, because
positions shift as new runs push old ones out.

## 6. What this does not fix

`bot.py` is 1384 lines, and its panel state machine already has three states
plus a `_panel_before_item` unwinder. This spec adds a fourth. That is
affordable once and it is named here rather than discovered later — but a split
is its own change with its own risk, and doing it inside a feature spec is how
unrelated refactors get smuggled in. Not now.

## 7. Testing

`python -m pytest` stays the whole feedback loop: offline, no credentials.

**`DoneStore`:**
- a report round-trips with every field, `DoneRecord.item` included
- `recent()` returns newest first
- writing an eleventh report prunes the oldest by `ran_at`, leaving ten
- pruning happens on write, and a read never mutates the store

**Writing:**
- a run writes a report under its own `run_id`, joining executed records to
  proposals the way `_done_items` does
- the `agent/triaged` bookkeeping label is excluded from the rows
- a run that executed nothing still writes a report, so "it ran and did nothing"
  is distinguishable from "it never ran"

**The view:**
- `/done` with no reports says so rather than rendering an empty list
- `/done` with one, and with ten; opening a run; paging inside it
- a tap carrying a superseded `digest_id` is refused exactly as elsewhere

**The restart test — the one that would have caught the original defect:**
- write a report, build a **fresh** `Bot` over the same store, and confirm
  `/done` still shows that run AND a correction on it still teaches a rule.
  Under today's code the panel is empty and the correction is impossible.

**Corrections:**
- `keep` / `relabel` / `teach_trash` on a past run's item produce the same rule
  they produce on the live run's item
- the live digest's DONE button reads the store, not `_last_run`
- a thread Gmail has purged reports that plainly instead of raising (section 8)

## 8. Not doing

- **Undo.** `undo_action` exists and stays unwired (1.5). Correcting teaches the
  agent; undoing moves real mail from a historical view, which is a wider
  surface than this spec's problem needs. Its own decision, later.
- **Splitting `bot.py`.** Section 6.
- **A time-based window, or unlimited history.** Section 2.2.
- **Reading from the checkpointer or from `audit.jsonl` alone.** Section 1.3
  and 1.4 give the reasons: unaddressable after a restart, and no `run_id` plus
  no subjects, respectively.
- **Re-fetch robustness beyond an error message.** Teaching a rule from a past
  run re-fetches the `Thread` from Gmail by id. A thread trashed more than 30
  days ago and purged cannot be fetched. Unreachable inside a 10-run window, so
  it gets a clear message, not a design.
- **`/backlog` and verdict collection.** Still the loaded gun, still its own
  spec. `Bot._resume` stays unreached.

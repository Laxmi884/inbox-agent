# Telegram review UI — design

> Supersedes nothing. Extends `2026-08-26-inbox-agent-design.md` §7 (Interfaces),
> which anticipated this work: *"the interrupt payload is UI-agnostic JSON — a
> Telegram renderer must be able to consume it unchanged."* This spec is the
> test of that claim.

## 1. Purpose

Move the human review gate off the notebook and onto the owner's phone, so the
agent can be used for real on a daily basis rather than only when a Jupyter
kernel happens to be open.

This is the first of three steps in the productionisation milestone that now
sits between Stage A and Stage B:

1. **Telegram review UI** ← this spec
2. Live Gmail over MCP
3. A few days of real use, producing LangSmith datasets

The ordering matters. Telegram touches no live mail, so the review UX can be
dogfooded against the frozen snapshot before any real mailbox is at risk.

## 2. What must not change

The claim under test is that the Stage A graph is UI-agnostic. So:

- **`inbox_agent/graph.py` does not change.** Not one line. If this spec cannot
  be implemented without touching the graph, the §7 claim was false and that is
  itself the finding worth recording.
- **`inbox_agent/audit.py` does not change.** The chokepoint, the deny-list and
  the JSONL record are identical whichever UI is attached.
- **`inbox_agent/models.py` does not change.** `ReviewRequest` and
  `ReviewResponse` are already the contract.
- `render.py` is *joined*, not replaced. The notebook renderer stays; the
  Telegram renderer is a sibling. Both remain usable.

## 3. Constraints

- **Dry-run remains the default.** This milestone changes the UI, not the blast
  radius. `INBOX_DRY_RUN=true` throughout; flipping it is a later, separate
  decision made only after proposals have looked right for several runs.
- **One authorised chat.** The bot answers exactly one Telegram user id. Any
  update from any other id is dropped and logged. Without this, anyone who finds
  the bot can approve actions against the owner's mailbox.
- **The bot token is a credential.** `.env` only, never committed, masked
  wherever printed — same treatment as `LANGSMITH_API_KEY`.
- **Telegram `callback_data` is capped at 64 bytes.** This is a hard protocol
  limit and it shapes the design (§4.3).
- **No new persistence.** The existing `SqliteSaver` checkpoint and
  `PreferenceStore` carry all state. The bot is stateless between updates.

## 4. Architecture

### 4.1 One process

The bot process owns the graph, the checkpointer, the store and the audit log —
the same objects the notebook wires up in its Cell 2, constructed once at
startup.

```
  Telegram  ──updates──▶  bot process  ──▶  build_graph(...)
     ▲                         │                  │
     └────messages/edits───────┘           SqliteSaver + PreferenceStore
```

A single process is deliberate. `SqliteSaver` on one connection avoids the
writer-lock contention that a separate scheduler process would introduce, and
Stage A's whole run is short enough that concurrency buys nothing. If a
scheduled trigger is wanted later, it enqueues to this process rather than
opening its own checkpointer.

### 4.2 Transport: long polling, not webhook

Long polling (`getUpdates`). No public endpoint, no TLS certificate, no inbound
firewall hole, and it runs unchanged on a laptop or a home server.

This is also a **security simplification, not just a convenience**. Spec §9.5 of
the teaching notebook establishes that a resume payload crossing the interrupt
boundary is untrusted input. A webhook would make that boundary a public HTTP
endpoint reachable by anyone who learns the URL. Polling means the only way to
reach the resume path is to be Telegram, authenticated with the bot token, and
to pass the §3 chat-id check.

Webhook remains possible later; nothing in this design forecloses it, and §4.3's
index-based callbacks are what make it safe when it happens.

### 4.3 Callbacks are indices, not thread ids

A naive callback encodes the thread id: `approve:1a040d7d5d69e611`. That is 24
bytes, fits, and is **wrong**.

Instead, callbacks carry the *position* within the review request:

```
  a:3      approve item 3
  r:3      reject item 3
  l:3:7    label item 3 with policy category 7
  p / n    previous / next page
  A        approve all
```

The mapping from index back to `thread_id` is read from **the persisted review
request in the checkpoint** — the exact payload the human was shown — never from
the callback itself.

This makes the §9.5 trust boundary structural rather than validated. A forged or
replayed callback can only ever name an index; an index either falls inside the
batch the human was shown or it does not exist. There is no encoding of a
callback that references a thread outside that batch, because thread ids never
travel in callback data at all.

The graph's existing `"not part of the reviewed batch"` guard stays as the
second line of defence. Two independent mechanisms, same guarantee — the pattern
already used for the deny-list floor (config unions it, the chokepoint re-ORs
it).

### 4.4 Two renderers, one transport

Both renderers from the design discussion are built. They are pure functions
over `ReviewRequest`, exactly as `render.py` is today, and are selected by
`INBOX_TG_MODE=digest|paged`.

**Digest** — one message, all items numbered, an `approve all` button, and a
button that drops into paged mode for the flagged subset.

**Paged** — one message showing a single item with per-item buttons; advancing
edits that same message in place, so a 50-thread review is one notification and
one message thread.

They are not rivals. The expected steady state is **digest as the entry point,
paged for the items that need attention** — the digest clears the confident
majority in one tap, paged handles the handful worth reading. `render.py`
already flags `confidence < LOW_CONFIDENCE` (0.5), which is the natural
selector for "worth reading".

Building both and using both for a few days is how the question gets settled by
evidence rather than by argument — the same method the model registry applies to
model choice.

### 4.5 Making the learning visible

Every review message carries a header:

```
Inbox review · 50 threads · 12 decided by rules you taught me
```

The count is `sum(1 for i in request.items if i.source == "rule")` — already
present on every `ReviewItem`, and already carrying a `rule_id`.

This adds no capability. It surfaces one the system has had since Stage A and
has never shown anyone. It matters because the learning loop's whole value
proposition is that the review gets shorter and more citable over time, and
today that improvement is invisible to the person doing the reviewing. A
rule-decided row is tappable to show the provenance string — which correction
taught it, and when.

## 5. Interfaces

| Piece | Responsibility |
|---|---|
| `inbox_agent/telegram/bot.py` | polling loop, auth check, update dispatch |
| `inbox_agent/telegram/render_tg.py` | `ReviewRequest` → message text + keyboard, both modes |
| `inbox_agent/telegram/callbacks.py` | callback string ↔ intent; index resolution against persisted state |
| `inbox_agent/config.py` | `INBOX_TG_TOKEN`, `INBOX_TG_CHAT_ID`, `INBOX_TG_MODE` added to `Settings` |

New env vars, following the existing `INBOX_*` convention:

```
INBOX_TG_TOKEN=      # bot token from @BotFather
INBOX_TG_CHAT_ID=    # the single authorised user id
INBOX_TG_MODE=digest # digest | paged
```

Commands: `/triage` starts a run, `/status` reports whether a run is parked at
the review gate, `/cancel` abandons a parked run.

## 6. Testing

The bot is tested without Telegram. A fake transport records outbound messages
and injects synthetic callbacks, so the whole path — render, callback parse,
index resolution, `Command(resume=...)`, execute — runs in pytest at the speed
of the rest of the suite.

Cases that must be covered:

- an update from an unauthorised chat id is dropped and logged
- a callback index outside the batch resolves to nothing, and no action executes
- a replayed callback against an already-resumed run does not double-execute
- both renderers produce a `ReviewResponse` semantically identical to the
  notebook's `respond()` for the same inputs
- the header's rule count matches the number of `source == "rule"` items

The last one matters more than it looks: it is the assertion that the visible
claim about learning is true.

## 7. Risks

**The 64-byte callback limit is a hard protocol constraint, not a guideline.**
The index scheme fits comfortably; any future addition that wants to encode more
must be checked against it rather than assumed to fit.

**Telegram edits are rate-limited.** Paged mode edits one message repeatedly. A
fast reviewer tapping `next` can hit the limit; the renderer must degrade to a
brief pause rather than dropping the update.

**Single process is a single point of failure.** If the bot dies mid-review, the
run is parked in the checkpoint and resumes on restart — this is the durability
property §9.3 of the teaching notebook describes, and it should be verified
explicitly rather than assumed.

**A bot token in `.env` is a live credential** with authority over a mailbox once
step 2 lands. It belongs in the dedicated project environment, not in the
Anaconda base environment currently in use.

## 8. Out of scope

Live Gmail writes (step 2, separate spec). Multi-user support. Inline label
creation beyond the policy's fixed taxonomy. Scheduled/automatic triage runs —
`/triage` is manual for this milestone, because a UI nobody has used yet should
not also be firing on a timer. Rich media, threading, or message previews.

## 9. Open items for the implementation plan

- Which Telegram library: `python-telegram-bot` (batteries included, async) vs
  raw `httpx` against the Bot API (no dependency, ~100 lines for polling +
  sendMessage + editMessageText + answerCallbackQuery). Leaning raw, to keep the
  dependency surface small and the trust boundary readable.
- Whether `/triage` should accept a limit (`/triage 10`) for cheap iteration.
- How the paged renderer indicates progress without a second message.

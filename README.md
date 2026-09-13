# inbox-agent

An email triage agent that acts on the confident majority and asks about the rest.

`inbox-agent` fetches unread threads from your Gmail inbox, classifies each one
against a small policy-defined taxonomy, and acts immediately on the confident,
reversible majority — labelling, archiving, or drafting a reply. Anything it is
unsure about, anything that needs a human voice, and anything security-sensitive
is held and surfaced in Telegram as a one-tap approve/reject digest. Every
correction you make there is turned into a durable rule, so the same judgment
call is not asked twice.

It is designed to run unattended: a scheduler fires triage runs at fixed times
of day, and the digest arrives on your phone whether or not you were watching.

```
  Gmail ──▶ fetch ──▶ prefilter ──▶ classify ──▶ apply_rules ──▶ propose ──▶ partition
                       (rules,        (LLM,        (learned       (action      (autonomy
                        no LLM)       per-thread)   prefs)         plans)       ladder)
                                                                                   │
                          ┌────────────────────────────────────────────────────────┤
                          ▼                                                        ▼
                    auto_execute                                          review ⟨interrupt⟩
                   (act then report)                                     (wait for a human)
                          │                                                        │
                    enqueue_held                                                execute
                          │                                                        │
                          └───────────────▶ mark_triaged ──▶ learn ◀───────────────┘
                                                               │
                                                    Telegram digest + rules
```

---

## Table of contents

- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [The safety model](#the-safety-model)
- [Setup](#setup)
- [Check it before you run it](#check-it-before-you-run-it)
- [Run it](#run-it)
- [Going live](#going-live)
- [Operating it](#operating-it)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Quickstart

The safe path: run against a frozen snapshot with dry-run on, confirm the shape
of what it does, then point it at the real mailbox.

```bash
git clone <this-repo> && cd inbox-agent
pip install -e ".[ollama]"        # or ".[openai]"
cp .env.example .env              # defaults are snapshot + dry-run
bash tools/install-hooks.sh       # secret-scan pre-commit hook

# Fill in INBOX_TG_TOKEN and INBOX_TG_CHAT_ID in .env, then:
inbox-agent doctor                # tells you what is missing and why
inbox-agent bot                   # send /triage 10 in Telegram
```

With the defaults (`INBOX_GMAIL=snapshot`, `INBOX_DRY_RUN=true`) nothing reaches
Gmail: every action is classified, held, and audited through the normal flow,
then logged as `simulated` instead of dispatched. See
[Going live](#going-live) when you are ready.

---

## Architecture

### The pipeline

One run is a single LangGraph `StateGraph` invocation
(`inbox_agent/graph.py:610-632`). Nodes, in order:

| Node | What it does | Cost |
|---|---|---|
| `fetch` | Pulls unread threads from Gmail (or the frozen snapshot). | Gmail API |
| `triage` | `prefilter` then `classify`. The prefilter decides anything a learned rule already covers, with a citable rule id; only genuinely novel mail reaches the model. | 1 LLM call per *novel* thread |
| `apply_rules` | Applies learned preferences and instructions on top of the classification. | none |
| `propose` | Turns a category into a concrete action plan (`label`, `archive`, `draft`, `trash`). | none |
| `partition` | Splits the batch by the autonomy ladder — act alone, or hold for a human. | none |
| `auto_execute` | Executes the confident, reversible majority immediately. | Gmail API |
| `enqueue_held` | Queues the rest for review (incremental mode). | none |
| `review` | A durable `interrupt` — the run parks until a human answers (backlog mode). | none |
| `execute` | Executes what the human approved. | Gmail API |
| `mark_triaged` | Applies the `agent/triaged` label so a thread is never re-triaged. | Gmail API |
| `learn` | Turns corrections into durable rules and instructions. | none |

The split after `partition` is the whole design. `auto_execute` is
**act-then-report**: the confident majority is already done by the time you see
the digest, and the digest explains what happened and offers undo. The `review`
interrupt is **ask-first**, and because LangGraph checkpoints it, a parked run
can be resumed hours later from a different process.

### Why classification is per-thread

`classify` is scoped to a single thread with a tight context
(`inbox_agent/classify.py`), because a 12B local model is reliable at one small
structured judgment and unreliable across long loops. Everything the sender
wrote — body **and** the `From`, `Subject`, `Date` headers — is escaped and kept
below an untrusted-content warning, never joined to the instructions. Headers
are attacker-controlled too; treating them as trusted was a real bug.

The body budget is settled at `INBOX_BODY_BUDGET=4000`, measured rather than
guessed: against 13 human labels on threads where the arms disagreed, snippet
scored 2/13 while both 4000 and full scored 10/13. `full` also pushed 4 of 60
threads over the slow-classify threshold and overflowed the context window on a
57,900-character thread — losing the policy from the prompt and returning a
worse answer at confidence 1.0.

### Module map

```
inbox_agent/
├── graph.py            the pipeline above; owns node wiring and the interrupt
├── gmail.py            Gmail client + retry/backoff ladder; the only API caller
├── classify.py         the one place the model is asked to judge
├── prefilter.py        deterministic rule application — zero LLM calls
├── apply/propose       (in graph.py) category → concrete action plan
├── partition.py        the autonomy ladder as code
├── recency.py          age-based demotion; a two-year-old needs_reply is not one
├── audit.py            the single chokepoint every mutation passes through
├── store.py            preference memory; every rule carries its provenance
├── policy.py           versioned behaviour layer (Context Hub + local fallback)
├── schedule.py         when a run is owed — pure logic, no clock, no filesystem
├── models.py           the typed contracts shared across all of the above
├── config.py           settings resolution and validation
├── doctor.py           reads back the same functions the bot uses
├── google_auth.py      OAuth consent + refresh, scoped to gmail.modify
├── single_instance.py  one bot per install, enforced by flock
├── logging_setup.py    rotating operational log + SIGUSR1 stack dumps
└── telegram/
    ├── bot.py          the polling loop, panels, and callback handling
    ├── render_tg.py    digest/queue/done rendering
    ├── callbacks.py    callback_data encode/decode (index-based, digest-scoped)
    └── __main__.py     startup banner, preflight, run_polling
```

### Persistence

Nothing important lives in memory:

| What | Where | Why |
|---|---|---|
| Learned rules and instructions | `store/prefs.sqlite` | Survives restarts; every rule records which correction produced it, how often it fired, and whether it was ever overridden. |
| Held queue and done reports | `store/held.sqlite` | The digest is reconstructable after a crash; `/done` makes the last ten runs correctable. |
| Graph checkpoints | `inbox_agent/checkpoints.sqlite` | Makes the `review` interrupt durable across processes. |
| Schedule state | `store/schedule.json` | Which slots have run, and which are owed. |
| Audit trail | `audit.jsonl` (append-only) | Every attempt — permitted, refused, or simulated. |

All of these hold real email content and are gitignored.

### The Telegram loop

`run_polling` is a single thread: `_tick` drains the update batch, then runs the
scheduler — never during a batch (`bot.py:1660`). That serialisation is
deliberate, because it means a run can only begin at a point where no update is
being handled.

The consequence is worth knowing before you deploy: **while a run is in flight,
the bot does not answer taps.** A 50-thread run is typically 10–20 minutes, most
of it local-model classification. Taps are not lost — `getUpdates` is
offset-based, so Telegram redelivers — but the button spins until the loop comes
back. Callbacks carry the digest id they were drawn for, so a tap on a
superseded digest is refused and re-rendered rather than acted on at the wrong
index.

### The learning loop

A correction in Telegram is not just an undo. `learn` converts it into a durable
rule with provenance, which the `prefilter` then applies on later runs with zero
LLM calls. This is what makes the agent get cheaper and more accurate with use.

One boundary this cannot fix: `newsletter_valuable` vs `newsletter_noise` is not
a prompt-size problem. More body made the model more confident and less correct
there. That distinction belongs to the preference and rules engine, not the
classifier.

---

## The safety model

### What it will never do

Two actions are forbidden at the code level, not just by prompt instruction:

```python
ALWAYS_FORBIDDEN = frozenset({"send_message", "delete_forever"})
```

(`inbox_agent/config.py:37`). `INBOX_FORBIDDEN_ACTIONS` in your `.env` can add
more actions to the deny-list; it cannot remove either of these two — the
chokepoint ORs your configured set with `ALWAYS_FORBIDDEN` on every call, so a
blank, mistyped, or tampered `.env` can't reopen them
(`inbox_agent/config.py:214`, `inbox_agent/audit.py:128`).

The Gmail OAuth scope reinforces this at Google's edge, not just in this code:
the agent requests only `gmail.modify`, which grants read, label, archive,
trash, and draft-create, and grants neither `gmail.send` nor permanent delete
(`inbox_agent/google_auth.py:15-24`).

Every action — permitted, refused, or simulated — passes through one chokepoint,
`execute_action` (`inbox_agent/audit.py:94`), and every call writes an
append-only JSONL audit record, whether or not anything actually happened: a
forbidden action is logged as refused and never dispatched
(`audit.py:128-133`), and in dry-run mode the action is logged as `simulated`
and also never dispatched (`audit.py:135-138`). Only a non-dry-run, permitted
action reaches Gmail (`audit.py:140-148`).

### The autonomy ladder

Whether the agent may act alone on an item, or must hold it for you, is decided
by one pure function, `hold_reason` (`inbox_agent/partition.py:27`). The checks
run in this order, and the first match wins:

1. **`trash`** — held for your approval, *unless* a rule you previously taught
   authorised it (`item.source == "rule" and item.rule_id`). This is the only
   way `trash` is ever taken without asking: the agent cannot decide on its own
   that something is trash-worthy, but a pattern you've already approved once
   can be applied again automatically.
2. **`low_confidence`** — the classifier's confidence fell below the threshold,
   regardless of category.
3. **`needs_reply`** — held not because the action is risky (it's typically a
   harmless draft or label) but because the thread wants a human voice.
4. **`security_alert`** — same reasoning as `needs_reply`.

Authorisation always outranks attention: a low-confidence `needs_reply` is held
as `low_confidence`, never as the one-tap-approvable `needs_reply`
(`partition.py:30-32`). Anything that matches none of the four is acted on by
the agent alone.

### `INBOX_DRY_RUN` defaults to true

If `INBOX_DRY_RUN` is absent from your environment entirely, the agent behaves
as if it were `true`: only an explicit falsey value (`false`, `0`, `no`, `off`)
turns it off (`inbox_agent/config.py:203-205`). With dry-run on, every proposed
action is still classified, held, or approved through the normal flow and
written to the audit log — nothing is skipped — but nothing reaches Gmail. This
is the safe default for a first run against a real mailbox.

### Only one person can drive the bot

`INBOX_TG_CHAT_ID` is the single Telegram user id allowed to send commands or
tap buttons. Every update from any other id is dropped and logged
(`bot.py:_authorised`). Without it the bot would be open to anyone who finds it,
and "anyone" would be able to approve actions against your mailbox.

---

## Setup

Requires **Python 3.11+**.

```bash
git clone <this-repo>
cd inbox-agent
pip install -e ".[ollama]"   # or: pip install -e ".[openai]"
```

`ollama` and `openai` are the two LLM backend extras (`pyproject.toml:32-33`);
pick whichever you'll classify mail with. Both providers are imported lazily, so
installing neither still lets you import the package and run `doctor`.

### 1. An LLM backend

The default is local inference via [Ollama](https://ollama.com):

```bash
ollama serve
ollama pull gemma4:12b-mlx
```

Backend resolution is `openrouter → openai → ollama → offline` unless you pin
`INBOX_LLM_BACKEND`. `offline` means classification cannot run — `doctor`
reports this as fatal.

Ollama also powers the semantic rule index (`INBOX_EMBEDDINGS=auto`). Without
it, rules still match exactly by text; they just aren't semantically
searchable.

### 2. Google Cloud OAuth

So the agent can read and act on your mailbox:

1. Create/select a project at console.cloud.google.com and enable the
   **Gmail API**.
2. Google Auth Platform → Branding → fill in app name and contact email.
3. Google Auth Platform → Audience → External, then either publish the app or
   add yourself under **Test users**. (An unpublished "Testing" app has its
   refresh token revoked by Google after exactly seven days — `doctor` tracks
   this and warns before it happens.)
4. Create an OAuth client of type **Desktop app**, download it, and save it as
   `secrets/credentials.json`.
5. The first live run opens a browser consent flow and writes
   `secrets/token.json`.

Both files are gitignored; neither should ever be committed.

### 3. A Telegram bot

Message `@BotFather`, create a bot, and copy the token it gives you. Then get
your own numeric Telegram user id (for example from `@userinfobot`) — the bot
refuses every update from any other id.

### 4. `.env`

`cp .env.example .env`, then fill in `INBOX_TG_TOKEN` and `INBOX_TG_CHAT_ID` at
minimum. Everything else has a safe default.

### 5. Git hooks

```bash
bash tools/install-hooks.sh   # once per clone; git does not version hooks
```

This installs a pre-commit secret scan. Everything in `tools/` is stdlib-only
on purpose — those run as git hooks, on a fresh clone, before any install step.

---

## Check it before you run it

```bash
inbox-agent doctor
```

prints the effective value of the settings that decide what the bot can do to
your mailbox and whether it starts at all — the mode (`INBOX_GMAIL`,
`INBOX_DRY_RUN`), the deny-list (`INBOX_FORBIDDEN_ACTIONS`), where the audit
trail goes (`INBOX_AUDIT_LOG`), the backend and embeddings mode, credentials,
and OAuth expiry — where each came from (`environment` / `dotenv` / `default`),
and what would stop the bot from starting. It resolves nothing itself: it reads
back the same functions the bot uses, so it can't tell you something the bot
wouldn't also see (`inbox_agent/doctor.py:9-11`).

```
  INBOX_GMAIL               live   <- dotenv   THE REAL MAILBOX
! INBOX_DRY_RUN             False   <- environment   set in the shell, not in .env - this value will NOT survive a restart, and the file says something else or nothing at all
  INBOX_LLM_BACKEND         ollama   <- dotenv
  INBOX_BODY_BUDGET         0   <- default
  INBOX_TRIAGED_LABEL       agent/triaged   <- default
  INBOX_STORE_DIR           inbox_agent/store   <- default
  INBOX_FORBIDDEN_ACTIONS   delete_forever, send_message   <- default
  INBOX_AUDIT_LOG           inbox_agent/audit.jsonl   <- default
  INBOX_EMBEDDINGS          ollama   <- default
  INBOX_TG_TOKEN            AAAAAAA...ZZZZ  (46 chars)   <- dotenv
  INBOX_TG_CHAT_ID          000000000   <- dotenv
X INBOX_GOOGLE_CREDENTIALS  secrets/credentials.json   <- default   missing, and INBOX_GMAIL=live needs it
X INBOX_GOOGLE_TOKEN        secrets/token.json   <- default   missing, and INBOX_GMAIL=live needs it
! oauth consent             unknown   <- -   consent date unknown (token predates tracking) - re-consent to start predicting the 7-day revocation
  policy                    hub:abcdef0123...   <- context_hub

2 fatal, 2 warning(s).  Fix the fatals before starting the bot.
```

Exit code is `1` when any row is fatal, `0` otherwise — safe to use in a
pre-flight script. The two `X` rows above are fatal only because
`INBOX_GMAIL=live`; on a fresh clone still in `snapshot` mode they report `ok`,
since nothing in snapshot mode touches real credentials.

---

## Run it

```bash
inbox-agent bot
```

(equivalently, `python -m inbox_agent.telegram`). This polls Telegram and prints
a startup banner showing the resolved backend, dry-run state, schedule, and
embeddings mode. **The banner is the honest source** — it is printed by the
process itself, so it reflects what that process actually loaded.

Telegram commands:

| Command | Effect |
|---|---|
| `/triage [n]` | Classify up to `n` unread threads now. |
| `/held` | Show the held queue without starting a run. |
| `/done` | The last ten runs, with undo. |
| `/status` | Whether a run is parked for review, the OAuth countdown, and the schedule. |
| `/cancel` | Retire the buttons on the last digest. |

---

## Going live

Two settings gate the real mailbox: `INBOX_GMAIL=live` and
`INBOX_DRY_RUN=false`. Set both **in `.env`, not in your shell.**

The reason is precedence: `load_dotenv` runs with `override=False`, so a value
exported into your shell always wins over `.env` (`inbox_agent/config.py:25`,
`config.py:271-283`), and shell exports die with the shell. If you
`export INBOX_DRY_RUN=false` and restart later, the export is gone, `.env` still
says `true`, and the bot comes back in dry-run — silently. It will still poll,
still classify, still send digests that look completely normal, and just not
touch Gmail.

**Editing `.env` does not reach a bot that is already running.** `load_dotenv`
runs once, at import, so a long-running process holds the file as it was when it
started — and `doctor`, which reads the file, will confidently tell you the
opposite of what the bot is doing. Restart after a `.env` edit.

---

## Operating it

### Scheduling

`INBOX_SCHEDULE` is a comma-separated list of local times, e.g.
`09:00,12:00,15:00,19:00,23:00`. An owed run waits for you to be quiet before it
starts, and gives up waiting after a bounded delay, so a scheduled run never
interrupts you mid-tap.

### Running it as a service (macOS)

```bash
bash tools/install-launchd.sh              # install or re-render
launchctl print gui/$(id -u)/com.inbox-agent.bot
bash tools/install-launchd.sh --uninstall
```

The job restarts on its own, and `tools/run_bot.sh` runs `doctor` first and
refuses to start a misconfigured bot. A second bot is refused by an flock with
exit 3 — two processes on one Telegram token split the updates, so the owner's
taps reach whichever one got them.

Keep the checkout out of `~/Documents`, `~/Desktop` and `~/Downloads`. Those are
TCC-protected and a launchd agent has no consent grant for them: the job exits
126 "Operation not permitted" before any Python runs, and respawns every 60s
saying nothing. `install-launchd.sh` refuses rather than let that happen.

### Logs

Two places, on purpose:

- `~/Library/Logs/inbox-agent/boot.log` — the startup banner, the preflight, and
  any traceback. Nothing rotates it, which is why it only ever receives startup
  output.
- `INBOX_LOG_FILE` — the rotating operational log (5MB × 5).

If the bot ever appears wedged, `kill -USR1 <pid>` dumps a full stack trace of
every thread to `boot.log` (`inbox_agent/logging_setup.py:38`). A healthy idle
bot shows `get_updates` → `ssl.read`; anything else is where it is stuck.

---

## Configuration reference

Every setting is optional unless marked. `inbox-agent doctor` reports the
resolved value and its source for the ones that matter most.

| Variable | Default | What it controls |
|---|---|---|
| `INBOX_GMAIL` | `snapshot` | `snapshot` or `live`. The one switch that decides whether a real mailbox is on the other end. An unrecognised value is refused at load time. |
| `INBOX_DRY_RUN` | `true` | Nothing reaches Gmail while true. |
| `INBOX_TG_TOKEN` | — | **Required.** From `@BotFather`. |
| `INBOX_TG_CHAT_ID` | — | **Required.** The one Telegram user id allowed to drive the bot. |
| `INBOX_TG_MODE` | `digest` | `digest` or `paged`. |
| `INBOX_SCHEDULE` | empty | Local times for unattended runs, e.g. `09:00,19:00`. Empty disables scheduling. |
| `INBOX_LLM_BACKEND` | auto | `ollama`, `openrouter`, `openai`, or `offline`. |
| `INBOX_BODY_BUDGET` | `0` | Characters of body into the classifier prompt. `4000` is the measured optimum for live use. |
| `INBOX_EMBEDDINGS` | `auto` | `auto`, `ollama`, or `none`. Whether rule text is embedded for semantic search. |
| `INBOX_FORBIDDEN_ACTIONS` | `send_message,delete_forever` | Adds to the deny-list; cannot remove the two built-ins. |
| `INBOX_STALE_AFTER_DAYS` | `90` | Age past which a `needs_reply` stops being held as urgent. |
| `INBOX_HTTP_TIMEOUT` | `60` | Seconds before a Gmail socket is abandoned. Must be positive; unbounded is not offered — that is the setting that once wedged the bot for five hours. |
| `INBOX_GOOGLE_CREDENTIALS` | `secrets/credentials.json` | OAuth client (Desktop app). |
| `INBOX_GOOGLE_TOKEN` | `secrets/token.json` | Written by the consent flow, mode 600. |
| `INBOX_AUDIT_LOG` | `inbox_agent/audit.jsonl` | Append-only record of every action. |
| `INBOX_STORE_DIR` | `inbox_agent/store` | Rules, held queue, schedule state. |
| `INBOX_LOG_FILE` | empty (stderr) | Rotating operational log. Set it under launchd. |
| `INBOX_TRIAGED_LABEL` | `agent/triaged` | Label that marks a thread as handled. |
| `CONTEXT_HUB_SKILL` | `inbox-triage` | Context Hub repo holding the versioned policy. |
| `CONTEXT_HUB_TAG` | empty | Must stay blank — it resolves a commit hash or nothing, and any other value 404s. |

### Policy lives in two places on purpose

`inbox_agent/policies/default.md` is the authoring surface; Context Hub is the
publish target. Run `python tools/push_policy.py` after every policy edit. Drift
is loud, not fatal: `load_policy` prints `DRIFT` and serves the hub's copy.

---

## Troubleshooting

Every warning or fatal row `doctor` can print, and what it means:

| Row | Level | Meaning |
|---|---|---|
| Any setting, source `environment` | `!` | Set in your shell, not `.env`. Won't survive a restart; move it into `.env`. |
| `INBOX_GMAIL=live` | note | Just a heads-up that this run touches the real mailbox — not an error. |
| `INBOX_LLM_BACKEND=offline` | `X` | No LLM backend resolved (no Ollama running, no API key set). Classification cannot run. |
| `INBOX_EMBEDDINGS` resolved to `none` while configured `auto` | `!` | Nothing is listening on Ollama, so the rule index is skipped. Rules still match exactly by text — nothing is broken, just not semantically searchable yet. |
| `INBOX_TG_TOKEN` / `INBOX_TG_CHAT_ID` not set | `X` | The bot refuses to start without both. |
| `INBOX_GOOGLE_CREDENTIALS` / `INBOX_GOOGLE_TOKEN` missing | `X` (only when `INBOX_GMAIL=live`) | Snapshot mode never needs these; live mode does. |
| `oauth consent` unknown | `!` | The stored token predates consent-time tracking, so the 7-day "Testing" expiry can't be predicted. Re-consent once to start tracking it. |
| `oauth consent` expiring soon / expired | `!` / `X` | A Google "Testing"-status app has its refresh token revoked after 7 days. Publish the app, or re-consent weekly. |
| `policy` drifted | `!` | The policy pulled from Context Hub and the local `policies/default.md` have diverged. |

### Symptoms that aren't in `doctor`

| Symptom | Cause |
|---|---|
| Buttons spin and nothing happens for minutes | A run is holding the polling loop. Taps are redelivered when it finishes. |
| A tap does nothing and the log shows no callback line | The tap never reached Telegram. Every callback path logs, including refusals. |
| "That digest is out of date" | A newer run superseded the digest you tapped. The current queue is sent instead. |
| Gmail `403 rateLimitExceeded` in the log | Normal under load; the backoff ladder retries with jitter. Sustained 403s mean you're near your quota. |
| Bot appears dead | `kill -USR1 <pid>` and read `boot.log`. `get_updates` → `ssl.read` is healthy idle. |

---

## Development

```bash
python -m pytest        # 1005 tests, ~8s, no network, no credentials needed
```

That is the whole feedback loop. It is fast enough to run after every change, so
run it — a change is not done until it is green.

### Conventions

- Stdlib-only in `tools/` — those run as git hooks, on a fresh clone, before any
  install step.
- A bug fix lands with the test that would have caught it. Most defects here were
  found by running the thing against a real mailbox, not by the suite, so when a
  live failure teaches something, pin it.
- Commit messages say what changed and why, in the imperative, one line.

### Generated notebooks

`inbox_agent_stage_a_explained.ipynb` and `inbox_agent_stage_b_explained.ipynb`
are **generated**. Never edit the `.ipynb` — edit the builder and re-run it:

```bash
python build_teaching_notebook.py   # -> stage_a_explained
python build_stage_b_notebook.py    # -> stage_b_explained
```

`inbox_agent.ipynb` is the hand-written lab; that one is fine to edit. Opening a
generated notebook in Jupyter rewrites its unicode escapes and produces a
several-hundred-line diff with no content change — discard it.

---

## License

No license file is present. All rights reserved by default; add a `LICENSE` if
you intend others to reuse this.

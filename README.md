# inbox-agent

An email triage agent that acts on the confident majority and asks about the rest.

## What it does

`inbox-agent` fetches unread threads from your Gmail inbox, classifies each one
against a small policy-defined taxonomy, and acts immediately on the confident,
reversible majority — labelling, archiving, or drafting a reply. Anything it is
unsure about, anything that needs a human voice, and anything security-sensitive
is held and surfaced in Telegram as a one-tap approve/reject digest. Every
correction you make there is turned into a durable rule, so the same judgment
call is not asked twice.

## What it will never do

Two actions are forbidden at the code level, not just by prompt instruction:

```python
ALWAYS_FORBIDDEN = frozenset({"send_message", "delete_forever"})
```

(`inbox_agent/config.py:37`). `INBOX_FORBIDDEN_ACTIONS` in your `.env` can add
more actions to the deny-list; it cannot remove either of these two — the
chokepoint below ORs your configured set with `ALWAYS_FORBIDDEN` on every call,
so a blank, mistyped, or tampered `.env` can't reopen them
(`inbox_agent/config.py:214`, `inbox_agent/audit.py:128`).

The Gmail OAuth scope reinforces this at Google's edge, not just in this
code: the agent requests only `gmail.modify`, which grants read, label,
archive, trash, and draft-create, and grants neither `gmail.send` nor
permanent delete (`inbox_agent/google_auth.py:15-24`).

Every action — permitted, refused, or simulated — passes through one
chokepoint, `execute_action` (`inbox_agent/audit.py:94`), and every call
writes an append-only JSONL audit record, whether or not anything actually
happened: a forbidden action is logged as refused and never dispatched
(`audit.py:128-133`), and in dry-run mode the action is logged as `simulated`
and also never dispatched (`audit.py:135-138`). Only a non-dry-run, permitted
action reaches Gmail (`audit.py:140-148`).

## The autonomy ladder

Whether the agent may act alone on an item, or must hold it for you, is
decided by one pure function, `hold_reason` (`inbox_agent/partition.py:27`).
The checks run in this order, and the first match wins:

1. **`trash`** — held for your approval, *unless* a rule you previously taught
   authorised it (`item.source == "rule" and item.rule_id`). This is the only
   way `trash` is ever taken without asking: the agent cannot decide on its
   own that something is trash-worthy, but a pattern you've already approved
   once can be applied again automatically.
2. **`low_confidence`** — the classifier's confidence fell below the
   threshold, regardless of category.
3. **`needs_reply`** — held not because the action is risky (it's typically a
   harmless draft or label) but because the thread wants a human voice.
4. **`security_alert`** — same reasoning as `needs_reply`.

Authorisation always outranks attention: a low-confidence `needs_reply` is
held as `low_confidence`, never as the one-tap-approvable `needs_reply`
(`partition.py:30-32`). Anything that matches none of the four is acted on
by the agent alone.

## `INBOX_DRY_RUN` defaults to true

If `INBOX_DRY_RUN` is absent from your environment entirely, the agent
behaves as if it were `true`: only an explicit falsey value (`false`, `0`,
`no`, `off`) turns it off (`inbox_agent/config.py:203-205`). With dry-run on,
every proposed action is still classified, held, or approved through the
normal flow and written to the audit log — nothing is skipped — but nothing
reaches Gmail. This is the safe default for a first run against a real
mailbox.

## Setup

Requires Python 3.11+.

```bash
git clone <this-repo>
cd inbox-agent
pip install -e ".[ollama]"   # or: pip install -e ".[openai]"
```

`ollama` and `openai` are the two LLM backend extras (`pyproject.toml:32-33`);
pick whichever you'll classify mail with. Both providers are imported lazily,
so installing neither still lets you import the package and run `doctor`.

You'll also need:

- **Google Cloud OAuth**, so the agent can read and act on your mailbox:
  1. Create/select a project at console.cloud.google.com and enable the
     **Gmail API**.
  2. Google Auth Platform → Branding → fill in app name and contact email.
  3. Google Auth Platform → Audience → External, then either publish the app
     or add yourself under **Test users**. (An unpublished "Testing" app has
     its refresh token revoked by Google after exactly seven days — `doctor`
     tracks this and warns before it happens; see Troubleshooting below.)
  4. Create an OAuth client of type **Desktop app**, download it, and save it
     as `secrets/credentials.json`.
  5. The first live run opens a browser consent flow and writes
     `secrets/token.json`. Both files are gitignored; neither should ever be
     committed.
- **A Telegram bot**: message `@BotFather`, create a bot, and copy the token
  it gives you. Then get your own numeric Telegram user id (for example from
  `@userinfobot`) — the bot refuses every update from any other id.
- **`.env`**: `cp .env.example .env`, then fill in `INBOX_TG_TOKEN` and
  `INBOX_TG_CHAT_ID` at minimum. Everything else has a safe default.

## Check it before you run it

```bash
inbox-agent doctor
```

prints the effective value of every setting, where it came from
(`environment` / `dotenv` / `default`), and what would stop the bot from
starting. It resolves nothing itself — it reads back the same functions the
bot uses, so it can't tell you something the bot wouldn't also see
(`inbox_agent/doctor.py:9-11`). This is real output, captured from a
configured checkout with `INBOX_DRY_RUN=false` additionally exported into the
shell (to demonstrate the warning below); the Telegram token and chat id —
the only two fields that identify a real person or credential — are replaced
with placeholders of the same shape:

```
  INBOX_GMAIL               live   <- dotenv   THE REAL MAILBOX
! INBOX_DRY_RUN             False   <- environment   set in the shell, not in .env - this value will NOT survive a restart, and the file says something else or nothing at all
  INBOX_LLM_BACKEND         ollama   <- dotenv
  INBOX_BODY_BUDGET         0   <- default
  INBOX_TRIAGED_LABEL       agent/triaged   <- default
  INBOX_STORE_DIR           inbox_agent/store   <- default
  INBOX_EMBEDDINGS          ollama   <- default
  INBOX_TG_TOKEN            AAAAAAA...ZZZZ  (46 chars)   <- dotenv
  INBOX_TG_CHAT_ID          000000000   <- dotenv
X INBOX_GOOGLE_CREDENTIALS  secrets/credentials.json   <- default   missing, and INBOX_GMAIL=live needs it
X INBOX_GOOGLE_TOKEN        secrets/token.json   <- default   missing, and INBOX_GMAIL=live needs it
! oauth consent             unknown   <- -   consent date unknown (token predates tracking) - re-consent to start predicting the 7-day revocation
  policy                    hub:13ac11f11da376e588cb84825ed88d05298f27aaaa90c6f694abf572f34161d9   <- context_hub

2 fatal, 2 warning(s).  Fix the fatals before starting the bot.
```

The `!` line on `INBOX_DRY_RUN` is the one worth reading twice: it fired
because that run had `INBOX_DRY_RUN=false` **exported in the shell**, while
`.env` on disk still said something else. See "Going live" below for why
that distinction matters. The two `X` rows are fatal here only because
`INBOX_GMAIL=live` on this checkout; on a fresh clone still in `snapshot`
mode (the default), those two rows report `ok` instead, since nothing in
snapshot mode touches real credentials. Exit code is `1` when any row is
fatal, `0` otherwise — safe to use in a pre-flight script.
`python3 -m inbox_agent.doctor` runs the identical check (verified: both
produce byte-identical output); `inbox-agent doctor` is the packaged entry
point for it.

## Run it

```bash
inbox-agent bot
```

(equivalently, `python -m inbox_agent.telegram` — `bot` is a thin wrapper
around the same startup path, not a second copy of it). This polls Telegram
and prints a startup banner showing the resolved backend, dry-run state, and
embeddings mode. Once it's running, send `/triage 10` in Telegram to classify
up to 10 unread threads: the confident majority is acted on immediately
(or simulated, if `INBOX_DRY_RUN=true`), and the rest appear as a digest with
approve/reject buttons. Other commands: `/held` to see the held queue without
starting a new run, `/status`, `/cancel`.

## Going live

Two settings gate the real mailbox: `INBOX_GMAIL=live` and
`INBOX_DRY_RUN=false`. Set both **in `.env`, not in your shell.**

The reason is precedence: `load_dotenv` runs with `override=False`, so a
value exported into your shell always wins over `.env`
(`inbox_agent/config.py:25`, `config.py:271-283`), and shell exports die with
the shell. If you `export INBOX_DRY_RUN=false` and restart later, the export
is gone, `.env` still says `true`, and the bot comes back in dry-run —
silently. It will still poll, still classify, still send digests that look
completely normal, and just not touch Gmail. `doctor` is what catches this
before it happens: any setting sourced from `environment` gets a `!` warning
naming exactly this risk (`inbox_agent/doctor.py:50-54`).

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

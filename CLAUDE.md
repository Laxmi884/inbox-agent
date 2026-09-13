# Inbox agent

A Gmail triage agent: it classifies inbox threads, executes what it is confident
about, and asks a human on Telegram about the rest. `inbox_agent/` is the
package, `tests/` the suite, `docs/superpowers/` the specs and plans.

## Verify

    python -m pytest        # 1005 tests, ~8s, no network, no credentials needed

That is the whole feedback loop. It is fast enough to run after every change,
so run it — a change is not done until it is green.

For a config/credential problem, `inbox-agent doctor` reports every setting with
where it came from. Prefer it over reading `.env` by hand: the value that bites
is the one resolvable from two places.

## Things that will bite you

**The two `_explained` notebooks are generated. Never edit the `.ipynb`.**
`build_teaching_notebook.py` → `inbox_agent_stage_a_explained.ipynb`;
`build_stage_b_notebook.py` → `inbox_agent_stage_b_explained.ipynb`. Edit the
builder and re-run it. (`inbox_agent.ipynb` is the hand-written lab; that one is
fine to edit.) Opening a generated notebook in Jupyter rewrites its unicode
escapes and produces a several-hundred-line diff with no content change — throw
that away, don't commit it.

**`/backlog` must stay unwired.** `Bot._resume` is unreached and deliberately
broken; read the WARNING above it in `telegram/bot.py:1086`. `to_response()`
defaults every unnamed thread to *approve*, so resuming today would blanket-
approve a 500-thread historical sweep — the exact outcome previewing exists to
prevent. Wiring the command is one line and the most destructive edit available
in this repo. Plan 3 rebuilds verdict collection first.

**Never `export` `INBOX_*`.** `.env` already carries `INBOX_GMAIL=live` and
`INBOX_DRY_RUN=false`. A shell export beats the file, does not survive a
restart, and has silently changed which mailbox a run wrote to.

**And editing `.env` does not reach a bot that is already running.**
`load_dotenv` runs once, at import, so a long-running process holds the file as
it was when it started. `inbox-agent doctor` reads the file and will
confidently tell you the opposite of what the bot is doing. On 2026-09-04 a bot
started at 18:12 kept sending LangSmith traces for seventeen hours after the
file was set to `false` at 18:40. The banner is the honest source — it is
printed by the process — so check it, and restart after a `.env` edit.

**launchd runs the live bot; do not start one by hand.**

    launchctl print gui/$(id -u)/com.inbox-agent.bot    # state, restarts, exit code
    bash tools/install-launchd.sh                       # install or re-render
    bash tools/install-launchd.sh --uninstall           # hand it back

It restarts on its own, including after SIGKILL, and `tools/run_bot.sh` runs
`doctor` first and refuses to start a misconfigured bot. A second bot is
refused by an flock with exit 3 — two processes on one Telegram token split the
updates and the owner's taps reach whichever one got them.

Logs are in two places on purpose: `~/Library/Logs/inbox-agent/boot.log` has
the banner, the preflight and any traceback, and `INBOX_LOG_FILE` has the
rotating operational log. Nothing rotates the first, which is why it only ever
receives startup output.

If you must run one by hand, stop the job first, and double-fork so it lands on
PPID 1: `( nohup python3 -u -m inbox_agent.telegram >> /tmp/bot.log 2>&1 & )`.
An agent tool call's process group killed the live bot for an hour once, and
macOS has no `setsid(1)`.

**This checkout must stay out of `~/Documents`, `~/Desktop` and `~/Downloads`.**
Those are TCC-protected, and a launchd agent has no consent grant for them: the
job exits 126 "Operation not permitted" before any Python runs and respawns
every 60s saying nothing. The repo was moved out of `~/Documents/Projects` on
2026-09-05 for exactly this. `install-launchd.sh` refuses rather than let it
happen again.

**The body budget is settled: `INBOX_BODY_BUDGET=4000`.** Measured, not
guessed. Against 13 human labels on the threads where the arms disagreed,
snippet scored 2/13 and both 4000 and full scored 10/13 — so the body matters
and the last 50,000 characters do not. `full` also put 4 of 60 threads over
`SLOW_CLASSIFY_SECONDS` and overflowed the context window on a 57,900-char
thread, losing the policy from the prompt and returning a worse answer at
confidence 1.0. Do not re-open this without new evidence.

One thing that experiment could NOT fix: `newsletter_valuable` vs
`newsletter_noise` is not a prompt-size problem. More body made the model more
confident and less correct there. That boundary belongs to the preference and
rules engine.

**Experiment tooling.** `tools/ab_body.py --sample N` draws a stratified corpus
across category and year (fetching is cheap; `--arms` is hours of local GPU, so
it is opt-in and separate). `tools/label_ab.py` records human verdicts on the
disagreements. Everything they write lands in `inbox_agent/store/`, which is
gitignored because it holds real email — so the labelled reference set has **no
backup**, and re-labelling is the one cost that cannot be automated away.

**Policy lives in two places on purpose.** `inbox_agent/policies/default.md` is
the authoring surface; Context Hub is the publish target. Run
`python tools/push_policy.py` after every policy edit. Drift is loud, not fatal:
`load_policy` prints DRIFT and serves the hub's copy. `CONTEXT_HUB_TAG` must
stay blank — it resolves a commit hash or nothing, and any other value 404s.

**The history was rewritten on 2026-09-12 and the remote is PRIVATE.** A
Telegram bot token used to be in history; `git filter-repo` replaced it with a
placeholder of the same shape across all 208 commits, removed the CAB notebook
(which carried real Jira data and an email address in its executed outputs),
and scrubbed absolute home paths. Every commit SHA before that date changed —
SHAs quoted in older docs and plans no longer resolve. The pre-rewrite history
is at `~/Projects/inbox-agent-backup-20260912-200500`.

**The bot token is still NOT rotated.** Removing it from history does not
invalidate it. Rotate it in @BotFather (`/revoke`), update `.env`, restart the
bot, and only then consider making the repo public. Until that is done, keep
the GitHub repo private.

## Conventions

- Stdlib-only in `tools/` — those run as git hooks, on a fresh clone, before any
  install step.
- A bug fix lands with the test that would have caught it. Most defects here
  were found by running the thing against a real mailbox, not by the suite, so
  when a live failure teaches something, pin it.
- Commit messages say what changed and why, in the imperative, one line.
- `bash tools/install-hooks.sh` once per clone; git does not version hooks.

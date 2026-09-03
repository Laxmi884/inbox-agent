"""Generates inbox_agent_stage_b_explained.ipynb from cell definitions.

Same reason build_notebook.py exists for the CAB notebook: a 100-cell .ipynb is
JSON, and hand-editing JSON is how notebooks quietly acquire duplicate ids,
stale outputs and cells nobody meant to keep. The source of truth is this file.

The Stage A notebook (`inbox_agent_stage_a_explained.ipynb`) stops where the
Telegram bot begins. It is accurate about the engine - fetch, prefilter,
classify, the partition ladder, the chokepoint, the graph - and its section 9.7
then predicts what production would look like. Some of those predictions came
true, one did NOT, and the whole user-facing half of the system arrived
afterwards. This notebook is that half.

Run:  python3 build_stage_b_notebook.py
"""

import json
from pathlib import Path

CELLS = []


def md(source: str):
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str):
    CELLS.append(("code", source.strip("\n")))


# ---------------------------------------------------------------- intro
md(r'''
# Stage B, explained — the half you actually touch

`inbox_agent_stage_a_explained.ipynb` is the engine. It takes the pipeline apart
and explains every piece, and it is still accurate: `fetch → prefilter →
classify → apply_rules → propose → partition`, the audited chokepoint, the
autonomy ladder, LangGraph's state and checkpointing and `interrupt()`.

It stops at the point where a person enters the picture.

This notebook is that part: the Telegram bot, the digest, the held queue, the
callback protocol, the learning loop, live Gmail, and how you run the whole
thing on a machine that is not this one. Everything you do when you type
`/triage` happens in code Stage A never mentions.

**Same rule as Stage A: nothing here is retyped.** Every listing is
`inspect.getsource()` on the real imported object, and every PROOF cell runs a
real test by node id. If a listing looks wrong, the source changed and this
notebook is telling you the truth about it.

### Start here: one prediction Stage A got wrong

Stage A's section 9.7 looked ahead to production and said:

> *"The trust boundary stops being theoretical. **A Telegram webhook is a public
> endpoint.** Section 9.5's guard is the thing standing between a replayed
> callback and an action on an email nobody approved."*

The guard is real and section 16 covers it. But the webhook never happened, and
it did not happen **for the reason that sentence raises**. Read the first lines
of the module that ended up being written:
''')

code(r'''
# --- helpers, same contract as the Stage A notebook ----------------------
import inspect, subprocess, sys, json
from pathlib import Path


def src(obj, *, head=None):
    """Print the real source of the real object. No retyping, no drift."""
    code = inspect.getsource(obj)
    if head:
        code = "\n".join(code.splitlines()[:head]) + "\n    ..."
    print(code)


def test(node_id):
    """Run one pytest node and report the outcome unambiguously.

    NOT the Stage A notebook's `-q` version. This project's pytest config
    suppresses the summary line, so `-q` prints dots and no count - and a
    reader cannot tell "1 passed" from "no tests ran". A PROOF cell that
    cannot distinguish those is the documentation form of a test that
    cannot fail, which is the one mistake this project keeps making.
    """
    r = subprocess.run([sys.executable, "-m", "pytest", node_id,
                        "--no-header", "-p", "no:cacheprovider"],
                       capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    verdict = lines[-1] if lines else "(no output)"
    ok = r.returncode == 0 and "no tests ran" not in r.stdout
    print(f"{'PASS' if ok else 'FAIL'}  {node_id.split('::')[-1][:66]}")
    print(f"      {verdict.strip()}")


def doc(module, lines=14):
    """The module docstring - where this project keeps its design arguments."""
    print("\n".join((module.__doc__ or "").strip().splitlines()[:lines]))


print("helpers ready")
''')

code(r'''
from inbox_agent.telegram import bot as bot_mod

doc(bot_mod, lines=12)
''')

md(r'''
**Long polling, not webhook — and the reason is the trust boundary itself.**

A webhook is a URL on the public internet. Anyone who learns it can POST to it,
so every request has to be treated as hostile and authenticated on the way in.
Long polling inverts that: the bot dials **out** to `api.telegram.org` and asks
"anything for me?". There is no inbound port, nothing to discover, and nothing
to forge at the transport layer.

So Stage A was right that the boundary matters and wrong about where it would
sit. That is worth more than a correct prediction would have been, because it
shows the choice being made: the guard in section 16 exists *anyway*, on top of a
transport that already removed the attack it was written for. Defence in depth
is not a slogan here — it is why `_authorised` and the digest-id check are
both still in the code when neither is load-bearing against a public endpoint.
''')

# ---------------------------------------------------------------- §14
md(r'''
---
# §14 · From "propose and wait" to "act and report"

Stage A's pipeline proposed fifty things and waited for fifty answers. Section 12
introduced `partition` — the autonomy ladder — which splits that batch by
authority. Stage B is what happens on each side of that split once a real person
is at the other end of it.

The single most important thing to understand about `/triage` today: **it acts
first.** By the time your phone buzzes, the confident and reversible majority has
already happened. The digest is a report, not a request.
''')

code(r'''
from inbox_agent.telegram.bot import Bot

# The comment above the graph invocation IS the design decision.
src_lines = inspect.getsource(Bot._start).splitlines()
print("\n".join(src_lines[:26]))
''')

md(r'''
Two consequences that are easy to miss:

**`mode="incremental"`, never `"backlog"`.** The graph still supports both — the
`interrupt()` path Stage A section 9.4 explains is intact and tested — but nothing
in the bot drives it. Section 21 covers why, and why you should not wire it up
casually.

**Failure is reported honestly.** Look at the `except` clause: it does not say
"nothing was executed", because that would be a lie. A run can raise *after*
`auto_execute` has already pushed actions through the chokepoint. This is not
theoretical — it happened on a live mailbox on 2026-09-03, and section 19 is that
story.
''')

# ---------------------------------------------------------------- §15
md(r'''
---
# §15 · Who is allowed to talk to it

The bot polls one chat and obeys one person. Everything else is dropped and
counted.
''')

code(r'''
src(Bot._authorised)
''')

md(r'''
### WHY — an allow-list of exactly one

`INBOX_TG_CHAT_ID` is not the bot's id. It is **your** Telegram user id, and it is
the same number no matter which bot you are talking to — the bot's identity comes
from its token. That distinction matters the moment you run a second bot for
testing: a dev bot needs a different *token* and the *same* chat id.

Without this check the bot is open to anyone who finds it. Telegram bot usernames
are enumerable, so "finds it" is not a hypothetical.
''')

code(r'''
# PROOF
test("tests/test_tg_bot.py::test_update_from_an_unauthorised_chat_is_dropped")
''')

# ---------------------------------------------------------------- §16
md(r'''
---
# §16 · The callback protocol — 64 bytes of hostile input

Every button in the digest carries a `callback_data` string. Telegram caps it at
**64 bytes**, and that cap drives three design decisions that look arbitrary
until you know it exists.
''')

code(r'''
from inbox_agent.telegram import callbacks
from inbox_agent.telegram.callbacks import encode, decode

doc(callbacks, lines=16)
''')

code(r'''
# What actually travels in a button.
for kind, args in [("open", (2,)), ("approve", (2,)), ("relabel", (2, 3)),
                   ("scope_narrow", (2,))]:
    data = encode(kind, *args, digest_id="7f2a")
    print(f"{data:34s} {len(data.encode()):2d} bytes  ->  {decode(data)}")
''')

md(r'''
### WHY #1 — a position travels, never a thread id

A callback carries **the index of the row you tapped**, and the bot resolves that
index against the list it rendered. A Gmail thread id in a button would be an
identifier for a real object in your mailbox, arriving from outside the process,
in a string an attacker controls. The index is meaningless without the list, and
the list is server-side state.

### WHY #2 — `decode` never raises

Callback data is untrusted input. A decoder that throws turns a malformed tap
into a traceback in the polling loop, which stops the bot. It returns a no-op
intent instead.

### WHY #3 — the digest id, which is a staleness guard

Every digest gets a short random id, and every button in it carries that id. Tap
a button in yesterday's digest and the bot recognises the id as stale and says so
— rather than resolving index 3 against **today's** list and acting on a
completely different email. This is the same class of bug as a replayed payload,
and it is reachable by accident, not just by malice: Telegram keeps old messages
on screen forever.
''')

code(r'''
# PROOF
test("tests/test_tg_callbacks.py::test_thread_ids_never_appear_in_callback_data")
test("tests/test_tg_callbacks.py::test_decode_never_raises_on_hostile_input")
test("tests/test_tg_callbacks.py::test_negative_index_cannot_wrap_around_to_a_real_thread")
test("tests/test_tg_bot.py::test_a_stale_tap_says_so_instead_of_doing_nothing")
''')

md(r'''
The 64-byte cap is protocol, not preference: exceed it and Telegram rejects the
message at send time, which on a phone looks like **a keyboard that simply does
not appear**. There is no error to see. That failure mode is why the cap is
asserted at the last page of a long digest, where indices reach three digits —
measuring page 0 alone never encodes a multi-digit index at all.
''')

code(r'''
# PROOF
test("tests/test_tg_render.py::test_every_rendered_callback_is_within_the_byte_cap")
test("tests/test_tg_render.py::test_every_item_view_callback_is_within_the_byte_cap")
''')

# ---------------------------------------------------------------- §17
md(r'''
---
# §17 · The digest and the held queue

Two lists, and the difference between them is **tense**.

| list | what it holds | the question it asks |
|---|---|---|
| **done** | what the run already did | *was that right?* |
| **held** | what the run refused to do alone | *should I?* |

The held queue is durable and survives restarts, because an item waiting on you
is work in flight, not a rendering detail.
''')

code(r'''
from inbox_agent.store import HeldQueue

src(HeldQueue, head=18)
''')

md(r'''
### WHY — the queue outlives the message

Stage A parked the run itself in a LangGraph checkpoint and rendered from it. That
worked, and it coupled "what is waiting for you" to "a suspended graph execution".
Restart the process, or send a second `/triage`, and the relationship between the
two became hard to reason about.

Held items are now their own store with their own namespace. The graph finishes
every run. Nothing is suspended. `/held` reads a queue, not a checkpoint.
''')

code(r'''
# PROOF - a held item survives a bot restart
test("tests/test_persistence.py")
''')

# ---------------------------------------------------------------- §18
md(r'''
---
# §18 · The learning loop, both paths

One human correction becomes one durable, attributable rule. Stage A section 11
showed this from the terminal. In the bot there are **two** doors into it, and
until 2026-09-03 only one of them was open.
''')

code(r'''
src(Bot._verdict, head=30)
''')

md(r'''
### The verdict vocabulary is buttons, not free text

`Keep in inbox`, `Label as …`, `Trash instead`. Each maps to exactly one action
sequence, so what gets taught is exactly what the button said. Free text would
need parsing, and a misparsed instruction becomes a durable rule that acts on
mail you have not seen yet.

### Then scope, asked separately

"Never archive this sender" and "never archive any valuable newsletter" are
different instructions behind the same tap, and only you know which you meant.
Asking costs one tap. Guessing costs a rule that reaches mail you never intended.
''')

code(r'''
src(Bot._ask_scope)
''')

md(r'''
### The door that was shut

Held items — the ones the agent **stopped to ask you about** — offered only
`Approve` and `Not this`. The correction vocabulary above was wired to the done
list alone, so on the one screen designed to ask you a question, "this is
learning, keep it in the inbox" was unsayable.

A bare `Not this` taught `actions=[none]`: it recorded what *not* to do and never
what to do. Found by dogfooding on 2026-09-03 — four taps produced four `none`
rules. One of them, on `hello@mail.langchain.com`, stops the agent trashing that
sender and will never label it `learning`. Half a lesson.

Worse and less visible: the held branch dropped the **rule id**, so a rule could
be corrected from the queue every morning and never lose a point of precision. A
learning loop that silently does not learn.

Both are fixed. A correction from the held queue now also **acts** — the thread is
still waiting, you have just answered the question that held it — then teaches,
then drains.
''')

code(r'''
# PROOF - the door, and what comes through it
test("tests/test_tg_render.py::test_a_held_item_offers_corrections_as_well_as_verdicts")
test("tests/test_tg_bot.py::test_a_held_item_can_be_corrected_not_just_approved")
test("tests/test_tg_bot.py::test_correcting_a_held_item_applies_it_to_that_thread_now")
test("tests/test_tg_bot.py::test_correcting_a_held_item_does_not_also_run_what_was_rejected")
test("tests/test_tg_bot.py::test_correcting_a_rule_held_item_overrides_that_rule")
''')

md(r'''
`Not this` survives deliberately. A refusal with no replacement **is** signal —
the digest design counts every reject as a candidate rule, and requiring an edit
is exactly why "skip" never taught this agent anything. It simply stopped being
the only sentence available.
''')

code(r'''
# PROOF - the bare reject still teaches what it always taught
test("tests/test_tg_bot.py::test_a_bare_reject_still_teaches_that_the_action_was_wrong")
''')

# ---------------------------------------------------------------- §19
md(r'''
---
# §19 · Live Gmail — where the snapshot stops protecting you

Stage A ran against a frozen 50-thread snapshot, and section 3 explained why: no
iteration can touch the real mailbox. `LiveGmailClient` implements the same
seven-method `Protocol`, so the swap is one argument to `build_graph`.

Everything below is what the snapshot was hiding.
''')

md(r'''
### 19.1 · Consent expires, and it does not warn you

The OAuth refresh token dies. If the Google Cloud consent screen is in
**Testing**, Google revokes it after seven days — but `invalid_grant` is also
what you get for a revoked grant and for a password change, so the error names
no cause on its own.

The failure mode is silence: the bot simply stops working, and you find out when
a `/triage` gets no reply.
''')

code(r'''
from inbox_agent import google_auth

src(google_auth.record_consent)
''')

md(r'''
A sidecar next to the token records **when consent was given**, because the token
file itself is rewritten on every refresh and its mtime tells you nothing about
when the seven-day clock started. That single timestamp is what makes expiry
predictable instead of a surprise.

Written on the consent path only — a refresh must never reset the clock, because
the clock is not measuring refreshes.
''')

code(r'''
# PROOF
test("tests/test_google_auth.py")
''')

md(r'''
### 19.2 · Rate limits, and a 403 that means two opposite things

On 2026-09-03 a live `/triage 20` died partway through, having already applied
**32 actions to 17 real threads**. The owner saw an HTTP error and a half-triaged
inbox.

The cause is worth internalising because it generalises far beyond Gmail:

> **403 is the one status where Gmail overloads two opposite meanings** — a
> permission denial that will never succeed, and "you are going too fast" which
> will succeed shortly. Gmail does **not** use 429 for its per-user limits.

The retry policy keyed on status alone, so a transient quota read as a permanent
refusal. The comment above it was right about every 4xx except this one.
''')

code(r'''
from inbox_agent import gmail

src(gmail._is_rate_limit)
src(gmail._with_backoff)
''')

md(r'''
Three things in there are the actual lesson:

**The `reason` field, not the status, decides.** `insufficientPermissions` fails
fast; a 403 with *no* reason also fails fast, because the safe reading of an
ambiguous 403 is the one that surfaces it.

**Two schedules.** A 5xx is a blip and clears in seconds. The limit that fired is
measured per **minute**, so retrying it on the 7-second schedule would still have
given up before it could possibly clear.

**Jitter that only lengthens.** Threads hydrate five at a time against one shared
quota. Un-jittered, all five fail together, sleep the identical interval and
collide again — which is how 32 actions went out in 12 seconds. Textbook full
jitter spreads both ways, but half that range *shortens* the schedule, and
shortening is the wrong way to err when the server has just said there were too
many requests.
''')

code(r'''
# PROOF
test("tests/test_gmail.py::test_a_rate_limit_403_is_retried_not_raised")
test("tests/test_gmail.py::test_a_permission_403_still_fails_immediately")
test("tests/test_gmail.py::test_a_rate_limit_waits_long_enough_for_a_per_minute_quota")
test("tests/test_gmail.py::test_a_server_error_keeps_the_short_schedule")
''')

md(r'''
Worth naming: `_with_backoff` had **no tests at all** before this. That is how a
policy this load-bearing kept a gap this size. A green suite is not evidence
about code the suite never reaches.
''')

# ---------------------------------------------------------------- §20
md(r'''
---
# §20 · Running it on a machine that is not this one

Stage A ran from a notebook in one directory. Making it something another person
can clone and run surfaced a class of bug that only exists once configuration has
more than one source.
''')

code(r'''
from inbox_agent import config

src(config.source_of)
''')

md(r'''
### WHY — the bug that made this necessary

`load_dotenv(override=False)` means **the shell environment wins over `.env`**.
So a value exported in a terminal months ago silently beats the file, and the
file is what you read when you want to know what the agent is doing.

This is not hypothetical either. During this project's own dogfooding, `.env`
said `INBOX_DRY_RUN=true` and `INBOX_GMAIL` was absent — and the running process
had `INBOX_GMAIL=live`, `INBOX_DRY_RUN=false` inherited from its shell. Reading
the file gave exactly the wrong answer about whether the agent was touching a
real mailbox.

`source_of` answers "where did this value actually come from" — `environment`,
`dotenv`, or `default` — and that is why there is **no config file**. A file
layered over env would make it three places instead of two.
''')

code(r'''
from inbox_agent import doctor
from inbox_agent.doctor import Check, render

# Deliberately NOT `render(run_checks())`. Against a real machine that prints
# the live token prefix, the chat id and the resolved policy commit into a cell
# output - and a notebook committed with its outputs is how those reach a repo.
# This project made exactly that mistake once already, with a test log.
#
# Synthetic checks show the format and the provenance column, which is the part
# worth understanding, and carry nothing real.
print(render([
    Check("INBOX_GMAIL",        "live",     "dotenv",  "ok", "THE REAL MAILBOX"),
    Check("INBOX_DRY_RUN",      "False",    "environment", "warn",
          "a shell export beats .env AND does not survive a restart"),
    Check("INBOX_LLM_BACKEND",  "ollama",   "dotenv",  "ok"),
    Check("INBOX_EMBEDDINGS",   "ollama",   "default", "ok"),
    Check("INBOX_TG_TOKEN",     "<redacted> (46 chars)", "dotenv", "ok"),
    Check("INBOX_GOOGLE_TOKEN", "secrets/token.json", "default", "fatal",
          "missing, and INBOX_GMAIL=live needs it"),
    Check("oauth consent",      "2026-09-10", "sidecar", "warn",
          "revocation due in 7 days if the app is still in Testing"),
]))
''')

md(r'''
Run it for real with `inbox-agent doctor` — the CLI entry point that
`pyproject.toml` installs. Two columns carry the weight:

**`<- source`** is the whole point. `environment` means a shell export won, which
beats `.env` *and* does not survive a restart — the exact combination that made
the original bug invisible.

**The mark** in column one: `X` fatal, `!` warning, blank for fine. `main()`
exits non-zero if and only if something is fatal, so it works in a script.
''')

md(r'''
`doctor` is the answer to "why is it not working", and its own source carries the
rule it must never break: **never let doctor be the thing that breaks.** A
diagnostic that crashes in the state it exists to diagnose is worse than none —
it was caught doing exactly that with `INBOX_EMBEDDINGS=ollama` and Ollama down,
printing no rows at all.
''')

code(r'''
# PROOF
test("tests/test_doctor.py")
''')

md(r'''
### Embeddings, and degrading instead of breaking

Rules are matched exactly by default; the embeddings index makes them match
semantically. `INBOX_EMBEDDINGS` defaults to `auto`: on a machine with Ollama
nothing changes, and a clone without it **degrades with a printed reason** rather
than breaking at the owner's first correction.
''')

code(r'''
src(config.resolve_embeddings)
''')

code(r'''
# PROOF - a rule can still be written with no embeddings backend at all
test("tests/test_store.py")
''')

# ---------------------------------------------------------------- §21
md(r'''
---
# §21 · What is deliberately not built

### `/backlog` — the loaded gun with the safety on

Stage A section 9.4 explains `interrupt()`, and the graph still supports
`mode="backlog"` with tests to match. There is **no `/backlog` command**, and the
method it would call is broken on purpose. Read the warning left for whoever
wires it:
''')

code(r'''
warning = inspect.getsource(bot_mod)
start = warning.index("# WARNING TO WHOEVER WIRES /backlog")
print(warning[start:warning.index("def _resume")])
''')

md(r'''
The heart of it: `to_response()` defaults every thread the intents do not name to
**approve**. The verdict-collection UI went away with the interrupt UI, so
`_resume` today would resume with an empty mapping — a blanket approval of the
entire parked batch, on a 500-thread historical sweep, which is *precisely the
outcome previewing that sweep exists to prevent*.

Wiring the command is a one-line change. Doing it would be the most destructive
single edit available in this codebase.

### The rest

- **Always-on** — no launchd agent yet, so the bot runs because somebody started
  it. No OAuth pre-expiry warning, though section 19.1's sidecar is now the input
  one needs.
- **Proactive** — scheduled runs need a `Trigger` seam and must serialise against
  a manual `/triage`; `Bot` holds per-conversation UI state and a collision is
  undefined today.
- **Evaluation** — tracing is wired and every audit record carries a
  `langsmith_run_id`, but nothing turns your corrections into a dataset yet.

Two findings from 2026-09-03 belong to that last one, and both were invisible to
the test suite:

**A model can run away and nothing stops it.** `_build_openrouter` sets neither
`max_tokens` nor `timeout`. Driven through `with_structured_output`, one model
generated to the 16 384-token ceiling instead of a short JSON object — ~179s per
affected thread, silently. A run sat for 18 minutes on 2.45s of CPU.

**`graph.invoke` is one opaque blocking call.** Between `triage start` and
`triage done` there is no per-thread progress line, so a run 18 minutes into a
stall is indistinguishable from one that started a second ago.
''')

# ---------------------------------------------------------------- close
md(r'''
---
# What to read next

- **The engine** — `inbox_agent_stage_a_explained.ipynb`, sections 1-13. Still
  accurate; section 9.7's forward look is the only part this notebook supersedes.
- **The lab** — `inbox_agent.ipynb` runs the pipeline end to end.
- **The safety model** — `README.md` leads with it, before setup, deliberately.
- **The design arguments** — they live in module docstrings and above the code
  they justify, not in a separate document that can drift.

### The one habit worth taking from this project

Every defect described in this notebook was found by **running the thing**, not
by the test suite — which was green throughout, and is now at 664 tests. The
suite is necessary and it is not evidence about code it never reaches:
`_with_backoff` had no tests, the held-item screen had tests that asserted the
keyboard and never pressed it, and `build_store`'s docstring says embeddings are
optional *precisely so the suite can run without them* — which is why no test
ever exercised the dependency that broke.

When a test covers a line, break the line and confirm the test fails. Two tests
in this project could not fail, and both were found that way.
''')


def build(path: Path) -> Path:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n")
    return path


if __name__ == "__main__":
    out = build(Path(__file__).parent / "inbox_agent_stage_b_explained.ipynb")
    print(f"wrote {out.name}: {len(CELLS)} cells "
          f"({sum(1 for k, _ in CELLS if k == 'code')} code, "
          f"{sum(1 for k, _ in CELLS if k == 'markdown')} markdown)")

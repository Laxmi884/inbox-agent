# Clone and run

2026-09-02

Project 1 of the productionalisation milestone. The agent has been running live
against the real mailbox since 2026-09-02 and works, but it is not a thing
anyone else can run and not a thing that survives its own restart. This spec
makes it installable, makes its configuration inspectable, and removes a
dependency that can silently break the learning loop.

Scope was fixed before drafting: packaging, an embeddings seam, a `doctor`
command that reports configuration *provenance*, a README, and a remote. No
config file, no profiles, no multi-tenancy — see section 7.

## 1. What is true today

Established by reading the code and by inspecting the running process, not
assumed.

### 1.1 The project is not installable

There is no `pyproject.toml`, no `requirements.txt`, and no `.venv`.
Dependencies are installed globally into an Anaconda interpreter. The dependency
set exists only as import statements across 19 modules. A person who clones this
repository has no supported way to make it run, and no record of which versions
it was verified against — which matters concretely, because `graph.py:198`
documents `checkpoint_id` behaviour verified against langgraph 1.2.11
specifically.

Verified installed versions, which become the pinned lower bounds:

```
langgraph                    1.2.11      langchain-core       1.6.0
langgraph-checkpoint-sqlite  3.1.0       langchain-openai     1.2.1
langsmith                    0.11.1      langchain-ollama     1.1.0
pydantic                     2.11.7      python-dotenv        1.2.2
google-api-python-client     2.200.0     google-auth          2.52.0
google-auth-oauthlib         1.4.1       google-auth-httplib2 0.4.2
httplib2                     0.22.0      httpx                0.28.1
```

### 1.2 The bot requires Ollama even when it is not the backend

`telegram/__main__.py:58` calls `get_embeddings()` unconditionally:

```python
prefs = PreferenceStore(open_store(settings.store_dir / "prefs.sqlite",
                                   get_embeddings()))
```

`get_embeddings()` (`config.py:285`) returns `OllamaEmbeddings` pointed at
`http://localhost:11434`, regardless of `INBOX_LLM_BACKEND`. Constructing it
performs no network call, so **startup succeeds**. The embedding call happens on
`put`, which means the first failure is at the first rule write.

A person who clones this, sets `OPENROUTER_API_KEY`, and never installs Ollama
therefore gets a bot that fetches, classifies, acts and reports correctly — and
then breaks the moment they correct it. The learning loop is the reason this
system exists, so this is the worst possible place for the failure to land.

### 1.3 Nothing reads the vectors

`store.py` reaches the store's search API in exactly two places — `_search_all`
(line 50) and `instructions()` (line 391) — and both pass only `namespace`,
`limit` and `offset`. `_search_all` is in turn the only reader of the rules
namespace (line 284) and the held queue (line 461). **No call site anywhere
passes a `query`.** `build_store`'s own docstring says as much: *"exact scope
matching is the Stage A path and needs no vectors."*

So the index in 1.2 is written on every rule write and never read. The
dependency is not a tradeoff being paid for a feature; it is currently pure
cost, and it is load-bearing on nothing.

### 1.4 `.env` does not describe what is running

`config.py:16` runs `load_dotenv(find_dotenv(), override=False)`. With
`override=False`, a key already present in the environment is not replaced, so
precedence is **shell environment > .env > code default**.

Observed on the live process (PID 30263) via `ps eww`:

```
INBOX_GMAIL=live
INBOX_DRY_RUN=false
```

Neither string existed in `.env`, which said `INBOX_DRY_RUN=true` and did not
mention `INBOX_GMAIL` at all — so the file's value resolved to the `snapshot`
default. The banner correctly printed `gmail: live`, because the exported values
won.

This is correct behaviour, and a useful feature: `INBOX_DRY_RUN=false python -m
inbox_agent.telegram` is a legitimate one-off. The failure is that **the winning
value lives only in a shell that does not survive a restart**. The launching
shell was already gone (PPID 1). A restart from a fresh shell — a crash, a
reboot, or launchd in Project 2 — would have come back on the snapshot in
dry-run, still polling, still classifying, still sending digests that look
entirely normal and touch nothing.

This is the same failure shape as the Context Hub 404 already fixed in
`policy.py`: *"the agent reported `local:...` while looking configured for the
hub."* A fallback that is individually correct, hiding a misconfiguration nobody
was told about.

(`.env` has since been corrected to declare both values. The class of bug is
what this spec addresses.)

### 1.5 Logs and the audit trail use different clocks

`audit.jsonl` stamps UTC with an explicit `Z`. The bot's log lines come from
`logging.basicConfig`, which uses local time and emits **no timezone marker**.
On EDT that is a silent four-hour offset between the two durable records of the
same run:

```
log:    2026-09-02 15:14:47      <- local, unmarked
audit:  2026-09-02T19:14:47Z     <- UTC, explicit
```

Correlating them requires knowing the offset from outside the data. This was
found by getting it wrong: an audit query filtered on the log's timestamps
returned a different run from four hours earlier, and the numbers looked
plausible. Noted here; the fix belongs with Project 2's logging work.

### 1.6 There is no remote

`git remote -v` is empty. 584 tests, 13 merged branches and the entire
project history exist on one laptop with no off-machine copy.

## 2. Packaging

`pyproject.toml`, PEP 621, hatchling backend, `requires-python = ">=3.11"`.

Core dependencies are those in 1.1 minus the two model providers. Both provider
packages are imported *lazily inside functions* already (`config.py`
`_build_ollama`, `_build_openrouter`, `get_llm`), so they become extras with no
code change:

```toml
[project.optional-dependencies]
ollama = ["langchain-ollama>=1.1.0"]   # local models
openai = ["langchain-openai>=1.2.1"]   # covers OpenRouter, which uses ChatOpenAI
dev    = ["pytest>=7.4"]
```

Lower bounds, not pins: the versions in 1.1 are what the suite and the live runs
were verified against, and going below them is unsupported.

A new `inbox_agent/cli.py` exposes `inbox-agent bot` and `inbox-agent doctor`.
`bot` delegates to the existing `telegram.__main__.main()` rather than
reimplementing it. **`python -m inbox_agent.telegram` continues to work
unchanged** — it is in active use and appears in the running process's command
line; the console script is additive.

## 3. The embeddings seam

`INBOX_EMBEDDINGS` takes `auto` (the default), `ollama`, or `none`. An
unrecognised value raises at load time naming the variable, in the established
style of `_resolve_gmail` and `_resolve_triaged_label`.

The three values differ in what happens when Ollama is not listening, which is
the only interesting axis:

| value | Ollama up | Ollama down |
|---|---|---|
| `auto` (default) | embeddings on | **falls back to `none`, prints why** |
| `ollama` | embeddings on | **raises at startup** |
| `none` | off | off |

`auto` is not a new idea; it is `resolve_backend()`'s existing contract applied
to the same daemon. That function already probes `ollama_available()` and prints
a named fallback when `INBOX_LLM_BACKEND=ollama` finds nothing listening, and
reusing the shape means there is one story about what happens when Ollama is
absent rather than two.

**On a machine with Ollama running — this one — `auto` resolves to Ollama and
nothing changes.** Vectors keep being written exactly as they are today, so the
backfill problem in 3.2 never arises here. A clone without Ollama degrades to
`none` with a printed reason instead of breaking at the first correction, which
is 1.2. Someone who wants embeddings to be non-negotiable pins `ollama` and gets
a hard failure instead of a quiet degrade.

The fallback is safe *specifically because of 1.3*: nothing queries the index, so
a run without it loses no capability that exists today. That finding no longer
has to justify the default — it justifies why degrading is acceptable.

### 3.1 The failure must move to startup

Independent of the default, and the real fix to 1.2. Today `get_embeddings()`
constructs an `OllamaEmbeddings` without touching the network, so the process
starts happily and dies at the first `put` — a correction, hours later, in the
one code path the whole system exists for.

Both `auto` and `ollama` therefore probe `ollama_available()` (already in
`config.py`, 1.5s timeout) **at construction**. Whether the outcome is a
fallback or an exception, it is decided while a human is watching the banner,
not while they are tapping a button in Telegram.

The vector code is kept, not deleted, so semantic rule matching stays one
setting away — with the caveat in 3.3.

### 3.2 Backfill

`SqliteStore` embeds on `put` and does not backfill. Any rule written while
embeddings were off has no vector, so a store that has run in both modes is
half-indexed — and a half-indexed store searched semantically returns confident,
incomplete results, which is the invisible-failure shape this codebase otherwise
refuses to ship.

Not a problem on this machine under the chosen default, since `auto` will
resolve to Ollama. It becomes one for anyone who runs a while on `none` and
later switches. Re-embedding is a loop calling `_put` over `rules()`, so
whenever semantic matching is actually wired up it must ship with that backfill
step. Recorded here rather than solved here.

### 3.3 Semantic matching is a safety change, not a feature flag

Flagged so that turning this on later is a deliberate decision rather than an
assumed next step. `matching()` decides which learned rule fires, and a firing
rule acts. `partition.hold_reason` grants **trash authority** to rule-sourced
decisions precisely because a rule is an exact, citable instruction the owner
gave. Fuzzy matching turns that into "this thread resembles one you taught me
about, so I will trash it", and it undermines `MIN_PRECISION` demotion too,
which tracks precision per rule on the assumption that the rule fires
deterministically.

### 3.4 Reading an indexed store without an index

No longer on the default path, but now reachable two ways: `auto` degrading on a
machine whose Ollama has stopped, and anyone setting `none` on a store that has
already been written with an index. `prefs.sqlite` on this machine is such a
store.

Opening it with `index=None` must still return every rule. The vectors live in a
separate table from the key-value rows and `_search_all` reads by namespace
only, so this should hold — but "should" is not evidence, and this is the path
that could make real learned rules invisible. It gets a dedicated test
(section 6) against a fixture store built *with* an index.

The `auto` route makes this sharper, not softer: a laptop where `ollama serve`
died between restarts silently takes this path. Rules must survive it.

## 4. `doctor`

The new capability, and the direct answer to 1.4.

Provenance is determined exactly rather than inferred, by capturing the
environment before `load_dotenv` mutates it:

```python
_ENV_AT_IMPORT = frozenset(os.environ)          # must precede the next line
load_dotenv(find_dotenv(), override=False)

def source_of(key: str) -> Literal["environment", ".env", "default"]:
    if key in _ENV_AT_IMPORT:                return "environment"
    if key in dotenv_values(find_dotenv()):  return ".env"
    return "default"
```

Comparing values after the fact cannot distinguish "the environment set it to
the same thing the file says" from "the file won". Capturing the key set before
the merge can. The snapshot must be taken on the line above `load_dotenv`, and
that ordering is the whole mechanism, so it carries a comment saying so.

`inbox-agent doctor` reports, for every setting: name, value (through the
existing `mask()` for secrets), and source. It flags:

- **⚠ override** — an environment value shadowing a *different* `.env` value.
  This is 1.4, made visible in one line.
- **⚠ OAuth expiry** — `google_auth.py:52` documents that Google revokes the
  refresh token of an External app in Testing after exactly seven days. Doctor
  reports the token file's age and the projected revocation date, so the death
  is predicted rather than discovered.
- **⚠ policy drift** — `Policy.drifted` already carries this; doctor surfaces it
  alongside the source.
- **⚠ degraded embeddings** — `INBOX_EMBEDDINGS=auto` that resolved to `none`
  because Ollama was not listening. Doctor reports the **resolved** mode, not the
  configured one; on a laptop whose `ollama serve` has died these differ, and
  the resolved one is what the store is actually doing.
- **✗ fatal** — a selected backend that is unreachable, absent Google
  credentials, missing Telegram token or chat id.

Exit code is non-zero when any fatal is present, so Project 2 can use it as a
launchd pre-flight check.

## 5. README and remote

A README leading with the safety model — the `ALWAYS_FORBIDDEN` deny-list,
`INBOX_DRY_RUN`, and the autonomy ladder in `partition.py` — before setup
instructions. A reader deciding whether to point this at their own mailbox needs
that first, and it is currently documented only in specs and docstrings.

Then setup (Google OAuth, BotFather, backend choice), running, and
troubleshooting keyed to what `doctor` prints.

A private GitHub remote, and a first push. 1.6 is a single-disk-failure risk on
the whole project and costs minutes to remove.

## 6. Testing

Unit:

- `_resolve_embeddings`: all three valid values, the `auto` default, and an
  invalid value that raises and names the variable
- the resolution matrix in section 3, with `ollama_available()` patched both
  ways — six cases, and the two that matter are `auto` + down (falls back,
  prints) and `ollama` + down (**raises, and raises at startup rather than at
  first put**)
- `source_of`: environment, `.env`, and default — including the case where both
  the environment and the file define a key with the *same* value, which a
  post-hoc value comparison gets wrong and this mechanism must get right
- doctor's fatal-versus-warning classification and its exit code, including that
  it reports the *resolved* embeddings mode rather than the configured one — on
  a machine where Ollama has died those differ, and the resolved one is the
  truth

Section 3.4: build a store *with* an index, write rules, reopen with
`index=None`, assert every rule reads back.

Regression: all 584 existing tests stay green.

**Manual, and this is the test that matters.** A green suite would not have
caught 1.2, because the suite constructs stores without embeddings by design
(`build_store`'s docstring: *"Embeddings are optional so the test suite runs
without Ollama"*). The suite is structurally incapable of reaching this bug.

Setup for both runs below: a second BotFather token, `INBOX_GMAIL=snapshot`,
`INBOX_DRY_RUN=true`, a store directory inside the worktree, and a backend that
is not Ollama, so stopping Ollama does not also remove the classifier.

1. **Ollama stopped, `auto`.** Startup must say it fell back. `/triage`, correct
   an item, confirm the rule is written and reads back. This is 1.2 on the old
   code and a clean degrade on the new — and it is the run a clone would make.
2. **Ollama running, `auto`.** Must resolve to Ollama and write vectors, proving
   the default preserves today's behaviour rather than merely claiming to.
3. **Ollama stopped, `INBOX_EMBEDDINGS=ollama`.** Must fail at startup, in the
   banner, naming Ollama — not at the correction in step 1.

Step 1 needs Ollama down. Do it when no scheduled or manual triage is expected,
or the live bot on `main` loses its classifier for the duration.

## 7. Not doing

- **A config file.** Rejected on the evidence of 1.4: the failure was a value
  resolvable from two places, and a TOML file layered over the environment makes
  that three. `doctor` answers the real question — *where did this value come
  from* — without adding a source.
- **Named profiles.** Would fix the shape of 1.4, but is a larger change than
  clone-and-run requires.
- **Multi-tenancy.** `INBOX_TG_CHAT_ID` is singular by construction. Each person
  runs their own process; that was the decision taken at the start of this
  milestone.
- **Deleting the vector code, or turning embeddings off by default.** An earlier
  draft defaulted to `none` on the strength of 1.3. Overruled deliberately:
  keeping Ollama the default on a machine that has it means this project changes
  no observable behaviour there at all, and the backfill trap in 3.2 never opens.
  The clone-and-run problem is solved by `auto` degrading, not by everyone
  losing the index.
- **Wiring up semantic rule matching.** Section 3.3 — it is a safety change, and
  it needs the backfill in 3.2 shipped with it.
- **The timezone fix (1.5) and the launchd unit.** Both belong to Project 2.

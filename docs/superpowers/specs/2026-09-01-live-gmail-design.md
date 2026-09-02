# Live Gmail

2026-09-01

Step 2 of the production-dogfooding milestone: replace the frozen snapshot with
the real mailbox. Everything above `GmailClient` was built for this — the
protocol at `gmail.py:47` exists so a live adapter drops in, and the whole of
Stage A has been running against `SnapshotGmailClient` as one implementation of
it. This spec is the second implementation, plus the OAuth subsystem it needs
and an MCP surface over it for Stage B.

The scope decisions were taken before drafting: `gmail.modify` only,
`INBOX_DRY_RUN=true` for the first live runs, bodies fetched but kept out of the
prompt, and a FastMCP server as a second deliverable.

## 1. What is true today

Established by reading the code and by querying the real mailbox through the
Gmail connector, not assumed.

### 1.1 The mailbox is four orders of magnitude larger than the snapshot

`users.labels.list` against the live account:

```
INBOX    threadsTotal 21058   threadsUnread 16748
UNREAD   threadsTotal 17811
```

The frozen snapshot is 50 threads — **0.24% of the inbox**. Two consequences
that the design has to answer rather than discover in production:

- `settings.inbox_query` (`in:inbox is:unread -label:agent/triaged`) matches
  roughly 16,748 threads on the first live run. `/triage` caps at
  `snapshot_size` and is fine. **`/backlog` is not**: it is specified to drain
  an untriaged inbox, which here means ~335 runs and ~16,748 `threads.modify`
  calls to apply the triaged label.
- Every model measurement in the registry was taken on a 50-thread sample drawn
  from a 21,058-thread population. The sample is not known to be representative
  and was never claimed to be.

### 1.2 `agent/triaged` does not exist

It is absent from `labels.list`. `mark_triaged` (`graph.py:353`) is the first
thing that will need it, on the first live run, for every thread processed. The
live client must create it rather than fail.

### 1.3 Label IDs cannot be derived from names, and have no reliable shape

Real user labels in this mailbox:

```
Label_1                      Notes
Label_5                      Property Listings
Label_6111317184412779502    Education/AI
Label_7181901001278056114    Learning
```

Both the short `Label_N` form and the long numeric form are in active use in one
account. **No heuristic on ID shape is safe**; the mapping must come from
`labels.list`. Thread reads return `labelIds` as IDs, so normalisation is
required on every read, not only on write.

### 1.4 Gmail's `q` accepts display names for `label:` — verified

The concern was that `-label:agent/triaged` in `settings.inbox_query` would need
an ID. Tested live against an existing nested label:

```
q: in:inbox is:unread -label:Education/AI   ->  returned threads, no error
```

So `inbox_query` passes to Gmail verbatim, and `Education/AI` also proves a
nested `/` name is a valid shape for `agent/triaged`. **The query needs no
translation layer.** Only `label_ids` on the way back does.

### 1.5 Six existing labels contain spaces

`Security Alerts`, `Property Listings`, `Job Alerts`, `Market Insights`,
`AI Innovations`, `Job Listings`.

This retroactively justifies `_resolve_triaged_label()` raising on whitespace
(`config.py:79`): `matches_query` splits the query on whitespace term-by-term,
so a spaced label name cannot round-trip. It also means the live label map will
legitimately contain names this project can never use as a triaged label — the
guard is on the configured value, not on the mailbox.

### 1.6 The prompt already prefers `body` over `snippet`

`classify.py:252`:

```python
f"<email_body>\n{_fence(thread.body or thread.snippet)}\n</email_body>"
```

The snapshot has `body == ""` for all 50 threads, so this has always resolved to
`snippet` and nobody has had to think about it. **The moment the live client
populates `body`, full email bodies enter the prompt silently.**

**Correction to an earlier draft of this section:** they do *not* arrive
untruncated. `_fence` clips at `MAX_BODY_CHARS = 4000` (`classify.py:18`,
applied at `classify.py:216`), so the 200KB `sizeEstimate` sampled above reaches
the prompt as 4000 characters, roughly 1000 tokens — comfortably inside
`num_ctx=8192`. There is no context overflow, and this spec should not have
claimed one.

The real cost is measurement continuity, and it is still decisive: 4000
characters against a snippet that maxes at 201 is a ~20x change in prompt size.
Every latency and cost figure in the model registry was measured at the smaller
one. Switching silently would invalidate the whole registry without anything
saying so.

"Fetch bodies, keep the prompt on snippets" is therefore not a no-op. It
requires an explicit change here.

### 1.8 Backlog senders do not recur, and concentration is weak

Three 50-thread samples of `in:inbox is:unread`, taken today, ~6 months back and
~2 years back:

| | today | 6 months | 2 years |
|---|---|---|---|
| top sender | linkedin jobalerts 12% | impactguru 12% | legeropinion 12% |
| senders covering half a page | 16 | 11 | 9 |
| shared senders vs today | - | 5 of 72 | **0 of 75** |

**Zero sender overlap between today's mail and two years ago.** Four shared
domains out of 63, all aggregators (beehiiv, linkedin, jobs2web) whose actual
sending addresses differ. These are subscription lifecycles: signed up, flooded,
drifted away.

This kills the obvious design, which was to triage the backlog head by sender,
convert each decision to a `Rule`, and let the rule store drain the rest.
Rules learned from current mail will not match the old backlog, and rules
learned from the old backlog have no future value.

It also rules out a per-sender approval UI: at 12% for the top sender and 9-16
senders per half-page, covering 16,748 threads across non-overlapping cohorts
would take hundreds of approvals.

Sample-size caveat: three samples of 50 from a 16,748 population. The direction
(0/75, 5/72) is strong enough to design against; the exact percentages are not.

### 1.7 There is exactly one construction site outside tests

`inbox_agent/telegram/__main__.py:50`. Five test files reference
`SnapshotGmailClient` and none of them need to change. This is the Protocol
paying off, and it is the reason this work is an addition rather than a
migration.

## 2. Decisions

### 2.1 A direct client, not MCP and not GmailToolkit

`LiveGmailClient` is written against `google-api-python-client`, implementing
the existing seven-method protocol.

**`langchain-google-community`'s `GmailToolkit` is rejected.** Its `get_tools()`
returns exactly five tools — `GmailCreateDraft`, `GmailSendMessage`,
`GmailSearch`, `GmailGetMessage`, `GmailGetThread`. No label, no archive, no
trash: it covers three of the seven methods and misses all four that mutate.
Its module-level `SCOPES = ["https://mail.google.com/"]` requests full mailbox
access including send and permanent delete, and it ships a `GmailSendMessage`
tool that is in `ALWAYS_FORBIDDEN` (`config.py:28`).

**A third-party Gmail MCP server is rejected for the digest path.** The
maintained forks advertise ~19 tools including `send_message`. MCP has no
mechanism to un-advertise a tool, so the deny-list would stop being structural
and become a filter maintained by hand against an upstream that can add tools in
a version bump. Separately, `fetch -> classify -> execute` is deterministic code
calling methods, with no model on the boundary; MCP's value is exposing tools
*to a model*, and inserting JSON-RPC where no model sits buys a failure mode.

### 2.2 `gmail.modify`, and nothing wider

`gmail.modify` grants read, label, archive, trash and draft-create — every
action on the ladder. It does **not** grant send (`gmail.send`) or permanent
delete (`https://mail.google.com/`).

`ALWAYS_FORBIDDEN` is `{send_message, delete_forever}`. The scope boundary and
the deny-list therefore coincide exactly, which means the two forbidden actions
become unavailable at Google's edge and not only in our code. This is a
strictly stronger guarantee than `audit.py:128` alone, and it is why no wider
scope is requested even though a wider one would be less work later.

### 2.3 The live client speaks label *names*, never IDs

`Thread.label_ids` holds display names in both implementations. The live client
resolves names to IDs on write and IDs to names on read, via a map built once
per client from `labels.list`.

Misses are handled differently by direction, and the difference matters:

- **Unknown ID on read** (a label created in Gmail since the map was built):
  refetch `labels.list` once, then fall back to surfacing the raw ID rather than
  dropping it. A dropped label is silent data loss into the classifier's
  `Current labels:` line.
- **Unknown name on write**: refetch once, and if still absent call
  `labels.create`. This is what makes 1.2 survivable — `agent/triaged` is
  created on the first `mark_triaged` and found in the map thereafter.

**Rejected: store raw IDs live and teach `matches_query` about them.** The
docstring at `gmail.py:19` states the contract this would break — both clients
answering the same query string is "the only thing that makes a snapshot test
evidence about live behaviour". Storing IDs live would mean a snapshot test
asserting on `agent/triaged` and a live run asserting on `Label_12` are no
longer the same assertion, and every existing test would quietly stop being
evidence.

Consequence to accept knowingly: a label renamed in Gmail changes identity from
this project's point of view. That is correct — the name is what the owner sees
and what the digest reports.

### 2.4 Bodies are fetched, stored, and kept out of the prompt

`build_prompt` takes a `body_budget: int = 0`. At 0 it uses `snippet`; above 0
it uses `body` truncated to the budget, falling back to `snippet` when `body` is
empty. The default holds the line at 1.6 for live.

It reproduces today's behaviour exactly **for the snapshot**, where `body` is
`""`. It does NOT for the test fixtures: five tests in `tests/test_classify.py`
pass `body=` and assert it reaches the prompt, four of them the prompt-injection
defences. Those must be parametrised over both fields rather than pinned to
`body` — which is strictly stronger coverage, because at budget 0 it is the
*snippet* that carries attacker-controlled text, and Gmail derives the snippet
from the body.

A parameter rather than a hardcoded `thread.snippet` so the snippet-vs-body
comparison is a config flip driven from LangSmith traces, which is step 5 of the
roadmap. Storing bodies now and spending tokens on them later means that
comparison needs no second fetch over 21,058 threads.

Bodies reach disk via the held queue. `.gitignore` already covers
`inbox_agent/store/` and `*.sqlite`.

### 2.5 The MCP server wraps our client, and shares the chokepoint

`inbox_agent/mcp_server.py`, FastMCP over stdio, seven tools mirroring the
protocol. `fastmcp 2.14.7` is already installed.

The load-bearing detail: **mutating tools route through `execute_action`**, so
the deny-list and the audit log apply identically. The MCP surface is an
alternative caller of the chokepoint, never a bypass. `send_message` is not
advertised because it was never written — 2.1's objection to a third-party
server does not apply to a server whose tool list is ours.

Not imported by the digest path. It exists so Stage B has a proven transport,
and so this mailbox can be driven by an external client under this project's
policy rather than a generic connector's.

### 2.6 `/backlog` is a deterministic bulk archive, with no model in it

Given 1.1, running `/backlog` as a triage sweep over 16,748 threads is not a
feature, it is an incident: roughly six hours of model time on the best measured
model to re-derive 16,748 individually-reasoned verdicts. Section 1.8 measures
why that work would also be worthless. The mode is redesigned rather than
disabled.

**A thread that is a year old and still unread in the inbox has already been
judged, by the owner, by not reading it.** `recency.py` states the same
intuition at 90 days: "whatever was waiting on the owner two years ago happened
or didn't." `/backlog` acts on that directly.

1. A **deterministic safety sieve** runs first, as a Gmail query and not a
   judgement: `is:starred`, `IMPORTANT`, any user label, or a thread the owner
   replied to is excluded and never touched.
2. Everything else past the cutoff is **archived, never trashed**. Archived mail
   stays in All Mail and stays searchable. A wrong archive costs nothing, and
   that is the entire reason a bulk operation is acceptable here.
3. **One confirmation per band**, not per thread: a true count, the sieve's
   exclusions, and ten random samples. The preview pages the full id set first
   (~37 calls) so the number shown is counted, not estimated.
4. **No rules are written from backlog decisions.** Per 1.8, cohorts do not
   recur, so such a rule is dead weight evaluated by `prefilter` on every future
   run forever.

Cutoff: **one year**, reassessed after the first sweep. Not `stale_after_days`'
90, deliberately - the 2-year cohort is uncontroversially dead by 1.8, while the
90-day-to-1-year band is not yet evidenced either way.

Two consequences worth stating, because they change work already budgeted:

- **`agent/triaged` is not applied to archived threads.** Archiving removes
  INBOX, so they leave `inbox_query` on their own. The 16,748 label writes in
  1.1 do not happen.
- **This does not need the `_resume` rebuild.** `bot.py:771` warns that
  `to_response()` defaults unnamed threads to approve, which is a blanket
  approval and catastrophic for a per-item review of a 500-thread sweep. For a
  single cohort-level yes/no, blanket approval of the cohort *is* the intended
  semantics. `_resume` remains unsafe for per-item use and Plan 3 still owes
  that work; `/backlog` no longer waits on it.

## 3. Components

| Module | Status | Responsibility |
|---|---|---|
| `inbox_agent/google_auth.py` | new | Credentials only. No Gmail knowledge. |
| `inbox_agent/gmail.py` | extended | `LiveGmailClient` beside `SnapshotGmailClient`. |
| `inbox_agent/mcp_server.py` | new | FastMCP surface over a `LiveGmailClient`. |
| `inbox_agent/config.py` | extended | `build_gmail_client()` + new settings. |
| `.env.example` | changed | document the four new variables. |
| dependencies | installed | `google-api-python-client`, `google-auth-oauthlib`. The repo has **no** `pyproject.toml`, `requirements.txt` or `setup.py`; this spec does not add one. |
| `inbox_agent/classify.py` | changed | `body_budget` parameter on `build_prompt`. |
| `inbox_agent/telegram/__main__.py` | changed | one line at `:50`. |
| `.gitignore` | changed | add `secrets/`. |

### 3.0 Configuration and dependencies

Four new variables, all defaulted so every existing construction of `Settings`
keeps working — the same rule `stale_after_days` and `store_dir` were added
under:

| Variable | Default | Meaning |
|---|---|---|
| `INBOX_GMAIL` | `snapshot` | `snapshot` or `live`. The one switch. |
| `INBOX_GOOGLE_CREDENTIALS` | `secrets/credentials.json` | OAuth client, Desktop app type. |
| `INBOX_GOOGLE_TOKEN` | `secrets/token.json` | Written by the consent flow. |
| `INBOX_BODY_BUDGET` | `0` | Chars of body into the prompt. 0 = snippet only. |

`build_gmail_client(settings)` branches on `INBOX_GMAIL` and is the only place
that decides. Defaulting to `snapshot` means an unconfigured checkout, and every
existing test, behaves exactly as it does today; going live is one deliberate
edit.

`google-auth` 2.52.0 is already present as a transitive dependency, but
**`google-api-python-client` and `google-auth-oauthlib` are new** and must be
installed. There is no dependency manifest in this repo to declare them in, and
introducing one is out of scope here — it would be the first, and it should be a
deliberate decision rather than a side effect of adding two packages. `fastmcp`
2.14.7 is already installed and needs no addition.

### 3.1 `google_auth.py`

```
get_credentials(scopes, client_secrets_path, token_path) -> Credentials
```

Load `token.json`; refresh if expired and refreshable; otherwise
`InstalledAppFlow.run_local_server(port=0)`, which opens the browser for one
consent and writes the token. Single responsibility: it knows about OAuth and
nothing about mail.

A missing `credentials.json` raises with the Cloud Console steps in the message,
in the style `load_snapshot` already uses (`gmail.py:61`) — **including "set
publishing status to In production"**, because an External app left in Testing
has its refresh token expired by Google after exactly seven days, and that
failure would otherwise surface a week later as an unexplained re-auth prompt.

### 3.2 `LiveGmailClient`

- `list_threads(limit, query)` — `threads.list(q=query, maxResults=limit)`,
  query passed verbatim, `matches_query` never called. Returns ids only.
- `get_thread(id)` — `threads.get(format="full")`. Headers from the first
  message; `label_ids` the union across messages mapped to names (a thread is
  UNREAD if any message is); body from a MIME walk preferring `text/plain`,
  falling back to stripped `text/html`, base64url-decoded.
- Hydration is a bounded `ThreadPoolExecutor(5)`: 50 threads is 1 list + 50
  gets, ~10s sequential. Simpler than `BatchHttpRequest` and well inside quota
  (10 units x 50 = 500, against 250/sec/user).
- `archive` — `modifyThread(removeLabelIds=["INBOX"])`.
- `trash` — `threads.trash()`, the real endpoint, not label manipulation, so
  `untrash()` reverses it when undo comes alive.
- `create_draft` — `drafts.create` with `threadId` and In-Reply-To/References.
- `HttpError` propagates. Bounded backoff on 429 and 5xx only; 4xx never
  swallowed.

## 4. Testing

`SnapshotGmailClient` tests are untouched.

- **`FakeGmailApi`** — fakes the `service.users().threads().list().execute()`
  chain, so `LiveGmailClient` is testable with no network and no credentials.
  This is the piece that makes everything else testable.
- **Contract tests** — the same assertions run against both clients wherever
  semantics must match: query answering, label naming, `archive` leaving the
  thread present, `trash` removing it from INBOX. This turns the `gmail.py:19`
  docstring's claim into something enforced.
- **MIME fixtures** — multipart/alternative, HTML-only, base64url padding,
  missing body, and an oversized body against the budget.
- **Label map** — creation on miss, refresh on miss, both ID shapes from 1.3,
  and a spaced name round-tripping to a name and not an ID.
- **`body_budget`** — 0 gives today's prompt byte-for-byte on a snapshot thread.

## 5. Rollout

A green suite is not evidence the thing works. Before any merge:

1. Auth once. Confirm `token.json` written and the consent screen is In
   production.
2. `list_threads(settings.inbox_query)` against the real inbox, dry-run on.
   Compare the returned shape to the snapshot, and confirm `label_ids` came back
   as names.
3. Confirm `agent/triaged` gets created, once, and is visible in Gmail.
4. A full `/triage` on the phone, dry-run on: real threads, real
   classifications, real digest, every audit record `simulated`.
5. Re-measure one model on real bodies and correct the registry notes that
   currently say "a floor measured on snippet-sized prompts".
6. Only then discuss `INBOX_DRY_RUN=false`.

## 6. Sequencing

This is two implementation plans, not one, and they should not be merged
together:

1. **The live client** — `google_auth.py`, `LiveGmailClient`, the label map, the
   `body_budget` parameter, the factory, the wiring, and section 4's tests.
   Ends at section 5 step 4: a real digest, from real mail, on the phone.
2. **The backlog sweep** — 2.6's sieve, the counted preview, the cohort
   confirmation, and `messages.batchModify` paging. Needs plan 1's client and
   nothing else. Roughly 56 API calls and no model, so it is verifiable in one
   sitting.
3. **The MCP surface** — `mcp_server.py` over the client from plan 1, routed
   through `execute_action`. Independently verifiable and independently
   revertable.

Plan 1 is on the critical path to the milestone. Plan 2 is the one-time cleanup
and should not run until plan 1's rollout has proven the client against live
mail. Plan 3 is Stage B enablement and blocks nothing.

Splitting them means a problem in the FastMCP surface cannot hold up the live
mailbox result, and means the rollout in section 5 has one subject at a time.

## 7. Deliberately not in scope

- `INBOX_DRY_RUN=false`. A separate decision after step 5.
- Per-thread triage of the backlog. 2.6 replaces it with a bulk archive; 1.8 is
  why the per-thread version would be expensive and worthless at once.
- The 90-day-to-1-year band. The cutoff is one year until the first sweep says
  otherwise.
- Bodies in the prompt. The budget defaults to 0; turning it up is step 5's
  outcome, not this spec's.
- Stage B tool-calling. The MCP server is the transport; the agent that uses it
  is later work.
- Incremental sync via `history.list`. Every run re-queries. At `/triage` scale
  that is one API call, and `agent/triaged` already does the deduplication that
  a history cursor would.

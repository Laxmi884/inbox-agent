# SDD ledger — plan: docs/superpowers/plans/2026-08-26-inbox-agent-stage-a.md

Spec: docs/superpowers/specs/2026-08-26-inbox-agent-design.md (read, reachable)
Branch: feat/inbox-agent-stage-a
Base: 9a59554

## Setup rulings

Ruling: work in-place on branch `feat/inbox-agent-stage-a`, not a separate git
worktree — the deliverable is a notebook the user runs in their IDE at this path,
and `.env` is gitignored so it does not propagate to a worktree, which would break
`load_dotenv()` and split the user's attention across two directories.
Cost if wrong: main's working tree is occupied during execution; recoverable by
`git checkout main`.

## Pre-flight conflict scan

### Cross-task interface pairs (produces vs consumes)

| Pair | Interface | Finding |
|---|---|---|
| T1→T4 | `Settings` 8 fields | T4 test constructs all 8 positionally-named — match |
| T1→T6 | `Settings`, `ALWAYS_FORBIDDEN` | T6 test imports both from config — match |
| T1→T9 | `Settings` | graph reads `.snapshot_size`, `.backend`, `.dry_run` — all present |
| T2→T3 | `Thread` | `sender_domain` used in T3 test — defined in T2 |
| T2→T4 | `Action`, `AuditRecord`, `REVERSIBLE_ACTIONS` | all three defined in T2 — match |
| T2→T5 | `Rule`, `ActionKind`, `Thread` | `ActionKind` is module-level in T2 — match |
| T2→T8 | `ActionKind`, `Decision`, `Action` | match |
| T2→T9,T10 | `ReviewRequest/Response/Item` | field names identical across all three tasks |
| T3→T4 | `GmailClient` methods | `_dispatch` calls exactly the 5 mutators + `get_thread` — match |
| T5→T7 | `matching()`, `record_hit()` | both defined in T5 — match |
| T5→T9 | `rule_from_correction()` | signature `(thread, action, note)` used identically |
| T6→T8 | `Policy.text` | match |
| T7,T8→T9 | `prefilter()->tuple`, `classify_batch()->list[Decision]` | match |
| T4→T9 | `execute_action` kw-only signature | identical in both tasks |

### Per-task self-consistency

| Task | Tests vs code it specifies | Finding |
|---|---|---|
| T1 | 7 tests vs `config.py` | consistent; note `load_dotenv()` at import means tests inherit real `.env` — the two tests that care use `monkeypatch.delenv`/`setenv` |
| T2 | 6 tests vs `models.py` | consistent |
| T3 | 8 tests vs `gmail.py` | consistent; dict-union `|` requires 3.9+, env is 3.12 |
| T4 | 7 tests vs `audit.py` | consistent; `prior_labels` captured before `_dispatch`, so undo_token test holds |
| T5 | 9 tests vs `store.py` | **DEFECT — see ruling R1** |
| T6 | 4 tests vs `policy.py` | consistent; remote path unverified — see ruling R2 |
| T7 | 6 tests vs `prefilter.py` | consistent |
| T8 | 8 tests vs `classify.py` | consistent; fence-escape test holds against `_fence` |
| T9 | 7 tests vs `graph.py` | consistent; **stated suite total wrong — see ruling R3** |
| T10 | 7 tests vs `render.py` | consistent |

### Rulings from the scan

Ruling R1 (Task 5): `PreferenceStore.rules()` as written calls
`self._store.search(RULES_NS)` with no limit. LangGraph's `BaseStore.search`
defaults to `limit=10`, so the rule set would silently truncate at 10 — prefilter
would stop matching rules 11+ with no error. That is precisely the silent-wrong
behaviour the spec's audit pillar exists to prevent. Task 5 must pass an explicit
limit and paginate until exhausted. Carried into Task 5's dispatch.
Cost if wrong: none — an explicit limit is correct regardless.

Ruling R2 (Task 6): `_pull_from_context_hub` uses `SkillContext` attribute names
(`.files`, `.commit_hash`) that the Context Hub docs did not specify. The call is
wrapped in try/except with a local-file fallback, and the tests exercise the local
path, so Stage A is fully functional without it. Accepted as written; the remote
path is to be verified against a live LangSmith account before Stage C relies on it.
Cost if wrong: Context Hub pull silently falls back to local, printing a reason.

Ruling R3 (Task 9, Step 5): the plan states "55 passed" for the full suite; the
per-task counts sum to 62 at that point (T1 7 + T2 6 + T3 8 + T4 7 + T5 9 + T6 4 +
T7 6 + T8 8 + T9 7). The stated total is indicative, not a gate. Implementers
report actual counts; a mismatch with 55 is not a failure.
Cost if wrong: none — cosmetic.

## Progress

Ruling R4 (Task 3 operator step): the snapshot ships with `body: ""`, relying on
`snippet`. Reasons: (a) `classify.build_prompt` already falls back via
`thread.body or thread.snippet`, so the pipeline is unaffected; (b) fetching 50 full
bodies costs ~50 MCP round-trips and puts the user's entire recent correspondence
into the session transcript; (c) sender + subject + snippet determines the Stage A
judgment for the overwhelming majority of bulk mail.
Cost if wrong: lower classification quality on threads whose snippet is
uninformative; remediable by re-pulling bodies for a named subset.

Snapshot built: inbox_agent/snapshot/threads.json — 50 threads, 37 unique senders,
confirmed gitignored (.gitignore:20). Snippets HTML-unescaped and stripped of
Gmail's invisible padding characters.

Task 1: implementer DONE (commit a5d8c5b, 9 passed). Task reviewer dispatched.

Task 1: review returned SPEC ✅ / QUALITY changes-requested. 2 Important (both
plan-mandated), 2 Minor. Adjudicated below.

Ruling R5 (Task 1, mask): UPHELD — the reviewer is right and the plan is wrong.
`mask()` returns `value[:7] + "..." + value[-4:]`; for any secret of 11 chars or
fewer those slices overlap and reproduce the whole value. That directly violates
the global constraint "secrets must never be printed in full". Inherited verbatim
from build_notebook.py:432, so the same latent bug exists in the CAB notebook.
Fixing: values <= 11 chars return a fixed redaction with no characters revealed.
Cost if wrong: none — strictly safer, and real keys (lsv2_pt_*, sk-or-v1-*) are
long enough to be unaffected either way.

Ruling R6 (Task 1, test hermeticity): UPHELD. `config.py` calls `load_dotenv()` at
import, so tests inherit the repo's real INBOX_LLM_BACKEND and OPENROUTER_API_KEY
and silently exercise the live backend path — including a 1.5s `ollama_available()`
network probe. Fixing with `tests/conftest.py`: an autouse fixture pinning
INBOX_LLM_BACKEND=offline, which individual tests override via monkeypatch. This
also makes Tasks 4, 6 and 9 hermetic, so it is worth doing once, now.
Cost if wrong: a test could mask a real backend-resolution bug; mitigated because
the two tests that specifically assert resolution set the var explicitly.

Ruling R7 (Task 1, empty INBOX_DRY_RUN): UPHELD, and upgraded from the reviewer's
Minor. `_FALSEY` includes `""`, so `INBOX_DRY_RUN=` in a .env turns dry-run OFF —
the fail-dangerous direction, in the one setting that stands between the agent and
the user's real mailbox. An empty value carries no signal and must leave dry-run ON.
Removing `""` from `_FALSEY` does not break the brief's parametrised test, which
covers only "false"/"False"/"0"/"no".
Cost if wrong: someone who deliberately set an empty value to disable dry-run finds
it still enabled — they get a safe surprise, not a dangerous one.

Ruling R8 (Task 1, frozenset annotation): UPHELD as a trivial correctness fix,
bundled into the same round. `forbidden_actions: frozenset` -> `frozenset[str]`.
Cost if wrong: none.

Task 2: implementer DONE (commit ed2e014, 6 passed).

Ruling R9 (execution discipline): reviewers may run concurrently with an
implementer (they are read-only), but two implementers must never run at once —
concurrent `git commit` races on .git/index.lock. Task 1's fix and Task 2's
implementation overlapped safely only because the concurrent peer was a reviewer.
Task 3 waits for Task 1's fix commit to land.
Cost if wrong: serialised commits cost wall-clock time, nothing else.

Task 1: fix round 1/5 dispatched (4 findings: mask leak, test hermeticity,
empty-dry-run, frozenset annotation). Task 2: reviewer dispatched.

Task 1: fix round 1/5 complete (4 addressed: mask redaction, conftest hermeticity,
empty-dry-run, frozenset[str]; commit fe0f006, 17 passed). Scoped re-review
dispatched. Controller spot-check of mask(): 8 and 5 char inputs -> '<redacted>',
38 char key -> 'lsv2_pt...1234', empty -> 'not set'. Boundary looks right.

Note for Task 6 dispatch: conftest pins INBOX_LLM_BACKEND but LANGSMITH_API_KEY
remains set from the real .env, so `load_policy(allow_remote=True)` would attempt a
live Context Hub call. The brief's only test on that path monkeypatches
`_pull_from_context_hub`, so it is safe as written — but Task 6's implementer must
be told not to add an unpatched allow_remote=True test.

Task 3: implementer dispatched (told to SKIP brief Step 1 — snapshot already built
by the controller and must not be touched or committed).

Task 2: review returned SPEC ✅ / QUALITY changes-requested. All findings
plan-mandated. Adjudicated below. One is a LOAD-BEARING plan defect my pre-flight
scan missed.

Ruling R10 (Task 2/4, LOAD-BEARING): `Action.kind` is typed `ActionKind`, a Literal
that excludes "send_message". Task 4's two security tests construct
`Action(kind="send_message", ...)` and expect the chokepoint to raise
ForbiddenActionError — but pydantic rejects the Action first with ValidationError,
so `execute_action` is never reached. Controller verified directly: constructing
that Action raises ValidationError today. As written, Task 4's deny-list tests
CANNOT pass, and the deny-list would be untested — in the one component whose whole
job is standing between the agent and the user's mailbox.

Fix: `Action.kind` becomes `str`; `ActionKind` stays the narrow Literal and remains
the type of `ThreadJudgment.action` (Task 8) and `Rule.action` (Task 5). Consequences,
all of them good:
  - Gemma still can only ever *judge* within the narrow safe set.
  - A forbidden kind can now be constructed, so it reaches the chokepoint and is
    refused there — which is what "enforced in code, not in the prompt" means, and
    it makes the defence-in-depth genuinely testable.
  - `REVERSIBLE_ACTIONS` stops being decorative: it now genuinely excludes
    send_message and delete_forever, so `AuditRecord.reversible` can be False.
  - The reviewer's Minor about `AuditRecord.action: str` vs `Rule.action: ActionKind`
    resolves as a side effect.
Cost if wrong: a typo'd action kind is no longer caught at construction; it surfaces
at the chokepoint as ForbiddenActionError instead, which is logged. Acceptable —
that is the layer that is supposed to be authoritative.

Ruling R11 (Task 2, vacuous test): UPHELD. `test_thread_fingerprint_is_stable_and_
sender_scoped` uses the identical subject "Hi" for both threads, so it never
exercises the digit-stripping the docstring advertises; a naive hash with no
normalisation would pass it. Reviewer confirmed by probe that the implementation
does work — the test simply proves nothing about it. Fix: assert digit-collapse
("Invoice 8821" vs "Invoice 8822" collapse) and sender-scoping (same subject,
different sender -> different fingerprint).
Cost if wrong: none.

Ruling R12 (Task 2, fingerprint normalisation): UPHELD. `fingerprint` does not
collapse internal whitespace and does not strip Re:/Fwd: prefixes, so a reply
thread fingerprints differently from its parent. Limited impact today because
Task 5 defaults to sender-scoped rules, but the docstring claims "stable identity
for mail like this one" and it is cheap to make that true.
Cost if wrong: slightly broader collapsing than intended; sender scope still bounds
every rule, so cross-sender collision remains impossible.

Task 2 fix round 1 is BLOCKED on Task 3's implementer finishing (ruling R9 —
one committer at a time).

Task 1: fix round 1/5 (4 addressed, 0 open; commits a5d8c5b..fe0f006)
Task 1: complete (commits 9a59554..fe0f006, review clean)
Task 3: implementer DONE (commit 6faaa3b, 8 passed, full suite 25). Snapshot
confirmed unstaged and gitignored. Reviewer dispatched.

Task 3: review returned SPEC ✅ / QUALITY changes-requested. Both Important findings
plan-mandated (defects in my brief's reference implementation, faithfully
transcribed). Reviewer independently probed in-place mutation and the dict-union
`simulated` key survival — both confirmed sound.

Ruling R13 (Task 3, trash semantics): UPHELD. `trash()` adds TRASH but never removes
INBOX, so a trashed thread is simultaneously labelled INBOX and TRASH — a state real
Gmail cannot produce, and asymmetric with `archive()` which does remove INBOX. The
consequence is not cosmetic: `render.review_table` and any inbox-scoped filtering
would keep showing a trashed thread as present, and Task 4's `_undo_token` captures
prior labels for trash, so the restore path would be reasoning about incoherent
state. Fix: `trash()` removes INBOX and adds TRASH, mirroring `archive()`.
Cost if wrong: none — this strictly increases fidelity to Gmail. The brief's existing
assertion (`"TRASH" in label_ids`) still holds.

Ruling R14 (Task 3, misnamed test): UPHELD. `test_trash_is_recorded_and_reversible`
tests neither reversibility nor INBOX removal — it asserts only that TRASH was added,
so it cannot surface R13 at all. A test whose name claims more than it checks is
worse than no test, because it reads as coverage. Fix: assert TRASH added AND INBOX
removed, then perform the restore round-trip and assert the original label set comes
back.
Cost if wrong: none.

Ruling R15 (Task 3, minors): `_labels()` re-invoking `get_thread` is a redundant
indirection with no correctness impact — DEFERRED, not worth a round. Missing
`create_draft` shape assertion is thin coverage and cheap, so it is bundled into the
fix round rather than deferred.
Cost if wrong: negligible either way.

Task 3: minor (deferred): _labels() duplicates the get_thread lookup path on every
mutating call — harmless indirection, flagged to the final whole-branch review.

Task 3 fix round 1 BLOCKED on Task 2's fix committing (ruling R9).

Task 2: fix round 1/5 (4 addressed, 0 open; commits ed2e014..ad444b5)
Task 2: complete (commits a5d8c5b..ad444b5, review clean)
  Re-reviewer confirmed the key regression check: ActionKind is still the narrow
  Literal and still types Rule.action, so the model's judgment vocabulary did not
  widen — only the transport type did. That was the whole point of R10.

Task 3: fix round 1/5 (3 addressed, 0 open; commits 6faaa3b..3badd70)
Task 3: complete (commits fe0f006..3badd70, review clean)
Task 4: implementer dispatched (sonnet — safety core, higher stakes than the
transcription tier the earlier tasks used). Dispatch explicitly warns it not to
re-narrow Action.kind and not to weaken any test that fails.

Task 4: implementer DONE (commit 7c7c743, 7 passed, full suite 35). Controller
spot-check with dry_run=False (the dangerous configuration): send_message and
delete_forever both raised ForbiddenActionError, both were written to the audit log
with result="refused: ... on the deny-list" and reversible=False, and the thread's
labels were untouched. The safety core behaves as specified. Reviewer dispatched.

Task 4: review returned SPEC ✅ / QUALITY changes-requested. Reviewer ran real bypass
probes against the module rather than reading only. No probe found a path where a
mutation reached the client — but it found that the reason is not the mechanism the
design claims. Adjudicated below.

Ruling R16 (Task 4, deny-list is not what is stopping forbidden actions): UPHELD,
Important. Two linked defects:
  (a) `execute_action` matches `action.kind` against `settings.forbidden_actions` by
      exact string equality, and audit.py has no ALWAYS_FORBIDDEN floor of its own —
      it fully trusts a caller-supplied Settings, which is an unvalidated frozen
      dataclass. With an empty forbidden set, the documented deny-list does nothing.
  (b) Because of (a), a case variant "Send_Message" misses the deny-list. In LIVE
      mode it is still blocked, but only incidentally, by `_dispatch`'s closed
      if/elif falling through to "unknown action kind" — and it is logged as
      `error:` rather than `refused:`. In DRY-RUN mode it is not blocked at all:
      the dry-run branch returns before `_dispatch` runs, so the record reads
      `result="simulated"` and a deny-list evasion attempt is filed as benign
      activity.
No live exploit exists today — every producer of an Action kind (classify, prefilter,
rule) is typed to the narrow ActionKind Literal, so a case variant cannot arise. But
this module's stated invariant is "no action outside the allow-list can reach Gmail,
WHATEVER THE CALLER BELIEVES". Safety that depends on callers staying well-behaved is
not the invariant the docstring promises, and Stage B hands tool selection to a model.
Fix: normalise `action.kind.strip().lower()` for the check, and check against
`settings.forbidden_actions | ALWAYS_FORBIDDEN` so the chokepoint carries its own floor.
Cost if wrong: an action kind differing only by case or whitespace is refused rather
than dispatched. That is the correct direction for this module.

Ruling R17 (Task 4, undo_candidates returns unusable candidates): UPHELD, Important.
"draft" is in REVERSIBLE_ACTIONS so `reversible=True`, but `_undo_token` has no draft
branch and `_dispatch`'s return value is discarded, so the record carries
`undo_token={}` and still passes the `undo_candidates()` filter. An undo tool would
get a candidate with nothing to act on. Fix: require a non-empty undo_token in the
filter, with a comment recording that real draft-undo needs a draft id the client
does not yet return.
Cost if wrong: a draft is not offered as undoable — correct, since it currently isn't.

Ruling R18 (Task 4, test not selective): UPHELD, Important. The reviewer showed
`pytest.raises(ForbiddenActionError, match="send_message")` also passes when the
deny-list is entirely broken, because `_dispatch`'s fallback error text happens to
contain the string. The test's coverage survives only by accident of a sibling test.
Fix: assert `result.startswith("refused")` inside the same test, and add tests for
case/whitespace evasion and for an empty `forbidden_actions`.
Cost if wrong: none.

Ruling R19 (Task 4, minors): unused `dataclasses.field` import and the docstring
saying "allow-list" where the design says "deny-list" are bundled into the fix round
(both trivial, and the terminology one actively obscures which mechanism is
load-bearing). DEFERRED: the narrow crash window between a successful `_dispatch` and
`log.append` (the exception path is already covered; closing it fully needs a
two-phase write not worth it in a prototype), and the absence of file locking on the
audit log (single-process for Stage A).

Ruling R20 (Task 4, ExecutionContext.rule_provenance): NO CHANGE. My plan's prose
Interfaces block listed `rule_provenance` as an ExecutionContext field while the
plan's own code threads it as a separate kwarg on `execute_action`. The code is
right and Task 9 already calls it that way; the prose was wrong. Recorded, not fixed.
Cost if wrong: none.

Task 4: minor (deferred): crash window between successful dispatch and log.append.
Task 4: minor (deferred): no file locking on AuditLog.append (single-process only).
Task 4: minor (deferred): LiveGmailClient, when it exists, must not expose send or
  permanent-delete methods — today safety is partly a side effect of the protocol
  simply not having them. Flagged to the final review as a Stage-B prerequisite.

Task 4 fix round 1 BLOCKED on Task 6's implementer committing (ruling R9).
Task 5: implementer DONE (commit f8f9003, 45 passed). Controller verified the R1
pagination fix directly: 25 rules inserted -> 25 returned; rule #7 still matches.
Task 6: implementer dispatched.

Ruling R21 (REVISES R9): allow parallel implementers when they touch disjoint files,
with an explicit `.git/index.lock` retry instruction in every dispatch. R9's strict
serialisation was correct about the hazard but too blunt: three fix rounds are now
queued behind one another on files that never overlap, and the collision window is a
few milliseconds inside a 60-120s task. Each dispatch now tells the agent to retry
the commit after a short wait if the index is locked.
Cost if wrong: an agent hits a lock, retries, and succeeds; worst case one reports
BLOCKED on commit and I re-dispatch that single commit.

Task 5: review returned SPEC ✅ / QUALITY changes-requested. Reviewer probed the
pagination directly at (0,5) (5,5) (6,5) (10,5) (15,5) (1,1) (4,1) (25,1000) (0,1000)
— all correct counts, no infinite loop, exact-multiple cases terminate on a trailing
empty page. Also confirmed record_hit/mark_overridden do not desync the searchable
`text` field. The R1 fix is sound.

Ruling R22 (Task 5, as_table omits created_at): UPHELD, Important. The global
constraint requires each rule's provenance to include WHEN it was learned, and
as_table() is the owner's only window onto what the agent thinks it knows —
`created_at` exists on Rule and is used for sort order but is never surfaced. The
brief's test only checks column presence, so it could not catch this. Fixing now
rather than in Task 10, because deferring means changing both files together later.
Cost if wrong: one extra column in a table the user reads. None.

Ruling R23 (Task 5, minors): `add_rule` and `_put` have byte-identical bodies —
verbatim duplication of a logic block, which the review rubric treats as a defect;
bundled into the fix round as `add_rule` delegating to `_put`. The as_table test
asserts key presence but not values, so a broken as_table could pass — also bundled,
since R22 is touching that test anyway. DEFERRED: `rules()` re-paginates on every
`matching()`/`as_table()` call, O(n) per call — correct, and fine at Stage A scale.

Task 5: minor (deferred): rules() re-reads the full rule set on every matching()
  call; acceptable at Stage A scale, worth revisiting if the rule set grows large.

Task 4: fix round 1/5 (4 addressed, 0 open; commits 7c7c743..2591d07)
Controller re-tested every bypass the reviewer found, after the fix:
  exact kind dry-run / CASE variant dry-run / whitespace dry-run / case variant LIVE /
  EMPTY deny-list LIVE / empty deny-list dry-run  -> ALL refused, all with
  `refused: ... is on the deny-list` records. Legitimate archive still works
  (result=simulated, reversible=True, undo_token={'restore_labels': ['INBOX']}).
  The "Send_Message under dry-run reads as simulated" hole is closed.
Task 5: fix round 1/5 (3 addressed; commit 67b5edd). as_table now carries created_at.
Task 7: implementer DONE (commit 7891636, 6 passed, full suite 58).

Task 6: review returned SPEC ✅ / QUALITY approved-with-comments. Network safety
verified per-test: the one allow_remote=True test monkeypatches
_pull_from_context_hub before any langsmith.Client() call, and no other test touches
either symbol. try/except confirmed correctly scoped — it wraps only the remote pull,
so a bug in the local fallback would surface rather than be swallowed.

Ruling R24 (Task 6, policy taxonomy gaps): UPHELD, Important. The reviewer found that
automated security and bank mail fits no category cleanly — `needs_reply` requires "a
person waiting", `automated` is worded engineering-flavoured ("build, alert, and
system notifications"), and nothing designates `important_fyi` as the catch-all. There
is also no explicit fallback for mail matching nothing. This is not hypothetical for
this user: the actual 50-thread snapshot contains an x.ai new-login alert, an Atlassian
API-token confirmation, a Google OAuth advisory, a Google account-data-sharing notice,
and a Binance KYC demand. Six of fifty threads, and they are among the highest-value
mail in the inbox — plus the exact shape most often impersonated by phishing. A 12B
model forced to guess between important_fyi and automated on these will be
inconsistent, and inconsistency here trains bad rules.
Fix: add a `security_alert` category, add an explicit `other` fallback, and broaden
`automated` so it is not read as engineering-only.
Cost if wrong: a slightly larger taxonomy for a small model to choose from; mitigated
because the categories are disjoint and each has a one-line definition.

Ruling R25 (Task 6, minor): `test_policy_names_the_forbidden_actions` substring-checks
only "send" (which "sending" satisfies) and never checks deletion, despite its
docstring claiming the deny-list is stated in the prompt. Bundled into the fix round.

Ruling R26 (Task 6 -> Task 8 cross-task): the policy asserts an `<email_body>` contract
that Task 8 must honour exactly or the injection defence is inert. Carried into Task 8's
review as an explicit check rather than fixed here — it is Task 8's code.
Task 5: fix round 1/5 (3 addressed, 0 open; commits f8f9003..67b5edd)
Task 5: complete (commits 7c7c743..67b5edd, review clean)

Task 4: complete (commits 3badd70..2591d07, review clean)
  Re-reviewer probed {send_message, Send_Message, SEND_MESSAGE, " send_message",
  "send_message ", delete_forever} x {dry_run T/F} x {ALWAYS_FORBIDDEN, frozenset()}
  = 24/24 combinations refused with a `refused:` audit record. Legitimate actions
  unaffected; archive undo_token captured pre-mutation. An unknown kind
  ("frobnicate") still fails safely via the dispatch fallback.
Task 7: implementer DONE (commit 7891636). Reviewer dispatched.
Task 8: implementer DONE (commit 58b83f9, 8 passed, full suite 66). Reviewer
  dispatched with an explicit cross-task check of R26 — that classify.py's fence tag
  matches the policy's wording character-for-character.
  Controller pre-check: fence held against 4 payloads (plain override, fence-escape
  with a literal closing tag, fake system turn, 20k oversized body) — exactly one
  opening and one closing tag in every built prompt; 20k body truncated to a 4406
  char prompt.
Task 9: implementer dispatched (sonnet — most integration-heavy task; told the plan's
  "55 passed" figure is stale per R3, and told never to weaken a failing test).

Task 6: fix round 1/5 (2 addressed; commit 284d8ab). Policy now has 10 categories
including security_alert and an explicit `other` fallback; new stamp
local:5afcbb39121f (the version changing IS the content-addressing working).
Re-reviewer dispatched with an explicit check that the <email_body> section survived
unweakened, since Task 8's code contract depends on that exact wording.

Task 7: review returned SPEC ✅ / QUALITY changes-requested. Reviewer ran 8 probes:
partition integrity holds (nothing dropped, duplicated, or in both lists), duplicate
senders handled, hit-counting correct for one-rule-many-threads and
many-rules-one-thread (only the winner credited), zero LLM imports confirmed.

Ruling R27 (Task 7, untested tie-break): UPHELD, Important. The "most recently created
rule wins" tie-break is implemented correctly but is not covered by any test — the
reviewer demonstrated that a broken implementation using `matches[0]` instead of
`max(..., key=created_at)` passes all six existing tests. This is precisely the
"would a broken implementation still pass?" defect, and the tie-break matters: it is
what makes the owner's most recent instruction authoritative over an older one.
Fix: add a test where two rules with different created_at both match one thread, and
assert the newer rule's action AND rule_id win.
Cost if wrong: none.

Ruling R28 (Task 7, minors): input validation for None threads/prefs — DECLINED,
YAGNI; the graph is the only caller and it always passes real objects, and a None
would raise immediately and loudly anyway. Tests asserting only critical Decision
fields — bundled into R27's new test, which will also assert category and reason.

Task 9: implementer DONE (commit d9dc158, 7 passed, full suite 74). Reported the
brief's code and tests worked verbatim against langgraph 1.2.11 with no API
deviations needed.
Ruling R29 (Task 9, run_triage): NO CHANGE. My plan's Interfaces block named a
`run_triage(graph, thread_id, limit)` helper that neither the plan's own code nor any
test references — the notebook drives `graph.invoke` directly. Same class of plan-prose
inconsistency as R20. The implementer was right to leave it out rather than invent an
unused function. Cost if wrong: none; the notebook calls invoke directly.

Ruling R30 (Task 6, hardcoded category list): ACCEPT AS IS, no further round. The
re-reviewer correctly noted the new test hardcodes the ten category names rather than
parsing them out of the markdown, so it catches a category being DELETED from the
prompt but not one being ADDED. On reflection the hardcoded form is adequate: deletion
is the dangerous direction (a category silently vanishing from the prompt while code
still expects it), and that is caught. An addition is benign — `ThreadJudgment.category`
is a free-form str, so a new category breaks nothing. Regex-parsing a markdown list to
enumerate categories would add its own fragility for no safety gain. I asked for
dynamic enumeration; on the evidence the simpler form is the better call.
Cost if wrong: someone adds a category to the prompt and the test does not notice.
Nothing downstream breaks when that happens.

Task 6: fix round 1/5 (1 addressed, 1 accepted-as-is; commits fe3d7ab..284d8ab)
Task 6: complete (commits 67b5edd..284d8ab, review clean)
  <email_body> section confirmed preserved verbatim, which is what Task 8's contract
  depends on.

Ruling R31 (Task 8, fence escapes only the CLOSING tag): UPHELD, CRITICAL. `_fence()`
escapes `</email_body>` but never `<email_body>`. A body containing a literal OPENING
tag yields a prompt with 2 opens and 1 close — a syntactically well-formed nested
fence, which is the classic delimiter-confusion injection. The real closing tag cannot
be forged, so this is not a full escape, but a 12B model is not a strict XML parser and
may well treat the inner fence as authoritative. Worse, the brief's own test
(`test_injection_attempt_cannot_close_the_fence`) counts ONLY closing tags, so it
passes while the invariant is broken — false confidence, which is the failure mode this
whole review process exists to prevent.
Note: my own controller probe had the same blind spot. I tested closing-tag forgery
and reported the fence sound; I never tried a literal opening tag. Confirmed after the
fact: opening-tag payload gives opens=2 closes=1, and a body merely quoting the
policy's own wording does too.
Fix: `_fence()` escapes BOTH tags; the test asserts opens == 1 AND closes == 1.
Cost if wrong: none — escaping both is strictly safer and costs nothing.

Ruling R32 (Task 8, minors): bundled into the same round — `build_prompt` return hint
`list` -> `list[BaseMessage]`; the `ThreadJudgment.action` Field description omits
"unlabel" from the five it lists, narrowing the model's guidance for no reason; and
`_to_actions` defaults a null label to "Triaged", a string that appears nowhere in the
policy's vocabulary — use the judgment's own category instead, which does.

Task 7: fix round 1/5 (1 addressed; commit 1ae8889, 75 passing). Implementer confirmed
the new tie-break test FAILS against a deliberately broken `matches[0]` and passes
against the correct `max(...)` — the test is proven capable of failing.
Task 7: complete (commits 2591d07..1ae8889, review clean)
Task 8: fix round 1/5 (2 addressed; commit bebb866, 84 passing). Controller re-probed
with the payload set that now includes the opening-tag case I originally missed:
closing tag / OPENING tag / both tags interleaved / policy wording echoed / 5x closing
/ tag straddling the truncation edge / plain override / fake system turn — 8/8 leave
exactly one fence pair. Re-reviewer dispatched.

Ruling R33 (CONTROLLER FIX, data protection): the LangGraph checkpointer writes
`inbox_agent/checkpoints.sqlite`, and graph state carries full Thread objects. I
verified the file created by Task 10's notebook run contained live sender addresses
and the owner's own address. It was NOT gitignored — my plan's gitignore section
covered snapshot/, store/ and *.jsonl but never anticipated the sqlite files the
notebook creates. Fixed directly in the controller session (commit 96d7674) rather
than routed through a task, because it is user-data exposure rather than a code
finding and the window mattered. All three data paths now verified ignored:
snapshot/threads.json, checkpoints.sqlite, audit.jsonl.
Cost if wrong: none — these files are regenerable local state.

Task 9: review returned SPEC ✅ / QUALITY changes-requested with TWO Criticals, both
reproduced by the reviewer against a live graph, not merely argued.

Ruling R34 (Task 9, unknown thread id — approve path): UPHELD, CRITICAL.
`decisions[thread_id].actions` at graph.py:119 is a bare index. A resume payload
naming a thread absent from `state["decisions"]` raises KeyError and crashes the whole
invocation. Reproduced: `KeyError: 'ghost-thread'`. Fails loud, which is the better of
the two failure modes, but a stale or replayed resume should not kill the run.
Fix: use `.get()` and skip unknown ids.

Ruling R35 (Task 9, unknown thread id — edit path): UPHELD, CRITICAL, and the most
serious finding of this run. The "edit" branch reads `response.edits` with `.get()`
and never checks that the thread was part of the reviewed batch. The reviewer
fabricated a "ghost-thread-2" that had never been proposed or shown to anyone, and the
graph executed a `trash` action on it, logged as actor="human".
This bypasses the propose -> human-review trust boundary that the interrupt exists to
enforce. The deny-list still blocks send_message and delete_forever, so the blast
radius is bounded to reversible actions — but "the human approved a batch" must mean
the executed set is a SUBSET of what the human was shown, and today it does not.
The approve path fails loud; the edit path fails OPEN. That asymmetry is the bug.
Fix: both branches validate membership in `state["decisions"]` before dispatching,
and anything not in the reviewed set is skipped and recorded.
Cost if wrong: a legitimate edit naming a thread outside the batch is ignored. That is
the correct direction — such an edit was never reviewed.

Ruling R36 (Task 9, refusals only printed): UPHELD, Important. A ForbiddenActionError
is caught and `print()`ed. `execute_action` does durably log the refusal first, so the
audit trail is intact — but TriageState surfaces nothing, so a notebook or a future
Telegram bot can only learn of a refusal by separately reading the JSONL file, and
print() goes nowhere in a bot deployment. Fix: add a `refused` list to TriageState.

Ruling R37 (Task 9, reject-with-edits teaches but does not act): CONFIRMED AS
INTENDED, comment only. Reject means "do not do this", so not executing is right;
the edits still express what the owner would have preferred, so learning from them is
right. The combination is also unreachable through the intended API — `render.respond()`
sets verdict="edit" whenever edits are present, so only a hand-built payload produces
reject+edits. Add a clarifying comment rather than changing behaviour.

Ruling R38 (Task 9, minors): `rule_provenance=decision.reason` is passed for every
actor kind, so an agent-attributed record carries a field named rule_provenance holding
the model's prose reason — misleading when read literally. Bundled: pass it only for
rule-sourced decisions. Also bundled: unused `Annotated` and `Any` imports.

Task 9: fix round 1/5 (4 addressed; commit 23e0bc0, 87 passing).

Ruling R39 (Task 9, forbidden kind crashes the learn node): UPHELD, opening fix
round 2. The implementer flagged this while testing rather than silently working
around it in production code — correct behaviour, and the finding is real. Controller
confirmed: `rule_from_correction(thread, "send_message", ...)` raises pydantic
ValidationError because `Rule.action` is the strict `ActionKind` Literal, while
`Action.kind` is `str` by design (ruling R10).

The reachable path: a human types an edit naming a forbidden kind in the notebook.
`execute` refuses it correctly and records it in `refused` — then `learn` runs, calls
`rule_from_correction` with that same kind, and the graph CRASHES in its final node,
after mutations have already been executed. The run ends half-complete with a
checkpoint in an inconsistent state.

This is the same class as R34/R35 — validate before touching — and it is the third
instance of it in this one node group, which is itself the finding: the execute/learn
pair trusts the resume payload's shape everywhere it touches it.
Fix: `learn_from_response` skips any edit whose kind is not a member of ActionKind,
and records the skip rather than dropping it silently.
Cost if wrong: a correction naming an unlearnable action teaches nothing, which is
correct — there is no valid rule to learn from "send this", an action the system will
never perform.
Task 9: fix round 2/5 (1 addressed; commit 43260b8, 88 passing).
Ruling R40 (Task 9, two `skipped` entry shapes): ACCEPT. The implementer flagged that
execute's ghost-thread skip and learn's unlearnable-kind skip carry slightly different
payloads, discriminated by a `stage` key rather than one uniform schema. A tagged union
is the right shape here — the two skips genuinely mean different things and a renderer
can switch on `stage`. Forcing one schema would flatten information.
Controller re-ran the ghost-thread attack against the fixed graph: audit log contains
only ['t1']; ghost-thread-2 NOT executed; skipped surfaced as
[{'thread_id':'ghost-thread-2','verdict':'edit','reason':'not part of the reviewed
batch'}]. The review gate now holds.
NOTE: Task 8's re-review was packaged but never dispatched before I was pulled to the
Task 10 stall. Dispatching it now alongside Task 9's.
Task 8: fix round 1/5 (2 addressed, 0 open; commits 58b83f9..bebb866)
Task 8: complete (commits 7891636..bebb866, review clean)
  Re-reviewer confirmed both fence tags now escaped symmetrically and, importantly,
  no over-correction: HTML like <b>Sale</b> and maths like `x < 5 and y > 3` pass
  through unchanged. Only the fence delimiters are neutralised, not all angle
  brackets — a legitimate email discussing the system is not mangled.
Task 9: fix rounds 1-2 (5 addressed, 0 open; commits d9dc158..43260b8)
Task 9: complete (commits 284d8ab..43260b8, review clean)
  Re-reviewer drove a live graph rather than reading the shipped tests. Ghost approve
  and ghost edit both execute nothing and leave the audit log empty for that id; a
  mixed payload still processes the legitimate thread; a send_message edit is refused,
  learns nothing, and surfaces in BOTH `refused` and `skipped` without raising
  anywhere. VALID_ACTION_KINDS confirmed equal to frozenset(get_args(ActionKind)) by
  direct equality against the live symbol, so it cannot drift from the Literal.
  Over-correction check passed: a fully legitimate run still executes both actions and
  learns the expected rule.
  Note the implementer's fix was better than what I specified — it placed ONE guard
  before verdict-specific branching rather than duplicating a check per branch, so
  approve and edit are covered identically and a future third verdict inherits it.

Task 10: implementer committed render.py, test_render.py and inbox_agent.ipynb at
3901945, then TERMINATED — the account hit its monthly spend limit mid-task. It never
executed the notebook and never wrote a report.

Ruling R41 (controller completes Task 10's verification inline): the account is at its
spend limit and a subagent already died from it. The remaining work was a verification
step plus a small, fully-diagnosed fix, so I did it in the controller session rather
than spawning agents that would likely fail the same way. This deviates from "never fix
findings in the controller session"; recording it rather than hiding it.
Cost if wrong: these two changes did not get an independent implementer's eyes. They
are covered by the final whole-branch review, and both are small and measured.

FIRST REAL RUN AGAINST GEMMA — two defects found, both mine, both in the plan:

Ruling R42 (config.py dropped reasoning=False): CRITICAL, fixed in 76e5b9b. gemma4 is a
hybrid thinker and reports capability ['completion','tools','thinking']. build_notebook.py
:563 sets reasoning=False for exactly this reason; I dropped it when adapting the
resolver. Left on, a SINGLE classification did not return within 9 minutes — twice,
measured. Off, 3.1s. Without this the prototype simply does not run.

Ruling R43 (ThreadJudgment.reason was required): CRITICAL, fixed in 76e5b9b. The model
reliably returns category, action and confidence but omits reason. Required, pydantic
raised OutputParserException and classify_thread's catch-all degraded a CORRECT
classification into a zero-confidence no-op. Every one of 50 threads was being thrown
away. reason now defaults to "", keeping the judgment and leaving the gap visible.

FINDING (no fix, for the record): Ollama's schema-constrained decoding
(format: <json schema>) is SILENTLY IGNORED by this MLX build — it returns prose and
does not honour the schema at all. Only the model's own ```json-fenced output is usable.
This is a substantive constraint on any local-model design and belongs in the Stage B
comparison.

FINDING (open, for the user): `reason` is empty on 50/50 threads. The audit trail
records what the agent did and which policy version produced it, but the model-side
"why" is blank. That is a real gap in the audit story and is the strongest argument
measured so far for a hosted model on the reasoning step.

FINDING: measured throughput 3.1s/thread; the full 50-thread inbox classifies in ~2.6
minutes with 0 parse failures. Local-model triage at this scale is comfortably viable.

FINDING: one clear miscategorisation — a Strava product nudge ("Your missing heart rate
data") classified as security_alert at 0.90. The security_alert category I added in R24
is over-triggering on account-adjacent product mail. Prompt-tuning work, exactly the
kind of thing the lab exists to surface.

Notebook: executed end to end via nbconvert, exit 0. Printed backend=ollama,
dry_run=True, forbidden=['delete_forever','send_message'],
policy=local:5afcbb39121f, langsmith key masked as lsv2_pt...c7f1 (51 chars) — the
Task 1 mask fix working on the real key. 50 actions executed, all simulated; 0 rules
learned (expected: the default cell approves everything and makes no corrections).

Final fix wave: complete (commits 76e5b9b..22b7a0a, re-review clean, 92 passing).
Re-reviewer confirmed the thread_id-under-checkpoint_id substitution IS documented in
code, real reasons render byte-for-byte unchanged, the conftest pin overrides .env,
and the deny-list / fence / thread-id validation / reason default are all untouched.
BRANCH READY FOR MERGE.

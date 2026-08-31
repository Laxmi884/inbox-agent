# Spike: can a small/cheap model sustain a tool-calling loop?

**Date:** 2026-08-31 · **Status:** answered · **Code here is throwaway**, kept
only so the measurement can be re-run. It is not Stage B.

## The question

Stage A's central bet is that the *graph* owns control flow and the model only
makes leaf judgements, because a 12B local model is reliable at one small
structured judgement and unreliable across a long tool-calling loop. The design
spec says so outright: *"Gemma may fail at Stage B. This is an expected finding,
not a project risk."*

Before building Stage B, find out cheaply whether that is true.

## Method

Five mutating `GmailClient` methods bound as real tools (`apply_label`,
`remove_label`, `archive`, `trash`, `create_draft`) plus `done`. System prompt is
the existing policy verbatim. One thread per loop, body fenced exactly as Stage A
fences it. 10 threads, same threads for every arm. Turn cap 6 — exceeding it *is*
the looping failure.

`send_message` and `delete_forever` were deliberately **not** bound, so that
naming them would count as a hallucinated tool call.

Run: `python spikes/stage_b_tool_loop_probe.py <model> 10` from the repo root.

## Results

Seven models measured, 70 thread-runs. One further model was requested and could
not be run.

| model | s/thread | called a tool | looped | hallucinated | call sequences |
|---|---|---|---|---|---|
| `google/gemma-3-12b-it` | **1.47** | 9/10 | 0 | 0 | 9× `label→done` — **corrupt args** |
| `z-ai/glm-5.2` | **4.86** | 10/10 | 0 | 0 | 6× `archive→done`, 4× `label→archive→done` |
| `openai/gpt-5.6-luna` | 6.83 | 10/10 | 0 | 0 | 8× `label→archive→done`, 1× `label→done`, 1× `label→archive→label→done` |
| `qwen/qwen3-30b-a3b-instruct` | 7.70 | 10/10 | 0 | 0 | 6× `label→archive→done`, 4× `label→done` |
| `gemma4:12b-mlx` *(local)* | 13.41 | 10/10 | 0 | 0 | 8× `label→archive→done`, 2× `label→done` |
| `z-ai/glm-5.3-flash` | 26.44 | 10/10 | 0 | 0 | 5× `archive→done`, 5× `label→done` |
| `meta/muse-glimmer-30b` | 26.47 | 10/10 | 0 | 0 | 9× `label→done`, 1× `archive→done` |
| `meta/muse-spark-1.2` | — | — | — | — | **not run** — HTTP 403 |

Stage A baseline for comparison: 4.53 s/thread, one action per thread.

`muse-spark-1.2` returns `403: This model requires you to complete the following
before use: 18+ age confirmation`. An account-settings gate, not a technical
result — it is unmeasured, not failed, and must not be read as either.

## Findings

### 1. The premise was wrong. Tool-looping is not where small models fail.

Across 70 thread-runs and seven models: **zero runaway loops, zero hallucinated
tool names, zero errors.** Nothing hit the turn cap. Every arm but one called a
tool on every thread.

The expected failure mode — a small model wandering, repeating itself, or
inventing tools — did not occur once. Whatever the argument for Stage A's rigid
pipeline is, "the model cannot handle a tool loop" is not it.

### 2. The local model is viable for Stage B, and does something Stage A cannot.

`gemma4:12b-mlx` completed 10/10 at 13.41 s/thread with varied, sensible labels.
It costs ~3× Stage A's per-thread latency (50 threads ≈ 11 min vs 3.8 min).

More interestingly, on 8 of 10 threads it chose `apply_label` **then** `archive`
— categorise, then clear the inbox. Stage A's schema cannot express that: one
`ThreadJudgment` yields one action, so it must pick label *or* archive. The tool
loop is strictly more expressive, and the model used that expressiveness
correctly without being told to.

### 3. Stage A competence does not predict tool-calling competence. It inverted.

This is the finding with the longest reach.

- **`gemma3` is the registry's BEST MEASURED for Stage A** (1.29 s/thread, 0
  parse failures, `reason` 10/10). In a tool loop it emits **corrupted
  arguments**: labels truncated at varying points with `{}` appended —
  `recruiter`→`r{}`, `security_alert`→`securi{}`, `needs_reply`→`nee{}`,
  `newsletter_valuable`→`newsletter{}`. It also skipped the tool entirely on 1
  of 10. Reproduced outside the harness with a one-line prompt, so it is not an
  artifact of this probe. Unresolved: whether the fault is the model or
  OpenRouter's function-call assembly for it. Either way it is unusable here —
  it would apply a label literally named `r{}` to a real mailbox.
- **`muse` is the registry's worst structured-output arm** (2/10 parse failures,
  answers in markdown prose, rescued only by LangChain coercion). In a tool loop
  it is **flawless**: 10/10, no hallucinations, clean labels.

Constrained-JSON competence and tool-calling competence are separate
capabilities. A model registry that ranks on one does not rank on the other, and
the Stage A notes must not be read as Stage B guidance.

### 4. `muse` is usable after all.

The registry records `muse` as unusable because OpenRouter returns HTTP 400
*"Reasoning is mandatory for this endpoint"*. That error came from trying to
**disable** reasoning. `_build_openrouter()` never sets the parameter, so the
tool-loop path reaches it fine. The registry note is accurate about what was
attempted and misleading about what is possible.

### 5. The R24 over-trigger survives the architecture change.

The Strava nudge *"Your missing heart rate data"* was labelled `security_alert`
by `gemma4` in the tool loop — the same misclassification `357c106` fixed in
Stage A via the prompt contract. The contract is attached to
`ThreadJudgment`, which the tool path does not use, so the fix did not travel.
Any Stage B build inherits this bug and will need its own answer.

### 6. Nothing constrains the label to the policy taxonomy, and it shows.

Stage A cannot emit an off-taxonomy label: `ThreadJudgment.category` is checked
against the policy and the label is derived from it. In the tool loop, `label` is
just a free string argument, and models used that freedom.

`gpt-5.6-luna` applied a label named **`INBOX`** — a Gmail *system* label, not a
category. Against a real mailbox that is at best a no-op and at worst confusing
state. `gemma3`'s corrupted `r{}` and `securi{}` (finding 3) are the same class
of defect arriving by a different route.

This is a genuine capability regression from Stage A, and it is the tool-loop's
structural weakness rather than any one model's failing. Stage B needs the
taxonomy enforced **in the tool signature** — an enum argument, validated at the
tool boundary — not merely stated in the policy prompt. That is the same lesson
as the deny-list: state it in code at the chokepoint, not in the prompt.

### 7. Parallel tool calls appear in the wild.

`glm-5.2` returned two tool calls in a single turn on 6 of 10 threads
(`turns=1, calls=2`). The probe handled it because it iterates over
`ai.tool_calls`, but a Stage B implementation that assumes one call per turn
would silently drop the second. Worth designing for rather than discovering.

## Recommendation

**Stage B is worth building, and the local model is not the reason to hesitate.**

But the spike does not say Stage B is *better* — it says it is *feasible*. These
10 threads have no ground truth, so nothing here scores quality; it measures
whether the loop holds together. That was the question, and the answer is yes.

Deciding Stage A vs Stage B still needs labels, which is what the
productionisation milestone (Telegram → live Gmail → a few days of real use)
exists to produce.

Two things to carry forward:

- **`z-ai/glm-5.2` is the arm to build against** if Stage B proceeds: 4.86
  s/thread — faster than Stage A's own 4.53 baseline while doing strictly more
  work — 10/10 clean, no corruption, no off-taxonomy labels. `qwen3` (7.70s) is
  the close second and `gpt-5.6-luna` (6.83s) is quick but emitted `INBOX` as a
  label. Local `gemma4:12b` works and is the offline fallback at 13.41s.
- **Enforce the taxonomy in the tool signature, not the prompt** (finding 6).
  This is the one place the tool loop is structurally weaker than Stage A, and
  it is fixable at the tool boundary.
- **Never reuse a Stage A model ranking for a tool-calling decision.** Finding 3
  is the reason, and it cost nothing to learn here rather than in production.
- **`muse-spark-1.2` remains unmeasured** pending the 18+ confirmation on the
  OpenRouter account. Re-run with
  `python spikes/stage_b_tool_loop_probe.py or:meta/muse-spark-1.2 10`.

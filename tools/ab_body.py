"""Does sending the email body actually classify better than the snippet?

The question has been open since live Gmail was switched on, and nothing could
answer it: every latency figure in the model registry was measured on
snippet-sized prompts, and the frozen snapshot has no bodies at all (0 of 50
populated, max 201 chars), so the one corpus available for experiments cannot
represent the thing being tested.

This is that experiment. Three properties make it trustworthy:

**It never writes to Gmail.** It calls classify_thread directly and never
touches the graph, so no action reaches execute_action and no mail moves. An
A/B that archived the same mail twice would be worse than no A/B.

**Both arms see byte-identical input.** Threads are fetched once, cached to
disk, and replayed. Re-running costs nothing, the arms cannot drift, and the
cache doubles as the bodies-included reference set the frozen snapshot never
was - which is what the rest of P4 needs to build evals on.

**It measures before it runs.** `--survey` reports the body-size distribution
and stops. Gemma runs at num_ctx=8192, which is roughly 32 000 characters for
the ENTIRE prompt - policy, standing instructions, JSON schema and email. A
long newsletter can push the policy out of the window, and an overflowing
prompt is exactly how the 16 384-token runaway of 2026-09-03 happened. Look at
the distribution before choosing to send bodies uncapped.

Arms are tagged in LangSmith (`ab:snippet`, `ab:full`, ...) so the traces are
comparable there too, not only in this script's output.

    python -m tools.ab_body --survey                  # sizes only, no model
    python -m tools.ab_body --sample 200              # stratified draw, cached
    python -m tools.ab_body --arms snippet,4000,full --arm-sample 60

Fetching and classifying are sized separately on purpose: `--sample` is cheap
and enumerates the whole mailbox, `--arm-sample` bounds the hours of model time
that follow. Cache the corpus once, measure on a slice, widen the slice when
the slice says something. `--fetch N` keeps the original recency draw.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "inbox_agent" / "store"
CACHE = STORE / "ab_threads.json"
RESULTS = STORE / "ab_results.json"
DISAGREEMENTS = STORE / "ab_disagreements.md"

# Gmail's own tabs. An account that never enabled categories returns nothing
# for four of these, which costs nothing - see allocate().
CATEGORIES = ("primary", "social", "promotions", "updates", "forums")

# Hydration pacing. Gmail's quota is measured per minute, so a burst that
# empties it stays empty for the rest of that minute and _with_backoff's six
# attempts expire waiting - retrying harder is the failure, not the fix. These
# ask for roughly 60 quota units a second against a limit that took 5 workers
# flat out to breach. A 200-thread sample costs about two minutes.
HYDRATE_CHUNK = 25
HYDRATE_PAUSE = 4.0
HYDRATE_WORKERS = 2


def _load_env() -> None:
    """Read .env the way the bot does, so this needs no exported shell vars."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    import os
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None:
            os.environ.setdefault(key, value)


def fetch(limit: int) -> list[dict]:
    """Pull threads from the real mailbox, read-only, and cache them.

    Uses the same client the bot uses, so what is cached is what the agent
    would actually have seen - not a hand-built fixture that agrees with the
    code by construction.
    """
    from inbox_agent.config import build_gmail_client, load_settings

    settings = load_settings()
    if settings.gmail != "live":
        raise SystemExit(
            f"INBOX_GMAIL={settings.gmail!r}: the snapshot has no bodies, which "
            f"is the whole thing being tested. Fetch against live.")
    client = build_gmail_client(settings)
    # Read-only: list_threads and nothing else. The write methods on this client
    # are never called from here, which is what makes the experiment safe to
    # re-run against the real mailbox.
    threads = client.list_threads(limit=limit, query=settings.inbox_query)
    rows = [t.model_dump(mode="json") for t in threads]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(
        {"version": 2,
         "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "strata": {}, "threads": rows}, indent=1), encoding="utf-8")
    print(f"cached {len(rows)} threads -> {CACHE}")
    return rows


def _strata(years: int, base_query: str) -> list[tuple[str, str]]:
    """Gmail queries that cut the mailbox by category and by year.

    Stratifying by query rather than by post-hoc bucketing is forced, not
    chosen: threads.list returns {id, snippet, historyId} and nothing else, so
    a thread's category and date are unknown until it is hydrated - and
    hydrating in order to decide whether to hydrate is the cost this sampler
    exists to avoid. Gmail already indexes both facets, so ask it 25 narrow
    questions instead of one broad one.

    The last bucket is open-ended (`before:`) so no mail is unreachable however
    old the account is.
    """
    year = datetime.date.today().year
    spans = [(str(year - i), f"after:{year - i}/01/01 before:{year - i + 1}/01/01")
             for i in range(years - 1)]
    oldest = year - years + 2
    spans.append((f"<{oldest}", f"before:{oldest}/01/01"))
    return [(f"{cat}/{name}", f"{base_query} category:{cat} {span}")
            for cat in CATEGORIES for name, span in spans]


def allocate(pools: dict[str, list[str]], n: int, rng: random.Random) -> dict[str, str]:
    """Round-robin draw across strata -> {thread_id: stratum}.

    Equal per stratum, not proportional. A mailbox that is 80% promotions would
    otherwise yield a sample that is 80% promotions - which is the same
    over-representation the recency fetch suffers from, wearing a different
    hat. The question being asked is "where does a body change the verdict",
    and that needs coverage of the rare strata, not fidelity to the common one.

    Exhausted strata drop out and their quota redistributes on the next pass,
    so an empty category needs no special case. Ids are deduped because a
    thread whose messages span a year boundary matches two date strata.
    """
    remaining = {label: list(ids) for label, ids in pools.items() if ids}
    for ids in remaining.values():
        rng.shuffle(ids)
    chosen: dict[str, str] = {}
    while remaining and len(chosen) < n:
        for label in list(remaining):
            if len(chosen) >= n:
                break
            ids = remaining[label]
            chosen.setdefault(ids.pop(), label)
            if not ids:
                del remaining[label]
    return chosen


def sample(n: int, seed: int, years: int, per_stratum: int) -> list[dict]:
    """Stratified random draw across the whole inbox, read and unread alike.

    The recency fetch (--fetch) takes the newest N of `in:inbox is:unread`,
    which is the distribution the live bot faces each morning but a poor
    measuring stick for this experiment: newest unread inbox mail skews to
    newsletters and promotions, where the subject alone already decides the
    verdict and a body can only fail to help. The threads a body should matter
    most for - real correspondence, long replies, an ask buried in paragraph
    three - are exactly the ones recency under-samples.

    `is:unread` is dropped so the draw can reach back through the mailbox;
    `in:inbox` is kept so the corpus still resembles what the agent triages,
    and the triaged label is still excluded so the agent's own past verdicts
    do not contaminate the set it is measured on.
    """
    from inbox_agent.config import build_gmail_client, load_settings

    settings = load_settings()
    if settings.gmail != "live":
        raise SystemExit(
            f"INBOX_GMAIL={settings.gmail!r}: the snapshot has no bodies, which "
            f"is the whole thing being tested. Sample against live.")
    client = build_gmail_client(settings)
    base = f"in:inbox -label:{settings.triaged_label}"

    pools: dict[str, list[str]] = {}
    print(f"enumerating {years * len(CATEGORIES)} strata (ids only, no bodies):")
    for label, query in _strata(years, base):
        ids = client.list_thread_ids(query=query, max_ids=per_stratum)
        pools[label] = ids
        print(f"  {label:<20} {len(ids):>5}" + ("  (capped)" if len(ids) >= per_stratum else ""))

    chosen = allocate(pools, n, random.Random(seed))
    total = sum(len(v) for v in pools.values())
    print(f"\n{total:,} threads seen, sampling {len(chosen)} "
          f"({len(set(chosen.values()))} strata represented, seed={seed})")
    if len(chosen) < n:
        print(f"asked for {n}; the mailbox has only {len(chosen)} matching threads")

    return _hydrate_paced(client, chosen, seed)


def _cached_rows(wanted: dict[str, str]) -> dict[str, dict]:
    """Rows already on disk for threads this draw wants, keyed by id.

    Resume is free because the draw is seeded: the same --sample N --seed S
    selects the same ids, so a re-run after a rate limit picks up exactly where
    the last one stopped instead of paying for the whole sample again.

    The seed fixes the draw given the same pools, not across time. The mailbox
    is live, so mail arriving between two runs shifts what list_thread_ids
    returns and the shuffle lands elsewhere - observed on 2026-09-04, two
    seed-0 draws 25 minutes apart differing by a few threads. Reproducing an
    experiment exactly means keeping the cache, not re-running the sampler.
    """
    if not CACHE.exists():
        return {}
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    rows = data if isinstance(data, list) else data.get("threads", [])
    return {r["id"]: r for r in rows if r.get("id") in wanted}


def _hydrate_paced(client, chosen: dict[str, str], seed: int) -> list[dict]:
    """Fetch the sample in paced chunks, writing the cache as it goes.

    Both halves are lessons from the 403 that killed the first 200-thread run
    on 2026-09-04, and both are cheap to honour.

    Pace, because the quota is per minute and five workers hydrating flat out
    re-consume it the instant it refills - see get_threads in gmail.py.

    Write as it goes, because that run hydrated most of 200 threads and cached
    none of them: one 403 near the end unwound the entire list and the whole
    fetch had to start over. The cost of a rate limit is now one chunk.
    """
    STORE.mkdir(parents=True, exist_ok=True)
    have = _cached_rows(chosen)
    if have:
        print(f"resuming: {len(have)} of {len(chosen)} already cached")
    todo = [tid for tid in chosen if tid not in have]

    def flush() -> None:
        CACHE.write_text(json.dumps({
            "version": 2,
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seed": seed, "strata": chosen,
            "threads": [have[tid] for tid in chosen if tid in have],
        }, indent=1), encoding="utf-8")

    for start in range(0, len(todo), HYDRATE_CHUNK):
        batch = todo[start:start + HYDRATE_CHUNK]
        for thread in client.get_threads(batch, workers=HYDRATE_WORKERS):
            have[thread.id] = thread.model_dump(mode="json")
        flush()
        print(f"  hydrated {len(have)}/{len(chosen)}", flush=True)
        if start + HYDRATE_CHUNK < len(todo):
            time.sleep(HYDRATE_PAUSE)

    flush()
    print(f"cached {len(have)} threads -> {CACHE}")
    return [have[tid] for tid in chosen if tid in have]


def load_cached() -> tuple[list[dict], dict[str, str]]:
    """Threads plus their stratum labels. A v1 cache is a bare list with none."""
    if not CACHE.exists():
        raise SystemExit(f"no cache at {CACHE}. Run with --sample N first.")
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    return data["threads"], data.get("strata", {})


def arm_subset(raw: list[dict], strata: dict[str, str], k: int,
               seed: int) -> list[dict]:
    """A stratified k of the cache, for the arms to run over.

    Fetching is cheap and classifying is not, so the two sizes are decoupled:
    cache the whole corpus once, measure on a slice, widen the slice when the
    slice says something. Cache order is preserved so two runs at different k
    report the same threads in the same order.
    """
    if k <= 0 or k >= len(raw):
        return raw
    pools: dict[str, list[str]] = {}
    for item in raw:
        pools.setdefault(strata.get(item["id"], "unstratified"), []).append(item["id"])
    chosen = allocate(pools, k, random.Random(seed))
    return [t for t in raw if t["id"] in chosen]


def survey(raw: list[dict], strata: dict[str, str] | None = None) -> None:
    """Body sizes, and what they imply for an 8192-token window."""
    from inbox_agent.classify import MAX_BODY_CHARS

    if strata:
        counts: dict[str, int] = {}
        for item in raw:
            counts[strata.get(item["id"], "unstratified")] = 1 + counts.get(
                strata.get(item["id"], "unstratified"), 0)
        print("stratum spread       " + ", ".join(
            f"{k} {v}" for k, v in sorted(counts.items())))

    bodies = [len(t.get("body") or "") for t in raw]
    snippets = [len(t.get("snippet") or "") for t in raw]
    empty = sum(1 for b in bodies if b == 0)
    # ~4 chars per token is the usual rough conversion; num_ctx=8192 leaves
    # roughly 32 000 characters for policy + instructions + schema + email.
    window_chars = 8192 * 4
    over = sum(1 for b in bodies if b > window_chars)
    over_cap = sum(1 for b in bodies if b > MAX_BODY_CHARS)

    print(f"threads              {len(raw)}")
    print(f"bodies empty         {empty}")
    if not any(bodies):
        print("\nEvery body is empty - this cache came from the snapshot, and "
              "the experiment cannot run on it.")
        return
    print(f"snippet  median      {statistics.median(snippets):,.0f} chars")
    print(f"body     median      {statistics.median(bodies):,.0f} chars")
    print(f"body     mean        {statistics.mean(bodies):,.0f} chars")
    print(f"body     max         {max(bodies):,} chars")
    print(f"body     p90         {sorted(bodies)[int(len(bodies) * 0.9) - 1]:,} chars")
    print()
    print(f"over the {MAX_BODY_CHARS:,}-char default cap   {over_cap}/{len(raw)}")
    print(f"over the whole ~{window_chars:,}-char window   {over}/{len(raw)}"
          + ("   <- these cannot be sent uncapped without pushing the policy "
             "out of the context window" if over else "   <- uncapped is safe "
             "on this sample"))


def run_arm(raw: list[dict], budget: int, label: str) -> list[dict]:
    from langsmith import tracing_context

    from inbox_agent.classify import classify_thread
    from inbox_agent.config import get_llm, load_settings
    from inbox_agent.models import Thread
    from inbox_agent.policy import load_policy

    settings = load_settings()
    policy = load_policy(settings)
    llm = get_llm()

    rows = []
    for index, item in enumerate(raw, 1):
        thread = Thread.model_validate(item)
        started = time.monotonic()
        # Tagged so the two arms are separable in LangSmith as well as here.
        with tracing_context(tags=[f"ab:{label}"],
                             metadata={"ab_arm": label, "ab_thread": thread.id}):
            decision = classify_thread(thread, llm, policy, None,
                                       body_budget=budget)
        elapsed = time.monotonic() - started
        rows.append({"thread_id": thread.id, "subject": thread.subject,
                     "category": decision.category,
                     "actions": [a.kind for a in decision.actions],
                     "confidence": decision.confidence, "seconds": elapsed})
        print(f"  {label:<8} {index}/{len(raw)}  {elapsed:5.1f}s  "
              f"{decision.category}", flush=True)
    return rows


def compare(results: dict[str, list[dict]]) -> None:
    names = list(results)
    base = names[0]
    print(f"\n{'arm':<10} {'median s':>9} {'total s':>9} {'agree with ' + base:>20}")
    for name in names:
        rows = results[name]
        times = [r["seconds"] for r in rows]
        if name == base:
            agree = "-"
        else:
            same = sum(1 for a, b in zip(results[base], rows)
                       if a["category"] == b["category"])
            agree = f"{same}/{len(rows)}"
        print(f"{name:<10} {statistics.median(times):9.1f} {sum(times):9.1f} {agree:>20}")

    for name in names[1:]:
        diffs = [(a, b) for a, b in zip(results[base], results[name])
                 if a["category"] != b["category"]]
        if not diffs:
            continue
        print(f"\ndisagreements, {base} -> {name}:")
        for a, b in diffs:
            print(f"  {a['subject'][:52]:<52} {a['category']} -> {b['category']}")


def _persist(results: dict[str, list[dict]], strata: dict[str, str]) -> None:
    """Write the run to disk.

    With LANGSMITH_TRACING=false there is no trace to go back to, so a
    scrollback buffer is the only record of a run that costs hours - and the
    disagreements are the part that wants reading slowly, by a human, later.
    That file is also the seed of the labelled set the rest of P4 needs:
    agreement between two arms is not correctness, and nothing here knows
    which arm was right.
    """
    STORE.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(
        {"ran_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "strata": strata, "arms": results}, indent=1), encoding="utf-8")

    names = list(results)
    base = names[0]
    lines = [f"# A/B disagreements, base arm `{base}`", ""]
    for name in names[1:]:
        rows = [(a, b) for a, b in zip(results[base], results[name])
                if a["category"] != b["category"]]
        lines += [f"## {base} -> {name}: {len(rows)} of {len(results[base])}", ""]
        for a, b in rows:
            lines += [f"- **{a['subject'][:90]}**  ",
                      f"  `{a['category']}` -> `{b['category']}`  ",
                      f"  stratum {strata.get(a['thread_id'], '?')}, "
                      f"thread `{a['thread_id']}`  ",
                      "  which is right? "]
        lines.append("")
    DISAGREEMENTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS.name} and {DISAGREEMENTS.name} to {STORE}")


def _budget(token: str) -> tuple[int, str]:
    from inbox_agent.config import BODY_FULL

    token = token.strip().lower()
    if token in ("snippet", "0"):
        return 0, "snippet"
    if token == "full":
        return BODY_FULL, "full"
    return int(token), token


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", type=int, metavar="N",
                    help="cache the newest N unread inbox threads (recency)")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="cache a stratified random N across category and year")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; reproduces a draw from the same pools")
    ap.add_argument("--years", type=int, default=5,
                    help="date strata; the last one is open-ended (default 5)")
    ap.add_argument("--per-stratum", type=int, default=500, metavar="N",
                    help="ids enumerated per stratum before drawing (default 500)")
    ap.add_argument("--survey", action="store_true",
                    help="report body sizes and stop, without calling the model")
    ap.add_argument("--arms", default=None,
                    help="comma-separated budgets: snippet, full, or a number. "
                         "Naming it is what starts the model run")
    ap.add_argument("--arm-sample", type=int, default=0, metavar="K",
                    help="run the arms over a stratified K of the cache, not all")
    args = ap.parse_args(argv)

    if args.fetch and args.sample:
        ap.error("--fetch and --sample both write the cache; choose one")

    _load_env()
    if args.sample:
        sample(args.sample, args.seed, args.years, args.per_stratum)
    elif args.fetch:
        fetch(args.fetch)
    raw, strata = load_cached()

    survey(raw, strata)
    if args.survey:
        return 0

    # Caching must not fall through into hours of model time. --sample 200 is a
    # two-minute read-only fetch; the arms behind it are an afternoon of local
    # GPU, and the two got run together by accident on 2026-09-04 because the
    # arms had a default. Filling the cache and starting an experiment are
    # different intentions, so the expensive one is now named explicitly.
    if args.arms is None:
        if args.fetch or args.sample:
            print("\ncache written. Add --arms snippet,4000,full to run the "
                  "experiment (hours), or --survey for sizes only.")
            return 0
        args.arms = "snippet,full"

    raw = arm_subset(raw, strata, args.arm_sample, args.seed)
    if args.arm_sample:
        print(f"\narms run over {len(raw)} of the cached threads "
              f"(--arm-sample {args.arm_sample}, seed {args.seed})")

    results = {}
    for token in args.arms.split(","):
        budget, label = _budget(token)
        print(f"\narm {label} (budget={budget}):")
        results[label] = run_arm(raw, budget, label)
    compare(results)
    _persist(results, strata)
    return 0


if __name__ == "__main__":
    sys.exit(main())

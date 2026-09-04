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
    python -m tools.ab_body --fetch 40                # cache 40 live threads
    python -m tools.ab_body --arms snippet,4000,full  # run and compare

Reads the cache unless --fetch is given, so the expensive step happens once.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "inbox_agent" / "store" / "ab_threads.json"


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
    payload = [t.model_dump(mode="json") for t in threads]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"cached {len(payload)} threads -> {CACHE}")
    return payload


def load_cached() -> list[dict]:
    if not CACHE.exists():
        raise SystemExit(f"no cache at {CACHE}. Run with --fetch N first.")
    return json.loads(CACHE.read_text(encoding="utf-8"))


def survey(raw: list[dict]) -> None:
    """Body sizes, and what they imply for an 8192-token window."""
    from inbox_agent.classify import MAX_BODY_CHARS

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
                    help="fetch N threads from the live mailbox into the cache")
    ap.add_argument("--survey", action="store_true",
                    help="report body sizes and stop, without calling the model")
    ap.add_argument("--arms", default="snippet,full",
                    help="comma-separated budgets: snippet, full, or a number")
    args = ap.parse_args(argv)

    _load_env()
    raw = fetch(args.fetch) if args.fetch else load_cached()

    survey(raw)
    if args.survey:
        return 0

    results = {}
    for token in args.arms.split(","):
        budget, label = _budget(token)
        print(f"\narm {label} (budget={budget}):")
        results[label] = run_arm(raw, budget, label)
    compare(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

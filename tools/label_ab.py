"""Human verdicts on the threads the A/B arms disagreed about.

The A/B measures agreement between arms, which is not correctness: when
snippet says `needs_reply` and the body says `automated`, nothing in the
experiment knows which is right. This is where a person says so, and the file
it writes is the first reference set the agent can actually be scored against
rather than compared to itself.

Three properties it needs, and the reasons are worth stating because each one
is a way an eval set quietly stops being evidence.

**The label is free text from the policy, not a choice between the two arms.**
Both can be wrong. Offering only their answers would bake the arms' shared
blind spots into the ground truth and guarantee a future model scores well on
them.

**It records the policy version.** A category means what the policy says it
means; `newsletter_valuable` under a rewritten policy is a different label
wearing the same name. A label with no policy attached cannot be re-checked.

**It keeps the disagreement that prompted it.** Which arms differed, and how,
is the reason the thread was interesting. Discarding it leaves a labelled row
that cannot answer "did the body help", which is the question it was collected
to answer.

    python -m tools.label_ab --review           # the unlabelled ones, with bodies
    python -m tools.label_ab --review --all     # including ones already done
    python -m tools.label_ab --set <id> receipt --note "it is a payment receipt"
    python -m tools.label_ab --status
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "inbox_agent" / "store"
CACHE = STORE / "ab_threads.json"
RESULTS = STORE / "ab_results.json"
LABELS = STORE / "ab_labels.json"
POLICY = ROOT / "inbox_agent" / "policies" / "default.md"

EXCERPT_CHARS = 700


def categories() -> list[str]:
    """The categories the policy defines, read from the authoring surface.

    Same derivation as policy_categories() in telegram/__main__.py, and for
    the same reason: a label the policy does not define is not a label, it is
    a typo that will be discovered months later by whatever consumes this file.
    """
    section = POLICY.read_text(encoding="utf-8").split("## Categories", 1)
    if len(section) < 2:
        return []
    return re.findall(r"^-\s+`([a-z_]+)`", section[1].split("\n## ", 1)[0], re.M)


def _load(path: Path, what: str) -> dict:
    if not path.exists():
        raise SystemExit(f"no {what} at {path}. Run tools.ab_body first.")
    return json.loads(path.read_text(encoding="utf-8"))


def disagreements() -> list[dict]:
    """Every thread where at least two arms landed on different categories."""
    res = _load(RESULTS, "results")
    cache = _load(CACHE, "thread cache")
    threads = {t["id"]: t for t in cache["threads"]}
    arms = res["arms"]
    names = list(arms)
    by_thread: dict[str, dict] = {}
    for name in names:
        for row in arms[name]:
            by_thread.setdefault(row["thread_id"], {})[name] = row["category"]

    out = []
    for tid, verdicts in by_thread.items():
        if len(set(verdicts.values())) < 2:
            continue
        thread = threads.get(tid, {})
        out.append({
            "thread_id": tid,
            "subject": thread.get("subject", ""),
            "sender": thread.get("sender", ""),
            "date": thread.get("date", ""),
            "stratum": res.get("strata", {}).get(tid, "?"),
            "body_chars": len(thread.get("body") or ""),
            "body_excerpt": (thread.get("body") or "")[:EXCERPT_CHARS],
            "arms": verdicts,
        })
    return out


def load_labels() -> dict:
    if not LABELS.exists():
        return {"version": 1, "policy_version": None, "items": {}}
    return json.loads(LABELS.read_text(encoding="utf-8"))


def save_label(thread_id: str, truth: str, note: str = "") -> None:
    valid = categories()
    if valid and truth not in valid:
        raise SystemExit(f"{truth!r} is not a policy category. One of: "
                         + ", ".join(valid))
    found = {d["thread_id"]: d for d in disagreements()}
    if thread_id not in found:
        raise SystemExit(f"{thread_id!r} is not one of the disagreements")

    data = load_labels()
    item = found[thread_id]
    data["policy_version"] = _load(RESULTS, "results").get("policy_version")
    data["items"][thread_id] = {
        "subject": item["subject"],
        "stratum": item["stratum"],
        "body_chars": item["body_chars"],
        "arms": item["arms"],
        "truth": truth,
        "note": note,
        "labelled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    STORE.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(data, indent=1), encoding="utf-8")
    agreeing = sorted(a for a, c in item["arms"].items() if c == truth)
    print(f"{thread_id}  {truth}"
          + (f"   (agrees with {', '.join(agreeing)})" if agreeing
             else "   (no arm got this right)"))


def review(show_all: bool) -> None:
    labelled = load_labels()["items"]
    items = disagreements()
    todo = [d for d in items if show_all or d["thread_id"] not in labelled]
    print(f"{len(items)} disagreements, {len(labelled)} labelled, "
          f"{len(items) - len(labelled)} to go\n")
    for i, d in enumerate(todo, 1):
        print("=" * 78)
        print(f"[{i}/{len(todo)}]  {d['subject']}")
        print(f"  from    {d['sender']}")
        print(f"  {d['stratum']}, {d['body_chars']:,} chars, {d['date']}")
        for arm, cat in d["arms"].items():
            print(f"  {arm:<9} -> {cat}")
        if d["thread_id"] in labelled:
            print(f"  LABELLED -> {labelled[d['thread_id']]['truth']}")
        body = re.sub(r"\n{3,}", "\n\n", d["body_excerpt"]).strip()
        print("  ---")
        for line in body.splitlines()[:14]:
            print(f"  | {line[:96]}")
        print(f"\n  python -m tools.label_ab --set {d['thread_id']} <category>")
        print()
    if todo:
        print("categories:", ", ".join(categories()))


def status() -> None:
    labelled = load_labels()["items"]
    items = disagreements()
    print(f"{len(labelled)}/{len(items)} labelled")
    if not labelled:
        return
    arms = sorted({a for d in items for a in d["arms"]})
    print("\nhow often each arm was right, on the threads they disagreed about:")
    for arm in arms:
        right = sum(1 for tid, lab in labelled.items()
                    if lab["arms"].get(arm) == lab["truth"])
        print(f"  {arm:<9} {right}/{len(labelled)}")
    neither = [lab for lab in labelled.values()
               if lab["truth"] not in lab["arms"].values()]
    if neither:
        print(f"\n{len(neither)} where no arm was right - the interesting ones:")
        for lab in neither:
            print(f"  {lab['subject'][:60]}  -> {lab['truth']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", action="store_true", help="print threads to judge")
    ap.add_argument("--all", action="store_true", help="with --review, include done")
    ap.add_argument("--set", nargs=2, metavar=("THREAD_ID", "CATEGORY"))
    ap.add_argument("--note", default="", help="why, for the next reader")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    if args.set:
        save_label(args.set[0], args.set[1], args.note)
    elif args.status:
        status()
    else:
        review(args.all)
    return 0


if __name__ == "__main__":
    sys.exit(main())

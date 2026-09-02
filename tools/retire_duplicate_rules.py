#!/usr/bin/env python3
"""Retire rules that a later rule already replaced.

PreferenceStore.add_rule now retires any live rule sharing a new rule's scope
and pattern, so duplicates cannot be created any more. Stores written before
that change still hold them: correcting one sender twice left two live rules
deciding the same mail, with the newer winning only on `matching`'s
max(created_at) tie-break.

That is stable until the newer rule is demoted, at which point the older one
takes over and - having never fired, so never being demotable itself - keeps
deciding permanently. This retires the older ones exactly as add_rule would
have: overridden, never deleted, with the survivor naming what it replaced.

    python tools/retire_duplicate_rules.py --check   # report only, exit 1 if any
    python tools/retire_duplicate_rules.py           # apply
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report duplicates without retiring them; exit 1 if any")
    ap.add_argument("--store", default=None,
                    help="path to prefs.sqlite (default: the configured store)")
    args = ap.parse_args(argv)

    from inbox_agent.config import load_settings
    from inbox_agent.store import PreferenceStore, open_store

    path = Path(args.store) if args.store else load_settings().store_dir / "prefs.sqlite"
    if not path.exists():
        print(f"no store at {path}", file=sys.stderr)
        return 2
    prefs = PreferenceStore(open_store(path))

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for rule in prefs.rules():
        if not rule.overridden:
            groups[(rule.scope, rule.pattern)].append(rule)

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    if not dupes:
        print(f"{path}: no duplicate live rules.")
        return 0

    retired = 0
    for (scope, pattern), rules in dupes.items():
        rules.sort(key=lambda r: r.created_at)
        survivor, older = rules[-1], rules[:-1]
        print(f"\n{scope} {pattern!r}: {len(rules)} live rules")
        print(f"  keep   {survivor.id}  {survivor.created_at}  {survivor.summary}")
        for rule in older:
            print(f"  retire {rule.id}  {rule.created_at}  {rule.summary}")
        if args.check:
            retired += len(older)
            continue
        for rule in older:
            prefs.mark_overridden(rule.id)
            retired += 1
        if not survivor.supersedes:
            survivor.supersedes = older[-1].id
            prefs._put(survivor)
            print(f"  {survivor.id}.supersedes = {older[-1].id}")

    verb = "would retire" if args.check else "retired"
    print(f"\n{verb} {retired} rule(s); none deleted.")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())

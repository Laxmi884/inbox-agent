#!/usr/bin/env python3
"""Publish inbox_agent/policies/default.md to Context Hub as the inbox-triage skill.

The committed file is the authoring surface: it is what code review sees, what
`git log -p` explains, and what the content tests in tests/test_policy.py assert
against. Context Hub is the publish target - a derived artifact, the way the
teaching notebook is derived from build_teaching_notebook.py.

Run this after every policy edit. Skipping it does not break a run: load_policy
keeps serving the hub's older text and prints a DRIFT line naming both copies,
and the bot's startup banner repeats it. That is the whole point of the warning
- a forgotten push should be loud, not fatal.

    python tools/push_policy.py            # push if the texts differ
    python tools/push_policy.py --check    # report only; exit 1 on drift
    python tools/push_policy.py --force    # push even when identical

If the agent ever runs ON LangSmith, this direction inverts: the hub becomes the
source of truth and the committed file becomes a snapshot that CI checks. Keep
the equality check either way - policy and classifier are coupled (a new
category needs both a policy edit and code that handles it), so a policy that
can change independently of the code can name a category nothing implements.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "inbox_agent" / "policies" / "default.md"
SKILL = "inbox-triage"


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report drift without pushing; exit 1 if they differ")
    ap.add_argument("--force", action="store_true",
                    help="push even when the two copies already match")
    args = ap.parse_args(argv)

    _load_env()
    import os
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set; nothing to push to.", file=sys.stderr)
        return 2

    from langsmith import Client
    from langsmith.schemas import FileEntry

    local = POLICY.read_text(encoding="utf-8")
    client = Client()

    remote = None
    if client.skill_exists(SKILL):
        ctx = client.pull_skill(SKILL)
        files = getattr(ctx, "files", {}) or {}
        content = files.get("POLICY.md")
        remote = getattr(content, "content", content)

    if remote is None:
        print(f"{SKILL}: no POLICY.md on the hub yet.")
    elif remote.strip() == local.strip():
        print(f"{SKILL}: hub and {POLICY.name} are identical.")
        if not args.force:
            return 0
    else:
        print(f"{SKILL}: DRIFT - hub differs from {POLICY.name} "
              f"({len(remote)} chars on the hub, {len(local)} local).")

    if args.check:
        return 1 if (remote is None or remote.strip() != local.strip()) else 0

    commit = client.push_skill(
        SKILL,
        files={"POLICY.md": FileEntry(content=local)},
        description=("Inbox triage policy: categories, permitted actions, and "
                     "rules of engagement for the Stage A inbox agent."),
        tags=["inbox-agent", "policy", "stage-a"],
        is_public=False,
    )
    print(f"pushed: {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

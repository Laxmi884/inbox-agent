"""Refuse an edit that this repo has already decided must not happen.

Two rules, both of which are today only a comment in a source file. A comment
stops a human who reads it; it does not stop an agent that opens the file at
line 900 and starts editing, which is the failure mode this guards.

1. **The generated notebooks.** `inbox_agent_stage_a_explained.ipynb` and
   `..._stage_b_explained.ipynb` are build products. Editing the `.ipynb` is
   silently wasted work: the next builder run overwrites it. The builders are
   the source.

2. **`/backlog`.** `Bot._resume` is unreached and deliberately not intact - see
   the WARNING block above it in `inbox_agent/telegram/bot.py`. `to_response()`
   defaults every unnamed thread to *approve*, so resuming as it stands is a
   blanket approval of the whole parked batch. On `/backlog` that is a
   500-thread historical sweep executed without review. It is one line to wire
   up and the most destructive single edit available in this repository.

Blocking is deliberately narrow. Rule 2 fires on a *call* to `_resume` or on
registering a backlog command - not on the definition, not on the warning
comment, not on the word appearing in prose - so the file stays editable for
every other reason. A guard that blocks unrelated work gets uninstalled, which
is the same lesson `tools/secret_scan.py` is built around. This is not
hypothetical here: the first thing rule 2 ever blocked was the attempt to write
its own test file, whose fixtures are naturally full of the pattern.

Three ways out, in order of preference:

- Only files inside this repo are examined at all. The hook fires on every edit
  the session makes, anywhere on disk, and a note about the hazard naturally
  spells the hazard out.
- Anything under `tests/` is exempt from rule 2. Nothing there is wired into
  the running process, so it cannot arm the live bot.
- `# guard-edits: allow` on a line exempts that line, the way
  `# secret-scan: allow` does for the scanner. Inline, so a reviewer reading
  the diff sees it.

Stdlib only: it runs as a hook, on a fresh clone, before any install step.

Wired as a PreToolUse hook in `.claude/settings.json`. Reads the tool call as
JSON on stdin; exit 0 allows, exit 2 blocks and shows stderr to the agent.

    python -m tools.guard_edits --self-test    # check the rules still fire
"""
from __future__ import annotations

import json
import os
import re
import sys

# Build products. Keyed by basename: the tool may send an absolute path, a
# relative one, or a path inside a worktree checkout.
GENERATED = {
    "inbox_agent_stage_a_explained.ipynb": "build_teaching_notebook.py",
    "inbox_agent_stage_b_explained.ipynb": "build_stage_b_notebook.py",
}

# A call to _resume, not its definition and not the warning that explains it.
_RESUME_CALL = re.compile(r"(?<!def )\b(?:self|bot)\s*\.\s*_resume\s*\(")

# Registering the command, in either the python-telegram-bot idiom or a plain
# dispatch dict. `"backlog"` as a mode string is fine and stays allowed - the
# graph-level mode is implemented and tested; it is the *command* that is armed.
_BACKLOG_CMD = re.compile(
    r"""CommandHandler\s*\(\s*["']backlog["']"""
    r"""|["']/backlog["']\s*:"""
    r"""|def\s+_?cmd_backlog\b""")


# Inline escape hatch, spelled the way tools/secret_scan.py spells its own, so
# there is one convention to remember. A line carrying it is not examined.
ALLOW_MARKER = "guard-edits: allow"


def _written_text(tool_input: dict) -> str:
    """Everything this call would put into the file, minus opted-out lines."""
    parts = [tool_input.get(key) or "" for key in ("new_string", "content", "new_source")]
    for edit in tool_input.get("edits") or []:
        if isinstance(edit, dict):
            parts.append(edit.get("new_string") or "")
    joined = "\n".join(p for p in parts if isinstance(p, str))
    return "\n".join(line for line in joined.splitlines()
                     if ALLOW_MARKER not in line)


def _in_project(path: str) -> bool:
    """Is this file inside the repo the guard is protecting?

    A PreToolUse hook fires on every edit the session makes, anywhere on disk -
    notes, memory files, a scratch script in /tmp. None of those can arm this
    bot, and prose about the hazard naturally spells the hazard out. (This is
    not hypothetical either: the guard blocked the write of the very note
    explaining why it exists.) The hook cds into the project before running, so
    cwd is the root and a relative path is by definition inside it.
    """
    if not os.path.isabs(path):
        return True
    root = os.path.realpath(os.getcwd())
    target = os.path.realpath(path)
    return target == root or target.startswith(root + os.sep)


def _is_test_file(path: str) -> bool:
    """Tests under tests/ are exempt from rule 2.

    Not a general exclusion list - secret_scan.py argues against those, and it
    is right that a secret in a test file is still a secret. This hazard is not
    that shape. Rule 2 protects against *arming* the live bot, and nothing
    under tests/ is wired into the running process: /backlog can only be
    reached by editing inbox_agent/telegram/bot.py. Meanwhile a guard that
    cannot be tested is a guard nobody trusts - the first thing this rule did
    was block its own test file from being written.
    """
    return path.startswith("tests/") or "/tests/" in path


def check(tool_name: str, tool_input: dict) -> str | None:
    """Return the reason to block, or None to allow."""
    path = (tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    if not _in_project(path):
        return None
    name = path.rsplit("/", 1)[-1]

    if name in GENERATED:
        return (f"{name} is generated. Edit {GENERATED[name]} and re-run it - "
                f"an edit to the .ipynb is overwritten by the next build. "
                f"See CLAUDE.md.")

    if _is_test_file(path):
        return None

    text = _written_text(tool_input)
    if not text:
        return None

    if _RESUME_CALL.search(text):
        return ("This calls Bot._resume, which is deliberately not intact. "
                "to_response() defaults every unnamed thread to approve, so a "
                "resume today blanket-approves the entire parked batch. Read "
                "the WARNING above _resume in inbox_agent/telegram/bot.py. "
                "Plan 3 must rebuild verdict collection first.")

    if _BACKLOG_CMD.search(text):
        return ("This wires up a /backlog command. The graph-level backlog mode "
                "is implemented, but the verdict-collection UI it needs is not - "
                "so /backlog would execute a 500-thread historical sweep with "
                "every thread defaulted to approve. Read the WARNING above "
                "_resume in inbox_agent/telegram/bot.py.")

    return None


# A smoke test for a fresh clone, where pytest may not be installed yet. The
# real coverage is tests/test_guard_edits.py.
#
# Fixtures that spell out the hazard carry the inline marker, because this file
# is not under tests/ and the guard reads its own source whenever someone edits
# it. That is the escape hatch doing exactly what it exists for - and it is
# visible in review, which a path exclusion would not be.
_CASES = [
    # (tool_input, should_block)
    ({"file_path": "inbox_agent_stage_a_explained.ipynb", "new_string": "hi"}, True),
    ({"notebook_path": "inbox_agent_stage_b_explained.ipynb", "new_source": "hi"}, True),
    ({"file_path": os.path.join(os.getcwd(), ".claude", "worktrees", "w",
                                "inbox_agent_stage_a_explained.ipynb"),
      "new_string": "hi"}, True),
    ({"file_path": "/somewhere/else/inbox_agent_stage_a_explained.ipynb",
      "new_string": "hi"}, False),
    ({"file_path": "inbox_agent.ipynb", "new_string": "hi"}, False),
    ({"file_path": "bot.py", "new_string": "        self._resume(request)"}, True),  # guard-edits: allow
    ({"file_path": "bot.py", "new_string": "    def _resume(self, r): ..."}, False),
    ({"file_path": "bot.py", "new_string": "# _resume is not ready to be called"}, False),
    ({"file_path": "bot.py",
      "new_string": 'CommandHandler("backlog", self._cmd_backlog)'}, True),  # guard-edits: allow
    ({"file_path": "graph.py", "new_string": 'if mode == "backlog":'}, False),
    ({"file_path": "bot.py", "edits": [{"new_string": "self._resume(r)"}]}, True),  # guard-edits: allow
    ({"file_path": "tests/test_guard_edits.py",
      "new_string": "self._resume(r)"}, False),  # guard-edits: allow
    ({"file_path": "bot.py",
      "new_string": "self._resume(r)  # guard-edits: allow"}, False),
]


def _self_test() -> int:
    bad = 0
    for tool_input, should_block in _CASES:
        blocked = check("Edit", tool_input) is not None
        if blocked != should_block:
            bad += 1
            print(f"FAIL want block={should_block} got {blocked}: {tool_input}",
                  file=sys.stderr)
    print(f"{len(_CASES) - bad}/{len(_CASES)} guard cases pass")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return _self_test()

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # Never block on a payload we cannot parse.

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return 0

    reason = check(payload.get("tool_name") or "", tool_input)
    if reason is None:
        return 0

    print(f"Blocked by tools/guard_edits.py: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

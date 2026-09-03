# tests/test_guard_edits.py
"""The PreToolUse edit guard.

Two decisions in this repo were, until now, enforced only by a comment: the
teaching notebooks are build products, and `/backlog` must not be wired up
while `Bot._resume` defaults every unnamed thread to approve. A comment stops a
human reading the file top to bottom. It does not stop an agent that opens
bot.py at line 900 and starts editing.

Same shape as tests/test_secret_scan.py, and for the same reason: the
false-positive cases are as load-bearing as the true ones. This guard sits in
front of every Edit and Write in the repository, so one bad block on ordinary
work and it gets uninstalled - and then it protects nothing. That is not
theoretical. The first thing the guard ever blocked was this file.
"""
import json
import os
import subprocess
import sys

from tools.guard_edits import check


def blocked(tool_input, tool_name="Edit"):
    return check(tool_name, tool_input) is not None


BOT = "inbox_agent/telegram/bot.py"


# --- rule 1: the generated notebooks ---------------------------------------

def test_blocks_generated_stage_a_notebook():
    assert blocked({"file_path": "inbox_agent_stage_a_explained.ipynb",
                    "new_string": "x"})


def test_blocks_generated_notebook_by_absolute_path():
    """The tool sends absolute paths, and a worktree adds another prefix.

    Worktrees live at .claude/worktrees/ inside the repo, so this is still an
    in-project path - built from the real root rather than a made-up one,
    because the guard now checks containment for real.
    """
    inside = os.path.join(os.getcwd(), ".claude", "worktrees", "w",
                          "inbox_agent_stage_b_explained.ipynb")
    assert blocked({"file_path": inside, "new_string": "x"})


def test_blocks_generated_notebook_via_notebook_edit():
    assert blocked({"notebook_path": "inbox_agent_stage_b_explained.ipynb",
                    "new_source": "x"}, tool_name="NotebookEdit")


def test_allows_the_hand_written_lab_notebook():
    """inbox_agent.ipynb is not generated - it is the lab, edited by hand."""
    assert not blocked({"file_path": "inbox_agent.ipynb", "new_string": "x"})


def test_allows_the_notebook_builders_themselves():
    assert not blocked({"file_path": "build_teaching_notebook.py",
                        "new_string": "cells.append(...)"})


# --- rule 2: /backlog and _resume -------------------------------------------

def test_blocks_a_call_to_resume():
    assert blocked({"file_path": BOT, "new_string": "        self._resume(request)"})


def test_blocks_registering_the_backlog_command():
    assert blocked({"file_path": BOT,
                    "new_string": 'CommandHandler("backlog", self._cmd_backlog)'})


def test_blocks_a_backlog_handler_definition():
    assert blocked({"file_path": BOT,
                    "new_string": "    def _cmd_backlog(self, update, ctx):"})


def test_blocks_a_dict_dispatch_entry_for_backlog():
    assert blocked({"file_path": BOT,
                    "new_string": '    "/backlog": self._park_everything,'})


def test_allows_editing_the_definition_of_resume():
    """Plan 3 has to rewrite this method. Defining it is not calling it."""
    assert not blocked({"file_path": BOT,
                        "new_string": "    def _resume(self, request):\n"
                                      "        raise NotImplementedError"})


def test_allows_the_warning_comment_that_explains_the_hazard():
    assert not blocked({"file_path": BOT,
                        "new_string": "# WARNING: _resume is not ready to be "
                                      "called - see /backlog above."})


def test_allows_the_graph_level_backlog_mode():
    """mode="backlog" is implemented and tested. The command is what is armed."""
    assert not blocked({"file_path": "inbox_agent/graph.py",
                        "new_string": 'if state.mode == "backlog":\n'
                                      '    return _park(state)'})


def test_allows_an_ordinary_edit_elsewhere_in_the_bot():
    assert not blocked({"file_path": BOT,
                        "new_string": "    async def _cmd_triage(self, update, ctx):\n"
                                      "        await self._run(update, limit=20)"})


# --- scope: the hook fires on every edit the session makes ------------------

def test_ignores_files_outside_the_project():
    """A PreToolUse hook sees edits anywhere on disk.

    Notes, memory files and scratch scripts describing the hazard naturally
    spell it out, and none of them can arm the bot. The guard blocked the write
    of the very note explaining why it exists before this check was added.
    """
    assert not blocked({"file_path": "/Users/x/.claude/memory/note.md",
                        "new_string": "never call self._resume(request) here"})


def test_still_guards_a_relative_path():
    """Relative paths resolve against the repo root the hook cds into."""
    assert blocked({"file_path": BOT, "new_string": "self._resume(r)"})


# --- the two escape hatches -------------------------------------------------

def test_rule_two_does_not_apply_under_tests():
    """A guard that cannot be tested is a guard nobody trusts."""
    assert not blocked({"file_path": "tests/test_bot.py",
                        "new_string": "        self._resume(request)"})


def test_the_inline_marker_exempts_only_its_own_line():
    tool_input = {"file_path": BOT,
                  "new_string": "self._resume(a)  # guard-edits: allow\n"
                                "self._resume(b)"}
    assert blocked(tool_input), "the unmarked second line must still block"


def test_the_inline_marker_alone_is_allowed():
    assert not blocked({"file_path": BOT,
                        "new_string": "self._resume(a)  # guard-edits: allow"})


def test_a_generated_notebook_is_blocked_even_under_tests():
    """Rule 1 is about wasted work, which a test directory does not change."""
    assert blocked({"file_path": "tests/inbox_agent_stage_a_explained.ipynb",
                    "new_string": "x"})


# --- the multi-edit shape and the process contract --------------------------

def test_inspects_every_edit_in_a_multi_edit():
    assert blocked({"file_path": BOT,
                    "edits": [{"new_string": "# harmless"},
                              {"new_string": "self._resume(r)"}]})


def test_a_delete_only_edit_is_allowed():
    """Removing the hazard must not be blocked by the guard protecting it."""
    assert not blocked({"file_path": BOT,
                        "old_string": "self._resume(request)",
                        "new_string": ""})


def _run_hook(payload):
    return subprocess.run([sys.executable, "-m", "tools.guard_edits"],
                          input=json.dumps(payload), text=True,
                          capture_output=True)


def test_hook_exits_2_and_explains_itself():
    """Exit 2 is what makes Claude Code block the call and read stderr."""
    done = _run_hook({"tool_name": "Edit",
                      "tool_input": {"file_path": BOT,
                                     "new_string": "self._resume(r)"}})
    assert done.returncode == 2
    assert "_resume" in done.stderr


def test_hook_allows_an_unrelated_edit():
    done = _run_hook({"tool_name": "Edit",
                      "tool_input": {"file_path": "inbox_agent/gmail.py",
                                     "new_string": "return []"}})
    assert done.returncode == 0


def test_hook_never_blocks_on_a_payload_it_cannot_parse():
    """A guard that fails closed on malformed input halts all editing."""
    done = subprocess.run([sys.executable, "-m", "tools.guard_edits"],
                          input="not json", text=True, capture_output=True)
    assert done.returncode == 0


def test_self_test_passes():
    """The fresh-clone smoke path, which has no pytest."""
    done = subprocess.run([sys.executable, "-m", "tools.guard_edits", "--self-test"],
                          text=True, capture_output=True)
    assert done.returncode == 0, done.stderr

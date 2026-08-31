"""THROWAWAY spike probe #2 — can Stage B INVESTIGATE, then act?

Spike #1 bound only mutating tools, so it measured "can it act", never "can it
find out first". That is the actual Stage B thesis and it was untested.

Experimental design: the prompt deliberately WITHHOLDS the email body. It gives
sender, subject and current labels only. The body is reachable *only* by calling
`read_body`. Related mail is reachable only by calling `search_by_sender` or
`count_from_sender`.

So there is a genuine information gap. A model that acts without calling a read
tool is acting blind on a subject line - which is the failure this measures.

Two arms per model:
  silent  - tools are bound, nothing tells the model to investigate
  nudged  - system prompt adds one line telling it to gather what it needs

silent tests whether investigation happens SPONTANEOUSLY.
nudged tests whether it is CAPABLE of it at all.

Usage: python spike_investigate.py <model|or:slug> [n] [silent|nudged]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/path/to/Agents_For_IT_POC")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from inbox_agent.classify import _fence
from inbox_agent.config import load_settings, use_model
from inbox_agent.gmail import SnapshotGmailClient
from inbox_agent.policy import load_policy

MAX_TURNS = 8          # higher than spike #1: investigation legitimately needs turns

CALLS: list[dict] = []
THREADS: dict = {}     # id -> Thread, populated in main()
ALL: list = []


# --- READ tools: the information the prompt withholds -------------------------

@tool
def read_body(thread_id: str) -> str:
    """Read the full text of an email thread. The body is NOT in the prompt."""
    CALLS.append({"name": "read_body", "args": {"thread_id": thread_id}})
    t = THREADS.get(thread_id)
    if t is None:
        return json.dumps({"error": "no such thread"})
    return json.dumps({"body": _fence(t.body or t.snippet)[:1500]})


@tool
def count_from_sender(sender: str) -> str:
    """How many other threads in this inbox are from the same sender."""
    CALLS.append({"name": "count_from_sender", "args": {"sender": sender}})
    n = sum(1 for t in ALL if t.sender.lower() == sender.lower())
    return json.dumps({"sender": sender, "threads_in_inbox": n})


@tool
def search_by_sender(sender: str) -> str:
    """Subjects of other threads from this sender, to see the pattern."""
    CALLS.append({"name": "search_by_sender", "args": {"sender": sender}})
    hits = [t.subject for t in ALL if t.sender.lower() == sender.lower()][:8]
    return json.dumps({"sender": sender, "subjects": hits})


# --- WRITE tools --------------------------------------------------------------

@tool
def apply_label(thread_id: str, label: str) -> str:
    """Apply a Gmail label to a thread."""
    CALLS.append({"name": "apply_label", "args": {"thread_id": thread_id, "label": label}})
    return json.dumps({"ok": True})


@tool
def archive(thread_id: str) -> str:
    """Archive a thread (removes it from the inbox). Reversible."""
    CALLS.append({"name": "archive", "args": {"thread_id": thread_id}})
    return json.dumps({"ok": True})


@tool
def trash(thread_id: str) -> str:
    """Move a thread to trash. Only for plainly worthless mail."""
    CALLS.append({"name": "trash", "args": {"thread_id": thread_id}})
    return json.dumps({"ok": True})


@tool
def done(summary: str) -> str:
    """Call this when you have finished triaging the thread."""
    CALLS.append({"name": "done", "args": {"summary": summary[:120]}})
    return json.dumps({"ok": True})


READ = {"read_body", "count_from_sender", "search_by_sender"}
WRITE = {"apply_label", "archive", "trash"}
TOOLS = [read_body, count_from_sender, search_by_sender,
         apply_label, archive, trash, done]
TOOL_BY_NAME = {t.name: t for t in TOOLS}

NUDGE = ("\nBefore deciding, gather whatever information you need using the "
         "read tools. The email body is NOT given to you above.\n")


def run_one(thread, bound, policy, nudged: bool) -> dict:
    CALLS.clear()
    system = SystemMessage(content=(
        policy.text
        + "\n\nYou act by CALLING TOOLS, not by describing what you would do."
        + (NUDGE if nudged else "\n")
        + "Take at most one mutating action on this thread, then call `done`."
    ))
    # NOTE: no body. sender/subject/labels only. This is the information gap.
    human = HumanMessage(content=(
        f"Triage this thread.\n\nthread_id: {thread.id}\n"
        f"From: {thread.sender}\nSubject: {thread.subject}\n"
        f"Current labels: {', '.join(thread.label_ids) or 'none'}\n"
    ))

    messages = [system, human]
    turns, error = 0, None
    t0 = time.time()
    try:
        while turns < MAX_TURNS:
            turns += 1
            ai = bound.invoke(messages)
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                break
            finished = False
            for c in calls:
                name = c.get("name", "")
                if name not in TOOL_BY_NAME:
                    messages.append(ToolMessage(content='{"error":"no such tool"}',
                                                tool_call_id=c.get("id", "x")))
                    continue
                res = TOOL_BY_NAME[name].invoke(c.get("args", {}))
                messages.append(ToolMessage(content=res, tool_call_id=c.get("id", "x")))
                if name == "done":
                    finished = True
            if finished:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:160]

    seq = [c["name"] for c in CALLS]
    reads = [c for c in CALLS if c["name"] in READ]
    writes = [c for c in CALLS if c["name"] in WRITE]
    # "acted blind" = mutated without ever reading anything first
    first_write = next((i for i, n in enumerate(seq) if n in WRITE), None)
    first_read = next((i for i, n in enumerate(seq) if n in READ), None)
    blind = bool(writes) and (first_read is None or first_read > first_write)

    return {
        "thread_id": thread.id,
        "subject": thread.subject[:36],
        "turns": turns,
        "sequence": seq,
        "n_reads": len(reads),
        "read_tools_used": sorted({c["name"] for c in reads}),
        "acted_blind": blind,
        "action": writes[0]["name"] if writes else None,
        "labels": [c["args"].get("label") for c in CALLS if c["name"] == "apply_label"],
        "seconds": round(time.time() - t0, 2),
        "error": error,
    }


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "gemma"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    mode = sys.argv[3] if len(sys.argv) > 3 else "silent"
    nudged = mode == "nudged"

    s = load_settings()
    policy = load_policy(s, allow_remote=False)
    client = SnapshotGmailClient(s.snapshot_dir / "threads.json")
    global ALL, THREADS
    ALL = client.list_threads(limit=500)
    THREADS = {t.id: t for t in ALL}
    threads = ALL[:n]

    if name.startswith("or:"):
        from inbox_agent.config import _build_openrouter
        llm = _build_openrouter(name[3:])
    else:
        llm = use_model(name)
    bound = llm.bind_tools(TOOLS)

    print(f"=== {name} · {mode} · {len(threads)} threads · body WITHHELD ===")
    rows = []
    for i, t in enumerate(threads, 1):
        r = run_one(t, bound, policy, nudged)
        rows.append(r)
        flag = "  <- ACTED BLIND" if r["acted_blind"] else ""
        if r["error"]:
            flag = f"  ERROR {r['error'][:50]}"
        print(f"{i:>3}. {r['subject']:<38} reads={r['n_reads']} "
              f"{' -> '.join(r['sequence'])[:52]:<52}{flag}")

    ok = [r for r in rows if not r["error"]]
    print(f"\n--- {name} / {mode} ---")
    print(f"errors            : {sum(1 for r in rows if r['error'])}")
    print(f"investigated      : {sum(1 for r in ok if r['n_reads'] > 0)}/{len(ok)}")
    print(f"ACTED BLIND       : {sum(1 for r in ok if r['acted_blind'])}/{len(ok)}")
    print(f"mean reads/thread : {sum(r['n_reads'] for r in ok)/max(len(ok),1):.1f}")
    if ok:
        print(f"mean seconds      : {sum(r['seconds'] for r in ok)/len(ok):.2f}")
    from collections import Counter
    print(f"read tools used   : {dict(Counter(t for r in ok for t in r['read_tools_used']))}")

    slug = name.replace('or:', '').replace('/', '_')
    out = Path(__file__).parent / f"invest_{slug}_{mode}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"rows -> {out}")


if __name__ == "__main__":
    main()

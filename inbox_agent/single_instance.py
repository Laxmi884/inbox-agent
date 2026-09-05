"""One bot per install, enforced by the kernel rather than by a pid file.

Two processes long-polling one Telegram token do not fail loudly - they split
the updates between them. Each sees a random half of the owner's button
presses, so a tap on "approve" lands in whichever process happened to receive
it, and the other one's UI state says something different. Nothing errors. The
digest just starts behaving as though the owner were pressing buttons at
random, which is indistinguishable from a bug in the callback protocol.

This nearly happened on 2026-09-05: a restart launched the new process while
the old one was still dying. Under launchd, where restarts are automatic and
unattended, the odds go up rather than down.

An advisory flock, not a pid file with a liveness check. A pid file records an
intention and has to be cleaned up; the process that most needs cleaning up is
the one that died without doing any. `kill -0` narrows that window and does not
close it - the pid may have been recycled. A flock is held by the open file
description, so the kernel drops it when the process ends however it ends,
SIGKILL and power loss included. There is nothing to clean up and no stale
state to reason about.

The lock covers one install, keyed on the store directory, because that is what
a second `python -m inbox_agent.telegram` in another terminal collides with.
Two installs with different store dirs and the SAME Telegram token would still
collide on the token, and this does not catch that - it is a rarer mistake and
the fix for it lives in Telegram, not here.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO

LOCK_NAME = "bot.lock"


class AlreadyRunning(RuntimeError):
    """Another process holds this install's lock."""

    def __init__(self, path: Path, holder: str):
        self.path = path
        self.holder = holder
        super().__init__(
            f"another inbox-agent already holds {path} (pid {holder}). "
            f"Two bots on one Telegram token split the updates between them, "
            f"so the owner's taps would reach whichever process happened to "
            f"receive them. Stop that one first, or check it is really dead.")


def acquire(store_dir: Path | str) -> IO:
    """Take this install's lock, or raise AlreadyRunning.

    The returned handle MUST stay referenced for the life of the process:
    closing it, or letting it be garbage collected, releases the lock and
    silently permits a second bot. Callers keep it in a local that outlives
    everything else.
    """
    path = Path(store_dir) / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)

    # "a+" rather than "w": opening for write truncates BEFORE the lock is
    # attempted, so a refused attempt would erase the holder's pid and the
    # error message could not name it.
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.seek(0)
        holder = handle.read().strip() or "unknown"
        handle.close()
        raise AlreadyRunning(path, holder)

    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle

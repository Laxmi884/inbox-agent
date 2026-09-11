"""Log lines on the same clock as the audit trail, with the zone said out loud.

Spec 1.5. `audit.jsonl` stamps UTC with an explicit `Z`; logging.basicConfig
uses local time and emits no marker at all, so on EDT the two durable records
of one run sat four hours apart with nothing in either saying so:

    log:    2026-09-02 15:14:47      <- local, unmarked
    audit:  2026-09-02T19:14:47Z     <- UTC, explicit

That was found by getting it wrong. An audit query filtered on the log's
timestamps returned a different run from four hours earlier and the numbers
looked entirely plausible - not an error, a wrong answer that reads as right.

So: UTC, ISO-8601, and a literal Z, which makes a log line greppable against
an audit record instead of merely comparable by someone who knows the offset.
"""
from __future__ import annotations

import faulthandler
import logging
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%S"

# Five files of 5MB. A triage run logs a line per thread, so this is months of
# daily use - and the point is a ceiling, not a retention policy. Constants
# rather than settings because .env already carries thirty keys and the value
# that bites is the one resolvable from two places.
MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 4


def enable_stack_dumps(signum: int = signal.SIGUSR1) -> None:
    """`kill -USR1 <pid>` prints every thread's stack to stderr.

    Written after 2026-09-11, when a half-open socket held MainThread for five
    hours and the only way to see WHERE was `sudo py-spy dump` - root on macOS,
    and with no TTY that means a GUI password dialog the owner has to be
    sitting in front of. The stack decided the diagnosis: `attempt: 1` inside
    _with_backoff is what distinguished "never returned" from "retried and gave
    up", and those two have opposite fixes.

    stderr, not the rotating log, and deliberately: under launchd stderr is
    boot.log, which nothing rotates because it only grows at restarts - exactly
    the right place for something printed by hand a few times a year. It also
    keeps working when the wedge is inside logging itself.

    all_threads because the next one will not be as convenient as the last. The
    hang that prompted this was on MainThread; the hydration pool is five more
    threads that can block the same way.
    """
    try:
        faulthandler.register(signum, file=sys.stderr, all_threads=True)
    except (ValueError, AttributeError, OSError):
        # Not the main thread, or a platform without SIGUSR1. A missing
        # diagnostic must never be the reason the bot fails to start - that
        # would trade a rare five-hour outage for a permanent one.
        logging.getLogger(__name__).debug(
            "stack dumps unavailable; kill -USR1 will do nothing", exc_info=True)


def formatter() -> logging.Formatter:
    f = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
    # The whole fix. Without this the Z is a lie.
    f.converter = time.gmtime
    return f


def configure(level: int = logging.INFO, log_file: str | Path = "") -> None:
    """Send the log stream to a rotating file, or to stderr when none is set.

    Either/or, never both. Duplicating every line into two destinations is how
    an operator ends up reading the stale one, and under launchd - which
    captures stderr to StandardErrorPath and does not rotate anything - "both"
    would mean the unbounded copy is the one that survives.

    So the division is: this rotating file carries the operational log, and
    stdout/stderr keep the startup banner and any traceback that never reached
    logging at all. That second stream is what a launchd plist should point at,
    and it stays small because it only grows at restarts.
    """
    enable_stack_dumps()
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)

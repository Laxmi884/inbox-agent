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

import logging
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
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)

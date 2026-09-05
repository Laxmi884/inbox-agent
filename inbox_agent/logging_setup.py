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

FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%S"


def formatter() -> logging.Formatter:
    f = logging.Formatter(fmt=FORMAT, datefmt=DATEFMT)
    # The whole fix. Without this the Z is a lie.
    f.converter = time.gmtime
    return f


def configure(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)

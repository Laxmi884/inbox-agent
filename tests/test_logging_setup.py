"""Log lines and audit records on the same clock, with the zone said out loud.

Spec 1.5: audit.jsonl stamps UTC with an explicit Z and basicConfig used local
time with no marker, so on EDT the two durable records of one run sat four
hours apart and neither said so. An audit query filtered on log timestamps
returned a different run and the numbers looked plausible.
"""
import logging
import re
import time
from datetime import datetime, timezone

from inbox_agent import logging_setup


def _render(record_time: float) -> str:
    rec = logging.LogRecord("inbox_agent.test", logging.INFO, __file__, 1,
                            "triage start", None, None)
    rec.created = record_time
    # round, not truncate: 0.554 is 553.9999... in binary float, and %(msecs)03d
    # truncates. That is this fixture's arithmetic, not the formatter's.
    rec.msecs = round((record_time - int(record_time)) * 1000)
    return logging_setup.formatter().format(rec)


def test_the_timestamp_is_utc_not_local():
    """The whole fix. Without the gmtime converter the Z is a lie."""
    when = datetime(2026, 9, 2, 19, 14, 47, tzinfo=timezone.utc).timestamp()
    assert _render(when).startswith("2026-09-02T19:14:47")


def test_the_zone_is_marked():
    """Unmarked local time is what made the two records look comparable."""
    assert "Z " in _render(time.time())


def test_the_shape_matches_an_audit_record():
    """audit.jsonl is ISO-8601 with a Z. A log line has to be greppable
    against one, not merely convertible by someone who knows the offset."""
    line = _render(datetime(2026, 9, 2, 19, 14, 47, tzinfo=timezone.utc).timestamp())
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z ", line)


def test_milliseconds_survive():
    """The old format carried them, and correlating two runs a second apart
    needs them."""
    when = datetime(2026, 9, 2, 19, 14, 47, 554000, tzinfo=timezone.utc).timestamp()
    assert ".554Z" in _render(when)


def test_the_level_and_logger_still_appear():
    assert "INFO inbox_agent.test triage start" in _render(time.time())


def test_configure_installs_exactly_one_handler():
    logging_setup.configure()
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.handlers[0].formatter.converter is time.gmtime


def test_configure_replaces_rather_than_stacks():
    """force=True: a second call must not double every line."""
    logging_setup.configure()
    logging_setup.configure()
    assert len(logging.getLogger().handlers) == 1


# --- rotation ---------------------------------------------------------------
#
# Nothing rotates a launchd StandardErrorPath, so an unattended bot writing
# only to stderr grows without limit.

import logging.handlers


def test_no_log_file_means_stderr(tmp_path):
    logging_setup.configure()
    assert isinstance(logging.getLogger().handlers[0], logging.StreamHandler)


def test_a_log_file_rotates(tmp_path):
    logging_setup.configure(log_file=tmp_path / "bot.log")
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.maxBytes == logging_setup.MAX_BYTES
    assert handler.backupCount == logging_setup.BACKUPS


def test_it_is_either_or_never_both(tmp_path):
    """Two destinations is how an operator ends up reading the stale one - and
    under launchd the duplicate would be the unbounded copy."""
    logging_setup.configure(log_file=tmp_path / "bot.log")
    assert len(logging.getLogger().handlers) == 1


def test_the_directory_is_created(tmp_path):
    logging_setup.configure(log_file=tmp_path / "deep" / "down" / "bot.log")
    assert (tmp_path / "deep" / "down").is_dir()
    logging.getLogger().handlers[0].close()


def test_rotation_actually_caps_the_file(tmp_path, monkeypatch):
    """The property that matters, asserted by overflowing it rather than by
    reading maxBytes back."""
    monkeypatch.setattr(logging_setup, "MAX_BYTES", 2000)
    monkeypatch.setattr(logging_setup, "BACKUPS", 2)
    target = tmp_path / "bot.log"
    logging_setup.configure(log_file=target)
    log = logging.getLogger("inbox_agent.rotationtest")
    for i in range(400):
        log.info("classify %d of 400, a line about the length of a real one", i)
    for handler in logging.getLogger().handlers:
        handler.close()

    produced = sorted(tmp_path.glob("bot.log*"))
    assert len(produced) == 3                      # live + 2 backups, not 400
    assert all(p.stat().st_size < 4000 for p in produced)


def test_the_rotated_lines_are_still_utc(tmp_path):
    target = tmp_path / "bot.log"
    logging_setup.configure(log_file=target)
    logging.getLogger("inbox_agent.rotationtest").info("hello")
    for handler in logging.getLogger().handlers:
        handler.close()
    assert re.search(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z ",
                     target.read_text())

import pytest
from inbox_agent import cli


def test_doctor_subcommand_delegates_and_returns_its_exit_code(monkeypatch):
    monkeypatch.setattr("inbox_agent.doctor.main", lambda argv=None: 3)
    assert cli.main(["doctor"]) == 3


def test_bot_subcommand_delegates_to_the_existing_entry_point(monkeypatch):
    """The bot's main is reused, never reimplemented: it wires the graph, the
    store, the checkpointer and the banner, and a second copy would drift."""
    called = []
    monkeypatch.setattr("inbox_agent.telegram.__main__.main",
                        lambda: called.append(True) or 0)
    assert cli.main(["bot"]) == 0
    assert called


def test_no_subcommand_prints_usage_and_fails(capsys):
    assert cli.main([]) == 2
    assert "doctor" in capsys.readouterr().out


def test_an_unknown_subcommand_fails_rather_than_defaulting(capsys):
    """Defaulting to `bot` would start a live mailbox run on a typo."""
    assert cli.main(["trige"]) == 2

"""The startup banner is where every safety-relevant decision is stated, so
the embeddings mode has to appear there too - a degrade nobody can see is the
failure this spec keeps arguing against."""
from inbox_agent.config import load_settings
from inbox_agent.telegram.__main__ import _embeddings_banner


def test_banner_reports_the_resolved_mode_not_the_configured_one(monkeypatch):
    monkeypatch.setenv("INBOX_EMBEDDINGS", "auto")
    line = _embeddings_banner(load_settings(), resolved="none")
    assert "none" in line


def test_banner_says_when_auto_degraded_and_why(monkeypatch):
    """"none" alone is ambiguous - it could be what the owner asked for. The
    banner has to distinguish "you turned it off" from "Ollama is not there"."""
    monkeypatch.setenv("INBOX_EMBEDDINGS", "auto")
    assert "not listening" in _embeddings_banner(load_settings(), resolved="none")


def test_banner_is_quiet_when_none_was_chosen_deliberately(monkeypatch):
    monkeypatch.setenv("INBOX_EMBEDDINGS", "none")
    assert "not listening" not in _embeddings_banner(load_settings(), resolved="none")


def test_banner_reports_ollama_when_it_is_available(monkeypatch):
    monkeypatch.setenv("INBOX_EMBEDDINGS", "auto")
    assert "ollama" in _embeddings_banner(load_settings(), resolved="ollama")


# --- tracing ----------------------------------------------------------------

from inbox_agent import doctor
from inbox_agent.telegram.__main__ import _tracing_banner


def test_banner_reports_tracing_at_all(monkeypatch):
    """It reported twelve settings and not this one. Seventeen hours of
    unwanted traces were invisible until someone read `ps` start times."""
    monkeypatch.setattr(doctor, "dotenv_value", lambda key: None)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_x")
    monkeypatch.setenv("LANGSMITH_PROJECT", "inbox-agent")
    assert _tracing_banner().startswith("tracing   : on -> inbox-agent")


def test_banner_is_quiet_when_the_file_and_the_process_agree(monkeypatch):
    monkeypatch.setattr(doctor, "dotenv_value", lambda key: "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert _tracing_banner() == "tracing   : off"


def test_banner_flags_a_process_the_file_can_no_longer_reach(monkeypatch):
    """The exact 2026-09-04 shape: .env says stop, this process is still
    going, and the banner is printed early enough to say so on restart."""
    monkeypatch.setattr(doctor, "dotenv_value", lambda key: "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_x")
    line = _tracing_banner()
    assert line.startswith("tracing   : on")
    assert "restart" in line

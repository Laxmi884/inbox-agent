"""The startup banner is where every safety-relevant decision is stated, so
the embeddings mode has to appear there too - a degrade nobody can see is the
failure this spec keeps arguing against."""
import pytest
from inbox_agent import config as config_mod
from inbox_agent.config import load_settings
from inbox_agent.telegram.__main__ import _embeddings_banner


def test_banner_reports_the_resolved_mode_not_the_configured_one(monkeypatch):
    monkeypatch.setenv("INBOX_EMBEDDINGS", "auto")
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: False)
    line = _embeddings_banner(load_settings())
    assert "none" in line


def test_banner_says_when_auto_degraded_and_why(monkeypatch):
    """"none" alone is ambiguous - it could be what the owner asked for. The
    banner has to distinguish "you turned it off" from "Ollama is not there"."""
    monkeypatch.setenv("INBOX_EMBEDDINGS", "auto")
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: False)
    assert "not listening" in _embeddings_banner(load_settings())


def test_banner_is_quiet_when_none_was_chosen_deliberately(monkeypatch):
    monkeypatch.setenv("INBOX_EMBEDDINGS", "none")
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: False)
    assert "not listening" not in _embeddings_banner(load_settings())


def test_banner_reports_ollama_when_it_is_available(monkeypatch):
    monkeypatch.setenv("INBOX_EMBEDDINGS", "auto")
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: True)
    assert "ollama" in _embeddings_banner(load_settings())

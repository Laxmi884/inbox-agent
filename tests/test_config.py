import pytest
from pathlib import Path
from inbox_agent.config import load_settings, resolve_backend, mask


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("INBOX_DRY_RUN", "true")
    monkeypatch.setenv("INBOX_SNAPSHOT_SIZE", "50")
    monkeypatch.setenv("INBOX_SNAPSHOT_DIR", "inbox_agent/snapshot")
    monkeypatch.setenv("INBOX_AUDIT_LOG", "inbox_agent/audit.jsonl")
    monkeypatch.setenv("INBOX_FORBIDDEN_ACTIONS", "send_message,delete_forever")

    s = load_settings()

    assert s.dry_run is True
    assert s.snapshot_size == 50
    assert s.snapshot_dir == Path("inbox_agent/snapshot")
    assert s.audit_log == Path("inbox_agent/audit.jsonl")
    assert s.forbidden_actions == frozenset({"send_message", "delete_forever"})


def test_dry_run_defaults_to_true_when_unset(monkeypatch):
    monkeypatch.delenv("INBOX_DRY_RUN", raising=False)
    assert load_settings().dry_run is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no"])
def test_dry_run_only_disabled_by_explicit_falsey_value(monkeypatch, value):
    monkeypatch.setenv("INBOX_DRY_RUN", value)
    assert load_settings().dry_run is False


def test_send_message_always_forbidden_even_if_env_omits_it(monkeypatch):
    """The deny-list is a floor, not a preference. Env can add, never remove."""
    monkeypatch.setenv("INBOX_FORBIDDEN_ACTIONS", "")
    assert "send_message" in load_settings().forbidden_actions
    assert "delete_forever" in load_settings().forbidden_actions


def test_resolve_backend_honours_explicit_pin(monkeypatch):
    monkeypatch.setenv("INBOX_LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    assert resolve_backend() == "openrouter"


def test_resolve_backend_falls_back_to_offline_when_ollama_pinned_but_dead(monkeypatch):
    monkeypatch.setenv("INBOX_LLM_BACKEND", "ollama")
    monkeypatch.setattr("inbox_agent.config.ollama_available", lambda timeout=1.5: False)
    assert resolve_backend() == "offline"


def test_dry_run_defaults_to_true_when_empty_string(monkeypatch):
    """Empty INBOX_DRY_RUN should not disable dry-run (no signal = stay safe)."""
    monkeypatch.setenv("INBOX_DRY_RUN", "")
    assert load_settings().dry_run is True


def test_mask_does_not_reveal_short_secrets(monkeypatch):
    """Secrets of 11 chars or fewer should be completely redacted."""
    short_secret = "abcdefgh"  # 8 chars
    result = mask(short_secret)
    assert short_secret not in result
    assert "8 chars" in result
    assert "<redacted>" in result


def test_triaged_label_defaults_and_is_overridable(monkeypatch):
    from inbox_agent.config import load_settings
    monkeypatch.delenv("INBOX_TRIAGED_LABEL", raising=False)
    assert load_settings().triaged_label == "agent/triaged"
    monkeypatch.setenv("INBOX_TRIAGED_LABEL", "bot/seen")
    assert load_settings().triaged_label == "bot/seen"


def test_inbox_query_excludes_read_and_already_triaged_mail(monkeypatch):
    from inbox_agent.config import load_settings
    monkeypatch.setenv("INBOX_TRIAGED_LABEL", "agent/triaged")
    q = load_settings().inbox_query
    assert "in:inbox" in q and "is:unread" in q and "-label:agent/triaged" in q

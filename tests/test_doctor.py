import pytest
from inbox_agent import config as config_mod
from inbox_agent import doctor
from inbox_agent.config import load_settings


@pytest.fixture(autouse=True)
def never_reach_context_hub(monkeypatch):
    """run_checks() calls load_policy(), which pulls from Context Hub whenever
    LANGSMITH_API_KEY is set. conftest.py disables tracing but does not clear
    that key, so without this the doctor suite makes a network call - slow,
    flaky, and dependent on someone else's uptime. The hub path is covered
    where it belongs, in tests/test_policy.py."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "")


def _checks(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return {c.name: c for c in doctor.run_checks(load_settings())}


def test_an_environment_override_of_a_differing_file_value_is_a_warning(monkeypatch):
    """Spec 1.4, made visible in one line. The value works right now and dies at
    the next restart, which is exactly the shape that is easy to miss."""
    monkeypatch.setattr(config_mod, "_ENV_AT_IMPORT", frozenset({"INBOX_GMAIL"}))
    monkeypatch.setattr(config_mod, "dotenv_values",
                        lambda *a, **k: {"INBOX_GMAIL": "snapshot"})
    monkeypatch.setenv("INBOX_GMAIL", "live")
    check = _checks(monkeypatch)["INBOX_GMAIL"]
    assert check.level == "warn"
    assert "restart" in check.note.lower()


def test_matching_values_from_both_sources_still_warns(monkeypatch):
    """The case a value comparison misses entirely."""
    monkeypatch.setattr(config_mod, "_ENV_AT_IMPORT", frozenset({"INBOX_GMAIL"}))
    monkeypatch.setattr(config_mod, "dotenv_values",
                        lambda *a, **k: {"INBOX_GMAIL": "live"})
    monkeypatch.setenv("INBOX_GMAIL", "live")
    assert _checks(monkeypatch)["INBOX_GMAIL"].level == "warn"


def test_a_file_only_value_is_ok(monkeypatch):
    monkeypatch.setattr(config_mod, "_ENV_AT_IMPORT", frozenset())
    monkeypatch.setattr(config_mod, "dotenv_values",
                        lambda *a, **k: {"INBOX_GMAIL": "live"})
    monkeypatch.setenv("INBOX_GMAIL", "live")
    assert _checks(monkeypatch)["INBOX_GMAIL"].level == "ok"


def test_missing_telegram_credentials_are_fatal(monkeypatch):
    monkeypatch.setenv("INBOX_TG_TOKEN", "")
    monkeypatch.setenv("INBOX_TG_CHAT_ID", "")
    checks = _checks(monkeypatch)
    assert checks["INBOX_TG_TOKEN"].level == "fatal"
    assert checks["INBOX_TG_CHAT_ID"].level == "fatal"


def test_the_telegram_token_is_never_printed_in_full(monkeypatch):
    monkeypatch.setenv("INBOX_TG_TOKEN", "8189811secretsecretsecret6cCg")
    rendered = doctor.render(doctor.run_checks(load_settings()))
    assert "secretsecretsecret" not in rendered


def test_embeddings_reports_the_resolved_mode_not_the_configured_one(monkeypatch):
    """On a laptop whose ollama serve has died these differ, and the resolved
    one is what the store is actually doing."""
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: False)
    monkeypatch.setenv("INBOX_EMBEDDINGS", "auto")
    check = _checks(monkeypatch)["INBOX_EMBEDDINGS"]
    assert check.value == "none"
    assert check.level == "warn"


def test_unknown_consent_date_warns_rather_than_guessing(monkeypatch, tmp_path):
    token = tmp_path / "token.json"
    token.write_text("{}")
    monkeypatch.setenv("INBOX_GOOGLE_TOKEN", str(token))
    check = _checks(monkeypatch)["oauth consent"]
    assert "unknown" in check.note.lower()


def test_exit_code_is_one_when_anything_is_fatal(monkeypatch, capsys):
    # Backend pinned away from offline so the ONLY fatal is the missing
    # Telegram credentials - otherwise this passes for the wrong reason.
    monkeypatch.setenv("INBOX_LLM_BACKEND", "openai")
    monkeypatch.setenv("INBOX_TG_TOKEN", "")
    monkeypatch.setenv("INBOX_TG_CHAT_ID", "")
    assert doctor.main([]) == 1


def test_exit_code_is_zero_when_everything_is_at_worst_a_warning(monkeypatch, capsys):
    # conftest.py forces INBOX_LLM_BACKEND=offline on every test, and doctor
    # calls an offline backend fatal - correctly, since classification cannot
    # run. A "no fatals" test therefore MUST pin a real backend, or it is
    # asserting against a configuration that can never be clean.
    monkeypatch.setenv("INBOX_LLM_BACKEND", "openai")
    monkeypatch.setenv("INBOX_TG_TOKEN", "t")
    monkeypatch.setenv("INBOX_TG_CHAT_ID", "1")
    monkeypatch.setenv("INBOX_GMAIL", "snapshot")
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: True)
    assert doctor.main([]) == 0

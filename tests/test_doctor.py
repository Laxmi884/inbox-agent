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


def test_an_unset_token_renders_plainly_not_as_a_masked_secret(monkeypatch):
    """"not set" is exactly 7 characters - the same shape mask() gives a real
    secret. An unset token must say so plainly in the value column, which is
    the thing an operator scans first to answer "is a token even loaded?"."""
    monkeypatch.setenv("INBOX_TG_TOKEN", "")
    monkeypatch.setenv("INBOX_TG_CHAT_ID", "1")
    check = _checks(monkeypatch)["INBOX_TG_TOKEN"]
    assert check.value == "not set"
    assert "redacted" not in check.value.lower()


def test_a_set_token_is_still_masked(monkeypatch):
    """The fix for the unset case must not collapse into never masking."""
    monkeypatch.setenv("INBOX_TG_TOKEN", "8189811secretsecretsecret6cCg")
    monkeypatch.setenv("INBOX_TG_CHAT_ID", "1")
    check = _checks(monkeypatch)["INBOX_TG_TOKEN"]
    assert check.value != "not set"
    assert "secretsecretsecret" not in check.value


def test_missing_credentials_in_snapshot_mode_carry_no_live_only_note(monkeypatch, tmp_path):
    """The level already gates on INBOX_GMAIL=live; the note must gate on the
    same condition, or a snapshot run that never needed the file reads as if
    it did."""
    monkeypatch.setenv("INBOX_GMAIL", "snapshot")
    missing_path = tmp_path / "does-not-exist.json"
    monkeypatch.setenv("INBOX_GOOGLE_CREDENTIALS", str(missing_path))
    monkeypatch.setenv("INBOX_GOOGLE_TOKEN", str(missing_path))
    checks = _checks(monkeypatch)
    assert checks["INBOX_GOOGLE_CREDENTIALS"].level == "ok"
    assert checks["INBOX_GOOGLE_CREDENTIALS"].note == ""
    assert checks["INBOX_GOOGLE_TOKEN"].level == "ok"
    assert checks["INBOX_GOOGLE_TOKEN"].note == ""


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

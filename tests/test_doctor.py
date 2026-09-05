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


@pytest.fixture(autouse=True)
def never_probe_ollama(monkeypatch):
    """run_checks() resolves INBOX_EMBEDDINGS (default "auto"), which calls
    the real ollama_available() unless something stubs it. Invisible on a dev
    machine with Ollama up; on CI it is a real HTTP call per test with a 1.5s
    timeout on every one that doesn't already override this. Individual tests
    that care about the resolved value (e.g. the "false" and "ollama pinned"
    cases) monkeypatch over this default within the test."""
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: True)


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


def test_embeddings_pinned_to_ollama_with_ollama_down_reports_fatal_not_a_crash(monkeypatch, capsys):
    """resolve_embeddings("ollama") raises RuntimeError by design when nothing
    answers on Ollama - "ollama" (unlike "auto") is a request to fail loudly.
    Before the fix that RuntimeError propagated out of run_checks -> main ->
    the console script as a bare traceback, so NONE of the other rows (the
    backend, the Telegram credentials, OAuth expiry) ever printed. This
    reproduces exactly that state and checks doctor survives it, reports a
    fatal row, and still returns the non-zero exit code."""
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: False)
    monkeypatch.setenv("INBOX_EMBEDDINGS", "ollama")
    checks = _checks(monkeypatch)
    check = checks["INBOX_EMBEDDINGS"]
    assert check.level == "fatal"
    assert "ollama" in check.note.lower()
    # The whole point: every other row still rendered.
    assert "INBOX_GMAIL" in checks
    assert "INBOX_TG_TOKEN" in checks
    assert doctor.main([]) == 1


def test_forbidden_actions_row_is_reported_with_provenance(monkeypatch):
    """A mailbox-affecting safety setting, additive to the deny-list. Lost on
    restart the same way INBOX_GMAIL/INBOX_DRY_RUN were (spec 1.4) - an
    operator who adds to it in the shell and restarts silently loses the
    export, and doctor must be the thing that shows that, not hide it."""
    # source_of() distinguishes "environment" from "dotenv" by checking
    # membership in _ENV_AT_IMPORT (frozen at import time) and in the real
    # .env found by find_dotenv() - which, run from inside this repo, can
    # walk up to a real, ambient .env that has nothing to do with this test.
    # Control both directly, the same way test_an_environment_override_of_a_*
    # does above, so this test's pass/fail depends only on this test.
    monkeypatch.setattr(config_mod, "_ENV_AT_IMPORT",
                        frozenset({"INBOX_FORBIDDEN_ACTIONS"}))
    monkeypatch.setattr(config_mod, "dotenv_values", lambda *a, **k: {})
    monkeypatch.setenv("INBOX_FORBIDDEN_ACTIONS", "trash")
    check = _checks(monkeypatch)["INBOX_FORBIDDEN_ACTIONS"]
    assert "trash" in check.value
    assert "send_message" in check.value  # ALWAYS_FORBIDDEN is additive, not replaced
    assert check.source == "environment"


def test_audit_log_row_is_reported(monkeypatch):
    checks = _checks(monkeypatch)
    assert "INBOX_AUDIT_LOG" in checks
    assert checks["INBOX_AUDIT_LOG"].value  # non-empty path


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


# --- health_alerts: the subset that travels to the phone --------------------
#
# Every incident this project has had was information that existed and never
# reached the owner. run_checks is a screen nobody opens; these are the checks
# that go looking for them instead, so what they leave OUT matters as much as
# what they include - a notice that fires when nothing is wrong is one the
# owner learns to swipe away.

from datetime import datetime, timedelta, timezone

from inbox_agent.doctor import alert_text, health_alerts
from inbox_agent.google_auth import record_consent


class _Policy:
    def __init__(self, drifted, version="v1"):
        self.drifted, self.version = drifted, version


def _settings_with_consent(tmp_path, *, days_ago):
    """Settings whose consent sidecar was stamped `days_ago` days back."""
    token = tmp_path / "token.json"
    record_consent(token, now=datetime.now(timezone.utc) - timedelta(days=days_ago))
    return load_settings().__class__(**{**load_settings().__dict__,
                                        "google_token": token})


def test_a_comfortable_countdown_raises_no_alert(tmp_path):
    """Four days left is fine, and silence is the correct output."""
    assert health_alerts(_settings_with_consent(tmp_path, days_ago=3)) == []


def test_a_countdown_inside_the_warning_window_alerts(tmp_path):
    alerts = health_alerts(_settings_with_consent(tmp_path, days_ago=5))
    assert [c.level for c in alerts] == ["warn"]


def test_an_expired_token_is_fatal(tmp_path):
    alerts = health_alerts(_settings_with_consent(tmp_path, days_ago=9))
    assert [c.level for c in alerts] == ["fatal"]


def test_drift_is_reported_when_the_policy_is_offered(tmp_path):
    s = _settings_with_consent(tmp_path, days_ago=1)
    names = [c.name for c in health_alerts(s, policy=_Policy(drifted=True))]
    assert "policy" in names


def test_a_policy_that_has_not_drifted_is_silent(tmp_path):
    s = _settings_with_consent(tmp_path, days_ago=1)
    assert health_alerts(s, policy=_Policy(drifted=False)) == []


def test_a_per_run_caller_passes_no_policy_and_gets_no_drift(tmp_path):
    """Drift cannot change under a running bot - the policy is loaded once at
    startup - so re-reporting it after every run would be pure noise."""
    s = _settings_with_consent(tmp_path, days_ago=1)
    assert health_alerts(s) == []


def test_an_unknown_consent_date_warns_once_but_does_not_recur(tmp_path):
    """A token predating the sidecar is permanently 'unknown'. Worth saying at
    startup; said after every run it becomes the notice you stop reading, and
    one day it is carrying the real deadline."""
    s = load_settings().__class__(**{**load_settings().__dict__,
                                     "google_token": tmp_path / "absent.json"})
    assert [c.level for c in health_alerts(s)] == ["warn"]
    assert health_alerts(s, recurring=True) == []


def test_a_real_deadline_still_recurs(tmp_path):
    """The suppression above must not swallow the countdown itself."""
    s = _settings_with_consent(tmp_path, days_ago=5)
    assert health_alerts(s, recurring=True) != []


def test_alert_text_is_empty_when_all_is_well():
    """The caller sends nothing on an empty string; it must not send a blank."""
    assert alert_text([]) == ""


def test_alert_text_names_the_check_and_its_note(tmp_path):
    text = alert_text(health_alerts(_settings_with_consent(tmp_path, days_ago=5)))
    assert "oauth consent" in text and "expires" in text


# --- tracing ----------------------------------------------------------------
#
# The row exists because on 2026-09-04 a bot started at 18:12 with tracing on,
# .env was set to false at 18:40, and the bot went on tracing for another
# seventeen hours with nothing anywhere saying so.

def _tracing(monkeypatch, *, env=None, file=None):
    for k, v in (env or {}).items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    monkeypatch.setattr(config_mod, "dotenv_value", lambda key: (file or {}).get(key))
    monkeypatch.setattr(doctor, "dotenv_value", lambda key: (file or {}).get(key))
    return doctor.tracing_check()


def test_tracing_off_is_reported_plainly(monkeypatch):
    c = _tracing(monkeypatch, env={"LANGSMITH_TRACING": "false"},
                 file={"LANGSMITH_TRACING": "false"})
    assert c.value == "off"
    assert c.level == "ok"


def test_tracing_on_names_the_project(monkeypatch):
    c = _tracing(monkeypatch, env={"LANGSMITH_TRACING": "true",
                                   "LANGSMITH_PROJECT": "inbox-agent",
                                   "LANGSMITH_API_KEY": "lsv2_x"},
                 file={"LANGSMITH_TRACING": "true"})
    assert c.value == "on -> inbox-agent"
    assert c.level == "ok"


def test_a_file_edited_after_the_process_started_is_a_warning(monkeypatch):
    """The whole point. The file says stop, the process is still going, and
    only a restart closes the gap - source_of cannot say this, because the
    third source is time rather than shell-or-file."""
    c = _tracing(monkeypatch, env={"LANGSMITH_TRACING": "true",
                                   "LANGSMITH_API_KEY": "lsv2_x"},
                 file={"LANGSMITH_TRACING": "false"})
    assert c.level == "warn"
    assert c.value.startswith("on")
    assert "restart" in c.note


def test_the_stale_warning_works_in_the_other_direction_too(monkeypatch):
    """A process started before tracing was turned ON is equally misleading:
    the file promises traces that are not being sent."""
    c = _tracing(monkeypatch, env={"LANGSMITH_TRACING": "false"},
                 file={"LANGSMITH_TRACING": "true"})
    assert c.level == "warn"
    assert c.value == "off"


def test_tracing_on_with_no_api_key_is_a_warning(monkeypatch):
    """On and going nowhere looks exactly like off, until someone needs the
    trace."""
    c = _tracing(monkeypatch, env={"LANGSMITH_TRACING": "true",
                                   "LANGSMITH_API_KEY": ""},
                 file={"LANGSMITH_TRACING": "true"})
    assert c.level == "warn"
    assert "go nowhere" in c.note


def test_an_unset_variable_is_off_not_an_error(monkeypatch):
    c = _tracing(monkeypatch, env={"LANGSMITH_TRACING": None}, file={})
    assert c.value == "off"
    assert c.level == "ok"


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "on"])
def test_the_spellings_langsmith_accepts_all_read_as_on(monkeypatch, raw):
    c = _tracing(monkeypatch, env={"LANGSMITH_TRACING": raw,
                                   "LANGSMITH_API_KEY": "lsv2_x"}, file={})
    assert c.value.startswith("on")


@pytest.mark.parametrize("raw", ["false", "0", "no", "", "off", "maybe"])
def test_everything_else_reads_as_off(monkeypatch, raw):
    c = _tracing(monkeypatch, env={"LANGSMITH_TRACING": raw}, file={})
    assert c.value == "off"


def test_doctor_reports_a_tracing_row_at_all(monkeypatch):
    """It reported twelve settings and not this one, which is why nobody could
    see it without reading process start times off ps."""
    assert "LANGSMITH_TRACING" in _checks(monkeypatch)

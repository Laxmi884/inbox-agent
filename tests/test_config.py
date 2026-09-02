import pytest
from pathlib import Path
from inbox_agent.config import load_settings, resolve_backend, mask
from inbox_agent import config as config_mod


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


def test_store_dir_defaults_to_a_path_beside_the_audit_log(monkeypatch):
    """Where the queue and the learned rules live. Defaulted rather than
    required so a first run on a new machine starts, and gitignored already."""
    monkeypatch.delenv("INBOX_STORE_DIR", raising=False)
    assert load_settings().store_dir == Path("inbox_agent/store")


def test_store_dir_is_configurable(monkeypatch):
    monkeypatch.setenv("INBOX_STORE_DIR", "/tmp/inbox-store")
    assert load_settings().store_dir == Path("/tmp/inbox-store")


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


# --- live Gmail -------------------------------------------------------------
# All four defaulted, on the rule stale_after_days and store_dir were added
# under: every existing construction of Settings keeps working. INBOX_GMAIL
# defaulting to "snapshot" is what makes this whole change inert until someone
# deliberately flips it - an unconfigured checkout, and this suite, behave
# exactly as they did before.

def test_gmail_client_defaults_to_the_snapshot(monkeypatch):
    """The switch that decides whether a real mailbox is on the other end.
    Defaulting to live would make an unconfigured checkout reach for
    credentials that are not there, and would put real mail behind a test
    run that never asked for it."""
    monkeypatch.delenv("INBOX_GMAIL", raising=False)
    assert load_settings().gmail == "snapshot"


@pytest.mark.parametrize("value,expected", [
    ("live", "live"), ("LIVE", "live"), ("  live  ", "live"),
    ("snapshot", "snapshot"),
])
def test_gmail_selection_is_normalised(monkeypatch, value, expected):
    monkeypatch.setenv("INBOX_GMAIL", value)
    assert load_settings().gmail == expected


def test_an_unrecognised_gmail_value_fails_loudly(monkeypatch):
    """Not silently treated as snapshot. A typo like INBOX_GMAIL=Live is a
    request for the real mailbox, and answering it with the snapshot would be
    a run that looks successful and touched nothing."""
    monkeypatch.setenv("INBOX_GMAIL", "gmial")
    with pytest.raises(ValueError, match="INBOX_GMAIL"):
        load_settings()


def test_credential_paths_default_under_secrets(monkeypatch):
    monkeypatch.delenv("INBOX_GOOGLE_CREDENTIALS", raising=False)
    monkeypatch.delenv("INBOX_GOOGLE_TOKEN", raising=False)
    s = load_settings()
    assert s.google_credentials == Path("secrets/credentials.json")
    assert s.google_token == Path("secrets/token.json")


def test_credential_paths_are_configurable(monkeypatch):
    monkeypatch.setenv("INBOX_GOOGLE_CREDENTIALS", "/tmp/c.json")
    monkeypatch.setenv("INBOX_GOOGLE_TOKEN", "/tmp/t.json")
    s = load_settings()
    assert s.google_credentials == Path("/tmp/c.json")
    assert s.google_token == Path("/tmp/t.json")


def test_body_budget_defaults_to_zero(monkeypatch):
    """0 means snippet only. The snapshot has empty bodies, so this reproduces
    what has always happened; live mail is where it starts mattering."""
    monkeypatch.delenv("INBOX_BODY_BUDGET", raising=False)
    assert load_settings().body_budget == 0


def test_body_budget_is_configurable(monkeypatch):
    monkeypatch.setenv("INBOX_BODY_BUDGET", "2000")
    assert load_settings().body_budget == 2000


def test_a_negative_body_budget_is_refused(monkeypatch):
    """Silently clamping to 0 would look like "snippet only" was chosen, when
    what happened is a misconfiguration nobody was told about."""
    monkeypatch.setenv("INBOX_BODY_BUDGET", "-1")
    with pytest.raises(ValueError, match="INBOX_BODY_BUDGET"):
        load_settings()


def test_embeddings_defaults_to_auto(monkeypatch):
    """Absent means auto: use Ollama when it is there, degrade when it is not.
    Defaulted rather than required so an existing checkout is unchanged."""
    monkeypatch.delenv("INBOX_EMBEDDINGS", raising=False)
    assert load_settings().embeddings == "auto"


@pytest.mark.parametrize("value,expected", [
    ("auto", "auto"), ("AUTO", "auto"), ("  ollama  ", "ollama"), ("none", "none"),
])
def test_embeddings_selection_is_normalised(monkeypatch, value, expected):
    monkeypatch.setenv("INBOX_EMBEDDINGS", value)
    assert load_settings().embeddings == expected


def test_an_unrecognised_embeddings_value_fails_loudly(monkeypatch):
    """Same reasoning as _resolve_gmail: a near miss must name the variable
    rather than quietly pick a mode the owner did not ask for."""
    monkeypatch.setenv("INBOX_EMBEDDINGS", "openai")
    with pytest.raises(ValueError) as exc:
        load_settings()
    assert "INBOX_EMBEDDINGS" in str(exc.value)
    assert "openai" in str(exc.value)


@pytest.mark.parametrize("configured,ollama_up,expected", [
    ("auto",   True,  "ollama"),
    ("auto",   False, "none"),     # degrade, and say so
    ("ollama", True,  "ollama"),
    ("none",   True,  "none"),     # never probes, never used
    ("none",   False, "none"),
])
def test_embeddings_resolution_matrix(monkeypatch, configured, ollama_up, expected):
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: ollama_up)
    assert config_mod.resolve_embeddings(configured) == expected


def test_pinned_ollama_raises_when_nothing_is_listening(monkeypatch):
    """The difference between `ollama` and `auto`. Pinning it means the owner
    wants the index, so a silent degrade would be answering a request for
    vectors with a store that has none."""
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: False)
    with pytest.raises(RuntimeError) as exc:
        config_mod.resolve_embeddings("ollama")
    assert "INBOX_EMBEDDINGS=ollama" in str(exc.value)
    assert "11434" in str(exc.value)


def test_auto_says_out_loud_that_it_degraded(monkeypatch, capsys):
    """A silent degrade is the failure this whole spec argues against."""
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: False)
    config_mod.resolve_embeddings("auto")
    assert "falling back" in capsys.readouterr().out


def test_get_embeddings_returns_none_when_resolved_to_none(monkeypatch):
    """The object, not the string. `None` is what _index() already expects."""
    monkeypatch.setattr(config_mod, "ollama_available", lambda *a, **k: False)
    assert config_mod.get_embeddings("auto") is None


def test_get_embeddings_probes_rather_than_deferring_to_first_write(monkeypatch):
    """The bug this task exists to kill: constructing OllamaEmbeddings performs
    no network call, so without an explicit probe the process starts fine and
    fails at the first correction instead."""
    probed = []
    monkeypatch.setattr(config_mod, "ollama_available",
                        lambda *a, **k: probed.append(True) or False)
    config_mod.get_embeddings("auto")
    assert probed, "get_embeddings must probe Ollama, not just construct a client"

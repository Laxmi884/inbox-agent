"""The model selector: pick a model by short name instead of editing .env."""
import pytest

from inbox_agent.config import (
    MODELS,
    ModelChoice,
    describe_models,
    resolve_model_choice,
)


def test_every_registry_entry_names_a_backend_we_can_build():
    for name, choice in MODELS.items():
        assert choice.backend in {"ollama", "openrouter", "openai"}, name
        assert choice.model_id, name


def test_resolve_returns_the_backend_and_model_id_for_a_known_name():
    c = resolve_model_choice("gemma")
    assert c.backend == "ollama"
    assert c.model_id == "gemma4:12b-mlx"


def test_resolve_is_case_and_whitespace_insensitive():
    """A name typed into a notebook cell should not fail on a stray space."""
    assert resolve_model_choice("  Gemma ").model_id == "gemma4:12b-mlx"


def test_unknown_name_fails_loudly_and_lists_what_is_valid():
    """Silently falling back to a default would waste a whole run on the wrong
    model - the exact failure that a dead OPENROUTER_MODEL slug already cost us."""
    with pytest.raises(KeyError) as e:
        resolve_model_choice("no-such-model")
    msg = str(e.value)
    assert "no-such-model" in msg
    for expected in ("gemma", "nemotron"):
        assert expected in msg


def test_registry_covers_the_models_actually_measured():
    """These four are the ones with recorded results; keep them addressable."""
    for name in ("gemma", "nemotron", "glm", "4o-mini"):
        assert name in MODELS


def test_describe_models_is_readable_and_flags_cost():
    rows = describe_models()
    assert len(rows) == len(MODELS)
    row = {r["name"]: r for r in rows}["gemma"]
    for col in ("name", "backend", "model_id", "cost"):
        assert col in row
    assert row["cost"] == "local"


def test_free_openrouter_models_are_marked_free():
    row = {r["name"]: r for r in describe_models()}["nemotron"]
    assert row["cost"] == "free"


def test_paid_openrouter_models_are_marked_paid():
    """A paid model must never be indistinguishable from a free one in the table."""
    row = {r["name"]: r for r in describe_models()}["4o-mini"]
    assert row["cost"] == "paid"


def test_use_model_builds_ollama_without_touching_env(monkeypatch):
    """Selecting a model must not depend on, or mutate, OLLAMA_MODEL."""
    monkeypatch.setenv("OLLAMA_MODEL", "some-other-model")
    built = {}

    class FakeChatOllama:
        def __init__(self, **kw):
            built.update(kw)

    import inbox_agent.config as cfg
    monkeypatch.setattr(cfg, "_build_ollama", lambda model: FakeChatOllama(model=model))
    cfg.use_model("gemma")

    assert built["model"] == "gemma4:12b-mlx"
    import os
    assert os.environ["OLLAMA_MODEL"] == "some-other-model"  # unchanged


def test_use_model_builds_openrouter_with_the_registry_id(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "stale/value")
    built = {}

    import inbox_agent.config as cfg
    monkeypatch.setattr(cfg, "_build_openrouter",
                        lambda model: built.update({"model": model}))
    cfg.use_model("nemotron")

    assert built["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_model_choice_carries_a_note_explaining_why_it_is_in_the_registry():
    """The registry doubles as the record of what has actually been measured."""
    assert isinstance(MODELS["gemma"], ModelChoice)
    assert MODELS["gemma"].note

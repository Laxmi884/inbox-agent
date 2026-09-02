# Clone and Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the inbox agent installable by someone else, make its effective configuration inspectable, and move the embeddings failure from "silently at the first correction" to "loudly at startup".

**Architecture:** Three independent threads that meet at the CLI. (a) `INBOX_EMBEDDINGS` gains an `auto|ollama|none` resolver that probes Ollama at construction rather than deferring to first write. (b) `config.py` captures `os.environ` above `load_dotenv` so every setting's origin is exactly knowable, and a new `doctor.py` renders that plus liveness checks. (c) `pyproject.toml` and a thin `cli.py` make both reachable as `inbox-agent`. Nothing in `graph.py`, `audit.py`, `partition.py` or the Telegram bot's logic is touched.

**Tech Stack:** Python 3.11+, langgraph 1.2.11, pydantic 2.11, python-dotenv 1.2, pytest 7.4, hatchling.

**Spec:** `docs/superpowers/specs/2026-09-02-clone-and-run-design.md`

## Global Constraints

- `requires-python = ">=3.11"`. Dependency floors are the verified versions in spec §1.1 — do not lower them.
- **All 584 existing tests stay green after every task.** Run the full suite, not just new tests.
- `python -m inbox_agent.telegram` must keep working unchanged, at every commit.
- Config resolvers follow the house pattern established by `_resolve_gmail` and `_resolve_triaged_label`: validate at load time, raise naming the offending variable, never silently fall back to a value the user did not ask for.
- Never write secrets to a log, a test fixture, or a committed file. `mask()` exists for this.
- `tests/conftest.py` forces `INBOX_LLM_BACKEND=offline` and `LANGSMITH_TRACING=false` on every test. Tests that need another backend must set it explicitly with `monkeypatch`.
- The `agent/triaged` label, the deny-list, and the autonomy ladder are out of scope. Do not modify `partition.py`, `graph.py`, or `audit.py`.

---

### Task 1: `INBOX_EMBEDDINGS` setting

Pure configuration. Adds the setting and its validator; changes no behaviour yet.

**Files:**
- Modify: `inbox_agent/config.py` (add `VALID_EMBEDDINGS`, `_resolve_embeddings`, `Settings.embeddings`, wire into `load_settings`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VALID_EMBEDDINGS: tuple[str, str, str]`; `_resolve_embeddings() -> str` returning one of `"auto" | "ollama" | "none"`; `Settings.embeddings: str` defaulting to `"auto"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_config.py -k embeddings -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'embeddings'`.

- [ ] **Step 3: Implement**

In `inbox_agent/config.py`, beside `VALID_GMAIL_CLIENTS`:

```python
VALID_EMBEDDINGS = ("auto", "ollama", "none")


def _resolve_embeddings() -> str:
    """Whether rule text gets a vector index, and what happens when Ollama is
    absent. See the design spec section 3.

    `auto` is the same contract resolve_backend() already offers for
    INBOX_LLM_BACKEND against the same daemon - probe, degrade, say so - so
    there is one story about a missing Ollama rather than two.
    """
    raw = os.getenv("INBOX_EMBEDDINGS", "auto").strip().lower()
    if raw not in VALID_EMBEDDINGS:
        raise ValueError(
            f"INBOX_EMBEDDINGS={raw!r} is not one of {VALID_EMBEDDINGS}. "
            "Use 'auto' to index rule text when Ollama is running and degrade "
            "when it is not, 'ollama' to require it, or 'none' to turn the "
            "index off."
        )
    return raw
```

Add the field to `Settings`, after `body_budget`:

```python
    # auto | ollama | none. Whether rule text is embedded for semantic search.
    # Nothing queries the index today (see the spec, section 1.3), so `auto`
    # degrading to `none` loses no capability that currently exists - it only
    # forfeits a future one. Defaulted so an existing checkout with Ollama
    # running behaves exactly as it always has.
    embeddings: str = "auto"
```

And in `load_settings()`, beside `body_budget=_resolve_body_budget(),`:

```python
        embeddings=_resolve_embeddings(),
```

- [ ] **Step 4: Run the new tests, then the whole suite**

Run: `python3 -m pytest tests/test_config.py -k embeddings -v`
Expected: PASS (6 tests).

Run: `python3 -m pytest`
Expected: `590 passed`.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/config.py tests/test_config.py
git commit -m "Add INBOX_EMBEDDINGS, defaulting to auto"
```

---

### Task 2: Resolve embeddings by probing, at construction

The real fix to spec §1.2. Today `get_embeddings()` builds a client without touching the network, so the process starts happily and dies at the first rule write — inside the learning loop, hours later. This decides it while a human is reading the banner.

**Files:**
- Modify: `inbox_agent/config.py` (`get_embeddings`, new `resolve_embeddings`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `_resolve_embeddings()` and `ollama_available()` from Task 1 / existing code.
- Produces: `resolve_embeddings(kind: Optional[str] = None) -> str` returning `"ollama"` or `"none"` (never `"auto"` — it is resolved away); `get_embeddings(kind: Optional[str] = None)` returning an `OllamaEmbeddings` or `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
from inbox_agent import config as config_mod


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_config.py -k "resolution_matrix or pinned_ollama or degraded or get_embeddings" -v`
Expected: FAIL — `AttributeError: module 'inbox_agent.config' has no attribute 'resolve_embeddings'`.

- [ ] **Step 3: Implement**

Replace the existing `get_embeddings()` in `inbox_agent/config.py`:

```python
def resolve_embeddings(kind: Optional[str] = None) -> str:
    """Which embeddings the store will ACTUALLY get: "ollama" or "none".

    Never returns "auto" - that is a request, not an outcome, and doctor has to
    report what happened rather than what was asked for.

    The probe is the point. OllamaEmbeddings constructs without touching the
    network, so before this existed the process started happily and raised at
    the first `put` - which is a correction, in Telegram, hours later, in the
    one code path this system exists for.
    """
    choice = kind if kind is not None else _resolve_embeddings()
    if choice == "none":
        return "none"                      # no probe: nothing to check
    if ollama_available():
        return "ollama"
    if choice == "ollama":
        raise RuntimeError(
            f"INBOX_EMBEDDINGS=ollama but nothing is listening on "
            f"{OLLAMA_BASE_URL}. Start `ollama serve`, or set "
            f"INBOX_EMBEDDINGS=auto to run without the rule index."
        )
    print(f"[resolve_embeddings] INBOX_EMBEDDINGS=auto and nothing is listening "
          f"on {OLLAMA_BASE_URL} - falling back to none. Rules still work; they "
          f"are matched exactly, not semantically. Start `ollama serve` to index.")
    return "none"


def get_embeddings(kind: Optional[str] = None):
    """Local embeddings for the rule index, or None. nomic-embed-text is 768-dim.

    `kind` is the configured value (auto|ollama|none); omit it to read the
    environment. Returns None whenever the resolved mode is "none", which is
    exactly what _index() already expects (store.py).
    """
    if resolve_embeddings(kind) == "none":
        return None

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
```

- [ ] **Step 4: Run the new tests, then the whole suite**

Run: `python3 -m pytest tests/test_config.py -k "resolution_matrix or pinned_ollama or degraded or get_embeddings" -v`
Expected: PASS (9 tests).

Run: `python3 -m pytest`
Expected: `599 passed`.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/config.py tests/test_config.py
git commit -m "Probe Ollama when embeddings are built, not at the first rule write"
```

---

### Task 3: Wire the setting into the bot, and prove rules survive a lost index

Two things that must land together: the bot honouring the setting, and evidence that a store written *with* an index still reads back *without* one. `auto` makes that second path reachable by accident — a laptop whose `ollama serve` died between restarts takes it silently.

**Files:**
- Modify: `inbox_agent/telegram/__main__.py:58`
- Test: `tests/test_store.py`, `tests/test_tg_banner.py` (new)

**Interfaces:**
- Consumes: `get_embeddings(kind)` and `Settings.embeddings` from Tasks 1–2.
- Produces: nothing new. This is the wiring task.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py`:

```python
def test_rules_written_with_an_index_read_back_without_one(tmp_path):
    """Spec section 3.4. `auto` degrading on a machine whose Ollama died makes
    this path reachable without anyone choosing it, so learned rules must
    survive it. A rule that becomes invisible is worse than one that errors:
    the agent would silently stop honouring a correction the owner made."""
    from inbox_agent.store import PreferenceStore, open_store
    from inbox_agent.models import ActionTemplate, Rule

    class FakeEmbeddings:
        """Deterministic, offline. The suite must not need Ollama."""
        def embed_documents(self, texts):
            return [[0.1] * 768 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 768

    path = tmp_path / "prefs.sqlite"

    indexed = PreferenceStore(open_store(path, FakeEmbeddings()))
    indexed.add_rule(Rule(id="r-keepme", scope="sender", pattern="a@b.com",
                          actions=[ActionTemplate(kind="archive", params={})],
                          provenance="written while indexed"))
    assert len(indexed.rules()) == 1

    plain = PreferenceStore(open_store(path))          # embeddings=None
    ids = [r.id for r in plain.rules()]
    assert ids == ["r-keepme"], f"rule lost when the index went away: {ids}"
```

Create `tests/test_tg_banner.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_tg_banner.py tests/test_store.py::test_rules_written_with_an_index_read_back_without_one -v`
Expected: the four banner tests FAIL with `ImportError: cannot import name '_embeddings_banner'`. The store test should already PASS — it is characterising existing behaviour, not driving new code. **If the store test fails, stop and report it**: that is spec §3.4's risk turning out to be real, it means learned rules go invisible when the index does, and it changes the design rather than being something to work around.

- [ ] **Step 3: Implement**

In `inbox_agent/telegram/__main__.py`, line 58, change:

```python
    prefs = PreferenceStore(open_store(settings.store_dir / "prefs.sqlite",
                                       get_embeddings()))
```

to:

```python
    prefs = PreferenceStore(open_store(settings.store_dir / "prefs.sqlite",
                                       get_embeddings(settings.embeddings)))
```

Add `resolve_embeddings` to the existing `from ..config import (...)` block, then define the banner helper above `main()`:

```python
def _embeddings_banner(settings) -> str:
    """The banner's embeddings line, as a string so it can be tested.

    Reports the RESOLVED mode, never the configured one: on a machine whose
    `ollama serve` has died those differ, and the resolved one is what the
    store is actually doing. "none" alone would be ambiguous - it is also what
    a deliberate INBOX_EMBEDDINGS=none looks like - so a degrade says why.
    """
    resolved = resolve_embeddings(settings.embeddings)
    if resolved == "none" and settings.embeddings == "auto":
        return ("embeddings: none   <- configured auto, but Ollama is not "
                "listening; rules still match exactly")
    return f"embeddings: {resolved}"
```

And call it in `main()`, beside the `store` line:

```python
    print(_embeddings_banner(settings))
```

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `python3 -m pytest tests/test_store.py tests/test_tg_banner.py -v`
Expected: PASS.

Run: `python3 -m pytest`
Expected: `604 passed`.

- [ ] **Step 5: Verify the banner by hand**

Run: `INBOX_EMBEDDINGS=none INBOX_TG_TOKEN=x INBOX_TG_CHAT_ID=1 INBOX_GMAIL=snapshot python3 -m inbox_agent.telegram 2>&1 | head -15`
Expected: an `embeddings: none` line in the banner. It will then fail to poll with a bad token — that is fine, the banner is what is under test. Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add inbox_agent/telegram/__main__.py tests/test_store.py tests/test_tg_banner.py
git commit -m "Honour INBOX_EMBEDDINGS in the bot, and prove rules outlive the index"
```

---

### Task 4: Configuration provenance

The mechanism behind `doctor`, and the answer to spec §1.4. Determined exactly, by capturing the environment above `load_dotenv`, rather than inferred by comparing values afterwards.

**Files:**
- Modify: `inbox_agent/config.py` (lines 14–16, the import and `load_dotenv` call)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `source_of(key: str) -> str` returning exactly `"environment"`, `"dotenv"`, or `"default"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_source_of_reports_a_shell_export_as_environment(monkeypatch):
    """The bug from spec 1.4: a value exported before launch wins over .env and
    dies with its shell. Naming it is the whole point of doctor."""
    monkeypatch.setattr(config_mod, "_ENV_AT_IMPORT", frozenset({"INBOX_GMAIL"}))
    assert config_mod.source_of("INBOX_GMAIL") == "environment"


def test_source_of_reports_a_file_value_as_dotenv(monkeypatch):
    monkeypatch.setattr(config_mod, "_ENV_AT_IMPORT", frozenset())
    monkeypatch.setattr(config_mod, "dotenv_values",
                        lambda *a, **k: {"INBOX_GMAIL": "live"})
    assert config_mod.source_of("INBOX_GMAIL") == "dotenv"


def test_source_of_reports_an_absent_key_as_default(monkeypatch):
    monkeypatch.setattr(config_mod, "_ENV_AT_IMPORT", frozenset())
    monkeypatch.setattr(config_mod, "dotenv_values", lambda *a, **k: {})
    assert config_mod.source_of("INBOX_BODY_BUDGET") == "default"


def test_environment_wins_even_when_the_file_agrees(monkeypatch):
    """The case a post-hoc value comparison gets wrong. If the shell and the
    file both say "live", comparing values cannot tell you which one is load
    bearing - but only the file survives a restart, so the distinction is the
    entire warning."""
    monkeypatch.setattr(config_mod, "_ENV_AT_IMPORT", frozenset({"INBOX_GMAIL"}))
    monkeypatch.setattr(config_mod, "dotenv_values",
                        lambda *a, **k: {"INBOX_GMAIL": "live"})
    assert config_mod.source_of("INBOX_GMAIL") == "environment"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_config.py -k source_of -v`
Expected: FAIL — `AttributeError: module 'inbox_agent.config' has no attribute 'source_of'`.

- [ ] **Step 3: Implement**

In `inbox_agent/config.py`, change the import on line 14 and the block around line 16:

```python
from dotenv import load_dotenv, find_dotenv, dotenv_values

# Captured BEFORE load_dotenv, and the ordering IS the mechanism: once
# load_dotenv has run there is no way to tell a value the shell exported from
# one the file supplied, because both are simply keys in os.environ. Comparing
# values afterwards cannot distinguish them either - when the shell and the
# file agree, the comparison says "same" and the warning that matters is lost.
# Only the file survives a restart, so which one won is the whole question.
# Do not move this line below load_dotenv.
_ENV_AT_IMPORT = frozenset(os.environ)

load_dotenv(find_dotenv(), override=False)
```

Then, beside `mask()`:

```python
def source_of(key: str) -> str:
    """Where the effective value of `key` came from: environment | dotenv | default.

    load_dotenv runs with override=False, so precedence is
    shell environment > .env > code default. A value that came from the shell
    does not survive a restart, which is how a bot came back on the snapshot in
    dry-run while still sending normal-looking digests. See the design spec 1.4.
    """
    if key in _ENV_AT_IMPORT:
        return "environment"
    if key in dotenv_values(find_dotenv()):
        return "dotenv"
    return "default"
```

- [ ] **Step 4: Run the new tests, then the whole suite**

Run: `python3 -m pytest tests/test_config.py -k source_of -v`
Expected: PASS (4 tests).

Run: `python3 -m pytest`
Expected: `608 passed`.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/config.py tests/test_config.py
git commit -m "Record where each setting's value actually came from"
```

---

### Task 5: Record when OAuth consent happened

Spec §4.1. `doctor` cannot warn about the seven-day revocation without a consent date, and no such date exists today: `token.json`'s `expiry` is the access token's hour, and `_write_token` rewrites the file on every refresh so its mtime tracks refreshes.

**Files:**
- Modify: `inbox_agent/google_auth.py` (the consent branch of `get_credentials`)
- Test: `tests/test_google_auth.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `consent_sidecar(token_path: Path) -> Path` returning `<token>.consent.json`; `record_consent(token_path: Path, *, now=None) -> None`; `consented_at(token_path: Path) -> Optional[datetime]` returning `None` when the sidecar is absent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_google_auth.py`:

```python
from datetime import datetime, timezone
from pathlib import Path


def test_consent_is_recorded_in_a_sidecar_not_in_the_token(tmp_path):
    """A sidecar because token.json's schema belongs to google-auth: it is
    produced by creds.to_json() and consumed by from_authorized_user_file, so a
    foreign key invites a breakage on upgrade for no benefit."""
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text('{"refresh_token": "x"}')

    google_auth.record_consent(token, now=datetime(2026, 9, 1, tzinfo=timezone.utc))

    assert google_auth.consent_sidecar(token) == tmp_path / "token.consent.json"
    assert '"refresh_token": "x"' in token.read_text(), "token.json must be untouched"
    assert google_auth.consented_at(token) == datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_consent_date_is_unknown_when_no_sidecar_exists(tmp_path):
    """Every token issued before this feature. Returning None makes doctor say
    'unknown' rather than invent a date - a confident wrong prediction about
    when the mailbox stops working is worse than no prediction."""
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text("{}")
    assert google_auth.consented_at(token) is None


def test_a_corrupt_sidecar_reads_as_unknown_rather_than_raising(tmp_path):
    """Doctor must survive a hand-edited or truncated sidecar. This file is
    diagnostics, never authorisation, so it can never be worth crashing over."""
    from inbox_agent import google_auth

    token = tmp_path / "token.json"
    token.write_text("{}")
    google_auth.consent_sidecar(token).write_text("not json{{")
    assert google_auth.consented_at(token) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_google_auth.py -k consent -v`
Expected: FAIL — `AttributeError: module 'inbox_agent.google_auth' has no attribute 'record_consent'`.

- [ ] **Step 3: Implement**

Add to `inbox_agent/google_auth.py`:

```python
def consent_sidecar(token_path: Path) -> Path:
    """Where the consent date lives: beside the token, never inside it.

    token.json's schema is google-auth's - creds.to_json() writes it and
    Credentials.from_authorized_user_file reads it - so an extra key there is a
    library upgrade away from breaking authentication for a diagnostic.
    """
    return Path(token_path).with_suffix(".consent.json")


def record_consent(token_path: Path, *, now: Optional[datetime] = None) -> None:
    """Stamp the moment consent was granted.

    Called ONLY from the consent path, never from the refresh path. Google's
    seven-day revocation for an app in Testing runs from consent and is not
    reset by refreshing, so stamping on refresh would promise six more days on
    the morning the token dies - worse than tracking nothing.
    """
    when = now or datetime.now(timezone.utc)
    path = consent_sidecar(token_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"consented_at": when.isoformat()}), encoding="utf-8")


def consented_at(token_path: Path) -> Optional[datetime]:
    """When consent was granted, or None if unrecorded or unreadable.

    None for every token issued before this existed. Diagnostics only - never
    read as authorisation - so a missing or corrupt file degrades to "unknown"
    rather than raising in a caller that is trying to explain what is wrong.
    """
    try:
        raw = json.loads(consent_sidecar(token_path).read_text(encoding="utf-8"))
        return datetime.fromisoformat(raw["consented_at"])
    except Exception:
        return None
```

Add `import json`, `from datetime import datetime, timezone`, and `Optional`/`Path` to the imports if absent.

Then in `get_credentials`, in the **consent** branch only (after `_write_token(token_path, creds)` that follows `_run_consent_flow`, at roughly line 163):

```python
    creds = _run_consent_flow(client_secrets_path=client_secrets_path,
                              scopes=scopes)
    _write_token(token_path, creds)
    # Consent path only. The refresh path above deliberately does not touch
    # this - see record_consent.
    record_consent(token_path)
    return creds
```

- [ ] **Step 4: Run the new tests, then the whole suite**

Run: `python3 -m pytest tests/test_google_auth.py -k consent -v`
Expected: PASS (3 tests).

Run: `python3 -m pytest`
Expected: `611 passed`.

- [ ] **Step 5: Commit**

```bash
git add inbox_agent/google_auth.py tests/test_google_auth.py
git commit -m "Record consent time in a sidecar, so token death can be predicted"
```

---

### Task 6: `doctor`

Its own module: `config.py` is already 497 lines and this is reporting, not resolution.

**Files:**
- Create: `inbox_agent/doctor.py`
- Test: `tests/test_doctor.py` (new)

**Interfaces:**
- Consumes: `load_settings()`, `source_of()`, `resolve_embeddings()`, `mask()`, `ollama_available()` (Tasks 1–4); `consented_at()` (Task 5); `load_policy()`.
- Produces: `Check` dataclass with fields `name: str`, `value: str`, `source: str`, `level: str` (`"ok" | "warn" | "fatal"`), `note: str`; `run_checks(settings) -> list[Check]`; `render(checks) -> str`; `main(argv=None) -> int` returning `1` when any check is `fatal`, else `0`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_doctor.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_doctor.py -v`
Expected: FAIL — `ImportError: cannot import name 'doctor'`.

- [ ] **Step 3: Implement**

Create `inbox_agent/doctor.py`:

```python
"""What is this agent actually configured to do, and where did that come from?

Exists because a value can be right and still be wrong: the bot ran live for
hours on INBOX_GMAIL=live and INBOX_DRY_RUN=false that were exported into a
shell and written nowhere, so a restart would have silently returned it to the
snapshot in dry-run - still polling, still classifying, still sending digests
that looked entirely normal and touched nothing. See the design spec 1.4.

Reporting only. This module resolves nothing and mutates nothing; every value
it prints is read back from the same functions the bot itself uses, because a
doctor that computes its own answers is checking a different program.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from . import config as config_mod
from .config import Settings, load_settings, mask, source_of
from .google_auth import consented_at
from .policy import load_policy

# Google revokes the refresh token of an External app left in "Testing" after
# exactly seven days (google_auth.py). Warn with room to act, not on the day.
TESTING_TOKEN_LIFETIME = timedelta(days=7)
WARN_WITHIN = timedelta(days=3)

_SECRET_KEYS = {"INBOX_TG_TOKEN"}


@dataclass(frozen=True)
class Check:
    name: str
    value: str
    source: str
    level: str        # "ok" | "warn" | "fatal"
    note: str = ""


def _setting(key: str, value, *, level: str = "ok", note: str = "") -> Check:
    """One reported setting, with provenance, and the override warning applied.

    The override check lives here rather than at each call site so it cannot be
    forgotten for one variable - which is how 1.4 stayed invisible.
    """
    source = source_of(key)
    shown = mask(str(value)) if key in _SECRET_KEYS else str(value)
    if source == "environment" and level == "ok":
        level, note = "warn", (
            "set in the shell, not in .env - this value will NOT survive a "
            "restart, and the file says something else or nothing at all")
    return Check(name=key, value=shown, source=source, level=level, note=note)


def run_checks(settings: Optional[Settings] = None) -> list[Check]:
    s = settings or load_settings()
    checks: list[Check] = [
        _setting("INBOX_GMAIL", s.gmail,
                 note="THE REAL MAILBOX" if s.gmail == "live" else ""),
        _setting("INBOX_DRY_RUN", s.dry_run),
        _setting("INBOX_LLM_BACKEND", s.backend,
                 level="fatal" if s.backend == "offline" else "ok",
                 note="no LLM backend available; classification cannot run"
                      if s.backend == "offline" else ""),
        _setting("INBOX_BODY_BUDGET", s.body_budget),
        _setting("INBOX_TRIAGED_LABEL", s.triaged_label),
        _setting("INBOX_STORE_DIR", s.store_dir),
    ]

    # Resolved, not configured. On a machine whose Ollama has died these differ,
    # and the resolved one is what the store is actually doing.
    resolved = config_mod.resolve_embeddings(s.embeddings)
    degraded = resolved == "none" and s.embeddings == "auto"
    checks.append(Check(
        name="INBOX_EMBEDDINGS", value=resolved, source=source_of("INBOX_EMBEDDINGS"),
        level="warn" if degraded else "ok",
        note=("configured auto, but nothing is listening on Ollama - rules are "
              "still matched exactly, so nothing is broken" if degraded else "")))

    for key, value in (("INBOX_TG_TOKEN", s.tg_token),
                       ("INBOX_TG_CHAT_ID", s.tg_chat_id)):
        checks.append(_setting(
            key, value or "not set",
            level="fatal" if not value else "ok",
            note="the bot refuses to start without it" if not value else ""))

    for key, path in (("INBOX_GOOGLE_CREDENTIALS", s.google_credentials),
                      ("INBOX_GOOGLE_TOKEN", s.google_token)):
        missing = not path.exists()
        checks.append(_setting(
            key, path,
            level="fatal" if (missing and s.gmail == "live") else "ok",
            note="missing, and INBOX_GMAIL=live needs it" if missing else ""))

    checks.append(_oauth_check(s))

    try:
        policy = load_policy(s)
        checks.append(Check("policy", policy.version, policy.source,
                            "warn" if policy.drifted else "ok",
                            "the hub and policies/default.md have diverged"
                            if policy.drifted else ""))
    except Exception as exc:                      # never let doctor be the thing that breaks
        checks.append(Check("policy", "unreadable", "-", "warn", str(exc)))

    return checks


def _oauth_check(s: Settings) -> Check:
    """How long until Google stops accepting the refresh token.

    Unknown is reported as unknown. Inventing a date would produce a confident
    wrong prediction about when the mailbox stops working, which is the exact
    failure this whole spec argues against.
    """
    granted = consented_at(s.google_token)
    if granted is None:
        return Check("oauth consent", "unknown", "-", "warn",
                     "consent date unknown (token predates tracking) - "
                     "re-consent to start predicting the 7-day revocation")
    dies = granted + TESTING_TOKEN_LIFETIME
    left = dies - datetime.now(timezone.utc)
    if left <= timedelta(0):
        return Check("oauth consent", granted.date().isoformat(), "-", "fatal",
                     f"refresh token expired ~{dies.date()}; re-consent needed")
    level = "warn" if left <= WARN_WITHIN else "ok"
    return Check("oauth consent", granted.date().isoformat(), "-", level,
                 f"expires ~{dies.date()} ({left.days}d left) if the app is "
                 f"still in Testing")


_MARK = {"ok": "  ", "warn": "! ", "fatal": "X "}


def render(checks: Sequence[Check]) -> str:
    width = max((len(c.name) for c in checks), default=0)
    lines = [f"{_MARK[c.level]}{c.name:<{width}}  {c.value}"
             f"   <- {c.source}" + (f"   {c.note}" if c.note else "")
             for c in checks]
    fatal = sum(c.level == "fatal" for c in checks)
    warn = sum(c.level == "warn" for c in checks)
    lines.append("")
    lines.append(f"{fatal} fatal, {warn} warning(s)."
                 + ("  Fix the fatals before starting the bot." if fatal else ""))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    checks = run_checks()
    print(render(checks))
    return 1 if any(c.level == "fatal" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the new tests, then the whole suite**

Run: `python3 -m pytest tests/test_doctor.py -v`
Expected: PASS (9 tests).

Run: `python3 -m pytest`
Expected: `620 passed`.

- [ ] **Step 5: Run it against the real configuration**

Run: `python3 -m inbox_agent.doctor`
Expected: a table. On this machine `INBOX_GMAIL` and `INBOX_DRY_RUN` should now read `<- dotenv` (they were corrected earlier), and `oauth consent` should read `unknown`. Read it and confirm it says something true; a doctor that lies is worse than no doctor.

- [ ] **Step 6: Commit**

```bash
git add inbox_agent/doctor.py tests/test_doctor.py
git commit -m "Add doctor: what is configured, where it came from, what is broken"
```

---

### Task 7: Packaging and the CLI

**Files:**
- Create: `pyproject.toml`, `inbox_agent/cli.py`
- Test: `tests/test_cli.py` (new)

**Interfaces:**
- Consumes: `doctor.main` (Task 6); `inbox_agent.telegram.__main__.main` (existing).
- Produces: `cli.main(argv=None) -> int`; console script `inbox-agent`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'cli'`.

- [ ] **Step 3: Implement the CLI**

Create `inbox_agent/cli.py`:

```python
"""The `inbox-agent` entry point.

Deliberately thin. `bot` delegates to inbox_agent.telegram.__main__.main, which
already wires the graph, the stores, the checkpointer and the startup banner -
a second copy of that wiring would drift from the first, and the wiring is
where the safety-relevant decisions are printed.

`python -m inbox_agent.telegram` keeps working and is not deprecated; this is
an addition, not a replacement.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

_USAGE = """usage: inbox-agent <command>

  doctor   report the effective configuration, where each value came from,
           and what would stop the bot from starting
  bot      run the Telegram bot (same as: python -m inbox_agent.telegram)
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else ""

    if command == "doctor":
        from . import doctor
        return doctor.main(args[1:])

    if command == "bot":
        from .telegram.__main__ import main as bot_main
        return bot_main()

    # No default. Falling through to `bot` on a typo would start a run against
    # the real mailbox because someone mistyped a diagnostic command.
    print(_USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `pyproject.toml`**

Create `pyproject.toml` at the repository root:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "inbox-agent"
version = "0.1.0"
description = "An email triage agent that acts on the confident majority and asks about the rest"
requires-python = ">=3.11"

# Floors are the versions this was verified against, not guesses. graph.py
# documents checkpoint_id behaviour confirmed against langgraph 1.2.11
# specifically, so going below these is unsupported.
dependencies = [
  "langgraph>=1.2.11",
  "langgraph-checkpoint-sqlite>=3.1.0",
  "langchain-core>=1.6.0",
  "langsmith>=0.11.1",
  "pydantic>=2.11",
  "python-dotenv>=1.2",
  "google-api-python-client>=2.200",
  "google-auth>=2.52",
  "google-auth-oauthlib>=1.4",
  "google-auth-httplib2>=0.4",
  "httplib2>=0.22",
  "httpx>=0.28",
]

[project.optional-dependencies]
# Both providers are imported lazily inside functions in config.py, so neither
# is needed to import the package - only to build the model it names.
ollama = ["langchain-ollama>=1.1.0"]
openai = ["langchain-openai>=1.2.1"]
dev    = ["pytest>=7.4"]

[project.scripts]
inbox-agent = "inbox_agent.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["inbox_agent"]
```

- [ ] **Step 5: Run the tests, then the whole suite**

Run: `python3 -m pytest tests/test_cli.py -v`
Expected: PASS (4 tests).

Run: `python3 -m pytest`
Expected: `624 passed`.

- [ ] **Step 6: Verify the package actually installs and the script works**

Run: `python3 -m pip install -e . --no-deps -q && inbox-agent doctor`
Expected: the same output as `python3 -m inbox_agent.doctor` in Task 6.

Run: `inbox-agent` and `inbox-agent nonsense`
Expected: usage text, exit code 2 (`echo $?`).

Run: `python3 -m inbox_agent.telegram --help 2>&1 | head -3` — confirm the old entry point still imports.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml inbox_agent/cli.py tests/test_cli.py
git commit -m "Make the project installable, with an inbox-agent entry point"
```

---

### Task 8: README and remote

**Files:**
- Create: `README.md`
- Test: none (documentation); verification is by following it.

**Interfaces:**
- Consumes: everything above.
- Produces: nothing code depends on.

- [ ] **Step 1: Write `README.md`**

Structure, in this order — the safety model comes first because a reader deciding whether to point this at their own mailbox needs it before setup:

1. **What it does** — one paragraph. Fetches unread inbox threads, classifies each, acts on the confident reversible majority, queues the rest for a one-tap verdict in Telegram, and learns a durable rule from every correction.
2. **What it will never do** — `ALWAYS_FORBIDDEN = {"send_message", "delete_forever"}` in `config.py` cannot be removed by configuration, only added to via `INBOX_FORBIDDEN_ACTIONS`. Every action passes one chokepoint, `execute_action`, which writes a JSONL audit record whether or not it acts.
3. **The autonomy ladder** — reproduce the precedence from `partition.hold_reason`: `trash` (unless a rule you taught authorised it) → `low_confidence` → `needs_reply` → `security_alert`. Anything unmatched, the agent does alone.
4. **`INBOX_DRY_RUN=true` is the default**, and what it means: every action is audited and none is performed.
5. **Setup** — Python 3.11+; `pip install -e ".[ollama]"` or `".[openai]"`; Google Cloud OAuth (Desktop app client, Gmail API enabled, `gmail.modify` scope) into `secrets/credentials.json`; a BotFather token and your own numeric Telegram user id; copy `.env.example` to `.env`.
6. **Check it before you run it** — `inbox-agent doctor`, with a real annotated example of its output, including one `!` line explaining the shell-vs-`.env` warning.
7. **Run it** — `inbox-agent bot`, then `/triage 10` in Telegram. Note that `python -m inbox_agent.telegram` is equivalent.
8. **Going live** — set `INBOX_GMAIL=live` and `INBOX_DRY_RUN=false` **in `.env`, not in your shell**, and say why: an exported value wins over the file and dies with the shell, so the next restart silently returns to the snapshot while still sending normal-looking digests.
9. **Troubleshooting** — one row per warning `doctor` can emit.

- [ ] **Step 2: Verify the README is followable, not just written**

Run each command in it verbatim from the worktree and confirm it does what the README claims. Fix the README where it does not. A setup document nobody has executed is a guess.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document the safety model first, then how to run it"
```

- [ ] **Step 4: Create the remote and push**

**Ask the user before this step** — it puts the repository, its full history, and its spec documents onto GitHub. Confirm the repository is **private**, and confirm `.gitignore` still covers `.env`, `secrets/`, `inbox_agent/store/`, `inbox_agent/snapshot/`, `*.sqlite` and `*.jsonl` before anything leaves the machine.

```bash
python3 tools/secret_scan.py          # the repo's own scanner, before pushing
gh repo create inbox-agent --private --source=. --remote=origin
git push -u origin main
```

Note: `main` is the branch to push, not this worktree branch. The worktree branch merges to `main` first (see below).

---

## After the plan

The manual verification from spec §6 is **not optional** and is not covered by any task above, because the suite is structurally incapable of reaching the bug in §1.2 — `build_store`'s docstring says embeddings are optional precisely so the suite runs without Ollama.

With a second BotFather token, `INBOX_GMAIL=snapshot`, `INBOX_DRY_RUN=true`, a store directory inside the worktree, and a non-Ollama backend so stopping Ollama does not also remove the classifier:

1. **Ollama stopped, `INBOX_EMBEDDINGS=auto`** — banner says it fell back; `/triage`; correct an item; the rule is written and reads back.
2. **Ollama running, `auto`** — resolves to `ollama`, proving the default preserves today's behaviour.
3. **Ollama stopped, `INBOX_EMBEDDINGS=ollama`** — fails in the banner, naming Ollama.

Run 1 needs Ollama down. Do it when no triage is expected, or the live bot on `main` loses its classifier for the duration.

Only then merge to `main` and restart the live bot.

"""Environment-backed configuration and model construction.

Mirrors the backend resolver already proven in build_notebook.py, renamed to the
INBOX_* namespace so the inbox agent and the CAB agent can run side by side with
different backends.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "gemma4:12b-mlx"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_DIMS = 768
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Actions the agent may never take, regardless of configuration. Env may add to
# this set; it may not remove from it. See spec section 2.
ALWAYS_FORBIDDEN = frozenset({"send_message", "delete_forever"})

_FALSEY = {"false", "0", "no", "off"}


@dataclass(frozen=True)
class Settings:
    backend: str
    dry_run: bool
    snapshot_dir: Path
    snapshot_size: int
    audit_log: Path
    forbidden_actions: frozenset[str]
    context_hub_skill: str
    context_hub_tag: str


def load_settings() -> Settings:
    raw_forbidden = os.getenv("INBOX_FORBIDDEN_ACTIONS", "")
    configured = {a.strip() for a in raw_forbidden.split(",") if a.strip()}

    # Absent means dry-run. Only an explicit falsey value turns it off.
    dry_raw = os.getenv("INBOX_DRY_RUN")
    dry_run = True if dry_raw is None else dry_raw.strip().lower() not in _FALSEY

    return Settings(
        backend=resolve_backend(),
        dry_run=dry_run,
        snapshot_dir=Path(os.getenv("INBOX_SNAPSHOT_DIR", "inbox_agent/snapshot")),
        snapshot_size=int(os.getenv("INBOX_SNAPSHOT_SIZE", "50")),
        audit_log=Path(os.getenv("INBOX_AUDIT_LOG", "inbox_agent/audit.jsonl")),
        forbidden_actions=ALWAYS_FORBIDDEN | frozenset(configured),
        context_hub_skill=os.getenv("CONTEXT_HUB_SKILL", "inbox-triage"),
        context_hub_tag=os.getenv("CONTEXT_HUB_TAG", "dev"),
    )


def ollama_available(timeout: float = 1.5) -> bool:
    """Cheap liveness probe so the notebook never hangs when Ollama is down."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def resolve_backend() -> str:
    """Which backend get_llm() will build. Honours INBOX_LLM_BACKEND, else auto-detects."""
    pinned = os.getenv("INBOX_LLM_BACKEND", "").strip().lower()
    if pinned == "ollama" and not ollama_available():
        print(f"[resolve_backend] INBOX_LLM_BACKEND=ollama but nothing is listening "
              f"on {OLLAMA_BASE_URL} - falling back to offline. Start `ollama serve`.")
        return "offline"
    if pinned:
        return pinned
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if ollama_available():
        return "ollama"
    return "offline"


def mask(value: Optional[str]) -> str:
    """Show enough of a key to confirm it loaded, never enough to leak it."""
    if not value:
        return "not set"
    if len(value) <= 11:
        return f"<redacted>  ({len(value)} chars)"
    return f"{value[:7]}...{value[-4:]}  ({len(value)} chars)"


def get_llm(backend: Optional[str] = None):
    """Build the chat model for the resolved backend. Always a BaseChatModel."""
    backend = backend or resolve_backend()

    if backend == "ollama":
        return _build_ollama(os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))

    if backend == "openrouter":
        return _build_openrouter(os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL))

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL), temperature=0)

    raise RuntimeError(
        "No LLM backend available. Start Ollama (`ollama serve`) or set "
        "OPENROUTER_API_KEY, then re-run."
    )


def _build_ollama(model: str):
    from langchain_ollama import ChatOllama

    print(f"[get_llm] Ollama ({model}) at {OLLAMA_BASE_URL}")
    return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            # One thread per call keeps this well under budget; see spec section 3.
            num_ctx=8192,
            # gemma4 is a hybrid thinker. Left on, it reasons at length before
            # emitting structured output and the call effectively never returns -
            # measured >9 minutes for a single classification. Off, it answers in
            # 1-3s. build_notebook.py:563 sets this for the same reason.
            reasoning=False,
        )


def _build_openrouter(model: str):
    from langchain_openai import ChatOpenAI

    print(f"[get_llm] OpenRouter ({model})")
    return ChatOpenAI(
        model=model,
        temperature=0,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={"X-Title": "Inbox Agent"},
    )


def get_embeddings():
    """Local embeddings for store semantic search. nomic-embed-text is 768-dim."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


# ---------------------------------------------------------------------------
# Model registry
#
# Pick a model by short name instead of editing .env and restarting the kernel:
#
#     from inbox_agent.config import use_model, describe_models
#     llm = use_model("nemotron")
#
# The registry doubles as the record of what has actually been run against the
# real snapshot - `note` says what was measured, not what is claimed. `.env`
# remains the default when no name is given, so nothing that exists breaks.
#
# `cost` is deliberately explicit. A dead or unexpectedly paid slug has already
# cost us one wasted run (stealth/ox-alpha was retired mid-project), so a name
# in here should tell you what it will do to your bill before you invoke it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelChoice:
    backend: str
    model_id: str
    cost: str   # "local" | "free" | "paid"
    note: str


MODELS: dict[str, ModelChoice] = {
    "gemma": ModelChoice(
        "ollama", "gemma4:12b-mlx", "local",
        "Local baseline. 3.1s/thread, 0 parse failures - but leaves `reason` "
        "empty on 50/50 threads, so the audit trail has no model-side why. "
        "Requires reasoning=False or a single call never returns.",
    ),
    "nemotron": ModelChoice(
        "openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", "free",
        "550B at no cost, and NOT viable for this workload. Measured 69s for a "
        "single classification: it spent 1138 reasoning tokens to emit ~40 "
        "tokens of JSON. A 50-thread pass is roughly 57 minutes. It also omits "
        "`reason`, so it does not even close the audit gap Gemma leaves. Same "
        "failure shape as local Gemma - a reasoning model is the wrong tool for "
        "one small structured judgment.",
    ),
    "glm": ModelChoice(
        "openrouter", "z-ai/glm-5.3-flash", "paid",
        "What the retired stealth/ox-alpha slot turned out to be.",
    ),
    "4o-mini": ModelChoice(
        "openrouter", "openai/gpt-4o-mini", "paid",
        "1.63s/thread, 0 parse failures, `reason` populated 50/50 - the "
        "audit gap Gemma leaves open. ~$0.02 for a 50-thread run.",
    ),
    "sonnet": ModelChoice(
        "openrouter", "anthropic/claude-sonnet-4.5", "paid",
        "Frontier reference. Not yet measured on this snapshot.",
    ),
}


def resolve_model_choice(name: str) -> ModelChoice:
    """Look up a registry entry. Unknown names fail loudly, never silently."""
    key = (name or "").strip().lower()
    if key not in MODELS:
        raise KeyError(
            f"{name!r} is not in the model registry. Known names: "
            f"{', '.join(sorted(MODELS))}. Add one to MODELS in config.py, "
            f"or pass a raw id via .env."
        )
    return MODELS[key]


def use_model(name: str):
    """Build a chat model by short name, bypassing .env entirely.

    Reads nothing from the environment except credentials, and mutates nothing -
    so switching models inside a notebook does not leave the process in a state
    that later cells silently inherit.
    """
    choice = resolve_model_choice(name)
    if choice.backend == "ollama":
        return _build_ollama(choice.model_id)
    if choice.backend == "openrouter":
        return _build_openrouter(choice.model_id)
    if choice.backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=choice.model_id, temperature=0)
    raise RuntimeError(f"registry entry {name!r} names unknown backend {choice.backend!r}")


def describe_models() -> list[dict]:
    """Rows for a readable table of what you can select."""
    return [
        {"name": n, "backend": c.backend, "model_id": c.model_id,
         "cost": c.cost, "note": c.note}
        for n, c in sorted(MODELS.items(), key=lambda kv: (kv[1].cost != "local", kv[0]))
    ]

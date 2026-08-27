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
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        print(f"[get_llm] Ollama ({model}) at {OLLAMA_BASE_URL}")
        return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            # One thread per call keeps this well under budget; see spec section 3.
            num_ctx=8192,
        )

    if backend == "openrouter":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        print(f"[get_llm] OpenRouter ({model})")
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={"X-Title": "Inbox Agent"},
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL), temperature=0)

    raise RuntimeError(
        "No LLM backend available. Start Ollama (`ollama serve`) or set "
        "OPENROUTER_API_KEY, then re-run."
    )


def get_embeddings():
    """Local embeddings for store semantic search. nomic-embed-text is 768-dim."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

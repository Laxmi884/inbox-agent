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
    # A needs_reply older than this stops being held in the inbox. 90 days is a
    # deliberate default: long enough that a real conversation is genuinely
    # over, short enough to matter when clearing a backlog. Defaulted so every
    # existing construction of Settings keeps working unchanged.
    stale_after_days: int = 90
    # Where the learned rules and the held queue live between runs. Both were
    # in-memory while the queue was only ever read by the run that filled it;
    # `agent/triaged` ended that, because a thread lost from the queue also
    # leaves the fetch query and is not re-fetched by any mode, so losing it is
    # no longer recoverable by running again. Defaulted, for the same reason
    # stale_after_days is: every existing construction of Settings keeps
    # working. Already gitignored - it holds real subjects and senders.
    store_dir: Path = Path("inbox_agent/store")
    # Telegram review UI. Defaults keep every existing construction of Settings
    # (tests, the notebook) working unchanged; the bot refuses to start without
    # a token and a chat id, rather than running open to anyone who finds it.
    tg_token: str = ""
    tg_chat_id: str = ""
    tg_mode: str = "digest"
    # Applied to every thread the agent has processed, so it leaves the fetch
    # query. Without it, threads that were labelled but left in the inbox - and
    # everything decided `none` - stay INBOX+UNREAD forever and are re-triaged,
    # re-charged and re-reported in every single digest until read by hand.
    #
    # In Gmail rather than a local set on purpose: it is visible, so "why did it
    # ignore this?" has an answer you can see in the mailbox; it survives losing
    # the local store; and `label` is already at "always" authority, so it grants
    # no new capability. Marking as READ would have worked too and was rejected -
    # it destroys unread as a signal for the human and is not on the ladder.
    triaged_label: str = "agent/triaged"
    # Which GmailClient build_gmail_client() returns. "snapshot" or "live".
    # Defaulted to snapshot deliberately: an unconfigured checkout, and the
    # whole existing suite, behave exactly as they did before this existed, and
    # nothing reaches for credentials that are not there. Going live is one
    # deliberate edit, which is the same shape as INBOX_DRY_RUN.
    gmail: str = "snapshot"
    # OAuth client of type "Desktop app", and the token the consent flow writes
    # beside it. Both live credentials for the real mailbox; `secrets/` is
    # gitignored. Paths rather than contents so neither is ever held in
    # Settings, which is printed in places.
    google_credentials: Path = Path("secrets/credentials.json")
    google_token: Path = Path("secrets/token.json")
    # Characters of message body allowed into the classifier prompt. 0 means
    # snippet only. The snapshot has empty bodies, so 0 reproduces what has
    # always happened; live mail is where it starts to matter. Kept a knob
    # rather than a constant so the snippet-vs-body comparison is a config flip
    # driven from LangSmith traces, not a code edit. See the design spec 2.4.
    body_budget: int = 0
    # auto | ollama | none. Whether rule text is embedded for semantic search.
    # Nothing queries the index today (see the spec, section 1.3), so `auto`
    # degrading to `none` loses no capability that currently exists - it only
    # forfeits a future one. Defaulted so an existing checkout with Ollama
    # running behaves exactly as it always has.
    embeddings: str = "auto"

    @property
    def inbox_query(self) -> str:
        return f"in:inbox is:unread -label:{self.triaged_label}"


def _resolve_triaged_label() -> str:
    """Empty and whitespace-containing values are both footguns, not valid
    configuration, so neither is passed through as-is.

    Empty (INBOX_TRIAGED_LABEL="") would turn inbox_query's `-label:` term into
    a no-op that excludes nothing, so fetch silently stops narrowing by
    triaged status, and mark_triaged would apply an empty label on top of it -
    falling back to the default keeps the query meaningful instead of failing
    open.

    Whitespace is legal in a real Gmail label but fatal here: the snapshot
    client's matches_query splits `query` on whitespace term-by-term
    (inbox_agent/gmail.py), so a labelled-with-a-space value would blow up
    fetch() with `ValueError: '...' is not a query term...` - an error that
    never names INBOX_TRIAGED_LABEL as the actual misconfiguration. Raising
    here, at load time, names it.
    """
    raw = os.getenv("INBOX_TRIAGED_LABEL", "agent/triaged").strip()
    if not raw:
        return "agent/triaged"
    if any(ch.isspace() for ch in raw):
        raise ValueError(
            f"INBOX_TRIAGED_LABEL={raw!r} contains whitespace, which the "
            "query parser cannot handle - it splits the fetch query on "
            "whitespace term-by-term. Use a label with no spaces (Gmail "
            "nested labels use '/', e.g. 'agent/triaged')."
        )
    return raw


VALID_GMAIL_CLIENTS = ("snapshot", "live")


def _resolve_gmail() -> str:
    """Which client to build. An unrecognised value is a misconfiguration, and
    falling back to "snapshot" would answer a request for the real mailbox with
    a run that looks successful and touched nothing. Same reasoning as
    _resolve_triaged_label: name the variable that is wrong, at load time."""
    raw = os.getenv("INBOX_GMAIL", "snapshot").strip().lower()
    if raw not in VALID_GMAIL_CLIENTS:
        raise ValueError(
            f"INBOX_GMAIL={raw!r} is not one of {VALID_GMAIL_CLIENTS}. "
            "Use 'snapshot' for the frozen evaluation set, or 'live' for the "
            "real mailbox (which also needs INBOX_GOOGLE_CREDENTIALS)."
        )
    return raw


VALID_EMBEDDINGS = ("auto", "ollama", "none")


def _resolve_embeddings() -> str:
    """Whether rule text gets a vector index, and what happens when Ollama is
    absent. See the design spec section 3.

    `auto` mirrors what resolve_backend() does when INBOX_LLM_BACKEND is
    pinned to "ollama" against the same daemon: probe, and if it is down,
    print a warning and degrade rather than fail. (resolve_backend()'s other
    paths differ - an unset INBOX_LLM_BACKEND that finds nothing live
    degrades to "offline" with no print - so the parallel is with that one
    pinned case, not with the function as a whole.)
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


# INBOX_BODY_BUDGET=full. A sentinel rather than a huge number so the banner,
# doctor and traces can say "full" instead of reporting a made-up ceiling that
# was never really the limit. Defined here rather than in classify.py because
# config is the base layer that imports nothing else in the package, and
# classify -> policy -> config already: the other direction would be a cycle.
BODY_FULL = -1


def describe_body_budget(budget: int) -> str:
    """How the banner, doctor and a trace should name the budget.

    One formatter so the three cannot disagree. "full" must never be reported
    as a number: the whole point of the sentinel is that there is no ceiling to
    name, and printing one would be the same class of lie as reporting a masked
    placeholder as a loaded token.
    """
    if budget == BODY_FULL:
        return "full body, uncapped"
    if budget == 0:
        return "0 chars into the prompt  (snippet only)"
    return f"{budget} chars of body into the prompt"


def _resolve_body_budget() -> int:
    """Characters of body into the prompt.

    0 is snippet only, a positive number is that many characters of body, and
    "full" is the entire body however long it is. Negative is refused rather
    than clamped: clamping to 0 would look exactly like "snippet only was
    chosen", when what actually happened is a misconfiguration nobody was told
    about. That is also why the sentinel is reachable ONLY through the word
    "full" - a bare -1 in the environment is far more likely to be a mistake
    than a request to send an unbounded prompt to a model with an 8192-token
    window.
    """
    raw = os.getenv("INBOX_BODY_BUDGET", "0").strip()
    if raw.lower() == "full":
        return BODY_FULL
    try:
        value = int(raw or "0")
    except ValueError:
        raise ValueError(
            f"INBOX_BODY_BUDGET={raw!r} is not an integer. Use 0 for snippet "
            "only, a character budget such as 2000, or 'full' for the whole "
            "body.") from None
    if value < 0:
        raise ValueError(
            f"INBOX_BODY_BUDGET={value} is negative. Use 0 for snippet only, "
            "or 'full' for the whole body.")
    return value


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
        store_dir=Path(os.getenv("INBOX_STORE_DIR", "inbox_agent/store")),
        forbidden_actions=ALWAYS_FORBIDDEN | frozenset(configured),
        context_hub_skill=os.getenv("CONTEXT_HUB_SKILL", "inbox-triage"),
        # Blank = latest. "dev" was the old default and is not a ref the hub
        # can resolve; see _pull_from_context_hub.
        context_hub_tag=os.getenv("CONTEXT_HUB_TAG", ""),
        stale_after_days=int(os.getenv("INBOX_STALE_AFTER_DAYS", "90")),
        triaged_label=_resolve_triaged_label(),
        gmail=_resolve_gmail(),
        google_credentials=Path(
            os.getenv("INBOX_GOOGLE_CREDENTIALS", "secrets/credentials.json")),
        google_token=Path(os.getenv("INBOX_GOOGLE_TOKEN", "secrets/token.json")),
        body_budget=_resolve_body_budget(),
        embeddings=_resolve_embeddings(),
        tg_token=os.getenv("INBOX_TG_TOKEN", "").strip(),
        tg_chat_id=os.getenv("INBOX_TG_CHAT_ID", "").strip(),
        tg_mode=os.getenv("INBOX_TG_MODE", "digest").strip().lower(),
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


def dotenv_value(key: str) -> Optional[str]:
    """What the .env FILE says for `key` right now, ignoring this process.

    source_of() answers "shell, file, or default", which covers the override
    that cost a live run in spec 1.4. It cannot answer a different question:
    is the file NEWER than the process reading it. load_dotenv runs once, at
    import, so a long-running bot holds whatever the file said when it
    started, and every later edit is invisible to it.

    Comparing this against os.environ is the only way to see that gap. See
    doctor.tracing_check, which was written after a bot spent 17 hours sending
    traces that .env had said to stop sending.
    """
    return dotenv_values(find_dotenv()).get(key)


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

    Prefer `build_embeddings` over this in production code: it resolves once
    and hands back both the resolved mode and the object, so the caller can
    report what actually happened (doctor, the startup banner). This function
    discards the resolved mode after using it, so a caller that also needs to
    know what was resolved ends up probing Ollama a second time to find out.
    Kept for its own test coverage of resolve_embeddings' contract, not
    because anything in inbox_agent/ still calls it.
    """
    if resolve_embeddings(kind) == "none":
        return None

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def build_embeddings(kind: Optional[str] = None) -> tuple[str, Optional[object]]:
    """Resolve once, and return both what was resolved and the object built.

    Callers need both: the store needs the embeddings, the banner needs to
    state what the store actually got. Resolving separately for each meant two
    Ollama probes, two copies of the degrade warning, and a window in which the
    two answers could disagree.
    """
    resolved = resolve_embeddings(kind)
    if resolved == "none":
        return resolved, None

    from langchain_ollama import OllamaEmbeddings

    return resolved, OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


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
        "Local baseline, and the only local model still standing. Full "
        "50-thread snapshot on ollama 0.33.1: 4.53s/thread warm, 0 parse "
        "failures, `reason` 50/50, mean confidence 0.97, all 9 policy "
        "categories used. WITHOUT the prompt contract the same 50 threads give "
        "`reason` 0/50 - unchanged from 0.32.15, even though 0.33.1 shipped the "
        "MLX structured-output fix (#16563 / PR #17929). ollama#15260 is the "
        "one that bites: reasoning=False silently voids schema enforcement, and "
        "reasoning=False is mandatory here or a single call never returns. "
        "Timing note: earlier 10-thread runs measured 3.16s/thread; the "
        "10-vs-50 gap spans both a different thread set and an Ollama upgrade, "
        "so it is not attributable to either alone.",
    ),
    "e4b": ModelChoice(
        "ollama", "gemma4:e4b-mlx", "local",
        "REJECTED on judgement, not on speed. The FASTEST thing measured "
        "anywhere in this project: 1.09s/thread warm, beating even the best "
        "hosted option (gemma-3-12b-it at 1.29s), and with the contract it "
        "parses 10/10 and fills `reason` 10/10. But its action head collapses. "
        "On a 10-thread diff against gemma4:12b-mlx it agreed on category 9/10 "
        "and on ACTION only 4/10, returning `label` for all ten threads where "
        "12b split 6 archive / 4 label. A triage agent that labels everything "
        "and archives nothing never clears the inbox - archive-vs-label IS the "
        "decision. Mean confidence 0.88 vs 0.97. Caveat: measured on 10 threads; "
        "the 50-thread confirmation was started and not finished. Also 9.5 GB "
        "on disk, LARGER than 12b-mlx despite the 'efficient' name. Not "
        "currently pulled.",
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
    "gemma3": ModelChoice(
        "openrouter", "google/gemma-3-12b-it", "paid",
        "BEST MEASURED. 1.29s/thread, 0/10 parse failures, `reason` 10/10, mean "
        "confidence 0.845, ~$0.003 per 50-thread run - faster AND a third the price "
        "of 4o-mini. Same 12B size as the local model but NOT a reasoning model: "
        "this is the control proving the local failure was reasoning, not size and "
        "not the Gemma family.",
    ),
    "qwen3": ModelChoice(
        "openrouter", "qwen/qwen3-30b-a3b-instruct-2507", "paid",
        "3.07s/thread, 0/10 failures, `reason` 10/10, confidence 0.80, ~$0.003. "
        "Non-reasoning 30B - the direct counterpart to muse-glimmer-30b with "
        "reasoning as the only variable changed, and it works.",
    ),
    "nemo": ModelChoice(
        "openrouter", "mistralai/mistral-nemo", "paid",
        "Cheapest at ~$0.001 per 50-thread run, but 11.2s/thread and 2/10 parse "
        "failures. The cheap floor has a real cost in reliability.",
    ),
    "muse": ModelChoice(
        "openrouter", "meta/muse-glimmer-30b", "paid",
        "Reasoning model. 7.1s/thread and 2/10 parse failures - it answers in "
        "markdown prose rather than JSON, and only LangChain's coercion "
        "rescues 8 of 10. Populates `reason` well, but mean confidence 0.578 vs "
        "0.834 for 4o-mini, and it defaults to archive over labelling. "
        "~$0.06 per 50-thread run: reasoning tokens bill as completion tokens, "
        "so it costs 3x what the headline per-token price suggests. Reasoning "
        "cannot be disabled here - OpenRouter returns HTTP 400 'Reasoning is "
        "mandatory for this endpoint'. The Ollama reasoning=False escape hatch "
        "that rescued the local model has no equivalent on this endpoint.",
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


# ---------------------------------------------------------------------------
# Gmail client construction
#
# The one place that decides snapshot vs live. GmailClient is a Protocol, so
# nothing above this line knows or cares which it got - that is why adding the
# live mailbox is an addition rather than a migration, and why there was exactly
# one construction site outside tests to change.
# ---------------------------------------------------------------------------


def _build_live_gmail_client(settings: Settings):
    """Split out so the factory's branching is testable without google libs."""
    from googleapiclient.discovery import build

    from .gmail import LiveGmailClient
    from . import google_auth

    creds = google_auth.get_credentials(
        client_secrets_path=settings.google_credentials,
        token_path=settings.google_token,
        scopes=[google_auth.GMAIL_MODIFY_SCOPE])
    # cache_discovery=False silences an oauth2client file-cache warning that is
    # noise on every start and has no bearing on anything here.
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    # http_factory is NOT optional in practice. googleapiclient's service holds
    # one httplib2.Http, which is not thread-safe, and LiveGmailClient hydrates
    # a page through a five-worker pool. Without a per-thread transport the
    # first real run dies with `SSL: WRONG_VERSION_NUMBER`, which names nothing
    # about threads - and every unit test still passes, because a fake has no
    # socket. See LiveGmailClient._http.
    return LiveGmailClient(
        service, http_factory=lambda: google_auth.authorized_http(creds))


def build_gmail_client(settings: Settings):
    """Snapshot or live, per INBOX_GMAIL.

    An unrecognised value raises rather than defaulting. Falling back to
    snapshot would look like a successful run against a mailbox that was never
    touched; falling back to live would touch a mailbox nobody asked it to.
    Neither is a failure you want to discover from a digest.
    """
    from .gmail import SnapshotGmailClient

    choice = (settings.gmail or "snapshot").strip().lower()
    if choice == "snapshot":
        return SnapshotGmailClient(settings.snapshot_dir / "threads.json")
    if choice == "live":
        return _build_live_gmail_client(settings)
    raise ValueError(
        f"INBOX_GMAIL={settings.gmail!r} is not a Gmail client. "
        f"Use 'snapshot' (the frozen evaluation set) or 'live' (the real "
        f"mailbox, which needs {settings.google_credentials})."
    )

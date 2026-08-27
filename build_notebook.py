"""Generates cab_release_notes_agent.ipynb from cell definitions (keeps the JSON honest)."""

import json
from pathlib import Path

CELLS = []


def md(source: str):
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str):
    CELLS.append(("code", source.strip("\n")))


# ---------------------------------------------------------------- 1. intro
md(r'''
# CAB Release Notes Agent — LangGraph POC

Automates the weekly **Change Advisory Board** prep: pull the release tickets for the CAB
window, gate them against release policy (QA sign-off + rollback plan), and hand the board a
formatted Markdown briefing.

```
START ──▶ fetch_data ──▶ validate_and_assess_risk ──▶ generate_cab_summary ──▶ END
```

| Node | Type | Responsibility |
|---|---|---|
| `fetch_data` | I/O (mocked) | Merge Jira + GitHub + ServiceNow into one ticket list |
| `validate_and_assess_risk` | Deterministic logic | Flag tickets missing approvals / rollback plans, score risk |
| `generate_cab_summary` | LLM | Write the executive Markdown report |

**Everything external is mocked.** No Jira/GitHub/ServiceNow credentials are needed, and the
notebook runs end-to-end with **no LLM API key** — it falls back to a local deterministic
chat model that implements the same `BaseChatModel` interface. Set `OPENAI_API_KEY` to run the
real thing; nothing else changes.
''')

# ---------------------------------------------------------------- 2. install
code(r'''
# Uncomment on a fresh environment.
# langchain-openai is what talks to OpenRouter too (OpenAI-compatible API).
# %pip install -q langgraph langchain-core langchain-openai langchain-ollama python-dotenv
''')

code(r'''
from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

print("langgraph + langchain-core loaded")
''')

# ---------------------------------------------------------------- 3. mocks
md(r'''
## 1. Mocked upstream systems

Three static payloads stand in for the real integrations. Each `fetch_*` function has the
signature the real client would have (`cab_date` in, list of records out), so swapping in
`jira.search_issues(...)` later is a one-line change per function.
''')

code(r'''
# --- Jira: the change tickets scheduled into the CAB window -----------------
MOCK_JIRA_ISSUES = [
    {
        "ticket_id": "REL-1042",
        "title": "Core banking backend: PostgreSQL 14 to 16 migration",
        "system": "core-backend",
        "owner": "Priya Nair",
        "team": "Platform Engineering",
        "risk_level": "high",
        "customer_impacting": True,
        "downtime_minutes": 45,
        "rollback_plan_provided": True,
        "rollback_summary": "pg_upgrade snapshot restore + replica promotion, rehearsed in staging 2026-08-11.",
        "approvals": ["QA", "Security", "Architecture"],
        "description": "Major version upgrade of the primary transactional database, including extension rebuilds and a planned 45-minute maintenance window.",
    },
    {
        "ticket_id": "REL-1043",
        "title": "Customer portal: fix truncated invoice totals on mobile",
        "system": "web-ui",
        "owner": "Diego Marchetti",
        "team": "Frontend Guild",
        "risk_level": "low",
        "customer_impacting": True,
        "downtime_minutes": 0,
        "rollback_plan_provided": True,
        "rollback_summary": "Revert the release tag; static assets are versioned and cached per-build.",
        "approvals": ["QA", "Product"],
        "description": "CSS and formatting fix for invoice totals overflowing their container below 480px viewport width.",
    },
    {
        "ticket_id": "REL-1044",
        "title": "Auth service: session tokens not invalidated on password reset",
        "system": "auth-service",
        "owner": "Sam Okonkwo",
        "team": "Identity",
        "risk_level": "high",
        "customer_impacting": True,
        "downtime_minutes": 0,
        "rollback_plan_provided": False,
        "rollback_summary": "",
        "approvals": ["QA", "Security"],
        "description": "Security fix so that resetting a password revokes all outstanding refresh tokens. Touches the shared session store.",
    },
    {
        "ticket_id": "REL-1045",
        "title": "Payments service: raise settlement batch size to 5000",
        "system": "payments",
        "owner": "Hannah Bergstrom",
        "team": "Payments",
        "risk_level": "medium",
        "customer_impacting": False,
        "downtime_minutes": 0,
        "rollback_plan_provided": True,
        "rollback_summary": "Config-only change; revert the feature flag value in the config service.",
        "approvals": ["Product"],
        "description": "Throughput tuning for the nightly settlement job ahead of quarter-end volumes.",
    },
]

# --- GitHub: PRs merged for those tickets -----------------------------------
MOCK_GITHUB_PULLS = {
    "REL-1042": [
        {"number": 8812, "repo": "core-backend", "title": "chore(db): pg16 migration scripts", "files_changed": 63, "additions": 2140, "deletions": 880, "reviewers": ["a.silva", "m.tan"]},
        {"number": 8815, "repo": "core-backend", "title": "feat(db): dual-read compatibility shim", "files_changed": 12, "additions": 410, "deletions": 36, "reviewers": ["a.silva"]},
    ],
    "REL-1043": [
        {"number": 4471, "repo": "customer-portal", "title": "fix(ui): invoice total overflow on small viewports", "files_changed": 3, "additions": 28, "deletions": 11, "reviewers": ["l.chen"]},
    ],
    "REL-1044": [
        {"number": 2290, "repo": "auth-service", "title": "fix(auth): revoke refresh tokens on password reset", "files_changed": 9, "additions": 302, "deletions": 74, "reviewers": ["r.gupta", "s.oduya"]},
    ],
    "REL-1045": [
        {"number": 1177, "repo": "payments", "title": "perf(settlement): batch size 1000 -> 5000", "files_changed": 2, "additions": 14, "deletions": 6, "reviewers": []},
    ],
}

# --- ServiceNow: the change requests wrapping each release ------------------
MOCK_SERVICENOW_CHANGES = {
    "REL-1042": {"change_id": "CHG0043210", "state": "Scheduled", "window": "2026-08-20 22:00-23:30 UTC", "cab_required": True},
    "REL-1043": {"change_id": "CHG0043211", "state": "Scheduled", "window": "2026-08-20 18:00-18:30 UTC", "cab_required": False},
    "REL-1044": {"change_id": "CHG0043212", "state": "Assess",    "window": "2026-08-20 21:00-21:30 UTC", "cab_required": True},
    "REL-1045": {"change_id": "CHG0043213", "state": "Draft",     "window": "2026-08-20 20:00-20:15 UTC", "cab_required": True},
}


def fetch_jira_issues(cab_date: str) -> List[Dict[str, Any]]:
    """MOCK of jira.search_issues(jql=...). Returns issues targeting the CAB window."""
    return [dict(issue) for issue in MOCK_JIRA_ISSUES]


def fetch_github_pulls(ticket_id: str) -> List[Dict[str, Any]]:
    """MOCK of GET /repos/{owner}/{repo}/pulls?q={ticket_id}."""
    return [dict(pr) for pr in MOCK_GITHUB_PULLS.get(ticket_id, [])]


def fetch_servicenow_change(ticket_id: str) -> Dict[str, Any]:
    """MOCK of GET /api/now/table/change_request?correlation_id={ticket_id}."""
    return dict(MOCK_SERVICENOW_CHANGES.get(ticket_id, {}))


print(f"{len(MOCK_JIRA_ISSUES)} mocked Jira issues, "
      f"{sum(len(v) for v in MOCK_GITHUB_PULLS.values())} PRs, "
      f"{len(MOCK_SERVICENOW_CHANGES)} change records")
''')

# ---------------------------------------------------------------- 4. state
md(r'''
## 2. Graph state

One `TypedDict` threaded through every node. Each node returns a **partial** state dict;
LangGraph merges it into the running state.
''')

code(r'''
class CABState(TypedDict):
    cab_date: str                      # e.g. "2026-08-20"
    raw_tickets: List[Dict[str, Any]]  # merged Jira + GitHub + ServiceNow records
    flagged_tickets: List[Dict[str, Any]]  # tickets missing approvals / rollback plans
    summary_report: str                # final Markdown briefing
''')

# ---------------------------------------------------------------- 5. node 1
md(r'''
## 3. Node 1 — `fetch_data`

Fans out across the three mocked systems and joins them on `ticket_id`.
''')

code(r'''
def fetch_data(state: CABState) -> Dict[str, Any]:
    """Collect every release ticket scheduled for the CAB date and enrich it."""
    cab_date = state["cab_date"]
    tickets: List[Dict[str, Any]] = []

    for issue in fetch_jira_issues(cab_date):
        ticket = dict(issue)
        ticket["cab_date"] = cab_date
        ticket["pull_requests"] = fetch_github_pulls(issue["ticket_id"])
        ticket["change_record"] = fetch_servicenow_change(issue["ticket_id"])
        ticket["code_churn"] = sum(
            pr["additions"] + pr["deletions"] for pr in ticket["pull_requests"]
        )
        tickets.append(ticket)

    print(f"[fetch_data] {len(tickets)} tickets collected for CAB {cab_date}")
    return {"raw_tickets": tickets}
''')

# ---------------------------------------------------------------- 6. node 2
md(r'''
## 4. Node 2 — `validate_and_assess_risk`

Deliberately **deterministic** rather than an LLM step: a compliance gate should be auditable
and reproducible, and the release policy is a short list of hard rules.

Policy applied:

1. **QA approval is mandatory** for every release.
2. **Security approval is mandatory** for high-risk releases.
3. **A rollback plan is mandatory** for every release.
4. The ServiceNow change record must be **`Scheduled`**, not sitting in `Draft` / `Assess`.
5. A PR needs at least one reviewer.

Each ticket also gets a `risk_score` (0–100) so the board sees the biggest items first; missing
governance artifacts push the score up and can escalate the effective risk band above whatever
the ticket author self-declared.
''')

code(r'''
REQUIRED_APPROVALS_ALWAYS = ["QA"]
REQUIRED_APPROVALS_HIGH_RISK = ["QA", "Security"]
ACCEPTABLE_CHANGE_STATES = {"Scheduled"}
RISK_BAND_BASE = {"low": 10, "medium": 35, "high": 60}


def _missing_approvals(ticket: Dict[str, Any]) -> List[str]:
    needed = (
        REQUIRED_APPROVALS_HIGH_RISK
        if ticket["risk_level"] == "high"
        else REQUIRED_APPROVALS_ALWAYS
    )
    held = {a.lower() for a in ticket.get("approvals", [])}
    return [a for a in needed if a.lower() not in held]


def _assess(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ticket with blockers, warnings and a numeric risk score attached."""
    blockers: List[str] = []
    warnings: List[str] = []
    score = RISK_BAND_BASE.get(ticket["risk_level"], 35)

    missing = _missing_approvals(ticket)
    if missing:
        blockers.append(f"Missing required approval(s): {', '.join(missing)}")
        score += 20 * len(missing)

    if not ticket.get("rollback_plan_provided"):
        blockers.append("No rollback plan documented")
        score += 25

    change = ticket.get("change_record") or {}
    state = change.get("state", "Unknown")
    if state not in ACCEPTABLE_CHANGE_STATES:
        blockers.append(f"ServiceNow {change.get('change_id', 'record')} is in '{state}', not Scheduled")
        score += 10

    if not ticket.get("pull_requests"):
        warnings.append("No linked pull requests")
    for pr in ticket.get("pull_requests", []):
        if not pr.get("reviewers"):
            warnings.append(f"PR #{pr['number']} ({pr['repo']}) merged with no reviewer")
            score += 10

    if ticket.get("code_churn", 0) > 2000:
        warnings.append(f"Large change: {ticket['code_churn']} lines touched")
        score += 10

    if ticket.get("downtime_minutes", 0) > 0:
        warnings.append(f"Requires {ticket['downtime_minutes']} min of planned downtime")
        score += 5

    score = max(0, min(100, score))
    assessed = dict(ticket)
    assessed["blockers"] = blockers
    assessed["warnings"] = warnings
    assessed["risk_score"] = score
    assessed["effective_risk"] = "high" if score >= 60 else "medium" if score >= 30 else "low"
    assessed["status"] = "flagged" if blockers else "ready"
    return assessed


def validate_and_assess_risk(state: CABState) -> Dict[str, Any]:
    """Split the ticket list into CAB-ready and flagged, scoring risk along the way."""
    assessed = [_assess(t) for t in state["raw_tickets"]]
    assessed.sort(key=lambda t: t["risk_score"], reverse=True)
    flagged = [t for t in assessed if t["status"] == "flagged"]

    print(f"[validate_and_assess_risk] {len(flagged)} flagged / {len(assessed)} total")
    for t in flagged:
        print(f"    {t['ticket_id']}: {'; '.join(t['blockers'])}")

    return {"raw_tickets": assessed, "flagged_tickets": flagged}
''')

# ---------------------------------------------------------------- 7. node 3
md(r'''
## 5. Node 3 — `generate_cab_summary`

The LLM step. It receives the assessed tickets as a JSON payload and writes the board briefing.

Configuration lives in a **`.env` file** at the project root, loaded by `python-dotenv`:

```bash
cp .env.example .env      # then edit
```

`.env` is gitignored, so API keys never reach a commit; `.env.example` is the committed template.
Anything already exported in your shell takes precedence over the file (`override=False`), which
is what makes CI and one-off runs work without editing anything:

```bash
CAB_LLM_BACKEND=offline jupyter nbconvert --execute cab_release_notes_agent.ipynb
```

Recognised keys: `CAB_LLM_BACKEND`, `CAB_DATE`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`,
`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENAI_API_KEY`, `OPENAI_MODEL`.

Four interchangeable backends. Set `CAB_LLM_BACKEND` to pin one explicitly, or leave it unset
and let `resolve_backend()` auto-detect:

| Backend | Selected when | Use it for |
|---|---|---|
| `ollama` | Ollama is reachable on `localhost:11434` | **Local dev — free, private, no rate limits** |
| `openrouter` | `OPENROUTER_API_KEY` set | Production / any hosted model |
| `openai` | `OPENAI_API_KEY` set | Production, OpenAI direct |
| `offline` | nothing else available | CI and reproducible smoke tests |

Auto-detect order is `openrouter → openai → ollama → offline`, so a configured key wins. While
you are **iterating on the prompt, pin local** in `.env` and pay nothing per run:

```ini
CAB_LLM_BACKEND=ollama
```

When the prompt is settled, switch two lines in the same file:

```ini
CAB_LLM_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
```

Nothing else in the graph changes — every backend is a `BaseChatModel`, so
`generate_cab_summary` never branches on which one it got.

OpenRouter speaks the OpenAI chat-completions protocol, so `langchain-openai`'s `ChatOpenAI`
drives it with nothing but a `base_url` and key swap. Pick any model with `OPENROUTER_MODEL`
(e.g. `anthropic/claude-sonnet-4.5`, `openai/gpt-4o-mini`, `google/gemini-2.5-flash`).
''')

code(r'''
# ---- Configuration ---------------------------------------------------------
# All settings come from .env (template: .env.example, gitignored so keys stay out of git).
# override=False means a variable already exported in your shell wins over the file.
try:
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path, override=False)
    print(f"loaded .env from {dotenv_path}" if dotenv_path else "no .env found - using defaults")
except ImportError:
    print("python-dotenv not installed - falling back to shell environment only")

# Override for one run without touching the file:
# os.environ["CAB_LLM_BACKEND"] = "offline"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Benchmarked on an M3 Pro / 18 GB: gemma4:12b (7.6 GB) writes tighter report prose than
# qwen3:14b (9.3 GB) and loads faster. Both call tools correctly. Swap via OLLAMA_MODEL.
DEFAULT_OLLAMA_MODEL = "gemma4:12b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def ollama_available(timeout: float = 1.5) -> bool:
    """Cheap liveness probe so the notebook never hangs when Ollama is not running."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def resolve_backend() -> str:
    """Which backend get_llm() will build. Honours CAB_LLM_BACKEND, else auto-detects."""
    pinned = os.getenv("CAB_LLM_BACKEND", "").strip().lower()
    if pinned == "ollama" and not ollama_available():
        print("[resolve_backend] CAB_LLM_BACKEND=ollama but nothing is listening on "
              f"{OLLAMA_BASE_URL} - falling back to offline. Start it with `ollama serve`.")
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


def _mask(value: Optional[str]) -> str:
    """Show enough of a key to confirm it loaded, never enough to leak it."""
    if not value:
        return "not set"
    return f"{value[:7]}...{value[-4:]}  ({len(value)} chars)"


print(f"backend         : {resolve_backend()}")
print(f"ollama reachable: {ollama_available()}  @ {OLLAMA_BASE_URL}")
print(f"ollama model    : {os.getenv('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL)}")
print(f"OPENROUTER_API_KEY: {_mask(os.getenv('OPENROUTER_API_KEY'))}")
print(f"OPENAI_API_KEY    : {_mask(os.getenv('OPENAI_API_KEY'))}")
print(f"cab date        : {os.getenv('CAB_DATE', '2026-08-20')}")
''')

code(r'''
class OfflineCABChatModel(BaseChatModel):
    """Deterministic stand-in for a hosted chat model, so the notebook runs with no API key.

    Reads the ```json payload``` block out of the incoming prompt and renders the CAB report
    from it. Same interface as ChatOpenAI, so the graph node is agnostic to which is in use.
    """

    @property
    def _llm_type(self) -> str:
        return "offline-cab-renderer"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        match = re.search(r"```json\s*(.*?)\s*```", prompt, re.DOTALL)
        payload = json.loads(match.group(1)) if match else {"cab_date": "unknown", "tickets": []}
        text = self._render(payload)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @staticmethod
    def _render(payload: Dict[str, Any]) -> str:
        cab_date = payload.get("cab_date", "unknown")
        tickets = payload.get("tickets", [])
        ready = [t for t in tickets if t["status"] == "ready"]
        flagged = [t for t in tickets if t["status"] == "flagged"]
        high = [t for t in ready if t["effective_risk"] == "high"]
        standard = [t for t in ready if t["effective_risk"] != "high"]
        downtime = sum(t.get("downtime_minutes", 0) for t in ready)

        L: List[str] = [f"# CAB Briefing — Release Week of {cab_date}", ""]

        L += ["## Executive Summary", ""]
        L += [
            f"- **{len(tickets)} changes** submitted for this CAB window: "
            f"**{len(ready)} ready to approve**, **{len(flagged)} blocked** on governance gaps.",
            f"- **{len(high)}** of the ready changes are high risk and warrant board discussion.",
            f"- Total planned customer downtime across ready changes: **{downtime} minutes**.",
        ]
        if flagged:
            L.append(
                "- Board action required: the blocked items below cannot proceed until their "
                "missing approvals or rollback plans are supplied."
            )
        else:
            L.append("- No governance blockers outstanding.")
        L.append("")

        L += ["## High Risk Items", ""]
        if high:
            for t in high:
                L += [
                    f"### {t['ticket_id']} — {t['title']}",
                    f"- **Owner:** {t['owner']} ({t['team']}) | **System:** {t['system']} "
                    f"| **Risk score:** {t['risk_score']}/100",
                    f"- **Window:** {t['change_record'].get('window', 'TBD')} "
                    f"({t['change_record'].get('change_id', 'no change record')})",
                    f"- **Impact:** {t['description']}",
                    f"- **Rollback:** {t.get('rollback_summary') or 'not documented'}",
                    f"- **Approvals held:** {', '.join(t['approvals']) or 'none'}",
                ]
                for w in t.get("warnings", []):
                    L.append(f"- **Watch:** {w}")
                L.append("")
        else:
            L += ["_No high-risk changes cleared for this window._", ""]

        L += ["## Blocked / Flagged Items", ""]
        if flagged:
            L += ["| Ticket | Title | Owner | Risk | Blockers |", "|---|---|---|---|---|"]
            for t in flagged:
                L.append(
                    f"| {t['ticket_id']} | {t['title']} | {t['owner']} | "
                    f"{t['risk_score']}/100 | {'; '.join(t['blockers'])} |"
                )
            L.append("")
            for t in flagged:
                L.append(f"- **{t['ticket_id']}** — recommended action: {t['blockers'][0]}. Defer until resolved.")
            L.append("")
        else:
            L += ["_Nothing blocked. All submissions met the release policy._", ""]

        L += ["## Standard Releases", ""]
        if standard:
            L += ["| Ticket | Title | System | Owner | Window |", "|---|---|---|---|---|"]
            for t in standard:
                L.append(
                    f"| {t['ticket_id']} | {t['title']} | {t['system']} | {t['owner']} | "
                    f"{t['change_record'].get('window', 'TBD')} |"
                )
            L += ["", "_Recommended for consent-agenda approval._", ""]
        else:
            L += ["_No standard low-risk releases this window._", ""]

        return "\n".join(L)


def get_llm() -> BaseChatModel:
    """Build the chat backend chosen by resolve_backend(). All return a BaseChatModel."""
    backend = resolve_backend()

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        print(f"[get_llm] using Ollama ({model}) at {OLLAMA_BASE_URL}")
        return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_ctx=16384,      # the JSON payload + report needs real context headroom
            reasoning=False,    # qwen3 is a hybrid thinker; keep it terse for report writing
        )

    if backend == "openrouter":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        print(f"[get_llm] using OpenRouter ({model})")
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            # Optional: attributes the request on openrouter.ai leaderboards.
            default_headers={
                "HTTP-Referer": "https://example.internal/cab-agent",
                "X-Title": "CAB Release Notes Agent",
            },
        )

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        print(f"[get_llm] using OpenAI ({model})")
        return ChatOpenAI(model=model, temperature=0)

    print("[get_llm] using OfflineCABChatModel (deterministic, no network)")
    return OfflineCABChatModel()
''')

code(r'''
CAB_SYSTEM_PROMPT = """You are the release manager preparing the Change Advisory Board briefing.

Write a Markdown report for a non-engineering executive audience with EXACTLY these sections,
in this order, using these headings:

# CAB Briefing - Release Week of {cab_date}
## Executive Summary
## High Risk Items
## Blocked / Flagged Items
## Standard Releases

Rules:
- Executive Summary: 3-5 bullets. Counts of ready vs blocked, headline risks, total downtime,
  and the single decision you need from the board.
- High Risk Items: one subsection per ready ticket whose effective_risk is "high". Cover owner,
  maintenance window, business impact, and the rollback plan in plain language.
- Blocked / Flagged Items: a Markdown table of every ticket with status "flagged", then one
  recommended action per ticket. Be explicit that these cannot proceed as-is.
- Standard Releases: a Markdown table of the remaining ready tickets, noted for consent-agenda
  approval.
- Use only facts from the payload. Never invent tickets, dates, names, or approvals.
- No preamble and no closing commentary; output the Markdown report only."""

CAB_HUMAN_PROMPT = """CAB date: {cab_date}

Assessed release tickets:

```json
{payload}
```"""

cab_prompt = ChatPromptTemplate.from_messages(
    [("system", CAB_SYSTEM_PROMPT), ("human", CAB_HUMAN_PROMPT)]
)


def _payload_for_llm(tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim each ticket to the fields the report needs, keeping the prompt small."""
    keep = (
        "ticket_id", "title", "system", "owner", "team", "description",
        "risk_level", "effective_risk", "risk_score", "status",
        "customer_impacting", "downtime_minutes", "approvals",
        "rollback_plan_provided", "rollback_summary",
        "blockers", "warnings", "change_record", "code_churn",
    )
    return [{k: t[k] for k in keep if k in t} for t in tickets]


def generate_cab_summary(state: CABState) -> Dict[str, Any]:
    """Turn the assessed tickets into the executive Markdown briefing."""
    payload = {
        "cab_date": state["cab_date"],
        "tickets": _payload_for_llm(state["raw_tickets"]),
    }
    chain = cab_prompt | get_llm()
    response = chain.invoke(
        {"cab_date": state["cab_date"], "payload": json.dumps(payload, indent=2)}
    )
    report = response.content if isinstance(response.content, str) else str(response.content)

    print(f"[generate_cab_summary] report generated ({len(report)} chars)")
    return {"summary_report": report}
''')

# ---------------------------------------------------------------- 8. graph
md(r'''
## 6. Wire the graph
''')

code(r'''
builder = StateGraph(CABState)

builder.add_node("fetch_data", fetch_data)
builder.add_node("validate_and_assess_risk", validate_and_assess_risk)
builder.add_node("generate_cab_summary", generate_cab_summary)

builder.add_edge(START, "fetch_data")
builder.add_edge("fetch_data", "validate_and_assess_risk")
builder.add_edge("validate_and_assess_risk", "generate_cab_summary")
builder.add_edge("generate_cab_summary", END)

app = builder.compile()
print("graph compiled:", list(app.get_graph().nodes))
''')

code(r'''
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as exc:  # no network / no renderer available
    print(f"PNG render unavailable ({type(exc).__name__}: {exc})\nMermaid source:\n")
    print(app.get_graph().draw_mermaid())
''')

# ---------------------------------------------------------------- 9. run
md(r'''
## 7. Run it
''')

code(r'''
from IPython.display import Markdown

initial_state: CABState = {
    "cab_date": os.getenv("CAB_DATE", "2026-08-20"),
    "raw_tickets": [],
    "flagged_tickets": [],
    "summary_report": "",
}

final_state = app.invoke(initial_state)

print("\n" + "=" * 78 + "\n")
print(final_state["summary_report"])
''')

code(r'''
# Same report, rendered.
display(Markdown(final_state["summary_report"]))
''')

md(r'''
## 8. Inspect the intermediate state

The report is the deliverable, but `flagged_tickets` is the machine-readable output that a real
deployment would push back to Jira as comments or to Slack as a nudge to the ticket owners.
''')

code(r'''
print(f"CAB date        : {final_state['cab_date']}")
print(f"Tickets fetched : {len(final_state['raw_tickets'])}")
print(f"Tickets flagged : {len(final_state['flagged_tickets'])}\n")

print(f"{'TICKET':<10} {'SCORE':>5}  {'STATUS':<8} {'RISK':<7} TITLE")
print("-" * 96)
for t in final_state["raw_tickets"]:
    print(f"{t['ticket_id']:<10} {t['risk_score']:>5}  {t['status']:<8} "
          f"{t['effective_risk']:<7} {t['title'][:52]}")

print("\nBlockers by ticket:")
for t in final_state["flagged_tickets"]:
    print(f"\n  {t['ticket_id']} — {t['title']}")
    for b in t["blockers"]:
        print(f"    [BLOCKER] {b}")
    for w in t["warnings"]:
        print(f"    [warn]    {w}")
''')

code(r'''
# Stream the same run to watch each node land.
for step in app.stream(initial_state, stream_mode="updates"):
    for node, update in step.items():
        touched = ", ".join(f"{k}({len(v) if isinstance(v, (list, str)) else 1})"
                            for k, v in update.items())
        print(f"-> {node:<26} updated: {touched}")
''')

# ---------------------------------------------------------------- 10. outro
md(r'''
## 9. Does the local model hold up? Structured output + tool calling

Markdown prose is the easy job. The two capabilities that decide whether a local model can carry
a real agent are **structured output** (typed objects, not text you have to regex) and **tool
calling**. Both are exercised below against whichever backend is active, so you can compare a
local run to a hosted one on identical inputs.

`with_structured_output()` on Ollama uses JSON-schema constrained decoding, so the *shape* is
guaranteed by the sampler rather than by the model's good intentions. A 12B model is as reliable
as a frontier model at conforming to a schema. Where it is still weaker is **judgement** — the
wording inside the fields, not the structure around them.

Measured on this machine (M3 Pro / 18 GB), one flagged ticket, warm model:

| | `gemma4:12b` (7.6 GB) | `qwen3:14b` (9.3 GB) |
|---|---|---|
| Structured output | 14–17 s | 18–20 s |
| Tool calling (2 parallel calls) | 4.6 s | 6.1 s |
| Correct tool + args | yes | yes |
| Honoured "one sentence" | on `rationale` only | no — 514 chars |

Both are good enough for the pipeline. `gemma4:12b` is the default here because it is smaller,
faster, and tighter in the field that ends up in front of the board.

Three schema lessons came out of getting this to work locally, all encoded below. They cost
nothing on a hosted model and are what makes a 12B model usable:

1. **Enums must be `Literal`, not prose.** Described in a `description`, the model returned
   `"REJECT / DEFER"`. As a `Literal` it becomes a JSON-schema enum and an invalid value is
   unrepresentable.
2. **Never use `max_length` for prose.** Constrained decoding truncates mid-word rather than
   inducing brevity — qwen3 emitted a sentence cut off at `"and the lack "`. Ask for brevity in
   the description, then normalise afterwards with a validator.
3. **Field names are part of the prompt.** Called `owner_action`, the field got
   `"Sam Okonkwo / Identity"` — the model read "owner" out of the *name* and grabbed the
   matching payload field. Renamed to `next_step`, the same model returns a real imperative.
''')

code(r'''
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _first_sentence(text: str) -> str:
    """Clip model prose to its first sentence.

    Local models reliably honour the *schema* but not prose-length *instructions*: gemma4:12b
    keeps appending an unasked-for "CAB Note: ..." paragraph, and qwen3:14b writes a 514-char
    'one sentence'. A length cap in the schema is the wrong fix (constrained decoding truncates
    mid-word), so normalise after generation instead. Cheap, deterministic, and a no-op on a
    model that already complied.
    """
    first_para = text.strip().split("\n\n")[0].strip()
    match = re.search(r"^(.+?[.!?])(\s|$)", first_para, re.DOTALL)
    return (match.group(1) if match else first_para).strip()


class BoardRecommendation(BaseModel):
    """A per-ticket recommendation for the CAB to vote on.

    Note the Literal on `decision`. Describing the allowed values in prose is NOT enough for a
    local model - gemma4:12b returned "REJECT / DEFER" when the options lived in the field
    description. As a Literal they become an enum in the JSON schema, so constrained decoding
    makes an invalid value unrepresentable rather than merely discouraged.

    Also note what is absent: no `max_length`. Under constrained decoding a length cap truncates
    the string mid-word instead of making the model concise - qwen3:14b hit a 200-char cap and
    emitted a cut-off sentence. Ask for brevity in the description, then validate it.
    """

    ticket_id: str = Field(description="The release ticket identifier, e.g. REL-1044")
    decision: Literal["approve", "approve_with_conditions", "defer", "reject"]
    rationale: str = Field(
        description="Exactly ONE short sentence, under 30 words, for a non-technical board member"
    )
    # Named `next_step`, NOT `owner_action`: with the field called `owner_action` the model read
    # "owner" out of the field name, found `owner`/`team` in the payload, and answered
    # "Sam Okonkwo / Identity" instead of an action. Field names are part of the prompt.
    next_step: str = Field(
        description=(
            "An imperative instruction stating what must be DONE to unblock this ticket. "
            "Start with a verb. Example: 'Document and attach a rollback plan, then move the "
            "change to Scheduled.' Never output a person's name or a team name."
        )
    )

    _clip = field_validator("rationale", "next_step")(
        classmethod(lambda cls, v: _first_sentence(v))
    )


def recommend(ticket: Dict[str, Any]) -> BoardRecommendation:
    """Ask the active model for a typed recommendation on one ticket."""
    structured = get_llm().with_structured_output(BoardRecommendation)
    return structured.invoke(
        "You are a release manager advising a Change Advisory Board.\n"
        "Recommend a decision for this change request.\n\n"
        f"```json\n{json.dumps(_payload_for_llm([ticket])[0], indent=2)}\n```"
    )


if resolve_backend() == "offline":
    print("Offline renderer does not implement structured output - "
          "set CAB_LLM_BACKEND to 'ollama' (or a hosted backend) to run this cell.")
else:
    for ticket in final_state["flagged_tickets"]:
        rec = recommend(ticket)
        print(f"{rec.ticket_id}  [{rec.decision}]")
        print(f"    why : {rec.rationale}")
        print(f"    next: {rec.next_step}\n")
''')

code(r'''
from langchain_core.tools import tool


@tool
def lookup_change_freeze(cab_date: str) -> str:
    """Check whether a date falls inside a corporate change-freeze window."""
    frozen = {"2026-12-22": "Year-end freeze", "2026-08-20": "No freeze in effect"}
    return frozen.get(cab_date, "No freeze in effect")


@tool
def count_open_incidents(system: str) -> int:
    """Return the number of open production incidents for a given system."""
    return {"core-backend": 2, "auth-service": 1, "payments": 0, "web-ui": 0}.get(system, 0)


if resolve_backend() == "offline":
    print("Offline renderer does not implement tool calling - pick another backend to run this.")
else:
    tool_llm = get_llm().bind_tools([lookup_change_freeze, count_open_incidents])
    reply = tool_llm.invoke(
        "Before I approve the core-backend release on 2026-08-20, check whether that date is in "
        "a change freeze and how many open incidents core-backend has. Use the tools."
    )
    if reply.tool_calls:
        for call in reply.tool_calls:
            print(f"tool call -> {call['name']}({call['args']})")
    else:
        print("Model answered without calling tools:\n", reply.content[:400])
''')

md(r'''
## 10. Going from POC to production

| Mock | Replace with |
|---|---|
| `fetch_jira_issues` | `jira.search_issues(f'fixVersion = "CAB-{cab_date}" AND status = "Ready for Release"')` |
| `fetch_github_pulls` | `GET /search/issues?q={ticket_id}+is:pr+is:merged` |
| `fetch_servicenow_change` | `GET /api/now/table/change_request?sysparm_query=correlation_id={ticket_id}` |
| `OfflineCABChatModel` | Set `OPENROUTER_API_KEY` (+ optional `OPENROUTER_MODEL`); `get_llm()` switches with no other change |

Natural next steps once the integrations are real:

- **Conditional edge** after `validate_and_assess_risk` → route to a `request_missing_artifacts`
  node that comments on flagged Jira tickets, then loops back.
- **Checkpointer** (`MemorySaver` / Postgres) so a CAB run can be paused for human review and
  resumed with the board's edits.
- **`interrupt_before=["generate_cab_summary"]`** to let the release manager amend the assessment
  before the report is written.
- **Structured output** — bind a Pydantic schema to the LLM so the report ships alongside typed
  per-ticket recommendations rather than Markdown alone.
''')

nb = {
    "cells": [
        {
            "cell_type": kind,
            "id": f"cell-{i:02d}",
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {}),
        }
        for i, (kind, src) in enumerate(CELLS)
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "cab_release_notes_agent.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} ({len(CELLS)} cells)")

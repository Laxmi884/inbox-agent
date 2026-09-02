"""The versioned behaviour layer (spec section 5.2).

The agent's policy - prompt, taxonomy, rules of engagement - is versioned
separately from its learned preferences, so any past run can be reproduced
against the exact policy that produced it. Context Hub is the remote of record;
a committed local file is the fallback so the notebook runs offline.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import Settings

LOCAL_POLICY = Path(__file__).parent / "policies" / "default.md"


@dataclass(frozen=True)
class Policy:
    text: str
    version: str
    source: Literal["context_hub", "local"]


def _pull_from_context_hub(settings: Settings) -> Policy:
    """Pull the policy skill at the configured tag. Raises if unavailable.

    A blank tag means the latest commit. Context Hub resolves a commit hash or
    nothing at all - there is no branch-like ref, so the "dev" this shipped with
    404'd on every run and load_policy quietly fell back to the local file. That
    fallback is correct, and it is exactly what hid the misconfiguration: the
    agent reported `local:...` while looking configured for the hub.

    Not pinning by default costs nothing in reproducibility, because the audit
    record stores the RESOLVED `hub:<commit>` of whatever actually ran. Set the
    tag to a commit hash only to force an older policy deliberately.
    """
    from langsmith import Client

    ctx = Client().pull_skill(settings.context_hub_skill,
                              version=settings.context_hub_tag or None)
    files = getattr(ctx, "files", {}) or {}
    for name in ("POLICY.md", "AGENTS.md", "SKILL.md"):
        if name in files:
            content = files[name]
            text = getattr(content, "content", content)
            commit = getattr(ctx, "commit_hash", settings.context_hub_tag)
            return Policy(text=text, version=f"hub:{commit}", source="context_hub")
    raise RuntimeError(
        f"skill {settings.context_hub_skill!r} has no POLICY.md/AGENTS.md/SKILL.md"
    )


def load_policy(settings: Settings, *, allow_remote: bool = True) -> Policy:
    """Context Hub if reachable and configured, else the committed local file."""
    if allow_remote and os.getenv("LANGSMITH_API_KEY"):
        try:
            return _pull_from_context_hub(settings)
        except Exception as exc:
            print(f"[policy] Context Hub unavailable ({exc}); using local policy.")

    text = LOCAL_POLICY.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return Policy(text=text, version=f"local:{digest}", source="local")

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
    # True when the hub's POLICY.md and the committed policies/default.md have
    # diverged. Carried on the object rather than only printed, so the bot's
    # startup banner can say it too: a warning that scrolls past on a long
    # startup is the same as no warning. Always False for a local-only load -
    # with no hub in play there are not two copies to disagree.
    drifted: bool = False


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
            return Policy(text=text, version=f"hub:{commit}",
                          source="context_hub", drifted=_drifted(text))
    raise RuntimeError(
        f"skill {settings.context_hub_skill!r} has no POLICY.md/AGENTS.md/SKILL.md"
    )


def _drifted(remote_text: str) -> bool:
    """Has the committed fallback diverged from what the hub is serving?

    The local file cannot be deleted - the content tests read it with
    allow_remote=False and the notebook has to run with no key and no network -
    so two copies of one text exist by construction. The only real question is
    whether they can disagree unnoticed, and that is exactly how the 404 above
    survived: the agent ran on the local file for weeks while looking wired to
    the hub.

    This is the one moment both texts are in hand, so the check costs a file
    read and no network. Whitespace at the edges is not drift; a push round-trip
    can add or drop a trailing newline and that is not a policy change.
    """
    try:
        local = LOCAL_POLICY.read_text(encoding="utf-8")
    except OSError:
        return False
    if local.strip() == (remote_text or "").strip():
        return False
    print(f"[policy] DRIFT: the hub policy and {LOCAL_POLICY.name} differ. "
          f"Running the HUB version (it is what the audit record names). "
          f"Push {LOCAL_POLICY} to the hub, or pull it down, to bring them "
          f"back into step.")
    return True


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

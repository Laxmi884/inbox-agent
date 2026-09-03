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
        # Mailbox-affecting safety settings: additive to the deny-list, and
        # the sink an operator checks when they think nothing happened. Both
        # are spec 1.4's failure applied to a different variable - a value
        # set in the shell and lost on restart - so both get a row for the
        # same reason INBOX_GMAIL and INBOX_DRY_RUN do.
        _setting("INBOX_FORBIDDEN_ACTIONS", ", ".join(sorted(s.forbidden_actions))),
        _setting("INBOX_AUDIT_LOG", s.audit_log),
    ]

    # Resolved, not configured. On a machine whose Ollama has died these differ,
    # and the resolved one is what the store is actually doing.
    #
    # resolve_embeddings() raises RuntimeError when INBOX_EMBEDDINGS is pinned
    # to "ollama" and nothing answers - by design, since "ollama" (unlike
    # "auto") is a request to fail loudly rather than degrade. That is the
    # right contract for the bot and the wrong one for doctor: doctor never
    # lets a fatal *condition* stop it from reporting the *other* eleven rows
    # (see the policy try/except below), so this one call is caught the same
    # way rather than propagating out of run_checks as a bare traceback.
    try:
        resolved = config_mod.resolve_embeddings(s.embeddings)
        degraded = resolved == "none" and s.embeddings == "auto"
        checks.append(Check(
            name="INBOX_EMBEDDINGS", value=resolved, source=source_of("INBOX_EMBEDDINGS"),
            level="warn" if degraded else "ok",
            note=("configured auto, but nothing is listening on Ollama - rules are "
                  "still matched exactly, so nothing is broken" if degraded else "")))
    except RuntimeError as exc:
        checks.append(Check("INBOX_EMBEDDINGS", "unavailable",
                            source_of("INBOX_EMBEDDINGS"), "fatal", str(exc)))

    # INBOX_TG_TOKEN is a secret key; pass its raw value straight to
    # _setting() so mask() sees the true absence and returns "not set"
    # plainly. Pre-substituting the string "not set" here would hand mask()
    # a truthy 7-character placeholder and it would render as a masked
    # secret - "<redacted> (7 chars)" - which is exactly the shape a real
    # loaded token has. On a screen whose job is fast triage of "is a token
    # even loaded?", that lie is the same failure this module exists to
    # catch. INBOX_TG_CHAT_ID is not a secret and is never masked, so
    # substituting a placeholder for display is safe there.
    checks.append(_setting(
        "INBOX_TG_TOKEN", s.tg_token,
        level="fatal" if not s.tg_token else "ok",
        note="the bot refuses to start without it" if not s.tg_token else ""))
    checks.append(_setting(
        "INBOX_TG_CHAT_ID", s.tg_chat_id or "not set",
        level="fatal" if not s.tg_chat_id else "ok",
        note="the bot refuses to start without it" if not s.tg_chat_id else ""))

    for key, path in (("INBOX_GOOGLE_CREDENTIALS", s.google_credentials),
                      ("INBOX_GOOGLE_TOKEN", s.google_token)):
        # The note must gate on the same condition as the level - a missing
        # file that INBOX_GMAIL=snapshot never needed is not a problem, and
        # an unmarked "ok" line should not read as if it were.
        needs_it = not path.exists() and s.gmail == "live"
        checks.append(_setting(
            key, path,
            level="fatal" if needs_it else "ok",
            note="missing, and INBOX_GMAIL=live needs it" if needs_it else ""))

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

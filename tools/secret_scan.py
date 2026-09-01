"""Refuse a commit that contains a credential.

Both leaks in this project were one mistake repeated: a value copied out of
`.env` - which was handled correctly throughout - into a file that then got
committed. A Telegram bot token reached git history that way; a Cohere key sits
in plaintext in an untracked notebook, one `git add .` from the same fate.
`.gitignore` does not help, because the file the value lands in is usually a
file you do want to commit.

Two design rules, both learned from scanners that get switched off:

1. **A scanner that cries wolf gets bypassed, and a bypassed scanner is worse
   than none.** This repo already contains `sk-or-v1-...` in `.env.example` and
   `sk-or-v1-test` in two test files. Anything that blocks those would have been
   disabled on its first day, so placeholders, test fixtures, git shas and
   ordinary prose are refused as candidates before entropy is even considered.
2. **Escape hatches must be visible in review.** The marker is inline
   (`# secret-scan: allow`) rather than a path exclusion list, because a
   reviewer reading the diff sees the marker, and nobody ever reads the
   exclusion list again.

Stdlib only, on purpose: a hook with dependencies is a hook that breaks on a
fresh clone, and it runs on every commit.

Usage:
    python -m tools.secret_scan            # scan what is staged (the hook)
    python -m tools.secret_scan --all      # scan every tracked file
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass

# Inline escape hatch. Both spellings, because the second is what detect-secrets
# users will reach for out of habit.
ALLOW_MARKERS = ("secret-scan: allow", "pragma: allowlist secret")

# Values that look like credentials and are not. Checked before entropy, since
# "sk-or-v1-test" has perfectly respectable entropy and is still a fixture.
_PLACEHOLDER = re.compile(
    r"^(test|example|sample|dummy|fake|changeme|placeholder|your|my|xxx+|todo|"
    r"redacted|none|null|abc123)", re.I)

# A credential is not a sentence, and it is not a hash. Hex-only strings are
# git shas, content hashes and checksums - all of which are long, all of which
# are public, and all of which would otherwise be the noisiest false positive
# in the repository.
_HEX_ONLY = re.compile(r"^[0-9a-f]+$", re.I)

# Field names that say the value is a credential. Narrow on purpose: `key` alone
# matches dictionary code everywhere, and `id` matches half the codebase.
_SECRET_FIELD = (r"(?:api[_-]?key|secret[_-]?key|access[_-]?key|auth[_-]?token|"
                 r"api[_-]?token|bot[_-]?token|password|passwd|secret|token)")


@dataclass(frozen=True)
class Finding:
    path: str
    line_no: int
    kind: str
    excerpt: str


def shannon_entropy(value: str) -> float:
    """Bits per character. A real key is near-uniform over its alphabet; an
    English sentence is not, and neither is a repeated placeholder."""
    if not value:
        return 0.0
    counts = {c: value.count(c) for c in set(value)}
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_credential(value: str, *, min_entropy: float = 3.2) -> bool:
    """Does this string look like a real secret rather than a stand-in?"""
    if len(value) < 20:
        return False
    if "..." in value or "…" in value or "<" in value or "$" in value:
        return False        # documented placeholder, or a shell/format variable
    if _PLACEHOLDER.match(value):
        return False
    if _HEX_ONLY.match(value):
        return False        # git sha, content hash, checksum
    if len(set(value)) < 8:
        return False        # aaaaaaaa..., xxxxxxxx...
    return shannon_entropy(value) >= min_entropy


def _redact(value: str) -> str:
    """Enough to recognise it, never enough to use it.

    A refused commit's message gets pasted into chats and issues. Printing the
    key there would spread it further than the commit would have.
    """
    return f"{value[:4]}…{value[-2:]}" if len(value) > 12 else "…"


# Vendor shapes, checked first because they need no entropy argument: the prefix
# is the evidence. Ordered - openrouter and anthropic before the bare `sk-`.
_VENDOR_RULES: list[tuple[str, re.Pattern]] = [
    ("telegram-bot-token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("openrouter-key", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{32,}")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{24,}")),
    ("openai-key", re.compile(r"\bsk-(?!or-v1-|ant-)[A-Za-z0-9]{32,}")),
    ("langsmith-key", re.compile(r"\blsv2_(?:pt|sk)_[A-Za-z0-9]{24,}")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY")),
]

# A credential with no recognisable prefix, identified by the name of the field
# it is being put into. Handles JSON-escaped notebook source, where the quote is
# \" rather than " - which is exactly where the Cohere key lives.
_ASSIGNMENT = re.compile(
    _SECRET_FIELD + r"""\s*[:=]\s*\\?["']([^"'\\\s]{20,})""", re.I)


def scan_text(text: str, path: str = "") -> list[Finding]:
    """Every credential-looking value in `text`, one Finding per line."""
    findings: list[Finding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(marker in line for marker in ALLOW_MARKERS):
            continue
        hit = None
        for kind, pattern in _VENDOR_RULES:
            m = pattern.search(line)
            if m and _is_credential(m.group(0), min_entropy=2.5):
                hit = (kind, m.group(0))
                break
            if m and kind == "private-key":
                hit = (kind, m.group(0))
                break
        if hit is None:
            m = _ASSIGNMENT.search(line)
            if m and _is_credential(m.group(1)):
                hit = ("assigned-secret", m.group(1))
        if hit is None:
            continue
        kind, value = hit
        findings.append(Finding(
            path=path, line_no=line_no, kind=kind,
            excerpt=line.strip().replace(value, _redact(value))[:120]))
    return findings


def _staged_files() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only",
                          "--diff-filter=ACM"],
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.splitlines() if p.strip()]


def _staged_added_lines(path: str) -> str:
    """Only the lines this commit ADDS.

    Scanning whole files would refuse a commit for a secret that is already in
    history - which is not this hook's job and cannot be fixed by editing the
    commit. Blame belongs to the change in hand.
    """
    out = subprocess.run(["git", "diff", "--cached", "--unified=0", "--", path],
                         capture_output=True, text=True, check=True)
    added = [l[1:] for l in out.stdout.splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    return "\n".join(added)


def main(argv: list[str]) -> int:
    if "--all" in argv:
        paths = subprocess.run(["git", "ls-files"], capture_output=True,
                               text=True, check=True).stdout.splitlines()
        findings = []
        for path in paths:
            try:
                findings += scan_text(open(path, encoding="utf-8",
                                           errors="ignore").read(), path)
            except (IsADirectoryError, FileNotFoundError):
                continue
    else:
        findings = []
        for path in _staged_files():
            findings += scan_text(_staged_added_lines(path), path)

    if not findings:
        return 0

    print("\nCOMMIT REFUSED - a credential is in the staged changes.\n")
    for f in findings:
        print(f"  {f.path}:{f.line_no}  [{f.kind}]")
        print(f"    {f.excerpt}\n")
    print("Move the value into .env (already gitignored) and read it with")
    print("os.environ, then re-stage. If the value is genuinely not a secret,")
    print(f"mark that line: {ALLOW_MARKERS[0]}\n")
    print("To commit anyway: git commit --no-verify")
    print("Do that only for a value you have confirmed is not live - a leaked")
    print("credential is not un-leaked by deleting it in a later commit.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

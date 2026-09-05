#!/bin/bash
# Pre-flight, then the bot. Run by launchd; safe to run by hand.
#
# launchd has no way to express "start only if the configuration is sound", so
# this wrapper is that expression. The clone-and-run spec designed doctor's
# exit code for exactly this: non-zero when any check is fatal, "so Project 2
# can use it as a launchd pre-flight check".
#
# Why it matters more unattended than interactively: a bot started by hand on a
# broken config gets read by the person who started it. One started at login,
# months later, does not. The failure that motivated the whole doctor module
# was a bot that came back on the snapshot in dry-run and went on sending
# digests that looked entirely normal and touched nothing.
#
# INBOX_* is deliberately NOT set here. .env is the only source, because a
# value set in two places is the bug this project keeps re-learning, and a
# plist is exactly the kind of second place nobody looks at again.
set -u

cd "$(dirname "$0")/.." || exit 78          # EX_CONFIG
PY="${INBOX_PYTHON:-python3}"

echo "=== inbox-agent preflight $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
if ! "$PY" -m inbox_agent.doctor; then
    echo "inbox-agent: doctor reports a fatal problem. Not starting." >&2
    exit 78
fi

exec "$PY" -u -m inbox_agent.telegram

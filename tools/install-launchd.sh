#!/bin/bash
# Render the plist for THIS checkout and hand it to launchd.
#
#   bash tools/install-launchd.sh            # render, install, start
#   bash tools/install-launchd.sh --dry-run  # print the plist and stop
#   bash tools/install-launchd.sh --uninstall
#
# The template is rendered rather than committed ready-made because a plist
# holds absolute paths, and a committed one would be correct on exactly one
# machine - the failure mode where a clone silently runs someone else's paths.
set -euo pipefail

LABEL="com.inbox-agent.bot"
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
TEMPLATE="$REPO/tools/launchd/$LABEL.plist.template"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$HOME/Library/Logs/inbox-agent"
DOMAIN="gui/$(id -u)"

case "${1:-}" in
  --uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$TARGET"
    echo "removed $LABEL. The bot is not running unless you started one by hand."
    exit 0
    ;;
esac

# The interpreter running the bot, resolved now rather than looked up from a
# PATH launchd does not have. A venv is honoured if one is active.
PYTHON="${INBOX_PYTHON:-$(command -v python3)}"
[ -x "$PYTHON" ] || { echo "no python3 on PATH" >&2; exit 1; }

render() {
  sed -e "s|__LABEL__|$LABEL|g" \
      -e "s|__REPO__|$REPO|g" \
      -e "s|__PYTHON__|$PYTHON|g" \
      -e "s|__PATH__|$(dirname "$PYTHON"):/usr/bin:/bin:/usr/sbin:/sbin|g" \
      -e "s|__LOGDIR__|$LOGDIR|g" \
      "$TEMPLATE"
}

if [ "${1:-}" = "--dry-run" ]; then
  render
  exit 0
fi

# macOS TCC. A launchd user agent gets no consent grant, so it cannot traverse
# ~/Documents, ~/Desktop, ~/Downloads or iCloud Drive - and the failure is
# neither obvious nor loud: the job spawns, exits 126 "Operation not permitted"
# before a single line of Python runs, and retries every ThrottleInterval
# forever. Found on 2026-09-05 by installing it from ~/Documents/Projects.
#
# Checked by path rather than by probing, because this script runs from a
# terminal, and Terminal usually HAS the grant - so a probe here would succeed
# and tell us nothing about what launchd will be allowed to do.
case "$REPO/" in
  "$HOME/Documents/"*|"$HOME/Desktop/"*|"$HOME/Downloads/"*|*"/Library/Mobile Documents/"*)
    cat >&2 <<MSG
Refusing to install: this checkout is in a macOS TCC-protected directory.

  $REPO

A launchd user agent has no consent grant for ~/Documents, ~/Desktop,
~/Downloads or iCloud Drive. The job would exit 126 "Operation not permitted"
before running any Python, and retry every 60s forever. Two ways out:

  1. Move the checkout somewhere unprotected - ~/dev, ~/src, ~/Projects - and
     re-run this. Cleanest, and it needs no security exception.

  2. Grant Full Disk Access to the interpreter in System Settings > Privacy &
     Security. Note what that means: the grant is to the binary, so giving it
     to a shared interpreter hands FDA to everything else that runs under it.

Until then the bot runs by hand. See CLAUDE.md for the double-fork.
MSG
    exit 1
    ;;
esac

# A running hand-started bot holds the flock, so launchd's copy would exit 3
# every ThrottleInterval until someone noticed. Say so now instead.
if pgrep -f "inbox_agent.telegram" >/dev/null 2>&1; then
  echo "A bot is already running (pid $(pgrep -f 'inbox_agent.telegram' | tr '\n' ' '))."
  echo "Stop it first, or launchd's copy will be refused by the single-instance"
  echo "guard once a minute. Then re-run this."
  exit 1
fi

mkdir -p "$(dirname "$TARGET")" "$LOGDIR"
render > "$TARGET"
plutil -lint "$TARGET" >/dev/null

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$TARGET"
launchctl enable "$DOMAIN/$LABEL"

echo "installed $TARGET"
echo "  python   $PYTHON"
echo "  repo     $REPO"
echo "  boot log $LOGDIR/boot.log        (banner, preflight, tracebacks)"
echo "  run log  \$INBOX_LOG_FILE         (rotating; set it in .env)"
echo
echo "  launchctl print $DOMAIN/$LABEL   # state, exit codes, restart count"
echo "  bash tools/install-launchd.sh --uninstall"

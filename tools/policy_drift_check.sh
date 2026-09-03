#!/usr/bin/env bash
# Warn when the policy is committed without being pushed to Context Hub.
#
# `python tools/push_policy.py --check` has been CI-ready since it was written
# and nothing ran it, because this repo has no CI. The commit is the only
# reliable moment left.
#
# It WARNS, it does not block. That is the design decision recorded in
# push_policy.py's docstring: a forgotten push is not fatal - load_policy keeps
# serving the hub's older copy and prints DRIFT, and the bot's startup banner
# repeats it - so a commit hook that refused the commit would be stricter than
# the system it is protecting, and would get bypassed.
#
# Three ways this stays out of the way:
#   - it runs only when the policy file itself is staged
#   - a missing LANGSMITH_API_KEY is silence, not a warning (a fresh clone has
#     no key and has nothing to push)
#   - a network failure is silence too; being offline is not a policy problem
set -uo pipefail

POLICY="inbox_agent/policies/default.md"

git diff --cached --name-only | grep -qx "$POLICY" || exit 0

# macOS ships no timeout(1); use it when coreutils is present, else run bare and
# lean on the langsmith client's own timeouts.
runner=""
for candidate in timeout gtimeout; do
    if command -v "$candidate" >/dev/null 2>&1; then
        runner="$candidate 20"
        break
    fi
done

output=$($runner python -m tools.push_policy --check 2>&1)
status=$?

# 1 is drift, and the only case worth a word. 0 is identical, 2 is no API key,
# anything else is the network or the hub having a bad day.
if [ "$status" -eq 1 ]; then
    cat >&2 <<EOF

  ---------------------------------------------------------------------
  POLICY DRIFT: $POLICY differs from Context Hub.

$(printf '  %s\n' "$output")

  Committing anyway. Runs will keep using the HUB's older copy until you:

      python tools/push_policy.py

  ---------------------------------------------------------------------

EOF
fi

exit 0

#!/usr/bin/env bash
# Install this repo's git hooks. Run once per clone:  bash tools/install-hooks.sh
#
# The hook lives in .git/hooks, which git does not version, so the logic it
# calls is kept in tools/ where it can be reviewed and tested, and only a thin
# shim is installed.
#
# Two checks, deliberately different in severity: a credential BLOCKS the
# commit, policy drift only warns. See each script's header for why.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
hook="$root/.git/hooks/pre-commit"
chmod +x "$root/tools/policy_drift_check.sh"
cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Refuse a commit that contains a credential. See tools/secret_scan.py.
python -m tools.secret_scan || exit 1

# Warn (never block) if the policy changed without a push. See the script.
"$(git rev-parse --show-toplevel)/tools/policy_drift_check.sh"
HOOK
chmod +x "$hook"
echo "installed $hook"

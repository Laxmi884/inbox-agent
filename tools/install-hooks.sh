#!/usr/bin/env bash
# Install this repo's git hooks. Run once per clone:  bash tools/install-hooks.sh
#
# The hook lives in .git/hooks, which git does not version, so the logic it
# calls is kept in tools/secret_scan.py where it can be reviewed and tested and
# only the two-line shim is installed.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
hook="$root/.git/hooks/pre-commit"
cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Refuse a commit that contains a credential. See tools/secret_scan.py.
exec python -m tools.secret_scan
HOOK
chmod +x "$hook"
echo "installed $hook"

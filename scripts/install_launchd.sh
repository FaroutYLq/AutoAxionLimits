#!/usr/bin/env bash
# Install (or refresh) the unattended daily/weekly launchd jobs for this user.
#   scripts/install_launchd.sh            # install from this checkout
#   scripts/install_launchd.sh --remove   # unload and delete the jobs
# The jobs run scripts/scheduled_run.sh from THIS checkout, which should be a
# dedicated clone kept on master (the script fast-forwards it on every run).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HOME/Library/LaunchAgents"
mkdir -p "$DEST" "$HOME/.aal_bench/scheduler/logs"
for kind in daily weekly; do
    label="com.autoaxionlimits.$kind"
    plist="$DEST/$label.plist"
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    if [[ "${1:-}" == "--remove" ]]; then
        rm -f "$plist"; echo "removed $label"; continue
    fi
    sed -e "s|__ROOT__|$ROOT|g" -e "s|__HOME__|$HOME|g" "$ROOT/scripts/launchd/$label.plist" > "$plist"
    launchctl bootstrap "gui/$(id -u)" "$plist"
    echo "installed $label -> $plist"
done
[[ "${1:-}" == "--remove" ]] || launchctl list | grep autoaxionlimits

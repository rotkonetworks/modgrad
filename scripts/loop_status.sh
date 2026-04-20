#!/usr/bin/env bash
# loop_status.sh — one-screen summary of in-flight training runs
# and recent git commits from the overnight loop. Designed for a
# user checking in after leaving the loop running.
#
# Usage:
#   ./scripts/loop_status.sh
#   watch -n 10 ./scripts/loop_status.sh   # auto-refresh every 10s

set -euo pipefail

cd "$(dirname "$0")/.."

hr() { printf '%.0s─' {1..60}; echo; }

hr
echo "  modgrad overnight-loop status  $(date '+%H:%M:%S')"
hr

echo "▸ running maze processes"
ps -eo pid,etime,pcpu,cmd 2>/dev/null | grep -E "target/release/mazes " | grep -v grep \
  | awk '{printf "  pid=%-7s uptime=%-10s cpu=%-5s %s\n", $1, $2, $3, substr($0, index($0,$4))}' \
  | head -6 || echo "  (none running)"
echo

for log in /tmp/loop_long_maze.log /tmp/sdf_ab/sdf_run.log; do
  if [ -f "$log" ]; then
    echo "▸ $log — last 4 step lines"
    grep -E '^step ' "$log" 2>/dev/null | tail -4 | sed 's/^/  /'
    # Also surface eval block if it's appeared
    if grep -q '^--- Evaluation' "$log" 2>/dev/null; then
      echo "  --- eval:"
      grep -A1 -E '^(First step|Per-step)' "$log" | sed 's/^/    /' | head -8
    fi
    echo
  fi
done

echo "▸ last 8 git commits on main"
git log --oneline -n 8 | sed 's/^/  /'
echo

echo "▸ workspace test summary (cached — run cargo test for fresh)"
# Just the headlines
for crate in modgrad-codec modgrad-ctm modgrad-device modgrad-prompts; do
  if [ -f "target/debug/deps/${crate//-/_}.d" ] || \
     [ -f "target/release/deps/${crate//-/_}.d" ]; then
    echo "  $crate: compiled"
  else
    echo "  $crate: not yet built in this tree"
  fi
done

hr
echo "  live viz: feh --reload 1 /tmp/mazes_live/combined.ppm"
echo "  dashboard: watch -n 10 ./scripts/loop_status.sh"
hr

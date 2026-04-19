#!/usr/bin/env bash
# autoresearch-check.sh — enforces program.md's "CANNOT change" contract.
#
# Runs against a commit range and exits non-zero if any file in the
# read-only set was touched. The driving agent calls this after its
# `git commit` so accidental edits to the ground-truth evaluation
# harness are caught before the run, not hours later when val_bpb
# rows in results.tsv turn out to be incomparable.
#
# Usage:
#   ./scripts/autoresearch-check.sh HEAD~1           # check last commit
#   ./scripts/autoresearch-check.sh <base>..<head>   # check range
#
# Exit codes:
#   0 — clean, no read-only files touched
#   1 — violation, agent should `git reset --hard HEAD~1` and retry
#   2 — usage error

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <commit-ish-or-range>" >&2
  exit 2
fi

RANGE_INPUT="$1"
# If input contains ".." assume it's already a range, otherwise treat as
# "from <commit> to HEAD" — matches the usual agent idiom of calling
# with HEAD~1 after a fresh commit.
if [[ "$RANGE_INPUT" == *..* ]]; then
  RANGE="$RANGE_INPUT"
else
  RANGE="${RANGE_INPUT}..HEAD"
fi

# Patterns for read-only files. Keep in sync with program.md's
# "CANNOT change" section. Ordered roughly by how load-bearing the
# file is — metric + summary formatter first, since silently changing
# them invalidates the whole experimental record.
READ_ONLY_PATTERNS=(
  '^crates/modgrad-training/src/metrics\.rs$'
  '^crates/modgrad-training/src/autoresearch\.rs$'
  '^crates/modgrad-training/src/checkpoint_bundle\.rs$'
  '^crates/modgrad-device/'
  '^examples/mazes/src/maze_gen\.rs$'
  '^program\.md$'
  '^val\.txt$'
  '^Cargo\.toml$'
  '^Cargo\.lock$'
  '^scripts/autoresearch-check\.sh$'
)

# Join patterns with | for a single grep -E call.
PATTERN=$(IFS='|'; echo "${READ_ONLY_PATTERNS[*]}")

changed=$(git diff --name-only "$RANGE" | { grep -E "$PATTERN" || true; })

if [[ -n "$changed" ]]; then
  echo "VIOLATION: touched read-only files in $RANGE:" >&2
  echo "$changed" | sed 's/^/  - /' >&2
  echo "" >&2
  echo "These files are part of the ground-truth evaluation harness or" >&2
  echo "project config. Editing them invalidates the experimental record." >&2
  echo "" >&2
  echo "Recover with: git reset --hard HEAD~1" >&2
  exit 1
fi

echo "OK: no read-only files touched in $RANGE"
exit 0

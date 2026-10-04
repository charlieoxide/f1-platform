#!/usr/bin/env bash
# Bulk-ingest qualifying + race sessions for a list of events into the local
# DB. Each call hits FastF1's live servers the first time (cached after),
# so this takes a few minutes and needs internet.
#
# Usage: bash scripts/ingest_season.sh 2024
set -euo pipefail

YEAR="${1:?Usage: bash scripts/ingest_season.sh <year>}"

# A reasonable first batch: enough races for the predictor to have real
# signal (varied grid positions, varied outcomes) without downloading an
# entire season on a disk-constrained machine. Extend this list later if
# the model needs more data.
EVENTS=(
  "Bahrain"
  "Saudi Arabia"
  "Australia"
  "Japan"
  "China"
  "Miami"
  "Monaco"
  "Canada"
  "Spain"
  "Austria"
  "Great Britain"
)

for event in "${EVENTS[@]}"; do
  for session_type in Q R; do
    echo "== ${YEAR} ${event} ${session_type} =="
    python -m src.ingest --session "${YEAR} ${event} ${session_type}" || \
      echo "!! failed: ${YEAR} ${event} ${session_type} (continuing)"
  done
done

echo "Done. Check disk headroom: df -h /"

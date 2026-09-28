#!/usr/bin/env bash
# Simple local backup: copy the DB to a timestamped file in a second local
# path. Wire this into a cron/systemd timer to run daily.
set -euo pipefail

SRC="data/f1.db"
DEST_DIR="${1:-$HOME/f1-platform-backups}"
mkdir -p "$DEST_DIR"

if [[ -f "$SRC" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  cp "$SRC" "$DEST_DIR/f1_${TS}.db"
  echo "Backed up $SRC -> $DEST_DIR/f1_${TS}.db"
  # keep last 14 backups only
  ls -1t "$DEST_DIR"/f1_*.db | tail -n +15 | xargs -r rm --
else
  echo "No DB found at $SRC, nothing to back up."
fi

#!/usr/bin/env bash
# Run before big installs (Docker builds, new pip packages). Free disk is a
# hard constraint on this machine (~11GB free at time of writing) — check
# `df -h /` before and after.
set -euo pipefail

echo "== Disk before =="
df -h / | tail -n1

echo "== apt cache =="
sudo apt-get clean
sudo apt-get autoremove --purge -y

echo "== old snap revisions (often multi-GB) =="
snap list --all 2>/dev/null | awk '/disabled/{print $1, $3}' | while read -r name rev; do
  echo "removing $name revision $rev"
  sudo snap remove "$name" --revision="$rev" || true
done

echo "== docker (if installed) =="
if command -v docker >/dev/null 2>&1; then
  docker system prune -a --volumes -f || true
fi

echo "== pip cache =="
pip cache purge || true

echo "== FastF1 cache (safe to clear, will re-download on next use) =="
rm -rf ./data/raw/fastf1_cache/* 2>/dev/null || true

echo "== Disk after =="
df -h / | tail -n1

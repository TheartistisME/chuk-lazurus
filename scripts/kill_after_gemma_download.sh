#!/usr/bin/env bash
# Monitor the in-flight Gemma download at /mnt/c/...ghostty-build... cache.
# When the .incomplete file disappears (download finished + renamed to blob
# hash), kill the demo process so it doesn't sit loading weights from the
# slow Windows-side cache.
#
# Usage:
#   bash scripts/kill_after_gemma_download.sh
#
# Exits cleanly after the kill or if the demo process isn't running.

set -euo pipefail

CACHE_BLOBS="/mnt/c/users/jehma/desktop/ghostty-build/vendetta-bundle/xdg/cache/huggingface/hub/models--google--gemma-4-E2B-it/blobs"
DEMO_PATTERN="\.venv/bin/python3 examples/inference/demo_c_aus3000"

echo "[watcher] cache blobs dir: $CACHE_BLOBS"
echo "[watcher] demo pattern:    $DEMO_PATTERN"

# Find demo PID up front
PID=$(pgrep -f "$DEMO_PATTERN" | head -1 || true)
if [ -z "$PID" ]; then
  echo "[watcher] no demo process matches pattern — nothing to kill. Exiting."
  exit 0
fi
echo "[watcher] tracking demo pid=$PID"
echo

LAST_SIZE=0
START_TIME=$(date +%s)

while true; do
  # Is demo still running?
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[watcher] demo pid $PID exited on its own. No kill needed."
    exit 0
  fi

  # Any .incomplete files still present?
  INC=$(ls "$CACHE_BLOBS"/*.incomplete 2>/dev/null | head -1 || true)

  if [ -z "$INC" ]; then
    # No .incomplete → download is done (or never started)
    ELAPSED=$(( $(date +%s) - START_TIME ))
    echo "[watcher] .incomplete file is gone — download finished (watched for ${ELAPSED}s)."
    echo "[watcher] killing demo pid $PID to free it from the slow Windows cache load"
    kill -TERM "$PID" || true
    sleep 3
    if kill -0 "$PID" 2>/dev/null; then
      echo "[watcher] TERM didn't land, sending KILL"
      kill -KILL "$PID" || true
    fi
    echo "[watcher] done. Now run scripts/run_aus3000_demo_fast.sh to use the Linux-side cache."
    exit 0
  fi

  # Still downloading — progress line
  SIZE=$(stat -c%s "$INC" 2>/dev/null || echo 0)
  DELTA=$(( SIZE - LAST_SIZE ))
  GB=$(awk -v s="$SIZE" 'BEGIN{printf "%.2f", s/1024/1024/1024}')
  RATE=$(awk -v d="$DELTA" 'BEGIN{printf "%.1f", d/1024/1024/3}')   # interval is 3s below
  TS=$(date +%H:%M:%S)
  echo "[$TS] $GB GB   (+${RATE} MB/s)   file: $(basename "$INC")"
  LAST_SIZE=$SIZE

  sleep 3
done

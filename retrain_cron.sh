#!/bin/bash
# Nightly retrain cron job
# Pulls clips from Pi, labels via Unifi, retrains, deploys if accuracy >= 90%

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/retrain_$(date +%Y%m%d_%H%M%S).log"

# Use arm64 venv
PYTHON="$SCRIPT_DIR/venv-arm64/bin/python"

echo "=== Retrain started at $(date) ===" | tee "$LOG_FILE"

cd "$SCRIPT_DIR"
"$PYTHON" retrain.py --cron --unifi-ip 192.168.1.1 >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "=== Retrain finished at $(date) with exit code $EXIT_CODE ===" | tee -a "$LOG_FILE"

# Keep last 30 days of logs
find "$LOG_DIR" -name "retrain_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE

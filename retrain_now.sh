#!/bin/bash
# Manual retrain — same as cron but with live output
cd "$(dirname "$0")"
exec ./venv-arm64/bin/python retrain.py --unifi-ip 192.168.1.1 "$@"

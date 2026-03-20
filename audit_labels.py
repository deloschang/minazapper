#!/usr/bin/env python3
"""
Audit training labels against Unifi ground truth.

Scans training_data/negative/ for clips that overlap with Unifi bark events
and moves them to training_data/mina/. Prevents mislabeled negatives from
poisoning the model.

Usage:
    python audit_labels.py                    # Dry run
    python audit_labels.py --fix              # Move mislabeled clips
    python audit_labels.py --tolerance 5000   # Wider matching window
"""

import re
import shutil
import argparse
from datetime import datetime
from pathlib import Path

from download_clips import UnifiProtectClient

TRAINING_DIR = Path("training_data")
POSITIVE_DIR = TRAINING_DIR / "mina"
NEGATIVE_DIR = TRAINING_DIR / "negative"

# Matches both detector clips and unifi-downloaded clips
CLIP_RE = re.compile(r"(?:S\d+_)?(?:unifi_)?(\d{8}_\d{6})")


def parse_timestamp(filename):
    """Extract epoch ms from any clip filename."""
    m = CLIP_RE.search(filename)
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    return int(dt.timestamp() * 1000)


def main():
    parser = argparse.ArgumentParser(description="Audit negative labels against Unifi bark events")
    parser.add_argument("--unifi-ip", default="192.168.1.1")
    parser.add_argument("--tolerance", type=int, default=5000,
                        help="Matching tolerance in ms (default: 5000)")
    parser.add_argument("--fix", action="store_true",
                        help="Move mislabeled clips from negative to mina")
    parser.add_argument("--days", type=int, default=30,
                        help="How far back to fetch Unifi events (default: 30)")
    args = parser.parse_args()

    # Parse all negative clips
    neg_clips = []
    for f in sorted(NEGATIVE_DIR.iterdir()):
        if f.suffix != ".wav":
            continue
        ts_ms = parse_timestamp(f.name)
        if ts_ms:
            neg_clips.append({"path": f, "ts_ms": ts_ms})

    if not neg_clips:
        print("No negative clips to audit.")
        return

    min_ts = min(c["ts_ms"] for c in neg_clips)
    max_ts = max(c["ts_ms"] for c in neg_clips)
    print(f"Auditing {len(neg_clips)} negative clips")
    print(f"Time range: {datetime.fromtimestamp(min_ts/1000)} to {datetime.fromtimestamp(max_ts/1000)}")

    # Fetch Unifi bark events
    client = UnifiProtectClient()
    client.login_local(args.unifi_ip)

    all_events = []
    offset = 0
    # Extend range by 1 day on each side to catch edge cases
    query_start = min_ts - 86_400_000
    query_end = max_ts + 86_400_000

    print(f"Fetching Unifi bark events...")
    while True:
        events = client.get_events(start_ts=query_start, end_ts=query_end, limit=100, offset=offset)
        if not events:
            break
        all_events.extend(events)
        if len(events) < 100:
            break
        offset += 100

    bark_events = [e for e in all_events if "alrmBark" in e.get("smartDetectTypes", [])]
    print(f"Found {len(bark_events)} Unifi bark events in range\n")

    if not bark_events:
        print("No bark events — all negatives are correct.")
        return

    # Load manual labels — skip anything human-reviewed
    manual_labels = {}
    manual_file = Path("manual_labels.json")
    if manual_file.exists():
        import json
        with open(manual_file) as f:
            raw = json.load(f)
        for sid, info in raw.items():
            manual_labels[sid] = info.get("label") if isinstance(info, dict) else info
        print(f"Respecting {len(manual_labels)} manual labels (won't override)\n")

    # Check each negative clip against bark events (skip manually labeled)
    import re
    session_re = re.compile(r"(S\d+)_")
    mislabeled = []
    for clip in neg_clips:
        # Never override human labels
        sm = session_re.match(clip["path"].name)
        if sm and sm.group(1) in manual_labels:
            continue

        for e in bark_events:
            if (e["start"] - args.tolerance) <= clip["ts_ms"] <= (e["end"] + args.tolerance):
                bark_time = datetime.fromtimestamp(e["start"] / 1000).strftime("%I:%M:%S %p")
                clip_time = datetime.fromtimestamp(clip["ts_ms"] / 1000).strftime("%I:%M:%S %p")
                mislabeled.append({
                    "clip": clip,
                    "event": e,
                    "bark_time": bark_time,
                    "clip_time": clip_time,
                })
                break

    if not mislabeled:
        print(f"All {len(neg_clips)} negative clips are correct (no overlap with bark events).")
        return

    print(f"Found {len(mislabeled)} mislabeled clips in negative/ that Unifi says are barking:\n")
    for m in mislabeled:
        print(f"  {m['clip']['path'].name}")
        print(f"    Clip time:  {m['clip_time']}")
        print(f"    Bark event: {m['bark_time']}")
        print()

    if args.fix:
        POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
        for m in mislabeled:
            src = m["clip"]["path"]
            dst = POSITIVE_DIR / src.name
            shutil.move(str(src), str(dst))
        print(f"Moved {len(mislabeled)} clips from negative/ to mina/")
    else:
        print(f"Run with --fix to move these to mina/")


if __name__ == "__main__":
    main()

"""
Auto-label detection clips using Unifi Protect's smart audio detection.

For each clip saved by bark_detector.py (the ones that triggered Pushover),
cross-references with Unifi's AI bark classification:
  - If Unifi also detected barking at that time → training_data/mina/
  - If Unifi did NOT detect barking → training_data/negative/
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone

from download_clips import UnifiProtectClient

CLIPS_DIR = Path("false_positives/detection_clips")
POSITIVE_DIR = Path("training_data/mina")
NEGATIVE_DIR = Path("training_data/negative")

# Filename patterns:
#   S001_20260318_010500_living_94%.wav
#   20260318_010102_kitchen_73%.wav
CLIP_RE = re.compile(
    r"(?:S\d+_)?(\d{8}_\d{6})_(\w+)_(\d+)%\.wav"
)


def parse_clip_timestamp(filename):
    """Extract UTC-ish timestamp (ms) and camera name from clip filename."""
    m = CLIP_RE.match(filename)
    if not m:
        return None, None, None
    ts_str, camera, confidence = m.group(1), m.group(2), int(m.group(3))
    from zoneinfo import ZoneInfo
    dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=ZoneInfo("UTC"))
    epoch_ms = int(dt.timestamp() * 1000)
    return epoch_ms, camera, confidence


def fetch_events_for_range(client, start_ms, end_ms):
    """Fetch ALL smartAudioDetect events (not just bark) in the time range."""
    all_events = []
    offset = 0
    batch_size = 100

    while True:
        events = client.get_events(
            start_ts=start_ms,
            end_ts=end_ms,
            limit=batch_size,
            offset=offset,
        )
        if not events:
            break
        all_events.extend(events)
        if len(events) < batch_size:
            break
        offset += batch_size

    return all_events


def clip_matches_bark_event(clip_ts_ms, bark_events, tolerance_ms=3000):
    """Check if clip timestamp falls within any bark event window (with tolerance)."""
    for event in bark_events:
        ev_start = event["start"] - tolerance_ms
        ev_end = event["end"] + tolerance_ms
        if ev_start <= clip_ts_ms <= ev_end:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label detection clips using Unifi bark detection"
    )
    parser.add_argument("--local-ip", help="Console's local IP for direct access")
    parser.add_argument("--2fa-code", dest="tfa_code", help="2FA code from email")
    parser.add_argument("--send-2fa", action="store_true", help="Just send 2FA code and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without moving files")
    parser.add_argument("--clips-dir", default=str(CLIPS_DIR), help="Directory with detection clips")
    parser.add_argument("--tolerance", type=int, default=3000,
                        help="Timestamp matching tolerance in ms (default: 3000)")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        print(f"Clips directory not found: {clips_dir}")
        return

    # Parse all clip filenames
    clips = []
    for f in sorted(clips_dir.iterdir()):
        if not f.suffix == ".wav":
            continue
        ts_ms, camera, confidence = parse_clip_timestamp(f.name)
        if ts_ms is None:
            print(f"  Skipping (can't parse): {f.name}")
            continue
        clips.append({"path": f, "ts_ms": ts_ms, "camera": camera, "confidence": confidence})

    if not clips:
        print("No clips found to label.")
        return

    print(f"Found {len(clips)} clips to label.")

    # Determine time range
    min_ts = min(c["ts_ms"] for c in clips) - 60_000
    max_ts = max(c["ts_ms"] for c in clips) + 60_000
    min_dt = datetime.fromtimestamp(min_ts / 1000)
    max_dt = datetime.fromtimestamp(max_ts / 1000)
    print(f"Time range: {min_dt} to {max_dt}")

    # Authenticate to Unifi
    client = UnifiProtectClient()
    if args.local_ip:
        client.login_local(args.local_ip, tfa_code=args.tfa_code, send_2fa_only=args.send_2fa)
    else:
        client.login()

    # Fetch events
    print("\nFetching Unifi smart audio events...")
    events = fetch_events_for_range(client, min_ts, max_ts)
    bark_events = [e for e in events if "alrmBark" in e.get("smartDetectTypes", [])]
    print(f"  Total audio events: {len(events)}")
    print(f"  Bark events: {len(bark_events)}")

    # Print bark event windows for context
    for ev in sorted(bark_events, key=lambda e: e["start"]):
        ev_start = datetime.fromtimestamp(ev["start"] / 1000)
        ev_end = datetime.fromtimestamp(ev["end"] / 1000)
        cam_desc = ev.get("description", {}).get("messageRaw", "")
        print(f"    Bark: {ev_start} - {ev_end}  {cam_desc}")

    # Label each clip
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    positive = 0
    negative = 0

    print(f"\nLabeling clips (tolerance: {args.tolerance}ms)...")
    for clip in clips:
        is_bark = clip_matches_bark_event(clip["ts_ms"], bark_events, args.tolerance)
        dest_dir = POSITIVE_DIR if is_bark else NEGATIVE_DIR
        label = "POSITIVE" if is_bark else "NEGATIVE"
        dest = dest_dir / clip["path"].name

        if is_bark:
            positive += 1
        else:
            negative += 1

        ts_str = datetime.fromtimestamp(clip["ts_ms"] / 1000).strftime("%H:%M:%S")
        print(f"  {label:8s}  {ts_str}  {clip['camera']:8s}  {clip['confidence']}%  {clip['path'].name}")

        if not args.dry_run:
            shutil.move(str(clip["path"]), str(dest))

    action = "Would move" if args.dry_run else "Moved"
    print(f"\n{action}: {positive} → mina/ (positive), {negative} → negative/ (false positive)")
    print(f"Training data totals:")
    mina_count = len(list(POSITIVE_DIR.glob("*.wav")))
    neg_count = len(list(NEGATIVE_DIR.glob("*.wav")))
    print(f"  mina/: {mina_count} files")
    print(f"  negative/: {neg_count} files")


if __name__ == "__main__":
    main()

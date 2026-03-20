#!/usr/bin/env python3
"""
Play back detection clips by session ID for manual review.

Usage:
    python play_clip.py S052          # Play all clips from session S052
    python play_clip.py S052 --list   # List clips without playing
    python play_clip.py S050-S055     # Play range of sessions
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PI_HOST = "del@192.168.10.10"
PI_KEY = os.path.expanduser("~/.ssh/minazap_pi")
PI_CLIPS = "~/minazap/detection_clips"
LOCAL_CACHE = Path("false_positives/detection_clips")

TRAINING_DIR = Path("training_data")
SEARCH_DIRS = [
    LOCAL_CACHE,
    TRAINING_DIR / "mina",
    TRAINING_DIR / "negative",
]


def find_clips(session_query):
    """Find clips matching session ID(s) across all directories."""
    # Parse range like S050-S055
    if "-" in session_query and session_query.count("S") == 2:
        parts = session_query.split("-")
        start = int(parts[0].lstrip("S"))
        end = int(parts[1].lstrip("S"))
        prefixes = [f"S{str(i).zfill(3)}_" for i in range(start, end + 1)]
    else:
        prefixes = [f"{session_query}_"]

    clips = []
    for search_dir in SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.iterdir()):
            if f.suffix != ".wav":
                continue
            if any(f.name.startswith(p) for p in prefixes):
                # Determine label from directory
                if "mina" in str(search_dir):
                    label = "positive"
                elif "negative" in str(search_dir):
                    label = "negative"
                else:
                    label = "unlabeled"
                clips.append({"path": f, "label": label})

    return clips


def fetch_from_pi(session_query):
    """Fetch clips from Pi when not found locally."""
    # Build glob patterns
    if "-" in session_query and session_query.count("S") == 2:
        parts = session_query.split("-")
        start = int(parts[0].lstrip("S"))
        end = int(parts[1].lstrip("S"))
        sessions = [f"S{str(i).zfill(3)}" for i in range(start, end + 1)]
    else:
        sessions = [session_query]

    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    fetched = []

    for sid in sessions:
        result = subprocess.run(
            ["scp", "-i", PI_KEY, f"{PI_HOST}:{PI_CLIPS}/{sid}_*.wav", str(LOCAL_CACHE)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            for f in sorted(LOCAL_CACHE.iterdir()):
                if f.suffix == ".wav" and f.name.startswith(f"{sid}_"):
                    fetched.append({"path": f, "label": "on pi"})

    if fetched:
        print(f"Fetched {len(fetched)} clips from Pi.\n")

    return fetched


def play_clip(path):
    """Play a WAV file using afplay (macOS) or aplay (Linux)."""
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)])
    else:
        subprocess.run(["aplay", str(path)])


def main():
    parser = argparse.ArgumentParser(description="Play detection clips by session ID")
    parser.add_argument("session", help="Session ID (e.g. S052) or range (e.g. S050-S055)")
    parser.add_argument("--list", action="store_true", help="List clips without playing")
    args = parser.parse_args()

    clips = find_clips(args.session)

    # Auto-fetch from Pi if not found locally
    if not clips:
        clips = fetch_from_pi(args.session)

    if not clips:
        print(f"No clips found for {args.session}")
        return

    print(f"Found {len(clips)} clips for {args.session}:\n")

    for i, clip in enumerate(clips, 1):
        tag = f"[{clip['label']:8s}]"
        print(f"  {i:3d}. {tag}  {clip['path'].name}")

    if args.list:
        return

    print(f"\nPlaying {len(clips)} clips... (Ctrl+C to stop)\n")
    try:
        for i, clip in enumerate(clips, 1):
            tag = f"[{clip['label']:8s}]"
            print(f"  Playing {i}/{len(clips)}: {tag} {clip['path'].name}")
            play_clip(clip["path"])
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

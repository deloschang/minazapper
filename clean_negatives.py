"""
Clean up negative clips:
1. Remove duplicates (same timestamp within 3 seconds = same event from different cameras)
2. Trim all clips to 2 seconds (random offset to keep variety)

Run: python clean_negatives.py
"""

import random
from pathlib import Path

import numpy as np
import soundfile as sf

random.seed(42)

NEGATIVE_DIR = Path("training_data/negative")
SAMPLE_RATE = 16000
TARGET_DURATION = 2.0  # seconds
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)


def get_timestamp_seconds(filename):
    """Extract timestamp as total seconds from filename like 20260315_030608_..."""
    parts = filename.split("_")
    if len(parts) < 2:
        return 0
    date_str, time_str = parts[0], parts[1]
    try:
        h, m, s = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
        d = int(date_str[6:8])
        return d * 86400 + h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return 0


def main():
    clips = sorted(NEGATIVE_DIR.glob("*.wav"))
    print(f"Total clips before cleanup: {len(clips)}")

    # Step 1: Remove duplicates (events within 3 seconds of each other)
    clips_by_ts = []
    for c in clips:
        ts = get_timestamp_seconds(c.stem)
        clips_by_ts.append((ts, c))

    clips_by_ts.sort(key=lambda x: x[0])

    keep = []
    removed_dupes = 0
    for i, (ts, clip) in enumerate(clips_by_ts):
        if i == 0:
            keep.append(clip)
            continue
        prev_ts = clips_by_ts[i - 1][0]
        if abs(ts - prev_ts) <= 3:
            clip.unlink()
            removed_dupes += 1
        else:
            keep.append(clip)

    print(f"Removed {removed_dupes} duplicates (within 3s of each other)")

    # Step 2: Trim all remaining clips to 2 seconds
    trimmed = 0
    removed_empty = 0
    for clip in keep:
        try:
            audio, sr = sf.read(str(clip), dtype="float32")
        except Exception:
            clip.unlink()
            removed_empty += 1
            continue

        if len(audio) < SAMPLE_RATE:  # less than 1 second
            clip.unlink()
            removed_empty += 1
            continue

        if len(audio) > TARGET_SAMPLES:
            # Pick a random 2-second window
            max_start = len(audio) - TARGET_SAMPLES
            start = random.randint(0, max_start)
            audio = audio[start:start + TARGET_SAMPLES]
            sf.write(str(clip), audio, SAMPLE_RATE)
            trimmed += 1

    remaining = list(NEGATIVE_DIR.glob("*.wav"))
    print(f"Trimmed {trimmed} clips to {TARGET_DURATION}s")
    print(f"Removed {removed_empty} empty/too-short clips")
    print(f"Final count: {len(remaining)} negative clips")


if __name__ == "__main__":
    main()

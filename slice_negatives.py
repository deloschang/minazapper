"""
Slice long negative clips into 2-second segments.
Also deduplicates events from multiple cameras (within 3s).

Run after downloading negatives:
    python slice_negatives.py [--input training_data/negative_raw] [--output training_data/negative]
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_DURATION = 2.0  # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
MIN_RMS = 0.005  # skip silent segments


def get_timestamp_seconds(filename):
    """Extract timestamp as total seconds from filename."""
    parts = filename.split("_")
    if len(parts) < 2:
        return 0
    date_str, time_str = parts[0], parts[1]
    try:
        h, m, s = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
        d = int(date_str[6:8])
        m_val = int(date_str[4:6])
        return m_val * 30 * 86400 + d * 86400 + h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data/negative_raw",
                        help="Directory with raw long WAV files")
    parser.add_argument("--output", default="training_data/negative",
                        help="Output directory for 2-second slices")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(input_dir.glob("*.wav"))
    if not clips:
        print(f"No WAV files found in {input_dir}")
        return

    # Deduplicate by timestamp (within 3 seconds = same event)
    clips_by_ts = [(get_timestamp_seconds(c.stem), c) for c in clips]
    clips_by_ts.sort(key=lambda x: x[0])

    deduped = []
    for i, (ts, clip) in enumerate(clips_by_ts):
        if i == 0 or abs(ts - clips_by_ts[i - 1][0]) > 3:
            deduped.append(clip)

    print(f"Total raw clips: {len(clips)}")
    print(f"After dedup: {len(deduped)}")

    # Slice each clip into 2-second non-overlapping segments
    total_slices = 0
    skipped_silent = 0

    for clip in deduped:
        try:
            audio, sr = sf.read(str(clip), dtype="float32")
        except Exception:
            continue

        if sr != SAMPLE_RATE:
            continue

        n_slices = len(audio) // CLIP_SAMPLES
        for j in range(n_slices):
            start = j * CLIP_SAMPLES
            segment = audio[start:start + CLIP_SAMPLES]

            # Skip silent segments
            rms = np.sqrt(np.mean(segment ** 2))
            if rms < MIN_RMS:
                skipped_silent += 1
                continue

            out_name = f"{clip.stem}_{j:03d}.wav"
            sf.write(str(output_dir / out_name), segment, SAMPLE_RATE)
            total_slices += 1

    print(f"Created {total_slices} x {CLIP_DURATION}s slices")
    print(f"Skipped {skipped_silent} silent segments")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validate training samples: remove empty, corrupted, or too-short WAV files.

Usage:
    python validate_samples.py              # Dry run
    python validate_samples.py --fix        # Remove bad files
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

TRAINING_DIR = Path("training_data")
MIN_FRAMES = 8000  # 0.5s at 16kHz


def is_bad_audio(filepath):
    """Check if audio file is empty, corrupted, or distorted.
    Returns (is_bad, reason) tuple."""
    try:
        info = sf.info(str(filepath))
        if info.frames == 0:
            return True, "empty"
        if info.frames < MIN_FRAMES:
            return True, "too_short"

        data, sr = sf.read(str(filepath))
    except Exception as e:
        return True, f"corrupted: {e}"

    rms = float(np.sqrt(np.mean(data ** 2)))

    # Near-silence — stream delivering empty audio
    if rms < 0.002:
        return True, f"silent (rms={rms:.5f})"

    # Spectral flatness
    fft = np.abs(np.fft.rfft(data))
    fft = fft[1:]  # drop DC
    geo_mean = float(np.exp(np.mean(np.log(fft + 1e-10))))
    arith_mean = float(np.mean(fft))
    flatness = geo_mean / (arith_mean + 1e-10)

    # Only flag as garbled when VERY confident:
    # quiet + flat = background noise from dead stream
    if rms < 0.005 and flatness > 0.45:
        return True, f"dead stream (rms={rms:.4f}, flatness={flatness:.3f})"

    # Extremely high flatness = pure noise/static regardless of volume
    if flatness > 0.65:
        return True, f"static (flatness={flatness:.3f})"

    return False, "ok"


def validate_dir(directory, fix=False):
    """Validate all WAV files in a directory. Returns count of bad files."""
    bad_files = []

    files = sorted(directory.glob("*.wav"))
    for i, f in enumerate(files):
        is_bad, reason = is_bad_audio(f)
        if is_bad:
            bad_files.append((f, reason))
        if (i + 1) % 5000 == 0:
            print(f"  Checked {i+1}/{len(files)}...")

    if bad_files and fix:
        for f, _ in bad_files:
            f.unlink()

    # Count by reason
    empty = sum(1 for _, r in bad_files if r == "empty")
    corrupted = sum(1 for _, r in bad_files if r.startswith("corrupted"))
    too_short = sum(1 for _, r in bad_files if r == "too_short")
    silent = sum(1 for _, r in bad_files if r.startswith("silent"))
    garbled = sum(1 for _, r in bad_files if r.startswith("noise"))

    return {
        "total": len(files),
        "empty": empty,
        "corrupted": corrupted,
        "too_short": too_short,
        "silent": silent,
        "garbled": garbled,
        "removed": len(bad_files) if fix else 0,
        "bad_files": bad_files,
    }


def validate_all(fix=False):
    """Validate all training data directories."""
    results = {}
    for subdir in ["mina", "negative"]:
        d = TRAINING_DIR / subdir
        if d.exists():
            results[subdir] = validate_dir(d, fix=fix)
    return results


def print_results(results, fix=False):
    for label, r in results.items():
        print(f"{label}/: {r['total']} files")
        bad_total = r["empty"] + r["corrupted"] + r["too_short"] + r["silent"] + r["garbled"]
        if bad_total:
            print(f"  Empty: {r['empty']}, Corrupted: {r['corrupted']}, Too short: {r['too_short']}, "
                  f"Silent: {r['silent']}, Garbled: {r['garbled']}")
            if fix:
                print(f"  Removed: {r['removed']}")
            elif r.get("bad_files"):
                for f, reason in r["bad_files"][:10]:
                    print(f"    {f.name}: {reason}")
                if len(r["bad_files"]) > 10:
                    print(f"    ... and {len(r['bad_files'])-10} more")
        else:
            print(f"  All clean.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate training samples")
    parser.add_argument("--fix", action="store_true", help="Remove bad files")
    args = parser.parse_args()

    results = validate_all(fix=args.fix)
    print_results(results, fix=args.fix)

"""
Interactive audio labeling tool.
Plays each audio clip and asks user to classify as barking, whining, or skip.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def play_audio(filepath: Path):
    """Play audio file using system player."""
    subprocess.run(["afplay", str(filepath)], check=False)


def main():
    audio_dir = Path("data/audio_unsorted")
    barking_dir = Path("data/barking")
    whining_dir = Path("data/whining")

    barking_dir.mkdir(parents=True, exist_ok=True)
    whining_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found in data/audio_unsorted/")
        sys.exit(1)

    # Filter out already-labeled files
    labeled = set()
    for d in [barking_dir, whining_dir]:
        for f in d.glob("*.wav"):
            labeled.add(f.name)

    unlabeled = [f for f in wav_files if f.name not in labeled]
    print(f"Found {len(unlabeled)} unlabeled clips ({len(labeled)} already labeled).\n")

    if not unlabeled:
        print("All clips are labeled!")
        return

    print("Controls:")
    print("  b = barking")
    print("  w = whining")
    print("  s = skip (uncertain)")
    print("  r = replay")
    print("  q = quit\n")

    for i, wav in enumerate(unlabeled):
        print(f"[{i+1}/{len(unlabeled)}] {wav.name}")
        play_audio(wav)

        while True:
            choice = input("  Label (b/w/s/r/q): ").strip().lower()
            if choice == "b":
                shutil.copy2(wav, barking_dir / wav.name)
                print("  → barking")
                break
            elif choice == "w":
                shutil.copy2(wav, whining_dir / wav.name)
                print("  → whining")
                break
            elif choice == "s":
                print("  → skipped")
                break
            elif choice == "r":
                play_audio(wav)
            elif choice == "q":
                print(f"\nLabeled so far: check data/barking/ and data/whining/")
                return
            else:
                print("  Invalid. Use b/w/s/r/q")

    print(f"\nDone labeling!")
    print(f"  Barking: {len(list(barking_dir.glob('*.wav')))} files")
    print(f"  Whining: {len(list(whining_dir.glob('*.wav')))} files")


if __name__ == "__main__":
    main()

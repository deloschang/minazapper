"""
Quick sanity check: play 50 random mina/ clips and tally accuracy.
Run in your terminal: python sanity_check.py
"""

import random
import subprocess
from pathlib import Path

MINA_DIR = Path("training_data/mina")
SAMPLE_SIZE = 50


def main():
    clips = list(MINA_DIR.glob("*.wav"))
    if not clips:
        print(f"No clips found in {MINA_DIR}")
        return

    random.seed(42)
    samples = random.sample(clips, min(SAMPLE_SIZE, len(clips)))

    print(f"Playing {len(samples)} random clips from {MINA_DIR}/")
    print("Controls: y=dog sound, n=not dog, r=replay, q=quit\n")

    kept = 0
    tossed = 0

    for i, clip in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {clip.name}")
        subprocess.run(["afplay", str(clip)], check=False)

        while True:
            choice = input("  Dog sound? (y/n/r/q): ").strip().lower()
            if choice == "y":
                kept += 1
                break
            elif choice == "n":
                tossed += 1
                break
            elif choice == "r":
                subprocess.run(["afplay", str(clip)], check=False)
            elif choice == "q":
                total = kept + tossed
                if total > 0:
                    print(f"\nResults: {kept} dog, {tossed} not dog, "
                          f"accuracy: {kept/total*100:.0f}%")
                return
            else:
                print("  Use y/n/r/q")

    total = kept + tossed
    if total > 0:
        print(f"\nResults: {kept} dog, {tossed} not dog, "
              f"accuracy: {kept/total*100:.0f}%")


if __name__ == "__main__":
    main()

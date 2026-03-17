"""
Review negative clips and move false negatives to mina/.
Run in your terminal: python review_negatives.py
"""

import subprocess
import shutil
from pathlib import Path

NEGATIVE_DIR = Path("training_data/negative")
MINA_DIR = Path("training_data/mina")


def main():
    clips = sorted(NEGATIVE_DIR.glob("*.wav"))
    if not clips:
        print(f"No clips found in {NEGATIVE_DIR}")
        return

    print(f"Reviewing {len(clips)} negative clips")
    print("Controls: y=move to mina, n=keep as negative, r=replay, q=quit\n")

    moved = 0
    kept = 0

    for i, clip in enumerate(clips):
        print(f"[{i+1}/{len(clips)}] {clip.name}")
        subprocess.run(["afplay", str(clip)], check=False)

        while True:
            choice = input("  Mina? (y/n/r/q): ").strip().lower()
            if choice == "y":
                shutil.move(str(clip), MINA_DIR / clip.name)
                moved += 1
                print("  → moved to mina/")
                break
            elif choice == "n":
                kept += 1
                break
            elif choice == "r":
                subprocess.run(["afplay", str(clip)], check=False)
            elif choice == "q":
                print(f"\nMoved to mina: {moved}, Kept as negative: {kept}, "
                      f"Remaining: {len(clips) - i - 1}")
                return
            else:
                print("  Use y/n/r/q")

    print(f"\nDone! Moved to mina: {moved}, Kept as negative: {kept}")


if __name__ == "__main__":
    main()

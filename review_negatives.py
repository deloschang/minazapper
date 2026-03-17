"""
Review negative clips and move false negatives to mina/.
Saves decisions to a log so you can resume where you left off.

Run in your terminal: python review_negatives.py
"""

import json
import subprocess
import shutil
from pathlib import Path

NEGATIVE_DIR = Path("training_data/negative")
MINA_DIR = Path("training_data/mina")
REVIEW_LOG = Path("training_data/review_log.json")


def load_log():
    if REVIEW_LOG.exists():
        with open(REVIEW_LOG) as f:
            return json.load(f)
    return {}


def save_log(log):
    with open(REVIEW_LOG, "w") as f:
        json.dump(log, f, indent=2)


def main():
    MINA_DIR.mkdir(parents=True, exist_ok=True)
    log = load_log()

    all_clips = sorted(NEGATIVE_DIR.glob("*.wav"))
    unreviewed = [c for c in all_clips if c.name not in log]

    print(f"Total negative clips: {len(all_clips)}")
    print(f"Already reviewed: {len(log)}")
    print(f"Remaining: {len(unreviewed)}")

    if not unreviewed:
        print("All clips reviewed!")
        return

    print("\nControls: y=move to mina, n=keep as negative, r=replay, q=quit\n")

    moved = 0
    kept = 0

    for i, clip in enumerate(unreviewed):
        print(f"[{i+1}/{len(unreviewed)}] {clip.name}")
        subprocess.run(["afplay", str(clip)], check=False)

        while True:
            choice = input("  Mina? (y/n/r/q): ").strip().lower()
            if choice == "y":
                shutil.move(str(clip), MINA_DIR / clip.name)
                log[clip.name] = "mina"
                save_log(log)
                moved += 1
                print("  → moved to mina/")
                break
            elif choice == "n":
                log[clip.name] = "negative"
                save_log(log)
                kept += 1
                break
            elif choice == "r":
                subprocess.run(["afplay", str(clip)], check=False)
            elif choice == "q":
                print(f"\nSession: moved {moved} to mina, kept {kept} as negative")
                print(f"Total reviewed: {len(log)}, Remaining: {len(unreviewed) - i - 1}")
                return
            else:
                print("  Use y/n/r/q")

    print(f"\nDone! Moved to mina: {moved}, Kept as negative: {kept}")
    print(f"Total reviewed: {len(log)}")


if __name__ == "__main__":
    main()

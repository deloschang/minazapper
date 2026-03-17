"""
Package training data into a zip file for Google Colab upload.
Run: python package_for_colab.py
"""

import zipfile
from pathlib import Path

TRAINING_DIR = Path("training_data")
OUTPUT = Path("training_data.zip")


def main():
    mina_files = list((TRAINING_DIR / "mina").glob("*.wav"))
    neg_files = list((TRAINING_DIR / "negative").glob("*.wav"))

    print(f"Mina clips:     {len(mina_files)}")
    print(f"Negative clips: {len(neg_files)}")

    if not mina_files:
        print("No mina clips found!")
        return

    print(f"\nCreating {OUTPUT}...")
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in mina_files:
            zf.write(f, f"training_data/mina/{f.name}")
        for f in neg_files:
            zf.write(f, f"training_data/negative/{f.name}")

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Done! {OUTPUT} ({size_mb:.0f} MB)")
    print(f"\nNext steps:")
    print(f"  1. Open train_colab.ipynb in Google Colab")
    print(f"  2. Upload {OUTPUT} when prompted")
    print(f"  3. Run all cells")
    print(f"  4. Download mina_classifier.tflite when done")


if __name__ == "__main__":
    main()

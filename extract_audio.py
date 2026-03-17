"""
Extract audio tracks from downloaded MP4 clips and convert to WAV.
"""

import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


def extract_audio(input_mp4: Path, output_wav: Path):
    """Extract audio from MP4 to WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(input_mp4),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # mono
        "-y",                     # overwrite
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    data_dir = Path("data")
    audio_dir = data_dir / "audio_unsorted"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Find all MP4 files in unsorted
    mp4_files = list((data_dir / "unsorted").glob("*.mp4"))
    if not mp4_files:
        print("No MP4 files found in data/unsorted/")
        sys.exit(1)

    print(f"Extracting audio from {len(mp4_files)} clips...")

    success = 0
    for mp4 in tqdm(mp4_files, desc="Extracting"):
        wav_path = audio_dir / mp4.with_suffix(".wav").name
        if wav_path.exists():
            success += 1
            continue
        if extract_audio(mp4, wav_path):
            success += 1
        else:
            print(f"  Failed: {mp4.name}")

    print(f"\nExtracted {success}/{len(mp4_files)} audio files to {audio_dir}/")
    print("Next: Run 'python label_audio.py' to sort into barking/whining categories.")


if __name__ == "__main__":
    main()

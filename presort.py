"""
Pre-sort raw Unifi recordings into mina/ (dog sounds) and negative/ clips.

Uses YAMNet-style audio classification via ONNX Runtime if available,
falls back to spectral bark detection (energy in 300-2000Hz band with
impulsive characteristics).

Usage:
    python presort.py [--input ./data/unsorted] [--threshold 0.5]

Pipeline:
    1. Scans each file in 1-second sliding windows
    2. Dog-sound windows → extract 2s clip → training_data/mina/
    3. Non-dog audio → sample one 2s clip per 30s → training_data/negative/
    4. Interactive verification pass on mina/ clips
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
WINDOW_SEC = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
CLIP_SEC = 2.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SEC)
NEGATIVE_INTERVAL_SEC = 30  # sample one negative clip per this many seconds

# Dog bark frequency characteristics
BARK_FREQ_LOW = 300    # Hz
BARK_FREQ_HIGH = 2000  # Hz

# YAMNet class indices for dog sounds
YAMNET_DOG_INDICES = [69, 70, 71, 72, 73, 74, 75]  # Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper


def load_audio(filepath: str) -> np.ndarray:
    """Load audio file to 16kHz mono float32."""
    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)
    except Exception:
        # Try ffmpeg for video files
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", filepath, "-vn", "-acodec", "pcm_s16le",
                 "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", "-"],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0:
                audio, _ = sf.read(
                    __import__("io").BytesIO(result.stdout),
                    dtype="float32",
                )
                return audio
        except Exception:
            pass
    return None


def try_load_yamnet():
    """Try to load YAMNet via tflite-runtime, TensorFlow, or ONNX Runtime."""
    model_path = Path("models/yamnet.tflite")
    onnx_path = Path("models/yamnet.onnx")

    # Try tflite-runtime first
    try:
        from tflite_runtime.interpreter import Interpreter
        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        return ("tflite", interp)
    except ImportError:
        pass

    # Try tensorflow
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        return ("tflite", interp)
    except (ImportError, Exception):
        pass

    # Try ONNX
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))
            return ("onnx", session)
        except Exception:
            pass

    return None


def yamnet_detect(interpreter_info, audio_window):
    """Run YAMNet on a 0.975s audio window. Returns max dog-class score."""
    kind, interp = interpreter_info

    # YAMNet expects exactly 15600 samples (0.975s at 16kHz)
    yamnet_samples = 15600
    if len(audio_window) < yamnet_samples:
        audio_window = np.pad(audio_window, (0, yamnet_samples - len(audio_window)))
    else:
        audio_window = audio_window[:yamnet_samples]

    audio_window = audio_window.astype(np.float32)

    if kind == "tflite":
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        interp.set_tensor(input_details[0]["index"], audio_window)
        interp.invoke()
        scores = interp.get_tensor(output_details[0]["index"])[0]
    elif kind == "onnx":
        input_name = interp.get_inputs()[0].name
        scores = interp.run(None, {input_name: audio_window})[0][0]
    else:
        return 0.0

    # Max score across dog-related classes
    dog_score = max(scores[i] for i in YAMNET_DOG_INDICES if i < len(scores))
    return float(dog_score)


def spectral_bark_detect(audio_window):
    """
    Detect dog barks/whines using spectral energy analysis.

    Dog vocalizations have:
    - Strong energy in 300-2000 Hz band
    - High ratio of bark-band energy to total energy
    - Impulsive character (high short-time energy variance)
    """
    if len(audio_window) < WINDOW_SAMPLES:
        audio_window = np.pad(audio_window, (0, WINDOW_SAMPLES - len(audio_window)))

    # Overall energy - skip very quiet windows
    rms = np.sqrt(np.mean(audio_window ** 2))
    if rms < 0.01:
        return 0.0

    # Compute spectrogram
    S = np.abs(librosa.stft(audio_window, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=1024)

    # Energy in bark frequency band
    bark_mask = (freqs >= BARK_FREQ_LOW) & (freqs <= BARK_FREQ_HIGH)
    bark_energy = np.mean(S[bark_mask, :] ** 2)
    total_energy = np.mean(S ** 2) + 1e-10

    bark_ratio = bark_energy / total_energy

    # Spectral centroid - barks tend to have centroid in 500-1500 Hz
    centroid = librosa.feature.spectral_centroid(S=S, sr=SAMPLE_RATE)[0]
    mean_centroid = np.mean(centroid)
    centroid_score = 1.0 if 400 < mean_centroid < 2000 else 0.3

    # Impulsiveness - barks are short bursts
    frame_energies = np.sum(S ** 2, axis=0)
    if np.mean(frame_energies) > 0:
        impulsiveness = np.std(frame_energies) / (np.mean(frame_energies) + 1e-10)
    else:
        impulsiveness = 0

    impulse_score = min(impulsiveness / 2.0, 1.0)

    # Combined score
    score = (bark_ratio * 0.5 + centroid_score * 0.25 + impulse_score * 0.25)

    # Scale to 0-1 range roughly
    score = min(score * 1.5, 1.0)

    return float(score)


def extract_clip(audio, center_sample, clip_samples, sr):
    """Extract a clip centered on center_sample."""
    half = clip_samples // 2
    start = max(0, center_sample - half)
    end = min(len(audio), start + clip_samples)
    start = max(0, end - clip_samples)

    clip = audio[start:end]
    if len(clip) < clip_samples:
        clip = np.pad(clip, (0, clip_samples - len(clip)))
    return clip


def presort(input_dir, output_dir, threshold, use_yamnet):
    """Run the presort pipeline."""
    input_path = Path(input_dir)
    mina_dir = Path(output_dir) / "mina"
    negative_dir = Path(output_dir) / "negative"
    mina_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio/video files
    extensions = {".mp4", ".wav", ".mp3", ".flac", ".m4a", ".mkv", ".mov"}
    files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not files:
        print(f"No audio/video files found in {input_path}")
        return

    print(f"Found {len(files)} files to process")

    # Try to load YAMNet
    yamnet = None
    if use_yamnet:
        yamnet = try_load_yamnet()
        if yamnet:
            print("Using YAMNet for classification")
        else:
            print("YAMNet not available, using spectral bark detection")
    else:
        print("Using spectral bark detection")

    detect_fn = (
        (lambda w: yamnet_detect(yamnet, w)) if yamnet
        else spectral_bark_detect
    )

    mina_count = 0
    negative_count = 0
    file_count = 0

    for filepath in files:
        file_count += 1
        print(f"\n[{file_count}/{len(files)}] {filepath.name}")

        audio = load_audio(str(filepath))
        if audio is None or len(audio) < SAMPLE_RATE:
            print("  Skipped (could not load or too short)")
            continue

        duration_s = len(audio) / SAMPLE_RATE
        print(f"  Duration: {duration_s:.1f}s")

        # Scan in 1-second windows with 0.5s hop
        hop = WINDOW_SAMPLES // 2
        dog_windows = []
        last_negative_sample = -NEGATIVE_INTERVAL_SEC * SAMPLE_RATE

        for start in range(0, len(audio) - WINDOW_SAMPLES + 1, hop):
            window = audio[start:start + WINDOW_SAMPLES]
            score = detect_fn(window)

            if score > threshold:
                center = start + WINDOW_SAMPLES // 2
                dog_windows.append((center, score))

        # Deduplicate overlapping dog detections (merge within 1s)
        merged = []
        for center, score in sorted(dog_windows):
            if merged and abs(center - merged[-1][0]) < SAMPLE_RATE:
                # Keep higher score
                if score > merged[-1][1]:
                    merged[-1] = (center, score)
            else:
                merged.append((center, score))

        # Extract dog clips
        for center, score in merged:
            clip = extract_clip(audio, center, CLIP_SAMPLES, SAMPLE_RATE)
            clip_name = f"{filepath.stem}_{center // SAMPLE_RATE:04d}s.wav"
            sf.write(str(mina_dir / clip_name), clip, SAMPLE_RATE)
            mina_count += 1

        # Extract negative clips (one per 30s, avoiding dog windows)
        dog_centers = set(c // SAMPLE_RATE for c, _ in merged)
        for sec in range(0, int(duration_s) - int(CLIP_SEC), NEGATIVE_INTERVAL_SEC):
            sample = sec * SAMPLE_RATE
            sec_mark = sec
            # Skip if too close to a dog detection
            if any(abs(sec_mark - dc) < 3 for dc in dog_centers):
                continue
            clip = extract_clip(audio, sample + CLIP_SAMPLES // 2, CLIP_SAMPLES, SAMPLE_RATE)
            # Only save if clip has some audio content
            if np.sqrt(np.mean(clip ** 2)) > 0.005:
                clip_name = f"{filepath.stem}_{sec:04d}s_neg.wav"
                sf.write(str(negative_dir / clip_name), clip, SAMPLE_RATE)
                negative_count += 1

        print(f"  Dog clips: {len(merged)}, Negative samples from this file: "
              f"{sum(1 for sec in range(0, int(duration_s) - int(CLIP_SEC), NEGATIVE_INTERVAL_SEC) if not any(abs(sec - dc) < 3 for dc in dog_centers))}")

    print(f"\n{'='*50}")
    print(f"Pre-sort complete!")
    print(f"  Mina (dog) clips:  {mina_count} → {mina_dir}/")
    print(f"  Negative clips:    {negative_count} → {negative_dir}/")
    print(f"{'='*50}")

    return mina_count


def verify_clips(mina_dir):
    """Interactive verification of mina/ clips."""
    mina_path = Path(mina_dir)
    clips = sorted(mina_path.glob("*.wav"))

    if not clips:
        print("No clips to verify.")
        return

    print(f"\n--- Verification Pass: {len(clips)} clips ---")
    print("Controls: y=keep, n=delete, r=replay, q=quit\n")

    kept = 0
    removed = 0

    for i, clip in enumerate(clips):
        print(f"[{i+1}/{len(clips)}] {clip.name}")

        # Play the clip
        subprocess.run(["afplay", str(clip)], check=False)

        while True:
            choice = input("  Keep? (y/n/r/q): ").strip().lower()
            if choice == "y":
                kept += 1
                break
            elif choice == "n":
                clip.unlink()
                removed += 1
                print("  → deleted")
                break
            elif choice == "r":
                subprocess.run(["afplay", str(clip)], check=False)
            elif choice == "q":
                print(f"\nStopped. Kept: {kept}, Removed: {removed}, "
                      f"Remaining: {len(clips) - i - 1}")
                return
            else:
                print("  Use y/n/r/q")

    print(f"\nVerification complete. Kept: {kept}, Removed: {removed}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-sort raw recordings into dog/negative clips"
    )
    parser.add_argument(
        "--input", default="./data/unsorted",
        help="Directory with raw .mp4/.wav files (default: ./data/unsorted)"
    )
    parser.add_argument(
        "--output", default="./training_data",
        help="Output directory (default: ./training_data)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection threshold 0-1 (default: 0.5)"
    )
    parser.add_argument(
        "--yamnet", action="store_true",
        help="Try to use YAMNet model (requires tflite-runtime or tensorflow)"
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip interactive verification pass"
    )
    args = parser.parse_args()

    mina_count = presort(args.input, args.output, args.threshold, args.yamnet)

    if mina_count and not args.skip_verify:
        verify = input("\nRun verification pass on mina/ clips? (y/n): ").strip().lower()
        if verify == "y":
            verify_clips(Path(args.output) / "mina")
    elif mina_count:
        print(f"\nSkipping verification. Run manually later:")
        print(f"  python presort.py --verify-only --output {args.output}")


if __name__ == "__main__":
    main()

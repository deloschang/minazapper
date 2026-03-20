#!/usr/bin/env python3
"""
End-to-end retrain pipeline:
  1. Pull detection clips from Pi
  2. Label using Unifi bark detection (positive/negative)
  3. Train model (memory-efficient, no Colab needed)
  4. Export TFLite
  5. Deploy to Pi and restart detector
"""

import os
import gc
import sys
import random
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from utils import compute_mfcc, SAMPLE_RATE, WINDOW_SIZE

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
PI_HOST = "del@192.168.10.10"
PI_KEY = os.path.expanduser("~/.ssh/minazap_pi")
CLIPS_DIR = Path("false_positives/detection_clips")
TRAINING_DIR = Path("training_data")
POSITIVE_DIR = TRAINING_DIR / "mina"
NEGATIVE_DIR = TRAINING_DIR / "negative"
MODEL_OUTPUT = Path("mina_classifier.tflite")
LABELS = ["mina", "negative"]

BATCH_SIZE = 32
LABELS_CHECKPOINT = Path(".labels_checkpoint")


def _labels_changed():
    """Check if manual labels have changed since last retrain."""
    import hashlib
    current_hash = ""
    for f in sorted([Path("manual_labels.json"), Path("manual_labels_training.json")]):
        if f.exists():
            current_hash += hashlib.md5(f.read_bytes()).hexdigest()

    if not LABELS_CHECKPOINT.exists():
        return True

    last_hash = LABELS_CHECKPOINT.read_text().strip()
    return current_hash != last_hash


def _save_deploy_snapshot():
    """Save which files are in mina/ and negative/ at deploy time."""
    import json
    snapshot = {}
    for label, d in [("bark", POSITIVE_DIR), ("not_bark", NEGATIVE_DIR)]:
        if d.exists():
            for f in d.iterdir():
                if f.suffix == ".wav":
                    snapshot[f.name] = label
    with open(".last_deploy_snapshot.json", "w") as f:
        json.dump(snapshot, f)
    print(f"  Deploy snapshot saved: {len(snapshot)} files")


def _save_labels_checkpoint():
    """Save current labels hash after successful deploy."""
    import hashlib
    current_hash = ""
    for f in sorted([Path("manual_labels.json"), Path("manual_labels_training.json")]):
        if f.exists():
            current_hash += hashlib.md5(f.read_bytes()).hexdigest()
    LABELS_CHECKPOINT.write_text(current_hash)

PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY", "")
PUSHOVER_APP_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_pushover(message, title="Mina Retrain"):
    """Send a Pushover notification."""
    import requests
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_APP_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": message,
            "title": title,
            "priority": -1,
        }, timeout=10)
    except Exception as e:
        print(f"  Pushover failed: {e}")


def send_telegram(message):
    """Send a Telegram notification."""
    import urllib.request, urllib.parse
    try:
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data=data)
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  Telegram failed: {e}")


def ssh(cmd, timeout=30):
    """Run command on Pi."""
    return subprocess.run(
        ["ssh", "-i", PI_KEY, "-o", "LogLevel=ERROR", PI_HOST, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def scp_from_pi(remote_glob, local_dir):
    """Copy files from Pi."""
    return subprocess.run(
        ["scp", "-i", PI_KEY, f"{PI_HOST}:{remote_glob}", str(local_dir)],
        capture_output=True, text=True, timeout=600,
    )


# --- Step 1: Pull clips ---

def pull_clips():
    print("=" * 60)
    print("Step 1: Pull detection clips from Pi")
    print("=" * 60)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    result = scp_from_pi("~/minazap/detection_clips/*.wav", CLIPS_DIR)
    # Also pull manual labels
    scp_from_pi("~/minazap/manual_labels.json", Path("."))
    count = len(list(CLIPS_DIR.glob("*.wav")))
    print(f"  {count} clips ready for labeling")
    return count > 0


# --- Step 2: Label clips ---

def label_clips(unifi_ip):
    print("\n" + "=" * 60)
    print("Step 2: Label clips via Unifi bark detection")
    print("=" * 60)

    import re
    import time as _time
    from download_clips import UnifiProtectClient

    clip_re = re.compile(r"(?:(S\d+)_)?(\d{8}_\d{6})_(\w+)_(\d+)%\.wav")

    clips = []
    session_ids = set()
    for f in sorted(CLIPS_DIR.iterdir()):
        if f.suffix != ".wav":
            continue
        m = clip_re.match(f.name)
        if not m:
            continue
        session_id = m.group(1)  # e.g. "S001" or None
        ts_str = m.group(2)
        from zoneinfo import ZoneInfo
        dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=ZoneInfo("UTC"))
        # Pi clips (S-prefixed) are UTC; this context is always Pi clips
        clips.append({
            "path": f,
            "ts_ms": int(dt.timestamp() * 1000),
            "camera": m.group(3),
            "session": session_id,
        })
        if session_id:
            session_ids.add(session_id)

    if not clips:
        print("  No clips to label.")
        return {"total": 0, "positive": 0, "negative": 0,
                "bark_timestamps": [], "session_range": "",
                "unifi_downloaded": 0}

    min_ts = min(c["ts_ms"] for c in clips) - 60_000
    max_ts = max(c["ts_ms"] for c in clips) + 60_000

    client = UnifiProtectClient()
    client.login_local(unifi_ip)

    # Fetch ALL audio events for the full time range of clips
    all_events = []
    offset = 0
    while True:
        events = client.get_events(start_ts=min_ts, end_ts=max_ts, limit=100, offset=offset)
        if not events:
            break
        all_events.extend(events)
        if len(events) < 100:
            break
        offset += 100

    bark_events = [e for e in all_events if "alrmBark" in e.get("smartDetectTypes", [])]
    print(f"  Unifi events: {len(all_events)} audio, {len(bark_events)} bark")

    # Collect bark event timestamps for summary
    bark_timestamps = []
    for e in sorted(bark_events, key=lambda x: x["start"]):
        t = datetime.fromtimestamp(e["start"] / 1000).strftime("%I:%M %p")
        bark_timestamps.append(t)

    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load manual labels from both sources (final source of truth)
    # Pi labels (from Pi review server) + local labels (from local review server)
    import json
    manual_labels = {}
    for labels_file in [Path("manual_labels.json"), Path("manual_labels_training.json")]:
        if labels_file.exists():
            with open(labels_file) as f:
                raw = json.load(f)
            for sid, info in raw.items():
                label = info.get("label") if isinstance(info, dict) else info
                # Per-clip labels: extract session ID for group-level matching
                if sid.startswith("clip:"):
                    # Also store per-clip label for audit_negatives
                    manual_labels[sid] = label
                    continue
                manual_labels[sid] = label
    print(f"  Manual labels loaded: {len(manual_labels)}")

    # --- Label clips: manual > UniFi > negative ---
    positive = 0
    negative = 0
    manual_used = 0
    for clip in clips:
        # Check manual label first
        if clip.get("session") and clip["session"] in manual_labels:
            is_bark = manual_labels[clip["session"]] == "bark"
            manual_used += 1
        else:
            # Fall back to UniFi
            is_bark = any(
                (e["start"] - 3000) <= clip["ts_ms"] <= (e["end"] + 3000)
                for e in bark_events
            )

        dest_dir = POSITIVE_DIR if is_bark else NEGATIVE_DIR
        shutil.move(str(clip["path"]), str(dest_dir / clip["path"].name))
        if is_bark:
            positive += 1
        else:
            negative += 1

    print(f"  Labeled: {positive} positive, {negative} negative ({manual_used} from manual labels)")

    # --- Download Unifi bark events as positive training data ---
    # Camera ID → name mapping
    CAMERA_NAMES = {
        "6987985800387203e400044e": "entry",
        "6987985800ec7203e4000450": "kitchen",
        "6987985800b87203e400044f": "living",
    }

    unifi_downloaded = 0
    if bark_events:
        print(f"  Downloading {len(bark_events)} Unifi bark events as positive samples...")
        for e in bark_events:
            event_start = datetime.fromtimestamp(e["start"] / 1000)
            event_id = e.get("id", "unknown")[:8]
            cam_name = CAMERA_NAMES.get(e.get("camera", ""), "unknown")
            filename = f"unifi_{event_start.strftime('%Y%m%d_%H%M%S')}_{event_id}_{cam_name}.wav"
            output_path = POSITIVE_DIR / filename

            if output_path.exists():
                continue

            if client.download_clip(e, str(output_path), audio_only=True):
                # Validate immediately — remove if empty/corrupted
                try:
                    import soundfile as _sf
                    info = _sf.info(str(output_path))
                    if info.frames < 8000:  # < 0.5s
                        output_path.unlink()
                    else:
                        unifi_downloaded += 1
                except Exception:
                    output_path.unlink()
            _time.sleep(0.3)

        print(f"  Downloaded {unifi_downloaded} new Unifi bark clips to mina/")

    # Skipping random non-bark UniFi events — the Pi's model already filters
    # these out, so they're not useful as negative training data. The real
    # negatives come from the Pi's own false positive detections.
    unifi_neg_downloaded = 0

    print(f"  Training data: {len(list(POSITIVE_DIR.glob('*.wav')))} mina, "
          f"{len(list(NEGATIVE_DIR.glob('*.wav')))} negative")

    # Session range
    sorted_sessions = sorted(session_ids) if session_ids else []
    session_range = ""
    if sorted_sessions:
        session_range = f"{sorted_sessions[0]} to {sorted_sessions[-1]}"

    return {
        "total": len(clips),
        "positive": positive,
        "negative": negative,
        "bark_timestamps": bark_timestamps,
        "session_range": session_range,
        "unifi_downloaded": unifi_downloaded,
        "unifi_neg_downloaded": unifi_neg_downloaded,
    }


# --- Step 2b: Audit negatives against Unifi ---

def audit_negatives(unifi_ip):
    """Audit training labels against ground truth.

    Label hierarchy (highest priority first):
      1. Manual labels from review server (human listened — final truth)
      2. UniFi AI bark detection (suggestion only)
      3. Default negative (if neither above applies)

    Manual labels NEVER get overridden. UniFi suggestions only apply
    to clips that haven't been manually reviewed.
    """
    import re
    import json
    from download_clips import UnifiProtectClient

    print("\n" + "=" * 60)
    print("Step 2b: Audit labels (manual > UniFi > default)")
    print("=" * 60)

    # Load manual labels from both sources — final source of truth
    manual_labels = {}  # session-level: S063 -> "bark"
    clip_labels = {}    # per-clip: filename -> "bark"
    for labels_file in [Path("manual_labels.json"), Path("manual_labels_training.json")]:
        if labels_file.exists():
            with open(labels_file) as f:
                raw = json.load(f)
            for sid, info in raw.items():
                label = info.get("label") if isinstance(info, dict) else info
                if sid.startswith("clip:"):
                    clip_labels[sid[5:]] = label  # filename -> label
                else:
                    manual_labels[sid] = label

    print(f"  Manual labels: {len(manual_labels)} sessions, {len(clip_labels)} per-clip")

    # Extract session ID from filenames
    session_re = re.compile(r"(S\d+)_")
    clip_re = re.compile(r"(?:S\d+_)?(?:unifi_)?(\d{8}_\d{6})")

    # --- Pass 1: Enforce manual labels across both dirs ---
    # Per-clip labels take priority over session-level labels
    manual_moves = 0
    for src_dir, src_label in [(NEGATIVE_DIR, "not_bark"), (POSITIVE_DIR, "bark")]:
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.iterdir()):
            if f.suffix != ".wav":
                continue

            # Check per-clip label first
            human_label = clip_labels.get(f.name)

            # Fall back to session-level label
            if not human_label:
                sm = session_re.match(f.name)
                if sm and sm.group(1) in manual_labels:
                    human_label = manual_labels[sm.group(1)]

            if not human_label:
                continue

            # File is in wrong dir per manual label
            if human_label == "bark" and src_label == "not_bark":
                shutil.move(str(f), str(POSITIVE_DIR / f.name))
                manual_moves += 1
            elif human_label == "not_bark" and src_label == "bark":
                shutil.move(str(f), str(NEGATIVE_DIR / f.name))
                manual_moves += 1

    if manual_moves:
        print(f"  Manual labels enforced: moved {manual_moves} clips")

    # --- Pass 2: UniFi suggestions for non-manually-labeled clips only ---
    tolerance = 5000  # 5s

    neg_clips = []
    for f in sorted(NEGATIVE_DIR.iterdir()):
        if f.suffix != ".wav":
            continue
        # Skip if manually labeled
        sm = session_re.match(f.name)
        if sm and sm.group(1) in manual_labels:
            continue
        m = clip_re.search(f.name)
        if not m:
            continue
        dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        neg_clips.append({"path": f, "ts_ms": int(dt.timestamp() * 1000)})

    if not neg_clips:
        print("  No unreviewed negatives to check against UniFi.")
        return

    min_ts = min(c["ts_ms"] for c in neg_clips) - 86_400_000
    max_ts = max(c["ts_ms"] for c in neg_clips) + 86_400_000

    client = UnifiProtectClient()
    client.login_local(unifi_ip)

    all_events = []
    offset = 0
    while True:
        events = client.get_events(start_ts=min_ts, end_ts=max_ts, limit=100, offset=offset)
        if not events:
            break
        all_events.extend(events)
        if len(events) < 100:
            break
        offset += 100

    bark_events = [e for e in all_events if "alrmBark" in e.get("smartDetectTypes", [])]
    print(f"  Checking {len(neg_clips)} unreviewed negatives against {len(bark_events)} UniFi bark events")

    unifi_moves = 0
    for clip in neg_clips:
        is_bark = any(
            (e["start"] - tolerance) <= clip["ts_ms"] <= (e["end"] + tolerance)
            for e in bark_events
        )
        if is_bark:
            shutil.move(str(clip["path"]), str(POSITIVE_DIR / clip["path"].name))
            unifi_moves += 1

    if unifi_moves:
        print(f"  UniFi suggestions applied: moved {unifi_moves} clips (negative → mina/)")
    else:
        print("  All unreviewed negatives are correct.")


# --- Step 2c: Validate training data ---

def validate_training_data():
    """Remove empty, corrupted, or too-short samples before training."""
    print("\n" + "=" * 60)
    print("Step 2c: Validate training samples")
    print("=" * 60)

    from validate_samples import validate_all, print_results
    results = validate_all(fix=True)
    print_results(results, fix=True)


# --- Step 3: Train ---

def load_audio(filepath):
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))
    else:
        audio = audio[:WINDOW_SIZE]
    return audio.astype(np.float32)


def augment_mfcc(mfcc, label):
    """tf.data augmentation on MFCC features (memory efficient)."""
    shift = tf.random.uniform([], -12, 12, dtype=tf.int32)
    mfcc = tf.roll(mfcc, shift, axis=0)
    noise = tf.random.normal(tf.shape(mfcc), stddev=0.1)
    mfcc = mfcc + noise
    return mfcc, label


def train_model():
    print("\n" + "=" * 60)
    print("Step 3: Train model")
    print("=" * 60)

    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU available: {gpus}")

    # Load audio → MFCC immediately (memory efficient)
    all_mfccs = []
    all_labels = []

    for label_idx, label_name in enumerate(LABELS):
        label_dir = TRAINING_DIR / label_name
        wav_files = sorted(label_dir.glob("*.wav"))
        print(f"Loading {label_name}: {len(wav_files)} files...")
        for i, wav_path in enumerate(wav_files):
            try:
                audio = load_audio(str(wav_path))
                mfcc = compute_mfcc(audio)
                all_mfccs.append(mfcc)
                all_labels.append(label_idx)
            except Exception:
                pass
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(wav_files)}...")

    X_all = np.array(all_mfccs, dtype=np.float32)[..., np.newaxis]
    y_all = np.array(all_labels, dtype=np.int32)
    del all_mfccs, all_labels
    gc.collect()

    print(f"\nTotal samples: {len(y_all)}")
    for i, name in enumerate(LABELS):
        print(f"  {name}: {np.sum(y_all == i)}")

    # Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=SEED, stratify=y_all
    )
    del X_all, y_all
    gc.collect()

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    del X_temp, y_temp
    gc.collect()

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # tf.data pipeline with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(10000, seed=SEED)
    train_ds = train_ds.map(augment_mfcc, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(124, 40, 1)),
        tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    header = "          " + "  ".join(f"{l:>8s}" for l in LABELS)
    print(header)
    for i, row in enumerate(cm):
        print(f"{LABELS[i]:>8s}  " + "  ".join(f"{v:>8d}" for v in row))
    print("\n" + classification_report(y_test, y_pred, target_names=LABELS, digits=4))

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_OUTPUT, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {len(tflite_model) / 1024:.1f} KB")

    with open("labels.txt", "w") as f:
        for label in LABELS:
            f.write(label + "\n")

    val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\nBest val accuracy: {val_acc:.4f}")
    return test_acc


# --- Step 4: Deploy ---

def deploy():
    print("\n" + "=" * 60)
    print("Step 4: Deploy to Pi")
    print("=" * 60)

    # Copy model
    subprocess.run(
        ["scp", "-i", PI_KEY, str(MODEL_OUTPUT), f"{PI_HOST}:~/minazap/"],
        check=True, timeout=30,
    )
    print("  Model uploaded.")

    # Restart detector via systemd
    ssh("sudo systemctl restart bark_detector")
    print("  Detector restarted.")

    # Clear clips on Pi
    ssh("rm -f ~/minazap/detection_clips/*.wav")
    print("  Pi clips cleared.")


# --- Step 5: Cleanup ---

def cleanup():
    print("\n" + "=" * 60)
    print("Step 5: Cleanup")
    print("=" * 60)
    for f in CLIPS_DIR.glob("*.wav"):
        f.unlink()
    print("  Local staging clips cleared.")


def main():
    parser = argparse.ArgumentParser(description="End-to-end retrain pipeline")
    parser.add_argument("--unifi-ip", default="192.168.1.1",
                        help="Unifi console IP (default: 192.168.1.1)")
    parser.add_argument("--skip-pull", action="store_true",
                        help="Skip pulling clips from Pi (use existing local clips)")
    parser.add_argument("--skip-label", action="store_true",
                        help="Skip labeling (clips already in training dirs)")
    parser.add_argument("--skip-deploy", action="store_true",
                        help="Train only, don't deploy to Pi")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip pull, label, and deploy — just train")
    parser.add_argument("--cron", action="store_true",
                        help="Non-interactive mode for cron: skip deploy if accuracy < 90%%")
    parser.add_argument("--min-clips", type=int, default=10,
                        help="Minimum new clips required to trigger retrain (default: 10)")
    args = parser.parse_args()

    if args.train_only:
        args.skip_pull = True
        args.skip_label = True
        args.skip_deploy = True

    if not args.skip_pull:
        has_clips = pull_clips()

    # Check if manual labels changed since last deploy
    labels_changed = _labels_changed()

    label_stats = None
    if not args.skip_label:
        clip_count = len(list(CLIPS_DIR.glob("*.wav")))
        if clip_count < args.min_clips and not labels_changed and args.cron:
            print(f"Only {clip_count} clips and no label changes. Skipping retrain.")
            return
        if clip_count > 0:
            label_stats = label_clips(args.unifi_ip)

    # Audit: fix any negatives that Unifi says are actually barking
    if not args.skip_label:
        audit_negatives(args.unifi_ip)

    # Validate: remove empty/corrupted samples before training
    validate_training_data()

    test_acc = train_model()

    if not args.skip_deploy:
        if test_acc < 0.90:
            if args.cron:
                print(f"Test accuracy {test_acc:.4f} < 90%. Skipping deploy.")
                # Still send summary even if we don't deploy
                if label_stats:
                    _send_summary(label_stats, test_acc, deployed=False)
                return
            print(f"\nTest accuracy {test_acc:.4f} is below 90%. Deploy anyway? [y/N]")
            if input().strip().lower() != "y":
                print("Aborted deploy.")
                return
        deploy()
        cleanup()
        _save_labels_checkpoint()
        _save_deploy_snapshot()

    # Send Pushover summary
    if label_stats:
        _send_summary(label_stats, test_acc, deployed=not args.skip_deploy)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def _send_summary(stats, test_acc, deployed):
    """Send Pushover summary of retrain results."""
    lines = [f"Retrain complete — {datetime.now().strftime('%b %d')}"]
    lines.append("")
    if stats.get("session_range"):
        lines.append(f"Sessions: {stats['session_range']}")
    lines.append(f"Clips processed: {stats['total']}")
    lines.append(f"True barks (Unifi confirmed): {stats['positive']}")
    lines.append(f"False positives (added to negative): {stats['negative']}")
    if stats.get("unifi_downloaded"):
        lines.append(f"Unifi bark clips downloaded: {stats['unifi_downloaded']}")
    if stats.get("unifi_neg_downloaded"):
        lines.append(f"Unifi non-bark clips downloaded: {stats['unifi_neg_downloaded']}")

    if stats["bark_timestamps"]:
        # Deduplicate nearby timestamps
        unique_times = list(dict.fromkeys(stats["bark_timestamps"]))
        lines.append(f"\nBark times: {', '.join(unique_times)}")

    lines.append(f"\nModel accuracy: {test_acc:.1%}")
    lines.append(f"Deployed: {'yes' if deployed else 'no'}")

    message = "\n".join(lines)
    print(f"\n--- Pushover Summary ---\n{message}\n")
    send_pushover(message, title="Mina Retrain Summary")
    send_telegram(f"🐕 {message}")


if __name__ == "__main__":
    main()

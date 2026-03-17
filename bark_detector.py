"""
Real-time bark detector for Raspberry Pi 4.

Pulls audio from 3 Unifi camera RTSP streams via go2rtc,
runs TFLite inference on 1-second windows, deduplicates
detections across cameras, and shows desktop pop-up notifications.

Usage:
    python bark_detector.py [--threshold 0.7] [--cooldown 5]
"""

import argparse
import subprocess
import threading
import time
import queue
import os
from datetime import datetime

import numpy as np
import librosa

# TFLite runtime — try ai_edge_litert first, then tflite_runtime, then tensorflow
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

# ─── Config ───────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser("~/minazap/mina_classifier.tflite")
LABELS_PATH = os.path.expanduser("~/minazap/labels.txt")

# MFCC params — must match training exactly
SAMPLE_RATE = 16000
WINDOW_SIZE = 16000  # 1 second
FRAME_LENGTH = 255
FRAME_STEP = 128
N_MELS = 40
N_MFCC = 40

# RTSP streams via go2rtc (audio only)
STREAMS = {
    "entry":   "rtsp://127.0.0.1:8554/entry",
    "kitchen": "rtsp://127.0.0.1:8554/kitchen",
    "living":  "rtsp://127.0.0.1:8554/living",
}

# Detection
DEFAULT_THRESHOLD = 0.7
DEFAULT_COOLDOWN = 5  # seconds between notifications


def compute_mfcc(audio):
    """Compute MFCC features. Must match training exactly."""
    audio = audio.astype(np.float32)
    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))
    elif len(audio) > WINDOW_SIZE:
        audio = audio[:WINDOW_SIZE]

    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=FRAME_LENGTH, hop_length=FRAME_STEP,
        n_mels=N_MELS, center=False,
    )
    return mfcc.T.astype(np.float32)  # (124, 40)


class BarkDetector:
    def __init__(self, model_path, threshold=0.7):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_idx = self.input_details[0]["index"]
        self.output_idx = self.output_details[0]["index"]
        self.threshold = threshold

        # Load labels
        self.labels = []
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH) as f:
                self.labels = [l.strip() for l in f if l.strip()]

        print(f"Model loaded: input={self.input_details[0]['shape']}, "
              f"output={self.output_details[0]['shape']}")
        print(f"Labels: {self.labels}")

    def predict(self, audio_1sec):
        """Run inference on 1 second of audio. Returns (is_bark, confidence)."""
        mfcc = compute_mfcc(audio_1sec)
        input_data = mfcc[np.newaxis, :, :, np.newaxis].astype(np.float32)

        self.interpreter.set_tensor(self.input_idx, input_data)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_idx)[0]

        # Index 0 = mina, Index 1 = negative
        mina_score = float(scores[0])
        is_bark = mina_score >= self.threshold
        return is_bark, mina_score


class AudioStreamReader:
    """Reads audio from an RTSP stream via ffmpeg."""

    def __init__(self, name, url, sample_rate=16000):
        self.name = name
        self.url = url
        self.sample_rate = sample_rate
        self.process = None
        self.running = False
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()

    def start(self):
        """Start ffmpeg process to read audio from RTSP stream."""
        self.running = True
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", str(self.sample_rate),
            "-ac", "1",               # mono
            "-f", "s16le",            # raw PCM output
            "-loglevel", "error",
            "pipe:1",
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(f"[{self.name}] Stream started")

    def _read_loop(self):
        """Continuously read audio data from ffmpeg stdout."""
        while self.running and self.process.poll() is None:
            raw = self.process.stdout.read(self.sample_rate * 2)  # 1 sec of 16-bit
            if not raw:
                time.sleep(0.1)
                continue

            # Convert raw PCM bytes to float32 normalized
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            with self.lock:
                self.buffer = np.concatenate([self.buffer, samples])
                # Keep only last 2 seconds in buffer
                max_buf = self.sample_rate * 2
                if len(self.buffer) > max_buf:
                    self.buffer = self.buffer[-max_buf:]

        if self.running:
            print(f"[{self.name}] Stream ended unexpectedly, restarting...")
            time.sleep(2)
            self.start()

    def get_window(self):
        """Get the latest 1-second window. Returns None if not enough data."""
        with self.lock:
            if len(self.buffer) < self.sample_rate:
                return None
            window = self.buffer[-self.sample_rate:].copy()
        return window

    def stop(self):
        self.running = False
        if self.process:
            self.process.kill()


def send_notification(message):
    """Send desktop notification on Raspberry Pi."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(f"🔔 {full_msg}")

    # Try notify-send (Linux desktop notification)
    try:
        subprocess.Popen(
            ["notify-send", "-u", "critical", "-t", "5000",
             "🐕 Mina Bark Detected!", message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass

    # Also write to a JSON file the camera-wall can poll
    import json
    notification = {
        "type": "bark",
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(os.path.expanduser("~/camera-wall/bark-alert.json"), "w") as f:
            json.dump(notification, f)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Real-time bark detector")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Detection threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                        help=f"Seconds between notifications (default: {DEFAULT_COOLDOWN})")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .tflite model")
    args = parser.parse_args()

    # Load model
    detector = BarkDetector(args.model, threshold=args.threshold)

    # Start audio streams
    streams = {}
    for name, url in STREAMS.items():
        reader = AudioStreamReader(name, url)
        reader.start()
        streams[name] = reader

    print(f"\nListening for barks (threshold={args.threshold}, cooldown={args.cooldown}s)...")
    print("Press Ctrl+C to stop.\n")

    last_notification_time = 0
    detection_window = {}  # camera -> last detection time

    try:
        while True:
            detections = []

            for name, reader in streams.items():
                window = reader.get_window()
                if window is None:
                    continue

                is_bark, confidence = detector.predict(window)

                if is_bark:
                    detections.append((name, confidence))
                    detection_window[name] = time.time()

            if detections:
                now = time.time()

                # Deduplicate: if multiple cameras detect within 1 second,
                # count as a single bark event
                recent = [
                    (name, t) for name, t in detection_window.items()
                    if now - t < 1.0
                ]

                if now - last_notification_time >= args.cooldown:
                    cameras = [d[0] for d in detections]
                    max_conf = max(d[1] for d in detections)
                    cam_str = ", ".join(cameras)

                    msg = f"Bark detected on {cam_str} ({max_conf:.0%} confidence)"
                    send_notification(msg)
                    last_notification_time = now

            # Run inference every 500ms (50% overlap)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for reader in streams.values():
            reader.stop()


if __name__ == "__main__":
    main()

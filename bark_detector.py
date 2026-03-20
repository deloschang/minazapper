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
import urllib.request
import urllib.parse
from datetime import datetime

import numpy as np
import librosa
import soundfile as sf

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
SILENCE_TIMEOUT = 30  # seconds of silence before session ends


# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


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


class SpectralBarkDetector:
    """Test mode detector using spectral energy analysis. No model needed."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        print(f"TEST MODE: Using spectral bark detection (no TFLite model)")
        print(f"Threshold: {threshold}")

    def predict(self, audio_1sec):
        """Detect barks via spectral energy in 300-2000Hz band."""
        if len(audio_1sec) < WINDOW_SIZE:
            audio_1sec = np.pad(audio_1sec, (0, WINDOW_SIZE - len(audio_1sec)))

        rms = np.sqrt(np.mean(audio_1sec ** 2))
        if rms < 0.01:
            return False, 0.0

        S = np.abs(librosa.stft(audio_1sec, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=1024)

        bark_mask = (freqs >= 300) & (freqs <= 2000)
        bark_energy = np.mean(S[bark_mask, :] ** 2)
        total_energy = np.mean(S ** 2) + 1e-10
        bark_ratio = bark_energy / total_energy

        centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=SAMPLE_RATE)[0])
        centroid_score = 1.0 if 400 < centroid < 2000 else 0.3

        frame_energies = np.sum(S ** 2, axis=0)
        impulsiveness = np.std(frame_energies) / (np.mean(frame_energies) + 1e-10)
        impulse_score = min(impulsiveness / 2.0, 1.0)

        score = min((bark_ratio * 0.5 + centroid_score * 0.25 + impulse_score * 0.25) * 1.5, 1.0)
        is_bark = score >= self.threshold
        return is_bark, float(score)


class AudioStreamReader:
    """Reads audio from an RTSP stream via ffmpeg."""

    MAX_BACKOFF = 120  # Max seconds between retries
    BACKOFF_RESET_AFTER = 300  # Reset backoff after 5 min of stable streaming

    def __init__(self, name, url, sample_rate=16000):
        self.name = name
        self.url = url
        self.sample_rate = sample_rate
        self.process = None
        self.running = False
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
        self._backoff = 2
        self._last_success = 0
        self._healthy = False
        self._new_data = False  # Set when new audio arrives
        self._new_samples = 0  # Count of new samples since last get_window

    def start(self):
        """Start ffmpeg process to read audio from RTSP stream."""
        self._cleanup()
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

    def _cleanup(self):
        """Kill any existing ffmpeg process."""
        if self.process:
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:
                pass
            self.process = None

    def _read_loop(self):
        """Continuously read audio data from ffmpeg stdout."""
        self._last_success = time.time()
        self._healthy = False

        while self.running and self.process.poll() is None:
            # Read 64ms chunks (1024 samples * 2 bytes) for low latency
            raw = self.process.stdout.read(1024 * 2)
            if not raw:
                time.sleep(0.1)
                continue

            # Stream is delivering data — mark healthy
            if not self._healthy:
                self._healthy = True
                self._last_success = time.time()

            # Reset backoff after sustained healthy streaming
            if time.time() - self._last_success > self.BACKOFF_RESET_AFTER:
                self._backoff = 2
                self._last_success = time.time()

            # Convert raw PCM bytes to float32 normalized
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            with self.lock:
                self.buffer = np.concatenate([self.buffer, samples])
                # Keep only last 2 seconds in buffer
                max_buf = self.sample_rate * 2
                if len(self.buffer) > max_buf:
                    self.buffer = self.buffer[-max_buf:]
                self._new_samples += len(samples)
                # Flag new data after ~62ms of new audio (1000 samples at 16kHz)
                if self._new_samples >= 1000:
                    self._new_data = True

        if self.running:
            self._cleanup()
            print(f"[{self.name}] Stream ended unexpectedly, retrying in {self._backoff}s...")
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, self.MAX_BACKOFF)
            self.start()

    def get_window(self):
        """Get the latest 1-second window. Returns None if no new data or not enough data."""
        with self.lock:
            if not self._new_data or len(self.buffer) < self.sample_rate:
                return None
            self._new_data = False
            self._new_samples = 0
            window = self.buffer[-self.sample_rate:].copy()
        return window

    def stop(self):
        self.running = False
        self._cleanup()


REVIEW_URL_BASE = "http://192.168.1.68:8086/review"


def send_notification(message, session_id=None):
    """Send notification via console, desktop, camera wall, and Pushover."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(f"🔔 {full_msg}")

    # Camera wall JSON overlay
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

    # Telegram (async, non-blocking)
    def _send_telegram():
        try:
            # Strip HTML tags for Telegram text, add review link
            import re as _re
            clean_msg = _re.sub(r"<[^>]+>", "", message)
            if session_id:
                clean_msg += f"\n\nReview: {REVIEW_URL_BASE}/{session_id}"
            data = urllib.parse.urlencode({
                "chat_id": TELEGRAM_CHAT_ID,
                "text": clean_msg,
            }).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data=data)
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"  Telegram error: {e}")
    threading.Thread(target=_send_telegram, daemon=True).start()



def main():
    parser = argparse.ArgumentParser(description="Real-time bark detector")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Detection threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                        help=f"Seconds between notifications (default: {DEFAULT_COOLDOWN})")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .tflite model")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: use spectral detection instead of TFLite model")
    parser.add_argument("--silence", type=float, default=SILENCE_TIMEOUT,
                        help=f"Seconds of silence before session ends (default: {SILENCE_TIMEOUT})")
    args = parser.parse_args()

    # Load detector
    if args.test:
        detector = SpectralBarkDetector(threshold=args.threshold)
    else:
        detector = BarkDetector(args.model, threshold=args.threshold)

    # Start audio streams
    streams = {}
    for name, url in STREAMS.items():
        reader = AudioStreamReader(name, url)
        reader.start()
        streams[name] = reader

    print(f"\nListening for barks (threshold={args.threshold}, silence={args.silence}s)...")
    print("Press Ctrl+C to stop.\n")

    # Session tracking
    in_session = False
    session_start = 0
    session_last_bark = 0
    session_detections = 0
    session_cameras = set()
    session_max_conf = 0
    # Save detection clips for review/retraining
    clips_dir = os.path.expanduser("~/minazap/detection_clips")
    os.makedirs(clips_dir, exist_ok=True)

    # Persist session counter across restarts
    counter_file = os.path.expanduser("~/minazap/session_counter.txt")
    try:
        with open(counter_file) as f:
            session_id_counter = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        session_id_counter = 0

    last_heartbeat = time.time()
    HEARTBEAT_INTERVAL = 3600  # 1 hour
    last_health_check = time.time()
    HEALTH_CHECK_INTERVAL = 120  # check every 2 min
    stream_down_notified = set()  # avoid repeat alerts

    try:
        while True:
            now = time.time()

            # Stream health check
            if now - last_health_check >= HEALTH_CHECK_INTERVAL:
                last_health_check = now
                for name, reader in streams.items():
                    if not reader._healthy and name not in stream_down_notified:
                        stream_down_notified.add(name)
                        send_notification(f"⚠️ Stream '{name}' is down — attempting reconnect")
                        print(f"[{name}] Stream unhealthy, restarting...")
                        reader.stop()
                        reader.start()
                    elif reader._healthy and name in stream_down_notified:
                        stream_down_notified.discard(name)
                        send_notification(f"✅ Stream '{name}' recovered")

            # Hourly heartbeat
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                last_heartbeat = now
                stream_status = ", ".join(
                    f"{n}: {'ok' if r._healthy else 'down'}" for n, r in streams.items()
                )
                clip_count = len([f for f in os.listdir(clips_dir) if f.endswith(".wav")])
                msg = (f"Heartbeat — uptime {int((now - last_heartbeat + HEARTBEAT_INTERVAL) // 3600)}h, "
                       f"streams: {stream_status}, "
                       f"clips: {clip_count}, "
                       f"sessions: {session_id_counter}")
                send_notification(msg)

            detections = []
            t_loop = time.time()

            for name, reader in streams.items():
                try:
                    window = reader.get_window()
                except Exception:
                    continue
                if window is None:
                    continue

                # Skip quiet audio — real barks have significant energy
                rms = float(np.sqrt(np.mean(window ** 2)))
                if rms < 0.007:
                    continue

                try:
                    t_inf = time.time()
                    is_bark, confidence = detector.predict(window)
                    inf_ms = (time.time() - t_inf) * 1000
                except Exception:
                    continue

                if is_bark:
                    detections.append((name, confidence, window, inf_ms))

            if detections:
                cameras = [d[0] for d in detections]
                max_conf = max(d[1] for d in detections)
                max_inf_ms = max(d[3] for d in detections)
                loop_ms = (time.time() - t_loop) * 1000
                print(f"  [{cameras[0]}] inference: {max_inf_ms:.1f}ms, loop: {loop_ms:.1f}ms")

                if not in_session:
                    # New bark session
                    in_session = True
                    session_start = now
                    session_detections = 0
                    session_cameras = set()
                    session_max_conf = 0
                    session_id_counter += 1
                    session_id = f"S{session_id_counter:03d}"
                    with open(counter_file, "w") as f:
                        f.write(str(session_id_counter))

                    cam_str = ", ".join(cameras)
                    review_url = f"{REVIEW_URL_BASE}/{session_id}"
                    msg = (f"[{session_id}] Barking started on {cam_str} ({max_conf:.0%} confidence)\n"
                           f"<a href=\"{review_url}\">Review & Label</a>")
                    send_notification(msg, session_id=session_id)

                # Save one clip per camera per second (avoid overlapping duplicates)
                for cam_name, conf, audio_window, _ in detections:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_key = f"{session_id}_{ts}_{cam_name}"
                    last_key = f"_last_save_{cam_name}"
                    if getattr(detector, last_key, None) == clip_key:
                        continue  # Already saved this camera+second
                    setattr(detector, last_key, clip_key)

                    # Skip quiet/garbled audio — real barks have rms > 0.01
                    rms = float(np.sqrt(np.mean(audio_window ** 2)))
                    if rms < 0.007:
                        continue
                    fft_mag = np.abs(np.fft.rfft(audio_window))[1:]
                    geo = float(np.exp(np.mean(np.log(fft_mag + 1e-10))))
                    arith = float(np.mean(fft_mag))
                    flatness = geo / (arith + 1e-10)
                    if (rms < 0.005 and flatness > 0.45) or flatness > 0.65:
                        continue

                    clip_path = os.path.join(clips_dir, f"{clip_key}_{conf:.0%}.wav")
                    try:
                        sf.write(clip_path, audio_window, SAMPLE_RATE)
                    except Exception:
                        pass

                # Update session stats
                session_last_bark = now
                session_detections += 1
                session_cameras.update(cameras)
                session_max_conf = max(session_max_conf, max_conf)

            elif in_session and (now - session_last_bark) >= args.silence:
                # Silence timeout — session ended, send summary
                duration = session_last_bark - session_start
                cam_str = ", ".join(sorted(session_cameras))
                review_url = f"{REVIEW_URL_BASE}/{session_id}"
                msg = (f"[{session_id}] Barking stopped. Duration: {duration:.0f}s, "
                       f"cameras: {cam_str}, "
                       f"peak confidence: {session_max_conf:.0%}\n"
                       f"<a href=\"{review_url}\">Review & Label</a>")
                send_notification(msg, session_id=session_id)
                in_session = False

            # Poll fast, inference gated by 250ms new audio per stream
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        # Log crash but let systemd restart us
        print(f"FATAL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for reader in streams.values():
            reader.stop()


if __name__ == "__main__":
    main()

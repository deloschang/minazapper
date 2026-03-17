"""
Download non-bark audio from Unifi Protect for the negative training set.

Pulls speech events (alrmSpeak) and motion events, extracts audio-only
as 16kHz mono WAV files.
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import requests
import urllib3
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USERNAME = os.getenv("UNIFI_USERNAME")
PASSWORD = os.getenv("UNIFI_PASSWORD")

OUTPUT_DIR = Path("training_data/negative_raw")


def login(host):
    """Login to local console, return session."""
    s = requests.Session()
    s.verify = False
    resp = s.post(
        f"https://{host}/api/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"Login failed: {resp.status_code}")
        sys.exit(1)
    csrf = resp.headers.get("X-CSRF-Token")
    if csrf:
        s.headers["X-CSRF-Token"] = csrf
    print("Logged in.")
    return s


def get_events(session, host, event_type, limit=500, days_back=180):
    """Fetch events of a given type."""
    now = int(time.time() * 1000)
    start = now - days_back * 86400 * 1000
    base = f"https://{host}/proxy/protect"

    all_events = []
    offset = 0

    while True:
        resp = session.get(
            f"{base}/api/events",
            params={
                "types": event_type,
                "limit": limit,
                "offset": offset,
                "start": start,
                "end": now,
                "orderDirection": "DESC",
            },
            timeout=60,
        )
        if resp.status_code != 200:
            break
        events = resp.json()
        if not events:
            break
        all_events.extend(events)
        if len(events) < limit:
            break
        offset += limit
        time.sleep(0.5)

    return all_events


def download_as_wav(session, host, event, output_path):
    """Download event clip as MP4 to temp file, then extract audio."""
    import tempfile
    camera_id = event.get("camera")
    start = event.get("start")
    end = event.get("end")

    if not all([camera_id, start, end]):
        return False

    # Ensure minimum 5 second clip, max 30 seconds
    duration = end - start
    if duration < 5000:
        end = start + 5000
    if duration > 30000:
        end = start + 30000

    base = f"https://{host}/proxy/protect"
    resp = session.get(
        f"{base}/api/video/export",
        params={"camera": camera_id, "start": start, "end": end},
        stream=True,
        timeout=60,
    )
    if resp.status_code != 200:
        return False

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)

    try:
        if os.path.getsize(tmp_path) < 1000:
            os.unlink(tmp_path)
            return False

        result = subprocess.run(
            ["ffmpeg", "-i", tmp_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", "-y", str(output_path)],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Download negative audio samples from Unifi Protect")
    parser.add_argument("--local-ip", default="192.168.1.1", help="Console IP")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--max-clips", type=int, default=500, help="Max clips to download")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = login(args.local_ip)

    # Fetch speech events
    print("Fetching speech events...")
    speech_events = get_events(session, args.local_ip, "smartAudioDetect", days_back=args.days)
    speech_events = [e for e in speech_events if "alrmSpeak" in e.get("smartDetectTypes", [])]
    print(f"  Found {len(speech_events)} speech events")

    # Fetch motion events (ambient activity noise)
    print("Fetching motion events...")
    motion_events = get_events(session, args.local_ip, "motion", days_back=args.days)
    print(f"  Found {len(motion_events)} motion events")

    # Interleave speech and motion for diversity
    for e in speech_events:
        e["_source"] = "speech"
    for e in motion_events:
        e["_source"] = "motion"

    all_events = []
    si, mi = 0, 0
    while len(all_events) < args.max_clips and (si < len(speech_events) or mi < len(motion_events)):
        # Alternate: 2 speech, 1 motion
        for _ in range(2):
            if si < len(speech_events):
                all_events.append(speech_events[si])
                si += 1
        if mi < len(motion_events):
            all_events.append(motion_events[mi])
            mi += 1

    all_events = all_events[:args.max_clips]
    print(f"\nDownloading {len(all_events)} clips as audio-only WAV...")

    downloaded = 0
    skipped = 0

    for event in tqdm(all_events, desc="Downloading"):
        event_id = event.get("id", "unknown")
        start_ts = event.get("start", 0)
        source = event.get("_source", "unknown")
        dt = datetime.fromtimestamp(start_ts / 1000)
        filename = f"{dt.strftime('%Y%m%d_%H%M%S')}_{event_id[:8]}_{source}.wav"
        output_path = OUTPUT_DIR / filename

        if output_path.exists():
            skipped += 1
            continue

        try:
            if download_as_wav(session, args.local_ip, event, output_path):
                downloaded += 1
            else:
                skipped += 1
                # Clean up empty files
                if output_path.exists() and output_path.stat().st_size < 1000:
                    output_path.unlink()
        except Exception as e:
            skipped += 1
            if output_path.exists():
                output_path.unlink()

        time.sleep(0.3)

    print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}")
    print(f"Negative samples saved to: {OUTPUT_DIR}/")
    total = len(list(OUTPUT_DIR.glob("*.wav")))
    print(f"Total negative clips: {total}")


if __name__ == "__main__":
    main()

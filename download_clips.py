"""
Download dog bark/whine detection clips from Unifi Protect via cloud API.

Authenticates through sso.ui.com, then pulls smart detection events
(alrmBark) and downloads the associated video clips.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import requests
import urllib3
from dotenv import load_dotenv
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

SSO_LOGIN_URL = "https://sso.ui.com/api/sso/v1/login"
CONSOLE_BASE = "https://unifi.ui.com/api"

USERNAME = os.getenv("UNIFI_USERNAME")
PASSWORD = os.getenv("UNIFI_PASSWORD")
CONSOLE_ID = os.getenv("UNIFI_CONSOLE_ID")
SITE_ID = os.getenv("UNIFI_SITE_ID")

DATA_DIR = Path("data")
BARKING_DIR = DATA_DIR / "barking"
WHINING_DIR = DATA_DIR / "whining"
UNSORTED_DIR = DATA_DIR / "unsorted"


class UnifiProtectClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Content-Type": "application/json",
        })
        self.csrf_token = None
        self.console_host = None

    def login(self):
        """Authenticate via Unifi SSO with 2FA support."""
        print("Authenticating with Unifi SSO...")

        # Step 1: SSO login
        payload = {
            "user": USERNAME,
            "password": PASSWORD,
            "rememberMe": False,
        }
        resp = self.session.post(SSO_LOGIN_URL, json=payload)

        # Handle 2FA requirement
        if resp.status_code == 499:
            data = resp.json()
            if data.get("required") == "2fa":
                authenticators = data.get("authenticators", [])
                # Find email authenticator
                email_auth = next(
                    (a for a in authenticators if a["type"] == "email"),
                    None,
                )
                if not email_auth:
                    print("No email 2FA authenticator found.")
                    sys.exit(1)

                auth_id = email_auth["id"]
                email_hint = email_auth.get("email", "your email")
                print(f"2FA required. Sending code to {email_hint}...")

                # Request 2FA code be sent
                send_resp = self.session.post(
                    f"https://sso.ui.com/api/sso/v1/2fa/{auth_id}/send",
                    json={},
                )
                if send_resp.status_code not in (200, 201, 204):
                    print(f"Failed to send 2FA code: {send_resp.status_code}")
                    print(send_resp.text[:500])
                    sys.exit(1)

                # Prompt user for the code
                code = input("Enter the 2FA code from your email: ").strip()

                # Submit 2FA code
                verify_resp = self.session.post(
                    f"https://sso.ui.com/api/sso/v1/2fa/{auth_id}/verify",
                    json={"code": code},
                )
                if verify_resp.status_code not in (200, 201):
                    print(f"2FA verification failed: {verify_resp.status_code}")
                    print(verify_resp.text[:500])
                    sys.exit(1)

                print("2FA verified successfully.")
            else:
                print(f"Unexpected 2FA response: {resp.text[:500]}")
                sys.exit(1)
        elif resp.status_code not in (200, 201):
            print(f"SSO login failed: {resp.status_code}")
            print(resp.text[:500])
            sys.exit(1)

        print("SSO login successful.")

        # Step 2: Get console proxy info
        # The cloud UI proxies requests to the local console
        self._resolve_console()

    def _resolve_console(self):
        """Resolve the console's cloud proxy endpoint."""
        # Try the standard cloud API proxy path
        resp = self.session.get(
            f"https://unifi.ui.com/api/consoles/{CONSOLE_ID}/proxy/protect/api/bootstrap",
            timeout=30,
        )
        if resp.status_code == 200:
            bootstrap = resp.json()
            self.console_host = f"https://unifi.ui.com/api/consoles/{CONSOLE_ID}/proxy/protect"
            nvr = bootstrap.get("nvr", {})
            print(f"Connected to NVR: {nvr.get('name', 'Unknown')}")
            print(f"Version: {nvr.get('version', 'Unknown')}")
            cameras = bootstrap.get("cameras", [])
            print(f"Cameras: {len(cameras)}")
            for cam in cameras:
                print(f"  - {cam.get('name', 'Unknown')} ({cam['id']})")
            return

        # Alternate: try the direct site-based path
        resp2 = self.session.get(
            f"https://unifi.ui.com/proxy/protect/api/bootstrap",
            headers={"X-Console-Id": CONSOLE_ID},
            timeout=30,
        )
        if resp2.status_code == 200:
            self.console_host = "https://unifi.ui.com/proxy/protect"
            print("Connected via alternate proxy path.")
            return

        print(f"Failed to connect to console. Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
        print("\nYou may need to provide your console's local IP address.")
        print("Run: python download_clips.py --local-ip <YOUR_CONSOLE_IP>")
        sys.exit(1)

    def login_local(self, host, tfa_code=None, send_2fa_only=False):
        """Authenticate directly to the local console with 2FA support."""
        self.local_host = host
        self.console_host = f"https://{host}/proxy/protect"
        print(f"Authenticating locally to {host}...")

        self.session.verify = False

        resp = self.session.post(
            f"https://{host}/api/auth/login",
            json={"username": USERNAME, "password": PASSWORD},
            timeout=10,
        )

        if resp.status_code == 499:
            data = resp.json()
            if data.get("code") == "MFA_AUTH_REQUIRED" or (
                isinstance(data.get("data"), dict) and data["data"].get("required") == "2fa"
            ):
                mfa_data = data.get("data", data)
                self._handle_local_2fa(host, mfa_data, tfa_code, send_2fa_only)
                return

        if resp.status_code != 200:
            print(f"Local login failed: {resp.status_code}")
            print(resp.text[:500])
            sys.exit(1)

        self._extract_tokens(resp)
        print("Local login successful.")

    def _handle_local_2fa(self, host, mfa_data, tfa_code=None, send_2fa_only=False):
        """Handle 2FA challenge on local console."""
        authenticators = mfa_data.get("authenticators", [])
        email_auth = next(
            (a for a in authenticators if a["type"] == "email"), None
        )
        if not email_auth:
            print("No email 2FA authenticator found. Available types:")
            for a in authenticators:
                print(f"  - {a['type']} (id: {a['id']})")
            sys.exit(1)

        auth_id = email_auth["id"]
        email_hint = email_auth.get("email", "your email")

        if send_2fa_only or not tfa_code:
            print(f"2FA required. Attempting to send code to {email_hint}...")

            # Try multiple endpoint patterns for sending 2FA code
            send_endpoints = [
                f"https://{host}/api/auth/2fa/{auth_id}/send",
                f"https://{host}/api/auth/login/2fa/{auth_id}/send",
                f"https://sso.ui.com/api/sso/v1/login/2fa/{auth_id}/send",
                f"https://sso.ui.com/api/sso/v1/2fa/send",
            ]

            sent = False
            for endpoint in send_endpoints:
                try:
                    resp = self.session.post(endpoint, json={}, timeout=10)
                    print(f"  Tried {endpoint} → {resp.status_code}")
                    if resp.status_code in (200, 201, 204):
                        print("2FA code sent successfully!")
                        sent = True
                        break
                except Exception as e:
                    print(f"  Tried {endpoint} → error: {e}")

            if not sent:
                print("Could not explicitly trigger 2FA send — code may have been sent automatically.")
                print("Check your email for a verification code.")

            if send_2fa_only or not tfa_code:
                print("\nRun again with: python download_clips.py --local-ip "
                      f"{host} --2fa-code <YOUR_CODE>")
                sys.exit(0)

        print(f"Verifying 2FA code: {tfa_code}")

        # Try multiple endpoint patterns for verifying 2FA code
        verify_endpoints = [
            (f"https://{host}/api/auth/login", {
                "username": USERNAME,
                "password": PASSWORD,
                "rememberMe": False,
                "ubic_2fa_token": tfa_code,
                "token": tfa_code,
            }),
            (f"https://{host}/api/auth/2fa/{auth_id}/verify", {
                "code": tfa_code,
            }),
            (f"https://sso.ui.com/api/sso/v1/login", {
                "user": USERNAME,
                "password": PASSWORD,
                "rememberMe": False,
                "ubic_2fa_token": tfa_code,
                "token": tfa_code,
            }),
        ]

        for endpoint, payload in verify_endpoints:
            resp = self.session.post(endpoint, json=payload, timeout=10)
            print(f"  Tried {endpoint} → {resp.status_code}")
            if resp.status_code == 200:
                self._extract_tokens(resp)
                print("2FA verification successful. Logged in.")
                return

        print("2FA verification failed on all endpoints.")
        print(f"Last response: {resp.status_code} - {resp.text[:500]}")
        sys.exit(1)

    def _extract_tokens(self, resp):
        """Extract auth tokens from login response."""
        self.csrf_token = resp.headers.get("X-CSRF-Token")
        if self.csrf_token:
            self.session.headers["X-CSRF-Token"] = self.csrf_token

        # Some consoles return token in response body
        try:
            data = resp.json()
            if "token" in data:
                self.session.headers["Authorization"] = f"Bearer {data['token']}"
        except (ValueError, KeyError):
            pass

    def get_events(self, start_ts=None, end_ts=None, limit=100, offset=0):
        """Fetch smart audio detection events for bark type."""
        params = {
            "orderDirection": "DESC",
            "limit": limit,
            "offset": offset,
            "types": "smartAudioDetect",
        }
        if start_ts:
            params["start"] = start_ts
        if end_ts:
            params["end"] = end_ts

        resp = self.session.get(
            f"{self.console_host}/api/events",
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"Failed to fetch events: {resp.status_code}")
            print(resp.text[:500])
            return []

        return resp.json()

    def get_all_events(self, days_back=90):
        """Paginate through all bark detection events."""
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - days_back * 86400) * 1000)

        all_events = []
        offset = 0
        batch_size = 100

        print(f"Fetching audio detection events from the last {days_back} days...")

        while True:
            events = self.get_events(
                start_ts=start_ts,
                end_ts=end_ts,
                limit=batch_size,
                offset=offset,
            )
            if not events:
                break

            # Filter for alrmBark events only
            bark_events = [
                e for e in events
                if "alrmBark" in e.get("smartDetectTypes", [])
            ]
            all_events.extend(bark_events)
            print(f"  Batch: {len(events)} total, {len(bark_events)} bark. Running total: {len(all_events)}")

            if len(events) < batch_size:
                break
            offset += batch_size
            time.sleep(0.5)  # be gentle on the API

        print(f"Total bark events found: {len(all_events)}")
        return all_events

    def download_clip(self, event, output_path, audio_only=False):
        """Download clip for a specific event. If audio_only, streams
        through ffmpeg to extract 16kHz mono WAV on the fly."""
        camera_id = event.get("camera")
        start = event.get("start")
        end = event.get("end")

        if not all([camera_id, start, end]):
            print(f"  Skipping event {event.get('id')}: missing camera/start/end")
            return False

        # Add small buffer around the event
        start_buffered = start - 2000  # 2s before
        end_buffered = end + 2000      # 2s after

        url = f"{self.console_host}/api/video/export"
        params = {
            "camera": camera_id,
            "start": start_buffered,
            "end": end_buffered,
        }

        try:
            resp = self.session.get(url, params=params, stream=True, timeout=60)
            if resp.status_code != 200:
                # Try alternate endpoint
                url2 = f"{self.console_host}/api/cameras/{camera_id}/video"
                resp = self.session.get(
                    url2,
                    params={"start": start_buffered, "end": end_buffered},
                    stream=True,
                    timeout=60,
                )
                if resp.status_code != 200:
                    print(f"  Failed to download clip: {resp.status_code}")
                    return False

            if audio_only:
                return self._stream_to_wav(resp, output_path)
            else:
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True

        except requests.exceptions.RequestException as e:
            print(f"  Download error: {e}")
            return False

    def _stream_to_wav(self, resp, output_path):
        """Pipe streamed MP4 through ffmpeg to extract audio as WAV."""
        import subprocess
        cmd = [
            "ffmpeg", "-i", "pipe:0",
            "-vn",                    # drop video
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", "16000",           # 16kHz
            "-ac", "1",               # mono
            "-y",                     # overwrite
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            for chunk in resp.iter_content(chunk_size=8192):
                proc.stdin.write(chunk)
            proc.stdin.close()
            proc.wait(timeout=30)
            return proc.returncode == 0
        except Exception:
            proc.kill()
            return False


def main():
    parser = argparse.ArgumentParser(description="Download bark detection clips from Unifi Protect")
    parser.add_argument("--local-ip", help="Console's local IP for direct access")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch (default: 90)")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--2fa-code", dest="tfa_code", help="2FA code from email")
    parser.add_argument("--send-2fa", action="store_true", help="Just send the 2FA code and exit")
    parser.add_argument("--audio-only", action="store_true",
                        help="Download as 16kHz mono WAV instead of MP4 (much smaller)")
    args = parser.parse_args()

    global DATA_DIR, BARKING_DIR, WHINING_DIR, UNSORTED_DIR
    DATA_DIR = Path(args.output_dir)
    BARKING_DIR = DATA_DIR / "barking"
    WHINING_DIR = DATA_DIR / "whining"
    UNSORTED_DIR = DATA_DIR / "unsorted"

    # Create directories
    for d in [BARKING_DIR, WHINING_DIR, UNSORTED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    client = UnifiProtectClient()

    if args.local_ip:
        client.login_local(args.local_ip, tfa_code=args.tfa_code, send_2fa_only=args.send_2fa)
    else:
        client.login()

    # Fetch all bark events
    events = client.get_all_events(days_back=args.days)

    if not events:
        print("No bark events found.")
        return

    # Save events metadata
    meta_path = DATA_DIR / "events_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved event metadata to {meta_path}")

    # Download clips
    ext = ".wav" if args.audio_only else ".mp4"
    mode_label = "audio-only WAV" if args.audio_only else "MP4 video"
    print(f"\nDownloading {len(events)} clips as {mode_label} to {UNSORTED_DIR}/...")
    downloaded = 0
    skipped = 0

    for event in tqdm(events, desc="Downloading"):
        event_id = event.get("id", "unknown")
        start_ts = event.get("start", 0)
        dt = datetime.fromtimestamp(start_ts / 1000)
        filename = f"{dt.strftime('%Y%m%d_%H%M%S')}_{event_id[:8]}{ext}"
        output_path = UNSORTED_DIR / filename

        if output_path.exists():
            skipped += 1
            continue

        if client.download_clip(event, output_path, audio_only=args.audio_only):
            downloaded += 1
        else:
            skipped += 1

        time.sleep(0.3)

    print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}")
    print(f"\nClips saved to: {UNSORTED_DIR}/")
    if args.audio_only:
        print(f"Next step: Run 'python presort.py --input {UNSORTED_DIR}' to auto-classify.")
    else:
        print(f"Next step: Run 'python extract_audio.py' to extract audio from clips.")
        print(f"Or run 'python presort.py --input {UNSORTED_DIR}' to auto-classify.")


if __name__ == "__main__":
    main()

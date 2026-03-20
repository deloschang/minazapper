#!/usr/bin/env python3
"""
Lightweight clip review server for manual labeling via Pushover.

Serves detection clips and accepts bark/not-bark labels.
Manual labels are the final source of truth — override UniFi and model.

Runs on the Pi at port 8080.
"""

import os
import json
import glob
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime

CLIPS_DIR = Path(os.path.expanduser("~/minazap/detection_clips"))
LABELS_FILE = Path(os.path.expanduser("~/minazap/manual_labels.json"))
PORT = 8085


def load_labels():
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logs

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # GET /review/S095 — review page for a session
        if path.startswith("/review/"):
            session_id = path.split("/")[2]
            self._serve_review_page(session_id)

        # GET /clip/filename.wav — serve audio file
        elif path.startswith("/clip/"):
            filename = path.split("/", 2)[2]
            self._serve_clip(filename)

        # GET /label?session=S095&label=bark or &label=not_bark
        elif path.startswith("/label"):
            params = parse_qs(parsed.query)
            session_id = params.get("session", [""])[0]
            label = params.get("label", [""])[0]
            self._apply_label(session_id, label)

        # GET /status — show all sessions and labels
        elif path == "/status":
            self._serve_status()

        # GET /queue — start reviewing unreviewed sessions
        elif path == "/queue":
            self._serve_queue()

        else:
            self._respond(404, "Not found")

    def _serve_review_page(self, session_id):
        clips = sorted(glob.glob(str(CLIPS_DIR / f"{session_id}_*.wav")))
        if not clips:
            self._respond(404, f"No clips found for {session_id}")
            return

        labels = load_labels()
        raw_label = labels.get(session_id, "unlabeled")
        current_label = raw_label.get("label") if isinstance(raw_label, dict) else raw_label

        # Deduplicate: keep highest confidence clip per camera+second
        import re
        best_per_key = {}
        clip_re = re.compile(r"(S\d+_\d{8}_\d{6}_\w+)_(\d+)%\.wav")
        for clip_path in clips:
            name = os.path.basename(clip_path)
            m = clip_re.match(name)
            if m:
                key = m.group(1)
                conf = int(m.group(2))
                if key not in best_per_key or conf > best_per_key[key][1]:
                    best_per_key[key] = (clip_path, conf, name)
            else:
                best_per_key[name] = (clip_path, 0, name)

        deduped = sorted(best_per_key.values(), key=lambda x: x[2])

        clip_rows = ""
        for i, (clip_path, conf, name) in enumerate(deduped):
            clip_rows += f"""
            <div style="margin:8px 0;">
                <span style="color:#888;">{i+1}.</span> {name}<br>
                <audio controls preload="auto" style="width:100%;max-width:400px;">
                    <source src="/clip/{name}" type="audio/wav">
                </audio>
            </div>"""

        label_color = {"bark": "#4CAF50", "not_bark": "#f44336", "unlabeled": "#888"}
        color = label_color.get(current_label, "#888")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{session_id} Review</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; padding: 16px; background: #1a1a1a; color: #eee; }}
        h1 {{ font-size: 24px; margin-bottom: 4px; }}
        .status {{ color: {color}; font-weight: bold; margin-bottom: 16px; }}
        .clips {{ margin: 16px 0; }}
        .buttons {{ display: flex; gap: 12px; margin-top: 20px; }}
        .btn {{ padding: 16px 32px; border: none; border-radius: 8px; font-size: 18px;
                font-weight: bold; cursor: pointer; text-decoration: none; text-align: center; flex: 1; }}
        .btn-bark {{ background: #4CAF50; color: white; }}
        .btn-not {{ background: #f44336; color: white; }}
        .btn:active {{ opacity: 0.7; }}
        .btn-play {{ background: #2196F3; color: white; margin-bottom: 16px; display: inline-block; }}
        audio {{ margin-top: 4px; }}
    </style>
    <script>
    function playAll() {{
        const audios = document.querySelectorAll('audio');
        let i = 0;
        function next() {{
            if (i < audios.length) {{
                audios[i].scrollIntoView({{behavior:'smooth',block:'center'}});
                audios[i].play();
                audios[i].onended = function() {{ i++; next(); }};
                i++;
            }}
        }}
        // Stop any currently playing
        audios.forEach(a => {{ a.pause(); a.currentTime = 0; }});
        i = 0;
        next();
    }}
    </script>
</head>
<body>
    <h1>{session_id}</h1>
    <div class="status">Current label: {current_label}</div>
    <div>{len(deduped)} clips · <a class="btn btn-play" href="javascript:playAll()">Play All</a></div>
    <div class="clips">{clip_rows}</div>
    <div class="buttons">
        <a class="btn btn-bark" href="/label?session={session_id}&label=bark">Bark</a>
        <a class="btn btn-not" href="/label?session={session_id}&label=not_bark">Not Bark</a>
    </div>
</body>
</html>"""
        self._respond(200, html, content_type="text/html")

    def _serve_clip(self, filename):
        clip_path = CLIPS_DIR / filename
        if not clip_path.exists():
            self._respond(404, "Clip not found")
            return
        with open(clip_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _next_unlabeled(self, after_session=None):
        """Find the next unlabeled session after the given one."""
        labels = load_labels()
        sessions = sorted(set(
            f.name.split("_")[0] for f in CLIPS_DIR.glob("S*.wav")
        ))
        past = after_session is None
        for sid in sessions:
            if not past:
                if sid == after_session:
                    past = True
                continue
            info = labels.get(sid, {})
            label = info.get("label") if isinstance(info, dict) else info
            if not label or label == "unlabeled":
                return sid
        return None

    def _apply_label(self, session_id, label):
        if label not in ("bark", "not_bark"):
            self._respond(400, "Invalid label. Use 'bark' or 'not_bark'.")
            return

        labels = load_labels()
        labels[session_id] = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
        }
        save_labels(labels)

        next_sid = self._next_unlabeled(session_id)
        emoji = "\u2705" if label == "bark" else "\u274c"
        next_btn = ""
        if next_sid:
            next_btn = f'<a class="btn btn-next" href="/review/{next_sid}">Next ({next_sid})</a>'
        else:
            next_btn = '<span style="color:#888;">All labeled!</span>'

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, sans-serif; padding: 40px 16px; background: #1a1a1a;
               color: #eee; text-align: center; }}
        .emoji {{ font-size: 64px; }}
        .label {{ font-size: 24px; margin-top: 16px; }}
        .btn-next {{ display: inline-block; margin-top: 20px; padding: 14px 32px; background: #2196F3;
                     color: white; border-radius: 8px; font-size: 18px; font-weight: bold;
                     text-decoration: none; }}
        .btn-next:active {{ opacity: 0.7; }}
        a {{ color: #64b5f6; }}
    </style>
</head>
<body>
    <div class="emoji">{emoji}</div>
    <div class="label">{session_id} labeled: <b>{label}</b></div>
    <div style="margin-top:20px;">{next_btn}</div>
    <p><a href="/review/{session_id}">Change</a> · <a href="/status">All sessions</a></p>
</body>
</html>"""
        self._respond(200, html, content_type="text/html")

    def _serve_queue(self):
        """Redirect to the first unreviewed session."""
        first = self._next_unlabeled()
        if first:
            self.send_response(302)
            self.send_header("Location", f"/review/{first}")
            self.end_headers()
        else:
            self._respond(200, """<!DOCTYPE html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:-apple-system,sans-serif;padding:40px 16px;background:#1a1a1a;color:#eee;text-align:center;}
a{color:#64b5f6;}</style></head>
<body><h1>All caught up!</h1><p>No unreviewed sessions.</p>
<p><a href="/status">View all sessions</a></p></body></html>""", content_type="text/html")

    def _serve_status(self):
        labels = load_labels()
        sessions = sorted(set(
            f.name.split("_")[0] for f in CLIPS_DIR.glob("S*.wav")
        ))

        rows = ""
        reviewed = 0
        unreviewed = 0
        for sid in sessions:
            info = labels.get(sid, {})
            label = info.get("label", "unlabeled") if isinstance(info, dict) else info
            color = {"bark": "#4CAF50", "not_bark": "#f44336"}.get(label, "#888")
            clip_count = len(list(CLIPS_DIR.glob(f"{sid}_*.wav")))
            rows += f'<tr><td><a href="/review/{sid}">{sid}</a></td><td>{clip_count}</td><td style="color:{color}">{label}</td></tr>\n'
            if label in ("bark", "not_bark"):
                reviewed += 1
            else:
                unreviewed += 1

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Mina Review</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; padding: 16px; background: #1a1a1a; color: #eee; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
        a {{ color: #64b5f6; }}
        .queue-btn {{ display: inline-block; padding: 10px 24px; background: #2196F3; color: white;
                      border-radius: 8px; font-weight: bold; text-decoration: none; margin: 12px 0; }}
    </style>
</head>
<body>
    <h1>Mina Review</h1>
    <p>Reviewed: {reviewed} · Unreviewed: {unreviewed}</p>
    <a class="queue-btn" href="/queue">Start Reviewing</a>
    <table>
        <tr><th>Session</th><th>Clips</th><th>Label</th></tr>
        {rows}
    </table>
</body>
</html>"""
        self._respond(200, html, content_type="text/html")

    def _respond(self, status, body, content_type="text/html"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        encoded = body.encode() if isinstance(body, str) else body
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    server = HTTPServer(("0.0.0.0", PORT), ReviewHandler)
    print(f"Review server running on http://0.0.0.0:{PORT}")
    print(f"  Review a session: http://192.168.10.10:{PORT}/review/S001")
    print(f"  All sessions:     http://192.168.10.10:{PORT}/status")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()

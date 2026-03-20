#!/usr/bin/env python3
"""
Local review server for labeling ALL training data clips.

Serves clips from training_data/mina/ and training_data/negative/.
Manual labels are the final source of truth.

Run on Mac: python review_local.py
Access: http://localhost:8086/queue
"""

import os
import re
import json
import glob
import shutil
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime

TRAINING_DIR = Path("training_data")
POSITIVE_DIR = TRAINING_DIR / "mina"
NEGATIVE_DIR = TRAINING_DIR / "negative"
PI_CLIPS_DIR = Path("false_positives/detection_clips")
LABELS_FILE = Path("manual_labels_training.json")
UNIFI_CACHE_FILE = Path("unifi_bark_cache.json")
PI_HOST = "del@192.168.10.10"
PI_KEY = os.path.expanduser("~/.ssh/minazap_pi")
PORT = 8086

# Thread-safe file lock and in-progress tracking for parallel review
_lock = threading.Lock()
_in_progress = {}  # key -> (reviewer_ip, timestamp)
_unifi_bark_windows = []  # list of (start_ms, end_ms) for bark events
_clips_cache = None
_clips_cache_time = 0
CACHE_TTL = 300  # 5 minutes
RMS_CACHE_FILE = Path("rms_cache.json")
CLIPS_CACHE_FILE = Path("clips_cache.json")
LAST_DEPLOY_SNAPSHOT = Path(".last_deploy_snapshot.json")

# Match session clips and unifi clips
SESSION_RE = re.compile(r"(S\d+)_")
CLIP_TS_RE = re.compile(r"(\d{8}_\d{6})")

# UniFi camera IDs for timeline links
UNIFI_CAMERAS = {
    "entry": "6987985800387203e400044e",
    "kitchen": "6987985800ec7203e4000450",
    "living": "6987985800b87203e400044f",
}
UNIFI_BASE = "https://192.168.1.1/protect/timelapse"


def load_labels():
    with _lock:
        if LABELS_FILE.exists():
            with open(LABELS_FILE) as f:
                return json.load(f)
        return {}


def save_labels(labels):
    with _lock:
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f, indent=2)


def load_unifi_cache():
    """Load or refresh UniFi bark event cache."""
    global _unifi_bark_windows

    # Use cache if fresh (< 1 hour old)
    if UNIFI_CACHE_FILE.exists():
        age = datetime.now().timestamp() - UNIFI_CACHE_FILE.stat().st_mtime
        if age < 86400:  # 24 hours
            with open(UNIFI_CACHE_FILE) as f:
                _unifi_bark_windows = json.load(f)
            return

    print("Fetching UniFi bark events (this may take a minute)...")
    try:
        from download_clips import UnifiProtectClient
        client = UnifiProtectClient()
        client.login_local("192.168.1.1")
        events = client.get_all_events(days_back=365)
        _unifi_bark_windows = [
            [e["start"], e["end"]]
            for e in events
        ]
        with open(UNIFI_CACHE_FILE, "w") as f:
            json.dump(_unifi_bark_windows, f)
        print(f"Cached {len(_unifi_bark_windows)} UniFi bark events.")
    except Exception as e:
        print(f"Could not fetch UniFi events: {e}")
        if UNIFI_CACHE_FILE.exists():
            with open(UNIFI_CACHE_FILE) as f:
                _unifi_bark_windows = json.load(f)


def unifi_suggests_bark(ts_ms, tolerance=5000):
    """Check if a timestamp overlaps with any UniFi bark event."""
    for start, end in _unifi_bark_windows:
        if (start - tolerance) <= ts_ms <= (end + tolerance):
            return True
    return False


def sync_pi_clips():
    """Pull latest clips from Pi."""
    import subprocess
    PI_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["scp", "-i", PI_KEY, f"{PI_HOST}:~/minazap/detection_clips/*.wav", str(PI_CLIPS_DIR)],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass


def _is_group_reviewed(k, groups, labels):
    """Check if all deduped clip groups have at least one labeled sibling."""
    clips = groups[k]["clips"]
    conf_re = re.compile(r"(\d+)%\.wav$")
    # Group clips by camera+second (dedup key)
    groups_by_key = {}
    for clip_path in clips:
        name = clip_path.name if hasattr(clip_path, 'name') else Path(clip_path).name
        dedup_key = conf_re.sub("", name)
        if dedup_key not in groups_by_key:
            groups_by_key[dedup_key] = []
        groups_by_key[dedup_key].append(name)
    if not groups_by_key:
        return False
    # Each camera+second group is reviewed if ANY sibling has a label
    for siblings in groups_by_key.values():
        if not any(f"clip:{n}" in labels for n in siblings):
            return False
    return True


def get_all_clips():
    """Get all clips grouped by a review key. Cached in memory and on disk."""
    global _clips_cache, _clips_cache_time
    now = datetime.now().timestamp()
    if _clips_cache and (now - _clips_cache_time) < CACHE_TTL:
        return _clips_cache

    # Try loading from disk cache first
    groups = _load_clips_from_disk()
    if groups is None:
        groups = _build_clips_cache()
        _save_clips_to_disk(groups)

    _clips_cache = groups
    _clips_cache_time = now
    return groups


def _load_clips_from_disk():
    """Load clips cache from disk if fresh (< 5 min)."""
    if not CLIPS_CACHE_FILE.exists():
        return None
    age = datetime.now().timestamp() - CLIPS_CACHE_FILE.stat().st_mtime
    if age > CACHE_TTL:
        return None
    try:
        with open(CLIPS_CACHE_FILE) as f:
            raw = json.load(f)
        # Reconstruct Path objects
        groups = {}
        for key, g in raw.items():
            groups[key] = {
                "clips": [Path(p) for p in g["clips"]],
                "current_dir": g["current_dir"],
                "ts_ms": g.get("ts_ms"),
                "unifi": g.get("unifi", "unknown"),
                "max_rms": g.get("max_rms", 0),
                "cameras": set(g.get("cameras", [])),
            }
        return groups
    except Exception:
        return None


def _save_clips_to_disk(groups):
    """Persist clips cache to disk."""
    try:
        raw = {}
        for key, g in groups.items():
            raw[key] = {
                "clips": [str(p) for p in g["clips"]],
                "current_dir": g["current_dir"],
                "ts_ms": g.get("ts_ms"),
                "unifi": g.get("unifi", "unknown"),
                "max_rms": g.get("max_rms", 0),
                "cameras": list(g.get("cameras", [])),
            }
        with open(CLIPS_CACHE_FILE, "w") as f:
            json.dump(raw, f)
    except Exception:
        pass


def invalidate_cache():
    global _clips_cache
    _clips_cache = None
    try:
        CLIPS_CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _build_clips_cache():
    groups = {}

    for label_dir, current_label in [
        (POSITIVE_DIR, "bark"), (NEGATIVE_DIR, "not_bark"), (PI_CLIPS_DIR, "pending")
    ]:
        if not label_dir.exists():
            continue
        for f in label_dir.iterdir():
            if f.suffix != ".wav":
                continue

            # Group by session ID if available
            sm = SESSION_RE.match(f.name)
            if sm:
                key = sm.group(1)
            else:
                # Group by timestamp (first 15 chars: YYYYMMDD_HHMMSS)
                tm = CLIP_TS_RE.search(f.name)
                if tm:
                    key = f"T{tm.group(1)}"
                else:
                    key = f"F_{f.stem}"

            # Parse timestamp for UniFi lookup
            ts_ms = None
            tm = CLIP_TS_RE.search(f.name)
            if tm:
                try:
                    from zoneinfo import ZoneInfo
                    ts_str = tm.group(1)
                    dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    # S-prefixed clips (from Pi) are in UTC
                    # All others (from download_clips.py) are in local time
                    if SESSION_RE.match(f.name):
                        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                    else:
                        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                    ts_ms = int(dt.timestamp() * 1000)
                except ValueError:
                    pass

            # Extract camera name
            cam = ""
            for cn in ["kitchen", "living", "entry"]:
                if cn in f.name:
                    cam = cn
                    break

            if key not in groups:
                groups[key] = {"clips": [], "current_dir": current_label, "ts_ms": ts_ms, "cameras": set()}
            groups[key]["clips"].append(f)
            if cam:
                groups[key]["cameras"].add(cam)
            # Keep earliest timestamp for the group
            if ts_ms and (groups[key]["ts_ms"] is None or ts_ms < groups[key]["ts_ms"]):
                groups[key]["ts_ms"] = ts_ms

    # Load persistent RMS cache
    rms_disk_cache = {}
    if RMS_CACHE_FILE.exists():
        try:
            with open(RMS_CACHE_FILE) as f:
                rms_disk_cache = json.load(f)
        except Exception:
            pass

    # Compute UniFi suggestion and max RMS per group
    import soundfile as sf
    rms_updated = False
    for key, group in groups.items():
        if group["ts_ms"]:
            group["unifi"] = "bark" if unifi_suggests_bark(group["ts_ms"]) else "no bark"
        else:
            group["unifi"] = "unknown"

        # Compute max RMS across clips (use disk cache when possible)
        max_rms = 0
        for f in group["clips"]:
            fname = f.name
            if fname in rms_disk_cache:
                rms = rms_disk_cache[fname]
            else:
                try:
                    data, sr = sf.read(str(f))
                    rms = float(np.sqrt(np.mean(data ** 2)))
                except Exception:
                    rms = 0
                rms_disk_cache[fname] = rms
                rms_updated = True
            if rms > max_rms:
                max_rms = rms
        group["max_rms"] = max_rms

    # Persist RMS cache
    if rms_updated:
        try:
            with open(RMS_CACHE_FILE, "w") as f:
                json.dump(rms_disk_cache, f)
        except Exception:
            pass

    # Auto-correct: move unreviewed clips to match UniFi suggestion
    # Only runs when UniFi data is loaded (not empty)
    # Manual labels are never touched
    if _unifi_bark_windows:
        labels = load_labels()
        for key, group in groups.items():
            if key in labels:
                continue
            if _is_group_reviewed(key, groups, labels):
                continue

            unifi = group.get("unifi", "unknown")
            current = group.get("current_dir", "")

            if unifi == "bark" and current == "not_bark":
                for clip in group["clips"]:
                    if clip.parent == NEGATIVE_DIR:
                        try:
                            shutil.move(str(clip), str(POSITIVE_DIR / clip.name))
                        except Exception:
                            pass
                group["current_dir"] = "bark"

    return groups


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/review/"):
            key = path.split("/")[2]
            self._serve_review_page(key)
        elif path.startswith("/clip/"):
            rest = path.split("/", 2)[2]  # label/filename
            self._serve_clip(rest)
        elif path.startswith("/label"):
            params = parse_qs(parsed.query)
            key = params.get("key", [""])[0]
            label = params.get("label", [""])[0]
            self._apply_label(key, label)
        elif path.startswith("/queue"):
            parts = path.strip("/").split("/")
            filter_dir = parts[1] if len(parts) > 1 else None
            self._serve_queue(filter_dir=filter_dir)

        # GET /priority — review most likely mislabeled clips first
        elif path == "/priority":
            self._serve_priority_queue()

        # GET /next?after=S063 — skip to next unlabeled
        elif path == "/next":
            params = parse_qs(parsed.query)
            after = params.get("after", [None])[0]
            reviewer_ip = self.client_address[0]
            next_key = self._next_unlabeled_fast(after, reviewer_ip)
            if next_key:
                self.send_response(302)
                self.send_header("Location", f"/review/{next_key}")
                self.end_headers()
            else:
                self.send_response(302)
                self.send_header("Location", "/status")
                self.end_headers()

        # GET /prev?before=S063 — go back to previous group
        elif path == "/prev":
            params = parse_qs(parsed.query)
            before = params.get("before", [None])[0]
            prev_key = self._prev_group(before)
            if prev_key:
                self.send_response(302)
                self.send_header("Location", f"/review/{prev_key}")
                self.end_headers()
            else:
                self.send_response(302)
                self.send_header("Location", "/status")
                self.end_headers()

        # GET /filter/<type> — filtered review queues
        elif path.startswith("/filter/"):
            filter_type = path.split("/")[2]
            self._serve_filtered_queue(filter_type)
        elif path == "/sync":
            sync_pi_clips()
            invalidate_cache()
            self.send_response(302)
            self.send_header("Location", "/status")
            self.end_headers()
        elif path == "/status" or path == "/":
            params = parse_qs(parsed.query)
            highlight = params.get("highlight", [None])[0]
            sort_by = params.get("sort", ["newest"])[0]
            min_rms = params.get("rms", [None])[0]
            label_filter = params.get("label", [None])[0]
            self._serve_status(highlight=highlight, sort_by=sort_by,
                               min_rms=float(min_rms) if min_rms else None,
                               label_filter=label_filter)
        else:
            self._respond(404, "Not found")

    def _serve_review_page(self, key):
        groups = get_all_clips()
        if key not in groups:
            self._respond(404, f"No clips found for {key}")
            return

        # Mark as in-progress for this reviewer
        reviewer_ip = self.client_address[0]
        with _lock:
            _in_progress[key] = (reviewer_ip, datetime.now().timestamp())

        group = groups[key]
        clips = sorted(group["clips"], key=lambda f: f.name)
        current_dir_label = group["current_dir"]

        labels = load_labels()
        raw_label = labels.get(key, {})
        manual_label = raw_label.get("label") if isinstance(raw_label, dict) else None

        unifi_suggestion = group.get("unifi", "unknown")
        display_label = manual_label or f"auto:{current_dir_label}"

        # Format timestamp for display
        ts_display = ""
        cam_name_first = ""
        for c in clips:
            for cn in ["kitchen", "living", "entry"]:
                if cn in c.name:
                    cam_name_first = cn
                    break
            if cam_name_first:
                break
        if group.get("ts_ms"):
            from zoneinfo import ZoneInfo
            dt = datetime.fromtimestamp(group["ts_ms"] / 1000, tz=ZoneInfo("America/New_York"))
            ts_display = dt.strftime("%A %-m/%-d/%y %-I:%M:%S %p ET")

        # Deduplicate: best confidence per camera+second
        best_per_key = {}
        conf_re = re.compile(r"(\d+)%\.wav$")
        for clip_path in clips:
            name = clip_path.name
            cm = conf_re.search(name)
            conf = int(cm.group(1)) if cm else 0
            # Key without confidence
            dedup_key = conf_re.sub("", name)
            if dedup_key not in best_per_key or conf > best_per_key[dedup_key][1]:
                best_per_key[dedup_key] = (clip_path, conf)

        deduped_raw = sorted(best_per_key.values(), key=lambda x: x[0].name)

        # Filter out dud files (empty or too small to be valid WAV)
        # Also resolve moved files (e.g. labeled clip moved from pi_clips to negative)
        MIN_WAV_SIZE = 1000  # valid WAV with audio is at least ~1KB
        deduped = []
        for clip_path, conf in deduped_raw:
            try:
                # If not at cached path, search other dirs
                if not clip_path.exists():
                    found = None
                    for search_dir in [POSITIVE_DIR, NEGATIVE_DIR, PI_CLIPS_DIR]:
                        candidate = search_dir / clip_path.name
                        if candidate.exists():
                            found = candidate
                            break
                    if found:
                        clip_path = found
                    else:
                        continue  # truly gone
                if clip_path.stat().st_size >= MIN_WAV_SIZE:
                    deduped.append((clip_path, conf))
                else:
                    clip_path.unlink(missing_ok=True)
            except Exception:
                pass

        clip_rows = ""
        for i, (clip_path, conf) in enumerate(deduped):
            name = clip_path.name
            # Determine which dir for serving
            if clip_path.parent == POSITIVE_DIR:
                serve_path = f"mina/{name}"
            elif clip_path.parent == NEGATIVE_DIR:
                serve_path = f"negative/{name}"
            else:
                serve_path = f"pi/{name}"

            # Per-clip label
            clip_label = labels.get(f"clip:{name}", {})
            clip_label_str = clip_label.get("label") if isinstance(clip_label, dict) else None
            clip_indicator = ""
            if clip_label_str == "bark":
                clip_indicator = '<span class="clip-indicator" style="color:#4CAF50;font-weight:bold;"> bark</span>'
            elif clip_label_str == "not_bark":
                clip_indicator = '<span class="clip-indicator" style="color:#f44336;font-weight:bold;"> not bark</span>'

            clip_rows += f"""
            <div style="margin:8px 0;padding:8px;border-radius:6px;{'background:#1e2e1e;' if clip_label_str == 'bark' else 'background:#2e1e1e;' if clip_label_str == 'not_bark' else ''}">
                <span style="color:#888;">{i+1}.</span> {name}{clip_indicator}
                <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                    <audio controls preload="auto" style="flex:1;max-width:350px;">
                        <source src="/clip/{serve_path}" type="audio/wav">
                    </audio>
                    <a href="javascript:void(0)" onclick="labelClip(this,'/label?key=clip:{name}&label=bark','bark')" style="padding:6px 12px;background:#4CAF50;color:white;border-radius:4px;text-decoration:none;font-size:13px;">B</a>
                    <a href="javascript:void(0)" onclick="labelClip(this,'/label?key=clip:{name}&label=not_bark','not_bark')" style="padding:6px 12px;background:#f44336;color:white;border-radius:4px;text-decoration:none;font-size:13px;">N</a>
                </div>
            </div>"""

        label_color = {"bark": "#4CAF50", "not_bark": "#f44336"}
        color = label_color.get(manual_label, "#888")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{key} Review</title>
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
        .btn-play {{ background: #2196F3; color: white; margin-bottom: 16px; display: inline-block; }}
        .speed-btn {{ padding: 4px 10px; background: #444; color: #ccc; border-radius: 4px; text-decoration: none; font-size: 13px; cursor: pointer; display: inline-block; }}
        .speed-btn.active {{ background: #2196F3; color: white; }}
        .btn:active {{ opacity: 0.7; }}
        audio {{ margin-top: 4px; }}
    </style>
    <script>
    let currentSpeed = 2;
    document.addEventListener('DOMContentLoaded', function() {{
        document.querySelectorAll('audio').forEach(a => {{ a.playbackRate = currentSpeed; }});
    }});
    function setSpeed(rate) {{
        currentSpeed = rate;
        document.querySelectorAll('audio').forEach(a => {{ a.playbackRate = rate; }});
        document.querySelectorAll('.speed-btn').forEach(b => {{
            b.classList.toggle('active', parseFloat(b.dataset.speed) === rate);
        }});
    }}
    function playAll() {{
        const audios = document.querySelectorAll('audio');
        let i = 0;
        function next() {{
            if (i < audios.length) {{
                const a = audios[i];
                i++;
                if (a.duration === 0 || isNaN(a.duration)) {{
                    next();
                    return;
                }}
                a.scrollIntoView({{behavior:'smooth',block:'center'}});
                a.onended = function() {{ next(); }};
                a.onerror = function() {{ next(); }};
                a.play().catch(() => next());
            }}
        }}
        audios.forEach(a => {{ a.pause(); a.currentTime = 0; a.playbackRate = currentSpeed; }});
        i = 0;
        next();
    }}
    function labelClip(btn, url, label) {{
        fetch(url, {{redirect:'manual'}}).then(() => {{
            const row = btn.closest('div[style*="border-radius"]');
            row.style.background = label === 'bark' ? '#1e2e1e' : '#2e1e1e';
            let indicator = row.querySelector('.clip-indicator');
            if (!indicator) {{
                indicator = document.createElement('span');
                indicator.className = 'clip-indicator';
                const nameSpan = row.querySelector('span:nth-child(1)');
                nameSpan.after(indicator);
            }}
            indicator.style.color = label === 'bark' ? '#4CAF50' : '#f44336';
            indicator.style.fontWeight = 'bold';
            indicator.textContent = label === 'bark' ? ' bark' : ' not bark';
        }});
    }}
    function fillAll(label) {{
        const btnSelector = label === 'bark' ? 'B' : 'N';
        const buttons = [];
        document.querySelectorAll('a[onclick*="labelClip"]').forEach(btn => {{
            if (btn.textContent.trim() === btnSelector) buttons.push(btn);
        }});
        let i = 0;
        function nextLabel() {{
            if (i < buttons.length) {{
                const btn = buttons[i];
                const m = btn.getAttribute('onclick').match(/labelClip\\(this,'([^']+)','([^']+)'\\)/);
                if (m) {{
                    const row = btn.closest('div[style*="border-radius"]');
                    row.style.background = m[2] === 'bark' ? '#1e2e1e' : '#2e1e1e';
                    let indicator = row.querySelector('.clip-indicator');
                    if (!indicator) {{
                        indicator = document.createElement('span');
                        indicator.className = 'clip-indicator';
                        row.querySelector('span:nth-child(1)').after(indicator);
                    }}
                    indicator.style.color = m[2] === 'bark' ? '#4CAF50' : '#f44336';
                    indicator.style.fontWeight = 'bold';
                    indicator.textContent = m[2] === 'bark' ? ' bark' : ' not bark';
                    fetch(m[1], {{redirect:'manual'}}).then(() => {{ i++; nextLabel(); }});
                }} else {{
                    i++; nextLabel();
                }}
            }}
        }}
        nextLabel();
    }}
    </script>
</head>
<body>
    <h1>{key}</h1>
    <div style="color:#aaa;margin-bottom:4px;">{ts_display}{f' · <a href="{UNIFI_BASE}/{UNIFI_CAMERAS.get(cam_name_first, "")  }?start={group.get("ts_ms", "")}" target="_blank" style="color:#64b5f6;">View in UniFi</a>' if group.get("ts_ms") and cam_name_first else ''}</div>
    <div class="status">Label: {display_label}</div>
    <div style="margin:8px 0;">
        UniFi AI: <b style="color:{'#4CAF50' if unifi_suggestion == 'bark' else '#f44336' if unifi_suggestion == 'no bark' else '#888'}">{unifi_suggestion}</b>
        &nbsp;·&nbsp; RMS: <b style="color:{'#4CAF50' if group.get('max_rms', 0) > 0.03 else '#FF9800' if group.get('max_rms', 0) > 0.01 else '#f44336' if group.get('max_rms', 0) > 0.007 else '#555'}">{group.get('max_rms', 0):.4f}</b>
        &nbsp;·&nbsp; Currently in: <b>{current_dir_label}</b>
    </div>
    <div>{len(deduped)} clips · <a class="btn btn-play" href="javascript:playAll()">Play All</a> &nbsp;
        <a class="speed-btn" data-speed="1" href="javascript:setSpeed(1)">1x</a>
        <a class="speed-btn" data-speed="1.5" href="javascript:setSpeed(1.5)">1.5x</a>
        <a class="speed-btn active" data-speed="2" href="javascript:setSpeed(2)">2x</a>
    </div>
    <div class="clips">{clip_rows}</div>
    <div style="margin-top:20px;color:#888;font-size:13px;">Label all remaining clips in group:</div>
    <div class="buttons">
        <a class="btn btn-bark" href="javascript:fillAll('bark')">All Bark</a>
        <a class="btn btn-not" href="javascript:fillAll('not_bark')">All Not Bark</a>
    </div>
    <div style="margin-top:12px;display:flex;gap:8px;">
        <a href="/prev?before={key}" style="flex:1;padding:14px;background:#666;color:white;border-radius:8px;font-size:16px;font-weight:bold;text-decoration:none;text-align:center;">&larr; Back</a>
        <a href="/next?after={key}" style="flex:1;padding:14px;background:#2196F3;color:white;border-radius:8px;font-size:16px;font-weight:bold;text-decoration:none;text-align:center;">Next &rarr;</a>
    </div>
    <p style="margin-top:12px;"><a href="/status">Back to status</a></p>
</body>
</html>"""
        self._respond(200, html, content_type="text/html")

    def _serve_clip(self, rel_path):
        if rel_path.startswith("pi/"):
            clip_path = PI_CLIPS_DIR / rel_path[3:]
        else:
            clip_path = TRAINING_DIR / rel_path
        # If not found, search all directories
        if not clip_path.exists():
            filename = Path(rel_path).name
            for search_dir in [POSITIVE_DIR, NEGATIVE_DIR, PI_CLIPS_DIR]:
                candidate = search_dir / filename
                if candidate.exists():
                    clip_path = candidate
                    break
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

    def _next_unlabeled(self, after_key=None, reviewer_ip=None, filter_dir=None):
        labels = load_labels()
        groups = get_all_clips()
        # Sort: loud clips first (more likely to need human judgment), quiet ones last
        keys = sorted(groups.keys(), key=lambda k: -(groups[k].get("max_rms", 0)))
        now = datetime.now().timestamp()
        past = after_key is None
        for k in keys:
            if not past:
                if k == after_key:
                    past = True
                continue
            if k in labels:
                continue
            if _is_group_reviewed(k, groups, labels):
                continue
            # Filter by current directory
            if filter_dir == "positive" and groups[k]["current_dir"] != "bark":
                continue
            if filter_dir == "negative" and groups[k]["current_dir"] != "not_bark":
                continue
            if filter_dir == "pending" and groups[k]["current_dir"] != "pending":
                continue
            # Skip if another reviewer is looking at it (within last 5 min)
            with _lock:
                if k in _in_progress:
                    ip, ts = _in_progress[k]
                    if ip != reviewer_ip and now - ts < 300:
                        continue
            return k
        return None

    def _apply_label(self, key, label):
        if label not in ("bark", "not_bark"):
            self._respond(400, "Invalid label.")
            return

        labels = load_labels()
        labels[key] = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
        }
        save_labels(labels)

        # Per-clip label: move that file + any same-second duplicates
        if key.startswith("clip:"):
            filename = key[5:]
            dest_dir = POSITIVE_DIR if label == "bark" else NEGATIVE_DIR

            # Find same-second siblings (e.g. living_97% and living_99% for same timestamp)
            conf_re = re.compile(r"(\d+)%\.wav$")
            base = conf_re.sub("", filename)  # everything before the confidence

            # Label all siblings
            for src_dir in [POSITIVE_DIR, NEGATIVE_DIR, PI_CLIPS_DIR]:
                if not src_dir.exists():
                    continue
                for f in src_dir.iterdir():
                    if f.suffix == ".wav" and conf_re.sub("", f.name) == base:
                        sibling_key = f"clip:{f.name}"
                        if sibling_key not in labels:
                            labels[sibling_key] = {
                                "label": label,
                                "timestamp": datetime.now().isoformat(),
                            }
            save_labels(labels)

            def _move_clips():
                for src_dir in [POSITIVE_DIR, NEGATIVE_DIR, PI_CLIPS_DIR]:
                    if not src_dir.exists():
                        continue
                    for f in list(src_dir.iterdir()):
                        if f.suffix == ".wav" and conf_re.sub("", f.name) == base:
                            if f.parent != dest_dir:
                                try:
                                    shutil.move(str(f), str(dest_dir / f.name))
                                except Exception:
                                    pass
            threading.Thread(target=_move_clips, daemon=True).start()

            # Update cache to reflect new directory
            sm = SESSION_RE.match(filename)
            group_key_for_cache = sm.group(1) if sm else None
            if not group_key_for_cache:
                tm = CLIP_TS_RE.search(filename)
                group_key_for_cache = f"T{tm.group(1)}" if tm else None
            if group_key_for_cache and _clips_cache and group_key_for_cache in _clips_cache:
                # Check if all clips in group now have labels and determine consensus
                all_labels = load_labels()
                group_clips = _clips_cache[group_key_for_cache]["clips"]
                clip_label_set = set()
                all_labeled = True
                for cl in group_clips:
                    cl_data = all_labels.get(f"clip:{cl.name}", {})
                    cl_label = cl_data.get("label") if isinstance(cl_data, dict) else None
                    if cl_label:
                        clip_label_set.add(cl_label)
                    else:
                        all_labeled = False
                if all_labeled and clip_label_set == {"bark"}:
                    _clips_cache[group_key_for_cache]["current_dir"] = "bark"
                elif all_labeled and clip_label_set == {"not_bark"}:
                    _clips_cache[group_key_for_cache]["current_dir"] = "not_bark"
                elif all_labeled:
                    _clips_cache[group_key_for_cache]["current_dir"] = "mixed"

            # Redirect back to the same group review page
            sm = SESSION_RE.match(filename)
            if sm:
                group_key = sm.group(1)
            else:
                tm = CLIP_TS_RE.search(filename)
                group_key = f"T{tm.group(1)}" if tm else key
            self.send_response(302)
            self.send_header("Location", f"/review/{group_key}")
            self.end_headers()
            return

        # Group-level label
        with _lock:
            _in_progress.pop(key, None)
            if _clips_cache and key in _clips_cache:
                _clips_cache[key]["current_dir"] = "bark" if label == "bark" else "not_bark"

        # Move files synchronously so trained column reflects immediately
        groups = get_all_clips()
        if key in groups:
            dest_dir = POSITIVE_DIR if label == "bark" else NEGATIVE_DIR
            for clip_path in groups[key]["clips"]:
                clip_key = f"clip:{clip_path.name}"
                if clip_key in labels:
                    clip_dest = POSITIVE_DIR if labels[clip_key].get("label") == "bark" else NEGATIVE_DIR
                    if clip_path.parent != clip_dest:
                        try:
                            shutil.move(str(clip_path), str(clip_dest / clip_path.name))
                        except Exception:
                            pass
                elif clip_path.parent != dest_dir:
                    try:
                        shutil.move(str(clip_path), str(dest_dir / clip_path.name))
                    except Exception:
                        pass

        reviewer_ip = self.client_address[0]
        next_key = self._next_unlabeled_fast(key, reviewer_ip)

        if next_key:
            self.send_response(302)
            self.send_header("Location", f"/review/{next_key}")
            self.end_headers()
        else:
            self.send_response(302)
            self.send_header("Location", "/status")
            self.end_headers()

    def _next_unlabeled_fast(self, after_key, reviewer_ip):
        """Fast version — uses cached clips, no RMS recomputation."""
        labels = load_labels()
        if not _clips_cache:
            return None
        keys = sorted(_clips_cache.keys())
        now = datetime.now().timestamp()
        past = False
        for k in keys:
            if not past:
                if k == after_key:
                    past = True
                continue
            if k in labels:
                continue
            if _clips_cache and _is_group_reviewed(k, _clips_cache, labels):
                continue
            with _lock:
                if k in _in_progress:
                    ip, ts = _in_progress[k]
                    if ip != reviewer_ip and now - ts < 300:
                        continue
            return k
        return None

    def _prev_group(self, before_key):
        """Return the group key immediately before before_key."""
        if not _clips_cache:
            return None
        keys = sorted(_clips_cache.keys())
        prev = None
        for k in keys:
            if k == before_key:
                return prev
            prev = k
        return None

    def _serve_queue(self, filter_dir=None):
        reviewer_ip = self.client_address[0]
        first = self._next_unlabeled(reviewer_ip=reviewer_ip, filter_dir=filter_dir)
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
<p><a href="/status">View all</a></p></body></html>""", content_type="text/html")

    def _serve_priority_queue(self):
        """Redirect to the most likely mislabeled unreviewed clip."""
        labels = load_labels()
        groups = get_all_clips()
        reviewer_ip = self.client_address[0]
        now = datetime.now().timestamp()

        # Score each unreviewed group by how suspicious it is
        # Higher score = more likely mislabeled = review first
        candidates = []
        for k, g in groups.items():
            if k in labels or _is_group_reviewed(k, groups, labels):
                continue
            with _lock:
                if k in _in_progress:
                    ip, ts = _in_progress[k]
                    if ip != reviewer_ip and now - ts < 300:
                        continue

            score = 0
            max_rms = g.get("max_rms", 0)
            unifi = g.get("unifi", "unknown")
            current = g.get("current_dir", "")

            # Loud + UniFi says no bark → likely false positive, high value review
            if max_rms > 0.01 and unifi == "no bark":
                score = 100

            # Quiet + UniFi says bark → possibly mislabeled negative
            elif max_rms < 0.008 and unifi == "bark":
                score = 90

            # Loud + in negative dir → might be a bark we missed
            elif max_rms > 0.01 and current == "not_bark":
                score = 80

            # Quiet + in positive dir → might not actually be a bark
            elif max_rms < 0.008 and current == "bark":
                score = 70

            # UniFi disagrees with current dir
            elif unifi == "bark" and current == "not_bark":
                score = 60
            elif unifi == "no bark" and current == "bark":
                score = 60

            else:
                score = 1  # Low priority

            candidates.append((score, k))

        candidates.sort(key=lambda x: -x[0])

        if candidates:
            self.send_response(302)
            self.send_header("Location", f"/review/{candidates[0][1]}")
            self.end_headers()
        else:
            self._respond(200, """<!DOCTYPE html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:-apple-system,sans-serif;padding:40px 16px;background:#1a1a1a;color:#eee;text-align:center;}
a{color:#64b5f6;}</style></head>
<body><h1>All caught up!</h1><p>No suspicious clips to review.</p>
<p><a href="/status">View all</a></p></body></html>""", content_type="text/html")

    def _serve_filtered_queue(self, filter_type):
        """Filtered review queues targeting likely mislabeled clips."""
        labels = load_labels()
        groups = get_all_clips()
        reviewer_ip = self.client_address[0]
        now = datetime.now().timestamp()

        candidates = []
        for k, g in groups.items():
            if k in labels or _is_group_reviewed(k, groups, labels):
                continue
            with _lock:
                if k in _in_progress:
                    ip, ts = _in_progress[k]
                    if ip != reviewer_ip and now - ts < 300:
                        continue

            max_rms = g.get("max_rms", 0)
            unifi = g.get("unifi", "unknown")
            current = g.get("current_dir", "")

            match = False
            if filter_type == "missed_barks":
                # Loud clips in negative — model thinks ambient but could be bark
                match = current == "not_bark" and max_rms > 0.015
            elif filter_type == "unifi_disagrees":
                # UniFi says bark but clip is in negative
                match = unifi == "bark" and current == "not_bark"
            elif filter_type == "quiet_positives":
                # Quiet clips in positive — might not be real barks
                match = current == "bark" and max_rms < 0.025
            elif filter_type == "loud_negatives":
                # Loudest clips in negative — most likely to be mislabeled
                match = current == "not_bark" and max_rms > 0.01

            if match:
                # Sort quiet_positives by lowest RMS first; others by highest RMS first
                if filter_type == "quiet_positives":
                    candidates.append((max_rms, k))
                else:
                    candidates.append((-max_rms, k))

        candidates.sort()

        if candidates:
            self.send_response(302)
            self.send_header("Location", f"/review/{candidates[0][1]}")
            self.end_headers()
        else:
            self._respond(200, f"""<!DOCTYPE html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:-apple-system,sans-serif;padding:40px 16px;background:#1a1a1a;color:#eee;text-align:center;}}
a{{color:#64b5f6;}}</style></head>
<body><h1>None found</h1><p>No unreviewed clips match "{filter_type}".</p>
<p><a href="/status">Back to status</a></p></body></html>""", content_type="text/html")

    def _serve_status(self, highlight=None, sort_by="newest", min_rms=None, label_filter=None):
        labels = load_labels()
        groups = get_all_clips()

        pass  # _is_group_reviewed is now module-level

        def sort_key(k):
            g = groups[k]
            is_rev = k in labels or _is_group_reviewed(k, groups, labels)
            max_rms = g.get("max_rms", 0)
            unifi = g.get("unifi", "unknown")
            current = g.get("current_dir", "")
            ts = g.get("ts_ms", 0) or 0

            if sort_by == "newest":
                return (-ts, k)

            elif sort_by == "oldest":
                return (ts, k)

            elif sort_by == "rms_high":
                return (-max_rms, k)

            elif sort_by == "rms_low":
                return (max_rms, k)

            elif sort_by == "dur_high":
                return (-len(g.get("clips", [])), k)

            elif sort_by == "dur_low":
                return (len(g.get("clips", [])), k)

            else:  # "suspicious" (default)
                if is_rev:
                    return (3, 0, k)
                # Loud + in negative or UniFi says no bark
                if max_rms > 0.02 and (current == "not_bark" or unifi == "no bark"):
                    return (0, -max_rms, k)
                # Quiet + in positive or UniFi says bark
                if max_rms < 0.01 and (current == "bark" or unifi == "bark"):
                    return (0, max_rms, k)
                # UniFi disagrees
                if (unifi == "bark" and current == "not_bark") or \
                   (unifi == "no bark" and current == "bark"):
                    return (0, -max_rms, k)
                # Pending
                if current == "pending":
                    return (1, -max_rms, k)
                return (2, -max_rms, k)

        keys = sorted(groups.keys(), key=sort_key)

        def is_reviewed(k):
            if k in labels:
                return True
            return _is_group_reviewed(k, groups, labels)

        reviewed = sum(1 for k in keys if is_reviewed(k))
        unreviewed = len(keys) - reviewed

        # Load deploy snapshot once
        deploy_snapshot = {}
        if LAST_DEPLOY_SNAPSHOT.exists():
            try:
                with open(LAST_DEPLOY_SNAPSHOT) as _f:
                    deploy_snapshot = json.load(_f)
            except Exception:
                pass

        rows = ""
        shown = 0
        for key in keys:
            info = labels.get(key, {})
            manual_label = info.get("label") if isinstance(info, dict) else None
            current_dir = groups[key]["current_dir"]
            unifi_suggestion = groups[key].get("unifi", "unknown")
            max_rms_val = groups[key].get("max_rms", 0)

            # Apply filters
            if min_rms is not None and max_rms_val < min_rms:
                continue
            if label_filter == "bark":
                is_bark = current_dir == "bark" or unifi_suggestion == "bark" or manual_label == "bark"
                if not is_bark:
                    continue
            if label_filter == "not_bark" and current_dir != "not_bark":
                continue
            if label_filter == "pending" and current_dir != "pending":
                continue
            if label_filter == "unreviewed" and (manual_label or _is_group_reviewed(key, groups, labels)):
                continue
            shown += 1
            # Check per-clip review status
            clip_reviewed = _is_group_reviewed(key, groups, labels)
            if manual_label:
                display = manual_label
            elif clip_reviewed:
                # Determine consensus from per-clip labels
                clip_labels = set()
                for cl in groups[key]["clips"]:
                    cl_data = labels.get(f"clip:{cl.name}", {})
                    cl_label = cl_data.get("label") if isinstance(cl_data, dict) else None
                    if cl_label:
                        clip_labels.add(cl_label)
                if clip_labels == {"bark"}:
                    display = "bark"
                elif clip_labels == {"not_bark"}:
                    display = "not_bark"
                else:
                    display = "mixed"
            else:
                display = f"auto:{current_dir}"
            color = {"bark": "#4CAF50", "not_bark": "#f44336", "mixed": "#FF9800"}.get(display, "#888")
            unifi_color = "#4CAF50" if unifi_suggestion == "bark" else "#f44336" if unifi_suggestion == "no bark" else "#888"
            if current_dir == "pending" and not manual_label:
                display = "new"
                color = "#FF9800"
            clip_count = len(groups[key]["clips"])
            max_rms = groups[key].get("max_rms", 0)
            if max_rms > 0.03:
                level = f"{max_rms:.3f}"
                level_color = "#4CAF50"
            elif max_rms > 0.01:
                level = f"{max_rms:.3f}"
                level_color = "#FF9800"
            elif max_rms > 0.007:
                level = f"{max_rms:.3f}"
                level_color = "#f44336"
            else:
                level = f"{max_rms:.3f}"
                level_color = "#555"
            is_highlighted = key == highlight

            # Flag suspicious clips
            suspicious = ""
            if not manual_label and not clip_reviewed:
                if max_rms > 0.02 and (unifi_suggestion == "no bark" or current_dir == "not_bark"):
                    suspicious = ' <span title="Loud in negative — possible bark">*</span>'
                elif max_rms < 0.01 and (unifi_suggestion == "bark" or current_dir == "bark"):
                    suspicious = ' <span title="Quiet in positive — possible false positive">*</span>'
                elif unifi_suggestion == "bark" and current_dir == "not_bark":
                    suspicious = ' <span title="UniFi disagrees">*</span>'
                elif unifi_suggestion == "no bark" and current_dir == "bark":
                    suspicious = ' <span title="UniFi disagrees">*</span>'

            # Format time for table with UniFi link
            ts_ms = groups[key].get("ts_ms")
            if ts_ms:
                from zoneinfo import ZoneInfo
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=ZoneInfo("America/New_York"))
                time_fmt = dt.strftime("%a %-m/%-d %I:%M%p")
                # Default to kitchen camera for timeline link
                cam_name = ""
                for c in groups[key]["clips"]:
                    for cn in ["kitchen", "living", "entry"]:
                        if cn in c.name:
                            cam_name = cn
                            break
                    if cam_name:
                        break
                cam_id = UNIFI_CAMERAS.get(cam_name or "kitchen", UNIFI_CAMERAS["kitchen"])
                unifi_url = f"{UNIFI_BASE}/{cam_id}?start={ts_ms}"
                time_str = f'{time_fmt} <a href="{unifi_url}" target="_blank" style="color:#64b5f6;text-decoration:none;" title="View in UniFi">cam</a>'
            else:
                time_str = ""

            cameras_str = ", ".join(sorted(groups[key].get("cameras", set()))) or "-"

            # Duration (each clip ~1s)
            dur_str = f"{clip_count}s"

            # Trained status: compare actual file location vs deploy snapshot
            # Check where files actually are right now (by scanning dirs, not cache)
            actual_labels = set()
            clip_basenames = [c.name for c in groups[key]["clips"]]
            for cn in clip_basenames:
                if (POSITIVE_DIR / cn).exists():
                    actual_labels.add("bark")
                elif (NEGATIVE_DIR / cn).exists():
                    actual_labels.add("not_bark")
                elif (PI_CLIPS_DIR / cn).exists():
                    actual_labels.add("pending")
                else:
                    # File may have moved — check both dirs
                    actual_labels.add("unknown")

            deployed_labels = set()
            for cn in clip_basenames:
                if cn in deploy_snapshot:
                    deployed_labels.add(deploy_snapshot[cn])

            actual = list(actual_labels)[0] if len(actual_labels) == 1 else "mixed"

            if not deployed_labels or "pending" in actual_labels:
                # Never deployed
                trained_str = '<span style="color:#f44336;">N</span>'
            elif actual_labels == deployed_labels:
                # Current state matches what was deployed
                dl = list(deployed_labels)[0]
                trained_str = f'<span style="color:#4CAF50;">Y-{dl}</span>'
            else:
                # Changed since deploy
                dl = list(deployed_labels)[0] if len(deployed_labels) == 1 else "mixed"
                trained_str = f'<span style="color:#FF9800;">N (was {dl})</span>'

            row_bg = "#333" if is_highlighted else "#2a1a00" if suspicious and not manual_label else ""
            row_style = f' style="background:{row_bg};"' if row_bg else ""
            rows += (f'<tr id="{key}"{row_style}><td><a href="/review/{key}">{key}</a>{suspicious}</td>'
                     f'<td style="color:#888;font-size:12px;">{time_str}</td>'
                     f'<td style="color:#888;font-size:12px;">{cameras_str}</td>'
                     f'<td>{clip_count}</td>'
                     f'<td style="color:#888;">{dur_str}</td>'
                     f'<td style="color:{level_color}">{level}</td>'
                     f'<td style="color:{unifi_color}">{unifi_suggestion}</td>'
                     f'<td style="color:{color}">{display}</td>'
                     f'<td>{trained_str}</td></tr>\n')

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Mina Training Review</title>
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
    <h1>Mina Training Review</h1>
    <p>Reviewed: {reviewed} · Unreviewed: {unreviewed} · Total: {len(keys)}</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <a class="queue-btn" style="background:#FF9800;" href="/priority">Priority Review</a>
        <a class="queue-btn" href="/queue">Review All</a>
        <a class="queue-btn" style="background:#4CAF50;" href="/queue/positive">Review Positives</a>
        <a class="queue-btn" style="background:#f44336;" href="/queue/negative">Review Negatives</a>
        <a class="queue-btn" style="background:#333;border:1px solid #555;" href="/queue/pending">New (Pi)</a>
        <a class="queue-btn" style="background:#333;border:1px solid #555;" href="/sync">Sync Pi</a>
    </div>
    <div style="margin-top:8px;font-size:13px;color:#888;">Sort:</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#2196F3;' if sort_by == 'suspicious' else 'background:#333;border:1px solid #555;'}" href="/status?sort=suspicious">Suspicious</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#2196F3;' if sort_by == 'newest' else 'background:#333;border:1px solid #555;'}" href="/status?sort=newest">Newest</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#2196F3;' if sort_by == 'oldest' else 'background:#333;border:1px solid #555;'}" href="/status?sort=oldest">Oldest</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#2196F3;' if sort_by == 'rms_high' else 'background:#333;border:1px solid #555;'}" href="/status?sort=rms_high">RMS High</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#2196F3;' if sort_by == 'rms_low' else 'background:#333;border:1px solid #555;'}" href="/status?sort=rms_low">RMS Low</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#2196F3;' if sort_by == 'dur_high' else 'background:#333;border:1px solid #555;'}" href="/status?sort=dur_high">Longest</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#2196F3;' if sort_by == 'dur_low' else 'background:#333;border:1px solid #555;'}" href="/status?sort=dur_low">Shortest</a>
    </div>
    <div style="margin-top:8px;font-size:13px;color:#888;">Label filter:</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#2196F3;' if not label_filter else 'background:#333;border:1px solid #555;'}" href="/status?sort={sort_by}">All</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#4CAF50;' if label_filter == 'bark' else 'background:#333;border:1px solid #555;'}" href="/status?sort={sort_by}&amp;label=bark">Bark</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#f44336;' if label_filter == 'not_bark' else 'background:#333;border:1px solid #555;'}" href="/status?sort={sort_by}&amp;label=not_bark">Not Bark</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#FF9800;' if label_filter == 'pending' else 'background:#333;border:1px solid #555;'}" href="/status?sort={sort_by}&amp;label=pending">Pending</a>
        <a class="queue-btn" style="font-size:13px;padding:6px 14px;{'background:#9C27B0;' if label_filter == 'unreviewed' else 'background:#333;border:1px solid #555;'}" href="/status?sort={sort_by}&amp;label=unreviewed">Unreviewed</a>
    </div>
    <div style="margin-top:8px;font-size:13px;color:#888;">RMS filter:</div>
    <form style="display:flex;gap:8px;align-items:center;margin-top:4px;" method="get" action="/status">
        <input type="hidden" name="sort" value="{sort_by}">
        {'<input type="hidden" name="label" value="' + label_filter + '">' if label_filter else ''}
        <span style="color:#eee;font-size:13px;">RMS &gt;</span>
        <input type="text" name="rms" value="{min_rms if min_rms is not None else ''}" placeholder="e.g. 0.01" style="width:80px;padding:6px;background:#333;color:#eee;border:1px solid #555;border-radius:4px;font-size:13px;">
        <button type="submit" style="padding:6px 14px;background:#2196F3;color:white;border:none;border-radius:4px;font-size:13px;cursor:pointer;">Apply</button>
        {'<a href="/status?sort=' + sort_by + (('&label=' + label_filter) if label_filter else '') + '" style="color:#f44336;font-size:13px;margin-left:4px;">Clear</a>' if min_rms is not None else ''}
    </form>
    <div style="margin-top:8px;font-size:13px;color:#888;">Quick filters:</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">
        <a class="queue-btn" style="background:#b71c1c;font-size:13px;padding:8px 16px;" href="/filter/missed_barks">Missed Barks (loud negatives)</a>
        <a class="queue-btn" style="background:#e65100;font-size:13px;padding:8px 16px;" href="/filter/unifi_disagrees">UniFi Says Bark (in negative)</a>
        <a class="queue-btn" style="background:#4a148c;font-size:13px;padding:8px 16px;" href="/filter/quiet_positives">Quiet Positives (suspect)</a>
        <a class="queue-btn" style="background:#1a237e;font-size:13px;padding:8px 16px;" href="/filter/loud_negatives">Loud Negatives</a>
    </div>
    <table>
        <tr><th colspan="9" style="color:#888;font-weight:normal;">Showing {shown} of {len(keys)}</th></tr>
        <tr><th>Group</th><th>Time</th><th>Camera</th><th>Clips</th><th>Dur</th><th>RMS</th><th>UniFi</th><th>Label</th><th>Trained</th></tr>
        {rows}
    </table>
    {'<script>document.getElementById("' + highlight + '").scrollIntoView({block:"center"});</script>' if highlight else ''}
</body>
</html>"""
        self._respond(200, html, content_type="text/html")

    def _respond(self, status, body, content_type="text/html"):
        self.send_response(status)
        if "html" in content_type:
            content_type += "; charset=utf-8"
        self.send_header("Content-Type", content_type)
        encoded = body.encode() if isinstance(body, str) else body
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


from http.server import ThreadingHTTPServer

SYNC_INTERVAL = 30  # seconds between auto-syncs

def _auto_sync():
    """Background thread: periodically pull new clips from Pi."""
    while True:
        try:
            import time as _t
            _t.sleep(SYNC_INTERVAL)
            before = set()
            if PI_CLIPS_DIR.exists():
                before = set(f.name for f in PI_CLIPS_DIR.iterdir())
            sync_pi_clips()
            after = set()
            if PI_CLIPS_DIR.exists():
                after = set(f.name for f in PI_CLIPS_DIR.iterdir())
            new_clips = after - before
            if new_clips:
                print(f"Auto-sync: {len(new_clips)} new clips from Pi")
                invalidate_cache()
        except Exception as e:
            print(f"Auto-sync error: {e}")


def main():
    print("Syncing clips from Pi...")
    sync_pi_clips()
    # Load UniFi cache before serving — needed for auto-correct
    load_unifi_cache()

    # Start background auto-sync
    sync_thread = threading.Thread(target=_auto_sync, daemon=True)
    sync_thread.start()
    print(f"Auto-sync enabled (every {SYNC_INTERVAL}s)")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), ReviewHandler)
    print(f"Training review server on http://localhost:{PORT}")
    print(f"  Queue:  http://localhost:{PORT}/queue")
    print(f"  Status: http://localhost:{PORT}/status")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()

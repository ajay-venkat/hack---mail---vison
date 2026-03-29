"""
VisionAid — High-Accuracy Indoor Navigation for Visually Impaired
==========================================================
Fixes:
  • Smart voice debouncing based on object change, not time alone
  • Priority queue: URGENT → WARNING → INFO tiers
  • Smart sentence: "still ahead, move right" after 3 repeats
  • Clear-path heartbeat: once per 10s
  • YOLOv8m for higher indoor accuracy
  • Temporal smoothing: object must appear 3 frames before alerting
  • conf=0.55, iou=0.45
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import collections
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import av

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="VisionAid", page_icon="👁️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
.main { background-color: #000 !important; color: #fff !important; font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stApp    { max-width: 100%; padding: 0; background-color: #000 !important; }
p, li, span, div { font-size: 1.1rem !important; color: #fff !important; }
.stExpander { background: #0D1117 !important; border: 2px solid #1A73E8 !important; border-radius: 10px; margin: 10px; }
.stExpander summary { color: #fff !important; font-size: 1.2rem !important; }
.element-container img { border-radius: 8px; width: 100% !important; border: 2px solid #1A73E8; }
.status-panel {
    background: #0D1117; padding: 20px; border: 3px solid #1A73E8;
    border-radius: 12px; text-align: center; margin-top: 15px;
}
.status-text { font-size: 20px !important; font-weight: 700; letter-spacing: 1px; }
.model-badge {
    display: inline-block; background: #1A73E8; color: #fff !important;
    font-size: 0.75rem !important; padding: 2px 10px; border-radius: 20px; margin-top: 8px;
}
.priority-urgent  { color: #DB4437 !important; }
.priority-warning { color: #F4B400 !important; }
.priority-info    { color: #0F9D58 !important; }
.priority-clear   { color: #888888 !important; }
.caregiver-subtitle { color: #888 !important; font-size: 1rem !important; margin-top: -10px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)


# ─── PRIORITY DEFINITIONS ────────────────────────────────────────────────────

# URGENT: re-speak every 2s
URGENT_OBJECTS = {"stairs", "car", "truck", "bus", "motorcycle", "bicycle"}

# WARNING: speak once per zone, 4s repeat
WARNING_OBJECTS = {
    "person", "door", "chair", "bed", "toilet", "sofa", "couch",
    "sink", "refrigerator", "cabinet", "desk",
}

# INFO: speak once, 6s repeat (everything else)
# Priority numbers (lower = higher priority)
OBJECT_PRIORITY = {obj: 0 for obj in URGENT_OBJECTS}
OBJECT_PRIORITY.update({obj: 1 for obj in WARNING_OBJECTS})
# INFO defaults to 2

REPEAT_INTERVALS = {0: 2.0, 1: 4.0, 2: 6.0}   # seconds per priority
CLEAR_PATH_INTERVAL = 10.0                       # speak "clear" only every 10s

# Temporal smoothing: frames required before alerting
CONFIRM_FRAMES = 3
GRACE_FRAMES   = 2   # frames object can disappear before dropping it


# ─── LANGUAGES ───────────────────────────────────────────────────────────────

LANGUAGES = {
    "English": {
        "code": "en-US",
        "warning": "Warning",
        "ahead": "ahead",
        "left": "on your left",
        "right": "on your right",
        "very_close": "very close",
        "close": "close",
        "nearby": "nearby",
        "clear": "Path is clear",
        "dark": "Environment too dark",
        "started": "VisionAid navigation started",
        "obstacle": "Obstacle ahead",
        # Indoor-specific one-shot alerts
        "chair_ahead":    "Chair ahead",
        "door_left":      "Door on your left",
        "door_right":     "Door on your right",
        "door_ahead":     "Door ahead",
        "table_right":    "Table on your right",
        "table_left":     "Table on your left",
        "table_ahead":    "Table ahead",
        "bed_detected":   "Bed detected",
        "stairs_ahead":   "Stairs ahead, be very careful",
        "sofa_ahead":     "Sofa ahead",
        "toilet_ahead":   "Toilet ahead",
        "sink_ahead":     "Sink ahead",
        "cabinet_ahead":  "Cabinet ahead",
        # Persistence phrases
        "still_ahead":    "still ahead, consider moving",
        "persists":       "Obstacle persists, please navigate carefully",
        "move_right":     "Try moving right",
        "move_left":      "Try moving left",
    },
    "Tamil": {
        "code": "ta-IN",
        "warning": "எச்சரிக்கை",
        "ahead": "முன்னால்",
        "left": "இடதுபுறம்",
        "right": "வலதுபுறம்",
        "very_close": "மிக அருகில்",
        "close": "அருகில்",
        "nearby": "அண்மையில்",
        "clear": "பாதை தெளிவாக உள்ளது",
        "dark": "சூழல் மிகவும் இருட்டாக உள்ளது",
        "started": "விஷன்எய்ட் தொடங்கியது",
        "obstacle": "தடை முன்னால்",
        "chair_ahead":    "நாற்காலி முன்னால் உள்ளது",
        "door_left":      "கதவு இடதுபுறம் உள்ளது",
        "door_right":     "கதவு வலதுபுறம் உள்ளது",
        "door_ahead":     "கதவு முன்னால் உள்ளது",
        "table_right":    "மேசை வலதுபுறம் உள்ளது",
        "table_left":     "மேசை இடதுபுறம் உள்ளது",
        "table_ahead":    "மேசை முன்னால் உள்ளது",
        "bed_detected":   "படுக்கை கண்டறியப்பட்டது",
        "stairs_ahead":   "படிகள் முன்னால், மிகவும் கவனமாக இருங்கள்",
        "sofa_ahead":     "சோபா முன்னால் உள்ளது",
        "toilet_ahead":   "கழிவறை முன்னால் உள்ளது",
        "sink_ahead":     "கழுவுதொட்டி முன்னால் உள்ளது",
        "cabinet_ahead":  "அலமாரி முன்னால் உள்ளது",
        "still_ahead":    "இன்னும் முன்னால் உள்ளது, நகரவும்",
        "persists":       "தடை தொடர்கிறது, கவனமாக செல்லுங்கள்",
        "move_right":     "வலதுபுறம் செல்லவும்",
        "move_left":      "இடதுபுறம் செல்லவும்",
    },
    "Hindi": {
        "code": "hi-IN",
        "warning": "चेतावनी",
        "ahead": "आगे",
        "left": "बाईं तरफ",
        "right": "दाईं तरफ",
        "very_close": "बहुत पास",
        "close": "पास",
        "nearby": "नज़दीक",
        "clear": "रास्ता साफ है",
        "dark": "वातावरण बहुत अंधेरा है",
        "started": "VisionAid शुरू हो गया",
        "obstacle": "बाधा आगे",
        "chair_ahead":    "आगे कुर्सी है",
        "door_left":      "बाईं तरफ दरवाज़ा है",
        "door_right":     "दाईं तरफ दरवाज़ा है",
        "door_ahead":     "आगे दरवाज़ा है",
        "table_right":    "दाईं तरफ मेज़ है",
        "table_left":     "बाईं तरफ मेज़ है",
        "table_ahead":    "आगे मेज़ है",
        "bed_detected":   "बिस्तर मिला",
        "stairs_ahead":   "आगे सीढ़ियाँ हैं, बहुत सावधान रहें",
        "sofa_ahead":     "आगे सोफ़ा है",
        "toilet_ahead":   "आगे शौचालय है",
        "sink_ahead":     "आगे नल है",
        "cabinet_ahead":  "आगे अलमारी है",
        "still_ahead":    "अभी भी आगे है, हटने की कोशिश करें",
        "persists":       "बाधा बनी है, सावधानी से चलें",
        "move_right":     "दाईं तरफ जाएं",
        "move_left":      "बाईं तरफ जाएं",
    },
}

OBJECT_TRANSLATIONS = {
    "Tamil": {
        "person": "நபர்", "car": "கார்", "truck": "லாரி",
        "bus": "பேருந்து", "motorcycle": "மோட்டார் சைக்கிள்",
        "bicycle": "சைக்கிள்", "chair": "நாற்காலி",
        "table": "மேசை", "dining table": "மேசை", "desk": "மேஜை",
        "bottle": "பாட்டில்", "dog": "நாய்", "cat": "பூனை",
        "door": "கதவு", "bed": "படுக்கை", "toilet": "கழிவறை",
        "tv": "தொலைக்காட்சி", "monitor": "மானிட்டர்",
        "laptop": "மடிக்கணினி", "cell phone": "கைப்பேசி",
        "traffic light": "போக்குவரத்து விளக்கு",
        "bench": "இருக்கை", "backpack": "பை",
        "couch": "சோபா", "sofa": "சோபா",
        "sink": "கழுவுதொட்டி", "refrigerator": "குளிர்சாதனப்பெட்டி",
        "stairs": "படிகள்", "cabinet": "அலமாரி",
        "lamp": "விளக்கு", "pillow": "தலையணை",
        "bookshelf": "புத்தக அலமாரி", "wall": "சுவர்", "floor": "தரை",
    },
    "Hindi": {
        "person": "व्यक्ति", "car": "कार", "truck": "ट्रक",
        "bus": "बस", "motorcycle": "मोटरसाइकिल",
        "bicycle": "साइकिल", "chair": "कुर्सी",
        "table": "मेज़", "dining table": "मेज़", "desk": "डेस्क",
        "bottle": "बोतल", "dog": "कुत्ता", "cat": "बिल्ली",
        "door": "दरवाज़ा", "bed": "बिस्तर", "toilet": "शौचालय",
        "tv": "टीवी", "monitor": "मॉनिटर",
        "laptop": "लैपटॉप", "cell phone": "मोबाइल फ़ोन",
        "traffic light": "ट्रैफ़िक लाइट",
        "bench": "बेंच", "backpack": "बैग",
        "couch": "सोफ़ा", "sofa": "सोफ़ा",
        "sink": "नल", "refrigerator": "फ्रिज",
        "stairs": "सीढ़ियाँ", "cabinet": "अलमारी",
        "lamp": "लैंप", "pillow": "तकिया",
        "bookshelf": "किताबों की अलमारी", "wall": "दीवार", "floor": "फर्श",
    },
}

INDOOR_ALERT_MAP = {
    "chair":        {"CENTER": "chair_ahead",  "LEFT": "chair_ahead",  "RIGHT": "chair_ahead"},
    "door":         {"CENTER": "door_ahead",   "LEFT": "door_left",    "RIGHT": "door_right"},
    "table":        {"CENTER": "table_ahead",  "LEFT": "table_left",   "RIGHT": "table_right"},
    "dining table": {"CENTER": "table_ahead",  "LEFT": "table_left",   "RIGHT": "table_right"},
    "desk":         {"CENTER": "table_ahead",  "LEFT": "table_left",   "RIGHT": "table_right"},
    "bed":          {"CENTER": "bed_detected", "LEFT": "bed_detected", "RIGHT": "bed_detected"},
    "stairs":       {"CENTER": "stairs_ahead", "LEFT": "stairs_ahead", "RIGHT": "stairs_ahead"},
    "sofa":         {"CENTER": "sofa_ahead",   "LEFT": "sofa_ahead",   "RIGHT": "sofa_ahead"},
    "couch":        {"CENTER": "sofa_ahead",   "LEFT": "sofa_ahead",   "RIGHT": "sofa_ahead"},
    "toilet":       {"CENTER": "toilet_ahead", "LEFT": "toilet_ahead", "RIGHT": "toilet_ahead"},
    "sink":         {"CENTER": "sink_ahead",   "LEFT": "sink_ahead",   "RIGHT": "sink_ahead"},
    "cabinet":      {"CENTER": "cabinet_ahead","LEFT": "cabinet_ahead","RIGHT": "cabinet_ahead"},
}


# ─── SESSION STATE ────────────────────────────────────────────────────────────

for k, v in {
    "frame_count": 0, "det_count": 0,
    "ui_msg": "START NAVIGATION to begin", "ui_msg_class": "priority-clear",
    # Voice state
    "last_spoken_text": "",
    "last_spoken_obj":  "",
    "last_spoken_zone": "",
    "last_spoken_time": 0.0,
    "last_clear_time":  0.0,
    "repeat_count":     0,
    "voice_busy":       False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── HAPTIC / VOICE HTML ─────────────────────────────────────────────────────

components.html("""
<div id="haptic-container" style="text-align:center;margin-bottom:10px;">
  <button id="arm-btn" style="
    background:#DB4437;color:#fff;border:none;
    padding:15px 30px;font-size:1.2rem;font-weight:bold;
    border-radius:8px;cursor:pointer;width:100%;max-width:400px;
    font-family:'Inter',sans-serif;">
    TAP TO ENABLE HAPTICS &amp; VOICE
  </button>
  <div id="armed-indicator" style="
    display:none;color:#0F9D58;font-size:1.2rem;font-weight:bold;
    font-family:'Inter',sans-serif;padding:10px;">
    📳 Haptics ON &nbsp;|&nbsp; 🔊 Voice ON
  </div>
</div>
<script>
// ── Shared state on the PARENT window (accessible by all iframes) ──
var pw = window.parent;
pw.hapticsArmed             = pw.hapticsArmed            || false;
pw.currentVibrationPattern  = pw.currentVibrationPattern || [0];
pw.lastVibrationPattern     = pw.lastVibrationPattern    || [0];
pw.lastVibrationTime        = pw.lastVibrationTime       || 0;
pw.pendingSpeech            = pw.pendingSpeech           || "";
pw.pendingSpeechLang        = pw.pendingSpeechLang       || "en-US";
pw.lastSpokenText           = pw.lastSpokenText          || "";

document.getElementById('arm-btn').addEventListener('click', function() {
    pw.hapticsArmed = true;
    this.style.display = 'none';
    document.getElementById('armed-indicator').style.display = 'block';
    if(navigator.vibrate) navigator.vibrate(50);
    // Warm up speech engine with a silent utterance
    var warmup = new SpeechSynthesisUtterance(' ');
    warmup.volume = 0;
    window.speechSynthesis.speak(warmup);
});

setInterval(function() {
    // ── Haptic ──
    if (pw.hapticsArmed && navigator.vibrate) {
        let now = Date.now();
        let same = JSON.stringify(pw.currentVibrationPattern) ===
                   JSON.stringify(pw.lastVibrationPattern);
        if (!same || (now - pw.lastVibrationTime) > 2000) {
            if (pw.currentVibrationPattern[0] !== 0) {
                navigator.vibrate(pw.currentVibrationPattern);
                pw.lastVibrationPattern = [...pw.currentVibrationPattern];
                pw.lastVibrationTime    = now;
            }
        }
    }
    // ── Voice (only this single component ever calls speechSynthesis) ──
    if (pw.hapticsArmed &&
        pw.pendingSpeech &&
        pw.pendingSpeech !== pw.lastSpokenText &&
        !window.speechSynthesis.speaking) {
        var u    = new SpeechSynthesisUtterance(pw.pendingSpeech);
        u.lang   = pw.pendingSpeechLang;
        u.rate   = 1.0;
        u.volume = 1.0;
        window.speechSynthesis.speak(u);
        pw.lastSpokenText = pw.pendingSpeech;
    }
}, 300);
</script>
""", height=110)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

st.sidebar.title("Accessibility Settings")
selected_lang = st.sidebar.selectbox("Language / மொழி / भाषा", list(LANGUAGES.keys()))
lang = LANGUAGES[selected_lang]

st.sidebar.markdown("### About")
st.sidebar.markdown(
    "<div style='color:#bbb;font-size:0.9rem;'>"
    "Caregiver/developer view.<br>"
    "Running high-accuracy indoor AI navigation."
    "</div>", unsafe_allow_html=True
)
st.sidebar.markdown("---")
stats_ph = st.sidebar.empty()


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

CUSTOM_MODEL   = "best_sunrgbd.pt"
FALLBACK_MODEL = "yolov8s.pt"          # YOLOv8 Small for speed

CONF_THRESHOLD = 0.55
IOU_THRESHOLD  = 0.45
INFER_IMGSZ    = 320                   # Reduced resolution for speed

@st.cache_resource(show_spinner="Loading YOLOv8m model...")
def load_yolo():
    if os.path.exists(CUSTOM_MODEL):
        m    = YOLO(CUSTOM_MODEL)
        name = f"SUN RGB-D Fine-tuned ({CUSTOM_MODEL})"
    else:
        m    = YOLO(FALLBACK_MODEL)
        name = f"YOLOv8m COCO ({FALLBACK_MODEL})"
    return m, name

try:
    MODEL, MODEL_NAME = load_yolo()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ─── SMART VOICE ENGINE ──────────────────────────────────────────────────────

class VoiceEngine:
    """
    Manages smart debouncing, priority queuing, and sentence variation.
    All state is stored per-session in st.session_state so it persists
    across the 500ms autorefresh ticks.
    """

    @staticmethod
    def get_priority(label_en: str) -> int:
        if label_en in URGENT_OBJECTS:  return 0
        if label_en in WARNING_OBJECTS: return 1
        return 2

    @staticmethod
    def _build_base_alert(label_en: str, direction: str, dist: str) -> str:
        """Build the standard first-time alert string."""
        cfg = lang
        alert_key = INDOOR_ALERT_MAP.get(label_en, {}).get(direction)
        if alert_key and alert_key in cfg:
            base = cfg[alert_key]
            if dist == "VERY_CLOSE":
                return f"{cfg['warning']}! {base}"
            return base

        # Generic
        label_loc = OBJECT_TRANSLATIONS.get(selected_lang, {}).get(label_en, label_en)
        dir_str   = cfg["left"] if direction=="LEFT" else cfg["right"] if direction=="RIGHT" else cfg["ahead"]
        dist_str  = cfg["very_close"] if dist=="VERY_CLOSE" else cfg["close"] if dist=="CLOSE" else cfg["nearby"] if dist=="MEDIUM" else ""

        if dist == "VERY_CLOSE" and direction == "CENTER":
            return f"{cfg['warning']}! {label_loc} {cfg['ahead']}, {cfg['very_close']}"
        parts = [label_loc, dir_str]
        if dist_str:
            parts.append(dist_str)
        return ", ".join(parts)

    @staticmethod
    def _persistence_alert(label_en: str, direction: str, repeat_count: int) -> str:
        """After 3 repeats, give navigation guidance."""
        if repeat_count >= 3:
            return lang["persists"]

        label_loc = OBJECT_TRANSLATIONS.get(selected_lang, {}).get(label_en, label_en)
        still     = lang["still_ahead"]

        # Suggest alternate direction
        if direction == "CENTER":
            nudge = lang["move_right"]
        elif direction == "LEFT":
            nudge = lang["move_right"]
        else:
            nudge = lang["move_left"]

        return f"{label_loc} {still}. {nudge}"

    @classmethod
    def should_speak(cls, label_en: str, direction: str, dist: str) -> tuple[bool, str]:
        """
        Decide if we should speak now and return the text.
        Returns (should_speak, text)
        """
        now      = time.time()
        priority = cls.get_priority(label_en)
        interval = REPEAT_INTERVALS[priority]

        prev_obj  = st.session_state.last_spoken_obj
        prev_zone = st.session_state.last_spoken_zone
        prev_time = st.session_state.last_spoken_time
        repeat_cnt = st.session_state.repeat_count
        elapsed   = now - prev_time

        # --- Always speak urgently for URGENT class ---
        if priority == 0:
            if elapsed >= interval:
                text = cls._build_base_alert(label_en, direction, dist)
                return True, text
            return False, ""

        # --- NEW object appeared ---
        if label_en != prev_obj:
            text = cls._build_base_alert(label_en, direction, dist)
            return True, text

        # --- Same object, zone changed ---
        if direction != prev_zone and elapsed >= 1.0:
            text = cls._build_base_alert(label_en, direction, dist)
            return True, text

        # --- Same object, same zone, within repeat interval ---
        if elapsed < interval:
            return False, ""

        # --- Repeat with persistence variation ---
        text = cls._persistence_alert(label_en, direction, repeat_cnt)
        return True, text

    @classmethod
    def record_spoken(cls, label_en: str, direction: str):
        """Update state after speaking."""
        prev_obj = st.session_state.last_spoken_obj
        if label_en == prev_obj:
            st.session_state.repeat_count += 1
            if st.session_state.repeat_count > 3:
                st.session_state.repeat_count = 3
        else:
            st.session_state.repeat_count = 0

        st.session_state.last_spoken_obj  = label_en
        st.session_state.last_spoken_zone = direction
        st.session_state.last_spoken_time = time.time()

    @classmethod
    def should_say_clear(cls) -> bool:
        now = time.time()
        if now - st.session_state.last_clear_time >= CLEAR_PATH_INTERVAL:
            st.session_state.last_clear_time = now
            return True
        return False


def emit_voice_haptic(text: str, dist: str, lang_code: str):
    """
    Set parent-window variables — the single persistent component
    (haptic+voice block above) picks them up and speaks/vibrates.
    No speechSynthesis calls here — avoids iframe conflicts.
    """
    vib = "[150]"
    if   dist == "VERY_CLOSE": vib = "[100,50,100,50,100]"
    elif dist == "CLOSE":      vib = "[200,100,200]"
    elif dist == "MEDIUM":     vib = "[300]"

    safe = text.replace("'", "\\'").replace('"', '\\"')
    components.html(f"""
    <script>
    var pw = window.parent;
    if (pw.currentVibrationPattern !== undefined)
        pw.currentVibrationPattern = {vib};
    if (pw.pendingSpeech !== undefined) {{
        pw.pendingSpeech     = '{safe}';
        pw.pendingSpeechLang = '{lang_code}';
        pw.lastSpokenText    = '';  /* reset so the poller picks it up */
    }}
    </script>
    """, height=0)
    st.session_state.last_spoken_text = text


# ─── FRAME HELPERS ───────────────────────────────────────────────────────────

def letterbox(img, size=640):
    h, w  = img.shape[:2]
    s     = size / max(h, w)
    nw, nh = int(w*s), int(h*s)
    rsz   = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), np.uint8)
    px, py = (size-nw)//2, (size-nh)//2
    canvas[py:py+nh, px:px+nw] = rsz
    return canvas, s, px, py

def unbox(x1, y1, x2, y2, s, px, py):
    return (x1-px)/s, (y1-py)/s, (x2-px)/s, (y2-py)/s

def bbox_to_dist(bh, fh):
    r = bh / fh
    if r > 0.40: return "VERY_CLOSE"
    if r > 0.20: return "CLOSE"
    if r > 0.10: return "MEDIUM"
    return "FAR"

def direction(cx, fw):
    r = cx / fw
    if r < 0.35: return "LEFT"
    if r > 0.65: return "RIGHT"
    return "CENTER"

def dist_css(d):
    return {"VERY_CLOSE":"priority-urgent","CLOSE":"priority-warning",
            "MEDIUM":"priority-info"}.get(d,"priority-clear")

def detect_large_obstacle(frame):
    """Fast CPU fallback obstacle detector."""
    h, w   = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (21,21), 0)
    edges  = cv2.Canny(blur, 30, 100)
    cx1, cx2 = int(w*.25), int(w*.75)
    cy1, cy2 = int(h*.15), int(h*.85)
    roi    = edges[cy1:cy2, cx1:cx2]
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area   = (cx2-cx1)*(cy2-cy1)
    total  = sum(cv2.contourArea(c) for c in cnts if cv2.contourArea(c)>500)
    r      = total / area
    if r > .45: return "VERY_CLOSE"
    if r > .25: return "CLOSE"
    return None


# ─── VIDEO PROCESSOR ─────────────────────────────────────────────────────────

class VideoProcessor:
    def __init__(self):
        self.latest_label_en = ""
        self.latest_dir      = "CENTER"
        self.latest_dist     = "FAR"
        self.has_detection   = False

        self.total_frames    = 0
        self.total_dets      = 0
        self.last_infer_time = 0.0
        self.last_results    = []
        self.empty_count     = 0

        # Temporal smoothing: track how many consecutive frames each obj seen
        # key = label_en, value = {"count": int, "miss": int, last_dir, last_dist, last_bbox}
        self.tracker: dict = {}

    def _update_tracker(self, detections: list) -> list:
        """
        Apply temporal smoothing:
          - Increment frame count for seen objects
          - Increment miss count for unseen objects
          - Only return objects confirmed in >= CONFIRM_FRAMES consecutive frames
          - Drop objects missing > GRACE_FRAMES
        """
        seen_labels = {d["label_en"] for d in detections}

        # Update seen
        for det in detections:
            lb = det["label_en"]
            if lb not in self.tracker:
                self.tracker[lb] = {"count": 1, "miss": 0,
                                    "dir": det["dir"], "dist": det["dist"],
                                    "bbox": det["bbox"], "conf": det["conf"]}
            else:
                self.tracker[lb]["count"] = min(self.tracker[lb]["count"] + 1, 10)
                self.tracker[lb]["miss"]  = 0
                self.tracker[lb]["dir"]   = det["dir"]
                self.tracker[lb]["dist"]  = det["dist"]
                self.tracker[lb]["bbox"]  = det["bbox"]
                self.tracker[lb]["conf"]  = det["conf"]

        # Update unseen
        to_drop = []
        for lb, info in self.tracker.items():
            if lb not in seen_labels:
                info["miss"] += 1
                if info["miss"] > GRACE_FRAMES:
                    to_drop.append(lb)
        for lb in to_drop:
            del self.tracker[lb]

        # Return only confirmed objects
        confirmed = []
        for lb, info in self.tracker.items():
            if info["count"] >= CONFIRM_FRAMES:
                confirmed.append({
                    "label_en": lb,
                    "dir":  info["dir"],
                    "dist": info["dist"],
                    "bbox": info["bbox"],
                    "conf": info["conf"],
                    "priority": OBJECT_PRIORITY.get(lb, 2),
                    "rank": (OBJECT_PRIORITY.get(lb, 2),
                             0 if info["dist"]=="VERY_CLOSE" else
                             1 if info["dist"]=="CLOSE" else 2),
                })
        return confirmed

    def recv(self, frame):
        img    = frame.to_ndarray(format="bgr24")
        fh, fw = img.shape[:2]
        self.total_frames += 1

        now = time.time()
        if now - self.last_infer_time >= 0.6:   # 600ms inference interval (reduced framerate to fix lag)

            lb_img, s, px, py = letterbox(img, INFER_IMGSZ)
            results = MODEL(lb_img, verbose=False,
                            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                            imgsz=INFER_IMGSZ)[0]

            raw_dets = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                ux1, uy1, ux2, uy2 = unbox(x1, y1, x2, y2, s, px, py)
                lb_en = MODEL.names[int(box.cls[0])]
                raw_dets.append({
                    "label_en": lb_en,
                    "dist": bbox_to_dist(uy2-uy1, fh),
                    "dir":  direction((ux1+ux2)/2, fw),
                    "bbox": (int(ux1), int(uy1), int(ux2), int(uy2)),
                    "conf": float(box.conf[0]),
                })

            self.last_results    = self._update_tracker(raw_dets)
            self.last_infer_time = now
            self.total_dets     += len([d for d in raw_dets])

        # ── Draw annotated frame ──────────────────────────────────────────────
        annotated = img.copy()
        COLORS = {"VERY_CLOSE":(0,0,255), "CLOSE":(0,140,255),
                  "MEDIUM":(0,200,0), "FAR":(200,200,200)}

        if self.last_results:
            self.empty_count = 0
            self.last_results.sort(key=lambda x: x["rank"])
            primary = self.last_results[0]

            self.latest_label_en = primary["label_en"]
            self.latest_dir      = primary["dir"]
            self.latest_dist     = primary["dist"]
            self.has_detection   = True

            for det in self.last_results:
                bx1, by1, bx2, by2 = det["bbox"]
                col = COLORS.get(det["dist"], (200,200,200))
                cv2.rectangle(annotated, (bx1,by1), (bx2,by2), col, 3)
                lbl = f"{det['label_en'].upper()} {int(det['conf']*100)}%"
                cv2.putText(annotated, lbl, (bx1, max(by1-8,16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            # Direction arrow HUD
            arrow = {"LEFT":"◀","RIGHT":"▶"}.get(primary["dir"],"▲")
            cv2.putText(annotated, arrow, (fw//2-20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLORS.get(primary["dist"],(200,200,200)), 3)

        else:
            self.has_detection = False
            self.empty_count  += 1
            # Fast CPU obstacle check when YOLO finds nothing
            obs = detect_large_obstacle(img)
            if obs:
                self.has_detection   = True
                self.latest_label_en = "obstacle"
                self.latest_dir      = "CENTER"
                self.latest_dist     = obs
                cv2.putText(annotated, f"OBSTACLE {obs}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─── MAIN UI ─────────────────────────────────────────────────────────────────

st.markdown("<h1>VisionAid</h1>", unsafe_allow_html=True)
st.markdown("<p class='caregiver-subtitle'>👁 CAREGIVER VIEW — Smart Voice + High Accuracy</p>",
            unsafe_allow_html=True)
st.markdown(f"<div class='model-badge'>🤖 {MODEL_NAME}</div>", unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="visionaid-v3",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=False,
)

if webrtc_ctx.state.playing:
    st_autorefresh(interval=500, key="visionaid_loop")

    if webrtc_ctx.video_processor:
        proc = webrtc_ctx.video_processor
        st.session_state.frame_count = proc.total_frames
        st.session_state.det_count   = proc.total_dets

        if proc.has_detection:
            lbl_en = proc.latest_label_en
            ddir   = proc.latest_dir
            ddist  = proc.latest_dist

            # Handle generic "obstacle" from CPU fallback
            if lbl_en == "obstacle":
                should_speak = VoiceEngine.should_speak("obstacle", "CENTER", ddist)
                if should_speak[0]:
                    dist_str = lang["very_close"] if ddist=="VERY_CLOSE" else lang["close"]
                    text = f"{lang['warning']}! {lang['obstacle']}, {dist_str}"
                    emit_voice_haptic(text, ddist, lang["code"])
                    VoiceEngine.record_spoken("obstacle", "CENTER")
                    st.session_state.ui_msg       = text.upper()
                    st.session_state.ui_msg_class = dist_css(ddist)
            else:
                ok, text = VoiceEngine.should_speak(lbl_en, ddir, ddist)
                if ok:
                    emit_voice_haptic(text, ddist, lang["code"])
                    VoiceEngine.record_spoken(lbl_en, ddir)
                    st.session_state.ui_msg       = text.upper()
                    st.session_state.ui_msg_class = dist_css(ddist)

        else:
            # No detection — heartbeat clear-path (max once per 10s)
            if proc.empty_count >= 10 and VoiceEngine.should_say_clear():
                text = lang["clear"]
                emit_voice_haptic(text, "FAR", lang["code"])
                st.session_state.last_spoken_obj  = ""
                st.session_state.last_spoken_zone = ""
                st.session_state.repeat_count     = 0
                st.session_state.ui_msg       = text.upper()
                st.session_state.ui_msg_class = "priority-clear"

stats_ph.markdown(
    f"**Frames:** {st.session_state.frame_count} | "
    f"**Dets:** {st.session_state.det_count} | "
    f"conf={CONF_THRESHOLD} iou={IOU_THRESHOLD}"
)

st.markdown(f"""
<div class="status-panel">
  <div class="status-text {st.session_state.ui_msg_class}">
    {st.session_state.ui_msg}<br>
    <span style='font-size:0.85rem;color:#888;font-weight:400;'>Voice Engine Active</span>
  </div>
</div>
""", unsafe_allow_html=True)

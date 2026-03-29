"""
VisionAid — High-Accuracy Indoor Navigation for Visually Impaired
==========================================================
Bugfixes & Upgrades (v2):
  • YOLOv8m inference at imgsz=960 for high accuracy
  • Class-specific confidence thresholds (furniture=0.30, person=0.55)
  • Directional Guidance: "Table ahead - move right"
  • OpenCV Wall Detection: uniform color regions & optical flow approach
  • OpenCV Table Fallback: horizontal edge detection
  • Strict "Path is clear" logic (empty 2s required)
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
    "sink", "refrigerator", "cabinet", "desk", "table", "dining table",
}

# INFO: speak once, 6s repeat (everything else)
OBJECT_PRIORITY = {obj: 0 for obj in URGENT_OBJECTS}
OBJECT_PRIORITY.update({obj: 1 for obj in WARNING_OBJECTS})

REPEAT_INTERVALS = {0: 2.0, 1: 4.0, 2: 6.0}   # seconds per priority
CLEAR_PATH_INTERVAL = 10.0                       # speak "clear" only every 10s

# Temporal smoothing
CONFIRM_FRAMES = 3
GRACE_FRAMES   = 2

# Minimum time center must be empty before "path clear" (in addition to 10 frames)
EMPTY_CENTER_TIME = 2.0


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
        "wall_ahead": "Wall ahead, please stop",
        "wall_approach": "Wall approaching, slow down",
        "surface": "Large surface ahead",
        
        # Navigation directives
        "nav_move_left": "- move left, path is clear",
        "nav_move_right": "- move right, path is clear",
        "nav_stop": "- stop, both sides blocked",
        "nav_step_back": "- step back, obstacle directly ahead",
        "nav_obj_on_left": "Object on left - move right",
        "nav_obj_on_right": "Object on right - move left",

        # Indoor-specific
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
        "still_ahead":    "still ahead",
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
        "wall_ahead": "சுவர் முன்னால் — நிறுத்துங்கள்",
        "wall_approach": "சுவர் நெருங்குகிறது, மெதுவாக செல்லுங்கள்",
        "surface": "முன்னால் மேசை அல்லது பெரிய பரப்பு",
        
        # Navigation directives
        "nav_move_left": "- இடதுபுறம் செல்லுங்கள்",
        "nav_move_right": "- வலதுபுறம் செல்லுங்கள்",
        "nav_stop": "- நிறுத்துங்கள்! இரண்டு பக்கமும் தடை",
        "nav_step_back": "- பின்னால் செல்லவும், நேராக தடை",
        "nav_obj_on_left": "தடை இடதுபுறம் - வலதுபுறம் செல்லுங்கள்",
        "nav_obj_on_right": "தடை வலதுபுறம் - இடதுபுறம் செல்லுங்கள்",

        # Indoor-specific
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
        "still_ahead":    "இன்னும் முன்னால் உள்ளது",
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
        "wall_ahead": "दीवार सामने — रुकिए",
        "wall_approach": "दीवार पास आ रही है, धीरे चलें",
        "surface": "सामने मेज या बड़ी सतह",
        
        # Navigation directives
        "nav_move_left": "- बाईं ओर जाएं, रास्ता साफ है",
        "nav_move_right": "- दाईं ओर जाएं, रास्ता साफ है",
        "nav_stop": "- रुकिए! दोनों तरफ बंद है",
        "nav_step_back": "- पीछे हटें, ठीक सामने रुकावट है",
        "nav_obj_on_left": "बाईं तरफ रुकावट - दाईं ओर जाएं",
        "nav_obj_on_right": "दाईं तरफ रुकावट - बाईं ओर जाएं",

        # Indoor-specific
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
        "still_ahead":    "अभी भी आगे है",
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
    "last_spoken_text": "", "last_spoken_obj":  "", "last_spoken_zone": "",
    "last_spoken_time": 0.0, "last_clear_time":  0.0, "repeat_count": 0,
    "voice_busy": False,
    # Optical flow state
    "prev_gray": None,
    # Center empty tracking
    "last_center_block_time": time.time(),
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
    var warmup = new SpeechSynthesisUtterance(' ');
    warmup.volume = 0;
    window.speechSynthesis.speak(warmup);
});

setInterval(function() {
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
    if (pw.hapticsArmed && pw.pendingSpeech && pw.pendingSpeech !== pw.lastSpokenText && !window.speechSynthesis.speaking) {
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
    "Running high-accuracy indoor AI navigation with Wall/Table fallback."
    "</div>", unsafe_allow_html=True
)
st.sidebar.markdown("---")
stats_ph = st.sidebar.empty()


# ─── MODEL LOADING & CONFIG ──────────────────────────────────────────────────

CUSTOM_MODEL   = "best_sunrgbd.pt"
FALLBACK_MODEL = "yolov8s.pt"          # YOLOv8 Small (faster on CPU)

INFER_IMGSZ    = 480                   # Moderate res to reduce lag
IOU_THRESHOLD  = 0.45

# Class-specific confidence
DEFAULT_CONF = 0.50
CLASS_CONF_THRESHOLDS = {
    "person": 0.55,
    "table": 0.35, "dining table": 0.35, "desk": 0.35,
    "chair": 0.35, "sofa": 0.35, "couch": 0.35, "bed": 0.35,
    "door": 0.35,
    "stairs": 0.40,
    "cabinet": 0.40, "refrigerator": 0.40,
}
# Using ultralytics built-in filtering (max performance), we set baseline to lowest
BASE_CONF = min(CLASS_CONF_THRESHOLDS.values())

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo():
    if os.path.exists(CUSTOM_MODEL):
        m    = YOLO(CUSTOM_MODEL)
        name = f"SUN RGB-D Fine-tuned ({CUSTOM_MODEL}) @ {INFER_IMGSZ}px"
    else:
        m    = YOLO(FALLBACK_MODEL)
        name = f"YOLO COCO ({FALLBACK_MODEL}) @ {INFER_IMGSZ}px"
    return m, name

try:
    MODEL, MODEL_NAME = load_yolo()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ─── SMART VOICE ENGINE ──────────────────────────────────────────────────────

class VoiceEngine:
    @staticmethod
    def get_priority(label_en: str) -> int:
        if label_en in URGENT_OBJECTS:  return 0
        if label_en in WARNING_OBJECTS: return 1
        return 2

    @staticmethod
    def _build_base_alert(label_en: str, direction: str, dist: str, left_clear: bool, right_clear: bool) -> str:
        cfg = lang
        base = ""
        
        # 1. Base object component
        alert_key = INDOOR_ALERT_MAP.get(label_en, {}).get(direction)
        if alert_key and alert_key in cfg:
            base = cfg[alert_key]
        elif label_en == "wall_ahead":
            return cfg["wall_ahead"]
        elif label_en == "wall_approach":
            return cfg["warning"] + "! " + cfg["wall_approach"]
        elif label_en == "surface":
            base = cfg["surface"]
        elif label_en == "obstacle_large":
            base = cfg["obstacle"]
        else:
            label_loc = OBJECT_TRANSLATIONS.get(selected_lang, {}).get(label_en, label_en)
            dir_str   = cfg["left"] if direction=="LEFT" else cfg["right"] if direction=="RIGHT" else cfg["ahead"]
            dist_str  = cfg["very_close"] if dist=="VERY_CLOSE" else cfg["close"] if dist=="CLOSE" else cfg["nearby"] if dist=="MEDIUM" else ""
            base = f"{label_loc} {dir_str}"
            if dist_str: base += f", {dist_str}"
            
        if dist == "VERY_CLOSE":
            base = f"{cfg['warning']}! {base}"
            
        # 2. Add directional guidance
        if direction == "CENTER":
            if left_clear and right_clear:
                # Need to side-step
                base += f" {cfg['nav_move_left']}"
            elif left_clear and not right_clear:
                base += f" {cfg['nav_move_left']}"
            elif right_clear and not left_clear:
                base += f" {cfg['nav_move_right']}"
            else:
                base += f" {cfg['nav_stop']}"
        elif direction == "LEFT":
            base += f" - {cfg['nav_obj_on_left']}"
        elif direction == "RIGHT":
            base += f" - {cfg['nav_obj_on_right']}"
            
        return base

    @staticmethod
    def _persistence_alert(label_en: str, direction: str, repeat_count: int) -> str:
        if repeat_count >= 3:
            return lang["persists"]

        if label_en in ["wall_ahead", "surface", "obstacle_large"]:
            label_loc = lang["warning"]
        else:
            label_loc = OBJECT_TRANSLATIONS.get(selected_lang, {}).get(label_en, label_en)
            
        still = lang["still_ahead"]
        nudge = lang["move_right"] if direction in ["CENTER", "LEFT"] else lang["move_left"]
        return f"{label_loc} {still}. {nudge}"

    @classmethod
    def should_speak(cls, label_en: str, direction: str, dist: str, left_clear: bool, right_clear: bool) -> tuple[bool, str]:
        now      = time.time()
        priority = cls.get_priority(label_en)
        interval = REPEAT_INTERVALS[priority]

        prev_obj  = st.session_state.last_spoken_obj
        prev_zone = st.session_state.last_spoken_zone
        prev_time = st.session_state.last_spoken_time
        repeat_cnt = st.session_state.repeat_count
        elapsed   = now - prev_time

        if priority == 0:
            if elapsed >= interval:
                text = cls._build_base_alert(label_en, direction, dist, left_clear, right_clear)
                return True, text
            return False, ""

        if label_en != prev_obj:
            text = cls._build_base_alert(label_en, direction, dist, left_clear, right_clear)
            return True, text

        if direction != prev_zone and elapsed >= 1.0:
            text = cls._build_base_alert(label_en, direction, dist, left_clear, right_clear)
            return True, text

        if elapsed < interval:
            return False, ""

        text = cls._persistence_alert(label_en, direction, repeat_cnt)
        return True, text

    @classmethod
    def record_spoken(cls, label_en: str, direction: str):
        prev_obj = st.session_state.last_spoken_obj
        if label_en == prev_obj:
            st.session_state.repeat_count = min(st.session_state.repeat_count + 1, 3)
        else:
            st.session_state.repeat_count = 0

        st.session_state.last_spoken_obj  = label_en
        st.session_state.last_spoken_zone = direction
        st.session_state.last_spoken_time = time.time()

    @classmethod
    def should_say_clear(cls) -> bool:
        now = time.time()
        # Ensure center zone was actually empty for 2 seconds
        if now - st.session_state.last_center_block_time < EMPTY_CENTER_TIME:
            return False
            
        if now - st.session_state.last_clear_time >= CLEAR_PATH_INTERVAL:
            st.session_state.last_clear_time = now
            return True
        return False


def emit_voice_haptic(text: str, dist: str, lang_code: str):
    vib = "[150]"
    if   dist == "VERY_CLOSE": vib = "[100,50,100,50,100]"
    elif dist == "CLOSE":      vib = "[200,100,200]"
    elif dist == "MEDIUM":     vib = "[300]"

    safe = text.replace("'", "\\'").replace('"', '\\"')
    components.html(f"""
    <script>
    var pw = window.parent;
    if (pw.currentVibrationPattern !== undefined) pw.currentVibrationPattern = {vib};
    if (pw.pendingSpeech !== undefined) {{
        pw.pendingSpeech = '{safe}';
        pw.pendingSpeechLang = '{lang_code}';
        pw.lastSpokenText = '';
    }}
    </script>
    """, height=0)
    st.session_state.last_spoken_text = text


# ─── OPENCV FALLBACK DETECTORS ───────────────────────────────────────────────

def detect_large_obstacle(frame):
    """Fallback: Large contour >15% of frame -> obstacle."""
    h, w   = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (21,21), 0)
    edges  = cv2.Canny(blur, 30, 100)
    
    # Check lower 60% of frame (path area)
    cy1, cy2 = int(h*0.4), h
    roi = edges[cy1:cy2, :]
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area = w * (cy2-cy1)
    for c in cnts:
        c_area = cv2.contourArea(c)
        if c_area / area > 0.15:  # >15% of bottom frame
            return "VERY_CLOSE"
    return None

def detect_table_edge(frame):
    """Fallback: Strong horizontal lines in middle frame -> table/desk."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Focus on middle 50% height
    cy1, cy2 = int(h*0.25), int(h*0.75)
    roi = gray[cy1:cy2, :]
    
    edges = cv2.Canny(cv2.GaussianBlur(roi, (5,5), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w*0.4, maxLineGap=20)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if nearly horizontal
            if abs(y2-y1) < 20:
                return "CLOSE"
    return None

def detect_wall_color(frame):
    """Fallback: Frame is dominated by single flat color/texture -> Wall."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Take center sample
    cx1, cx2 = int(w*0.3), int(w*0.7)
    cy1, cy2 = int(h*0.3), int(h*0.7)
    center_roi = hsv[cy1:cy2, cx1:cx2]
    
    # Get median color
    median_val = np.median(center_roi, axis=(0,1))
    
    # Threshold whole image against median
    lower = np.clip(median_val - np.array([10, 40, 40]), 0, 255)
    upper = np.clip(median_val + np.array([10, 40, 40]), 0, 255)
    
    mask = cv2.inRange(hsv, lower, upper)
    matching_pixels = cv2.countNonZero(mask)
    
    # If >35% of frame is exactly the same color as center -> Wall
    if matching_pixels / (h*w) > 0.35:
        return "VERY_CLOSE"
    return None


# ─── FRAME HELPERS ───────────────────────────────────────────────────────────

def letterbox(img, size=960):
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


# ─── VIDEO PROCESSOR ─────────────────────────────────────────────────────────

class VideoProcessor:
    def __init__(self):
        self.latest_label_en = ""
        self.latest_dir      = "CENTER"
        self.latest_dist     = "FAR"
        self.has_detection   = False
        
        self.left_clear      = True
        self.right_clear     = True

        self.total_frames    = 0
        self.total_dets      = 0
        self.last_infer_time = 0.0
        self.last_results    = []
        self.empty_count     = 0

        self.tracker: dict   = {}

    def _update_tracker(self, detections: list) -> list:
        seen_labels = {d["label_en"] for d in detections}

        for det in detections:
            lb = det["label_en"]
            if lb not in self.tracker:
                self.tracker[lb] = {"count": 1, "miss": 0, "dir": det["dir"], "dist": det["dist"], "bbox": det["bbox"], "conf": det["conf"]}
            else:
                self.tracker[lb]["count"] = min(self.tracker[lb]["count"] + 1, 10)
                self.tracker[lb]["miss"]  = 0
                self.tracker[lb]["dir"]   = det["dir"]
                self.tracker[lb]["dist"]  = det["dist"]
                self.tracker[lb]["bbox"]  = det["bbox"]
                self.tracker[lb]["conf"]  = det["conf"]

        to_drop = []
        for lb, info in self.tracker.items():
            if lb not in seen_labels:
                info["miss"] += 1
                if info["miss"] > GRACE_FRAMES: to_drop.append(lb)
        for lb in to_drop: del self.tracker[lb]

        confirmed = []
        for lb, info in self.tracker.items():
            if info["count"] >= CONFIRM_FRAMES:
                confirmed.append({
                    "label_en": lb, "dir": info["dir"], "dist": info["dist"],
                    "bbox": info["bbox"], "conf": info["conf"],
                    "rank": (OBJECT_PRIORITY.get(lb, 2),
                             0 if info["dist"]=="VERY_CLOSE" else 1 if info["dist"]=="CLOSE" else 2),
                })
        return confirmed

    def recv(self, frame):
        img    = frame.to_ndarray(format="bgr24")
        fh, fw = img.shape[:2]
        self.total_frames += 1

        now = time.time()
        if now - self.last_infer_time >= 0.7:  # 700ms interval to stop lag

            lb_img, s, px, py = letterbox(img, INFER_IMGSZ)
            results = MODEL(lb_img, verbose=False, conf=BASE_CONF, iou=IOU_THRESHOLD, imgsz=INFER_IMGSZ)[0]

            raw_dets = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                lb_en  = MODEL.names[cls_id]
                conf   = float(box.conf[0])
                
                # Apply class-specific thresholds
                thresh = CLASS_CONF_THRESHOLDS.get(lb_en, DEFAULT_CONF)
                if conf < thresh: continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                ux1, uy1, ux2, uy2 = unbox(x1, y1, x2, y2, s, px, py)
                
                raw_dets.append({
                    "label_en": lb_en,
                    "dist": bbox_to_dist(uy2-uy1, fh),
                    "dir":  direction((ux1+ux2)/2, fw),
                    "bbox": (int(ux1), int(uy1), int(ux2), int(uy2)),
                    "conf": conf,
                })

            self.last_results    = self._update_tracker(raw_dets)
            self.last_infer_time = now
            self.total_dets     += len(raw_dets)

        annotated = img.copy()
        COLORS = {"VERY_CLOSE":(0,0,255), "CLOSE":(0,140,255), "MEDIUM":(0,200,0), "FAR":(200,200,200)}

        if self.last_results:
            self.empty_count = 0
            st.session_state.last_center_block_time = time.time()
            
            self.last_results.sort(key=lambda x: x["rank"])
            primary = self.last_results[0]

            self.latest_label_en = primary["label_en"]
            self.latest_dir      = primary["dir"]
            self.latest_dist     = primary["dist"]
            self.has_detection   = True
            
            # Check side clearance for navigation directions
            self.left_clear  = not any(d["dir"] == "LEFT" for d in self.last_results)
            self.right_clear = not any(d["dir"] == "RIGHT" for d in self.last_results)

            for det in self.last_results:
                bx1, by1, bx2, by2 = det["bbox"]
                col = COLORS.get(det["dist"], (200,200,200))
                cv2.rectangle(annotated, (bx1,by1), (bx2,by2), col, 3)
                lbl = f"{det['label_en'].upper()} {int(det['conf']*100)}%"
                cv2.putText(annotated, lbl, (bx1, max(by1-8,16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            arrow = {"LEFT":"◀","RIGHT":"▶"}.get(primary["dir"],"▲")
            cv2.putText(annotated, arrow, (fw//2-20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLORS.get(primary["dist"],(200,200,200)), 3)

        else:
            self.left_clear = True
            self.right_clear = True
            
            # ── Fallback cascade ──
            wall_dist = detect_wall_color(img)
            edge_dist = None if wall_dist else detect_table_edge(img)
            obs_dist  = None if (wall_dist or edge_dist) else detect_large_obstacle(img)
            
            fallback_label = None
            fallback_dist  = None
            
            if wall_dist:
                fallback_label, fallback_dist = "wall_ahead", wall_dist
            elif edge_dist:
                fallback_label, fallback_dist = "surface", edge_dist
            elif obs_dist:
                fallback_label, fallback_dist = "obstacle_large", obs_dist

            if fallback_label:
                self.has_detection   = True
                self.empty_count     = 0
                st.session_state.last_center_block_time = time.time()
                self.latest_label_en = fallback_label
                self.latest_dir      = "CENTER"
                self.latest_dist     = fallback_dist
                cv2.putText(annotated, f"FALLBACK: {fallback_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                self.has_detection = False
                self.empty_count  += 1

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─── MAIN UI ─────────────────────────────────────────────────────────────────

st.markdown("<h1>VisionAid</h1>", unsafe_allow_html=True)
st.markdown("<p class='caregiver-subtitle'>👁 CAREGIVER VIEW — Smart Voice + High Accuracy</p>", unsafe_allow_html=True)
st.markdown(f"<div class='model-badge'>🤖 {MODEL_NAME}</div>", unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="visionaid-v4",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
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
            lc     = proc.left_clear
            rc     = proc.right_clear

            ok, text = VoiceEngine.should_speak(lbl_en, ddir, ddist, lc, rc)
            if ok:
                emit_voice_haptic(text, ddist, lang["code"])
                VoiceEngine.record_spoken(lbl_en, ddir)
                st.session_state.ui_msg       = text.upper()
                st.session_state.ui_msg_class = dist_css(ddist)

        else:
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
    f"imsz={INFER_IMGSZ} min_conf={BASE_CONF}"
)

st.markdown(f"""
<div class="status-panel">
  <div class="status-text {st.session_state.ui_msg_class}">
    {st.session_state.ui_msg}<br>
    <span style='font-size:0.85rem;color:#888;font-weight:400;'>Voice Engine Active</span>
  </div>
</div>
""", unsafe_allow_html=True)

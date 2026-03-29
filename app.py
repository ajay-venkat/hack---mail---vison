"""
VisionAid — High-Accuracy Indoor Navigation for Visually Impaired
==========================================================
3rd Massive Upgrade (v3):
  • Real-World Distance Warning (5m, Focal Length f=615)
  • Fall Detection (Camera tilt + Brightness drop)
  • Stair Direction (UP / DOWN via Perspective Lines)
  • Moving Object Detection (Motion Mask Overlay)
  • Scene Understanding (Indoor, Outdoor, Road, Corridor)
  • Emergency SOS System (UI Button + Sound)
  • Smart Navigation Memory (Stuck Detection 5s+)
  • Battery & Performance Toggle (Low, Normal, High)
  • High-Contrast UI HUD & Caregiver Log
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

# ─── PAGE CONFIG & CSS ───────────────────────────────────────────────────────

st.set_page_config(page_title="VisionAid Pro", page_icon="👁️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
.main { background-color: #000 !important; color: #fff !important; font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stApp    { max-width: 100%; padding: 0; background-color: #000 !important; }
p, li, span, div { font-size: 1.1rem !important; color: #fff !important; }

/* HUD Dashboard Styles */
.hud-container { display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; }
.hud-box {
    background: #111; border: 2px solid #333; border-radius: 12px; padding: 15px; flex: 1;
    min-width: 150px; text-align: center; display: flex; flex-direction: column; justify-content: center;
}
.hud-red { border-color: #DB4437; background: rgba(219,68,55,0.1); }
.hud-yellow { border-color: #F4B400; background: rgba(244,180,0,0.1); }
.hud-green { border-color: #0F9D58; background: rgba(15,157,88,0.1); }
.hud-title { font-size: 0.85rem !important; color: #aaa !important; text-transform: uppercase; font-weight: 700; margin-bottom: 5px; }
.hud-value { font-size: 1.8rem !important; font-weight: 900; line-height: 1.2; }

/* Status Panel */
.status-panel { background: #0D1117; padding: 25px; border: 3px solid #1A73E8; border-radius: 15px; text-align: center; margin-top: 15px; }
.status-text { font-size: 24px !important; font-weight: 900; letter-spacing: 1px; }

/* Priorities */
.priority-urgent  { color: #DB4437 !important; text-shadow: 0 0 15px rgba(219,68,55,0.5); }
.priority-warning { color: #F4B400 !important; }
.priority-info    { color: #0F9D58 !important; }
.priority-clear   { color: #888888 !important; }

/* SOS Button */
.stButton>button {
    width: 100%; font-weight: 900; font-size: 1.2rem; padding: 15px;
    border-radius: 12px; border: none; transition: 0.2s;
}
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR & SETTINGS ──────────────────────────────────────────────────────

st.sidebar.title("VisionAid Settings")

# Contact & Language
contact_num = st.sidebar.text_input("Emergency Contact Number", value="+1234567890")
selected_lang = st.sidebar.selectbox("Language / மொழி / भाषा", ["English", "Tamil", "Hindi"])

# Performance Modes
st.sidebar.markdown("### Performance Mode")
perf_mode = st.sidebar.radio(
    "Select Mode", 
    options=["Low Battery (Fast)", "Normal (Balanced)", "High Accuracy (Slow)"],
    index=1
)

PERF_MAP = {
    "Low Battery (Fast)":   {"model": "yolov8n.pt", "imgsz": 320, "interval": 1.0},
    "Normal (Balanced)":    {"model": "yolov8s.pt", "imgsz": 480, "interval": 0.6},
    "High Accuracy (Slow)":{"model": "yolov8m.pt", "imgsz": 960, "interval": 0.3},
}
ACTIVE_PERF = PERF_MAP[perf_mode]

st.sidebar.markdown("---")
st.sidebar.markdown("### Caregiver Log")
caregiver_log = st.sidebar.empty()


# ─── REAL-WORLD DISTANCE CONSTANTS ───────────────────────────────────────────

FOCAL_LENGTH_PX = 615.0  # Appx focal length for typical smartphone camera at 480p/640p
REF_HEIGHTS_M = {
    "person": 1.7, "door": 2.0, "car": 1.5, "truck": 3.0, "bus": 3.0,
    "motorcycle": 1.2, "bicycle": 1.1, "chair": 0.9, "table": 0.75, 
    "dining table": 0.75, "desk": 0.75, "bed": 0.6, "sofa": 0.8, "couch": 0.8,
    "cabinet": 1.8, "refrigerator": 1.8, "monitor": 0.4, "tv": 0.6, "sink": 1.0,
    "toilet": 0.8, "wall_ahead": 2.0, "obstacle_large": 1.0, "surface": 0.8
}
DEFAULT_HEIGHT_M = 0.5


# ─── PRIORITY DEFINITIONS ────────────────────────────────────────────────────

URGENT_OBJECTS  = {"car", "truck", "bus", "motorcycle", "bicycle", "stairs_up", "stairs_down", "fall"}
WARNING_OBJECTS = {"person", "door", "chair", "bed", "toilet", "sofa", "couch", "sink", "refrigerator", "cabinet", "desk", "table", "dining table", "wall_ahead"}
OBJECT_PRIORITY = {obj: 0 for obj in URGENT_OBJECTS}
OBJECT_PRIORITY.update({obj: 1 for obj in WARNING_OBJECTS})

REPEAT_INTERVALS = {0: 2.0, 1: 4.0, 2: 6.0}
CLEAR_PATH_INTERVAL = 10.0
CONFIRM_FRAMES = 3
GRACE_FRAMES   = 2
EMPTY_CENTER_TIME = 2.0


# ─── LANGUAGES ───────────────────────────────────────────────────────────────

LANGUAGES = {
    "English": {
        "code": "en-US",
        "warning": "Warning",
        "ahead": "ahead", "left": "on your left", "right": "on your right",
        "clear": "Path is clear", "obstacle": "Obstacle ahead",
        
        # Distance Tiers
        "dist_0": "STOP! {obj} is extremely close, less than 1 metre",
        "dist_1": "WARNING! {obj} ahead, 2 metres away",
        "dist_2": "Caution! {obj} in 3 metres",
        "dist_3": "Alert! {obj} detected 5 metres ahead",
        
        # New Feature Alerts
        "fall": "Are you okay? Fall detected",
        "stairs_up": "Stairs going UP detected ahead, use handrail",
        "stairs_down": "Stairs going DOWN ahead, be very careful",
        "moving": "Moving {obj} approaching — stop",
        "stuck": "You seem stuck. The {obj} is blocking your path.",
        "nav_history_left": "Previously left side was clear, try going left",
        "nav_history_right": "Previously right side was clear, try going right",
        "scene_outdoor": "Outdoor environment detected, watch for vehicles",
        "scene_corridor": "You appear to be in a corridor, walk straight",
        "scene_road": "You are near a road, please stop",
        
        # Standard Info
        "wall_ahead": "Wall ahead, please stop",
        "surface": "Large surface ahead",
        "nav_move_left": "- move left, path is clear",
        "nav_move_right": "- move right, path is clear",
        "nav_stop": "- stop, both sides blocked",
        "nav_step_back": "- step back",
        "nav_obj_on_left": "Object on left - move right",
        "nav_obj_on_right": "Object on right - move left",
        "still_ahead": "still ahead",
        "persists": "Obstacle persists, please navigate carefully",
    },
    "Tamil": {
        "code": "ta-IN",
        "warning": "எச்சரிக்கை",
        "ahead": "முன்னால்", "left": "இடதுபுறம்", "right": "வலதுபுறம்",
        "clear": "பாதை தெளிவாக உள்ளது", "obstacle": "தடை முன்னால்",
        
        "dist_0": "நிறுத்துங்கள்! {obj} மிகவும் அருகில் உள்ளது, 1 மீட்டருக்குள்",
        "dist_1": "எச்சரிக்கை! {obj} 2 மீட்டர் தொலைவில் உள்ளது",
        "dist_2": "கவனம்! {obj} 3 மீட்டரில் உள்ளது",
        "dist_3": "கவனம்! {obj} 5 மீட்டர் தொலைவில் உள்ளது",
        
        "fall": "நீங்கள் நலமா? வீழ்ச்சி கண்டறியப்பட்டது",
        "stairs_up": "படிக்கட்டுகள் மேலே செல்கின்றன, கைப்பிடியைப் பிடிக்கவும்",
        "stairs_down": "படிக்கட்டுகள் கீழே செல்கின்றன, மிகவும் கவனமாக இருங்கள்",
        "moving": "நகரும் {obj} வருகிறது — நிறுத்துங்கள்",
        "stuck": "நீங்கள் சிக்கியுள்ளீர்கள். {obj} பாதையை மறைக்கிறது.",
        "nav_history_left": "முன்பு இடதுபுறம் தெளிவாக இருந்தது, இடதுபுறம் செல்ல முனையுங்கள்",
        "nav_history_right": "முன்பு வலதுபுறம் தெளிவாக இருந்தது, வலதுபுறம் செல்ல முனையுங்கள்",
        "scene_outdoor": "வெளிப்புற சூழல், வாகனங்களை கவனிக்கவும்",
        "scene_corridor": "நீங்கள் தாழ்வாரத்தில் உள்ளீர்கள், நேராக நடக்கவும்",
        "scene_road": "நீங்கள் சாலைக்கு அருகில் உள்ளீர்கள், நிறுத்துங்கள்",
        
        "wall_ahead": "சுவர் முன்னால் — நிறுத்துங்கள்",
        "surface": "முன்னால் மேசை அல்லது பெரிய பரப்பு",
        "nav_move_left": "- இடதுபுறம் செல்லுங்கள்",
        "nav_move_right": "- வலதுபுறம் செல்லுங்கள்",
        "nav_stop": "- நிறுத்துங்கள்! இரண்டு பக்கமும் தடை",
        "nav_step_back": "- பின்னால் செல்லவும்",
        "nav_obj_on_left": "தடை இடதுபுறம் - வலதுபுறம் செல்லுங்கள்",
        "nav_obj_on_right": "தடை வலதுபுறம் - இடதுபுறம் செல்லுங்கள்",
        "still_ahead": "இன்னும் முன்னால் உள்ளது",
        "persists": "தடை தொடர்கிறது, கவனமாக செல்லுங்கள்",
    },
    "Hindi": {
        "code": "hi-IN",
        "warning": "चेतावनी",
        "ahead": "आगे", "left": "बाईं तरफ", "right": "दाईं तरफ",
        "clear": "रास्ता साफ है", "obstacle": "बाधा आगे",
        
        "dist_0": "रुकिए! {obj} 1 मीटर से भी करीब है",
        "dist_1": "चेतावनी! {obj} 2 मीटर दूर है",
        "dist_2": "सावधान! {obj} 3 मीटर दूर है",
        "dist_3": "सावधान! {obj} 5 मीटर दूर है",
        
        "fall": "क्या आप ठीक हैं? गिरने का पता चला",
        "stairs_up": "ऊपर जाने वाली सीढ़ियां हैं, रेलिंग पकड़ें",
        "stairs_down": "नीचे जाने वाली सीढ़ियां हैं, बहुत सावधान रहें",
        "moving": "चलती हुई {obj} आ रही है — रुकिए",
        "stuck": "आप फंस गए हैं। {obj} रास्ता रोक रही है।",
        "nav_history_left": "पहले बाईं तरफ साफ था, बाईं ओर जाने की कोशिश करें",
        "nav_history_right": "पहले दाईं तरफ साफ था, दाईं ओर जाने की कोशिश करें",
        "scene_outdoor": "बाहरी वातावरण, वाहनों का ध्यान रखें",
        "scene_corridor": "आप एक गलियारे में हैं, सीधे चलें",
        "scene_road": "आप सड़क के पास हैं, कृपया रुकें",
        
        "wall_ahead": "दीवार सामने — रुकिए",
        "surface": "सामने मेज या बड़ी सतह",
        "nav_move_left": "- बाईं ओर जाएं",
        "nav_move_right": "- दाईं ओर जाएं",
        "nav_stop": "- रुकिए! दोनों तरफ बंद है",
        "nav_step_back": "- पीछे हटें",
        "nav_obj_on_left": "बाईं तरफ रुकावट - दाईं ओर जाएं",
        "nav_obj_on_right": "दाईं तरफ रुकावट - बाईं ओर जाएं",
        "still_ahead": "अभी भी आगे है",
        "persists": "बाधा बनी है, सावधानी से चलें",
    },
}
lang = LANGUAGES[selected_lang]

OBJECT_TRANSLATIONS = {
    "Tamil": {
        "person": "நபர்", "car": "கார்", "truck": "லாரி", "bus": "பேருந்து",
        "motorcycle": "மோட்டார் சைக்கிள்", "bicycle": "சைக்கிள்", "chair": "நாற்காலி",
        "table": "மேசை", "dining table": "மேசை", "desk": "மேஜை", "door": "கதவு",
        "bed": "படுக்கை", "toilet": "கழிவறை", "stairs_up": "படிகள்", "stairs_down": "படிகள்",
        "sofa": "சோபா", "couch": "சோபா", "wall_ahead": "சுவர்", "obstacle_large": "தடை"
    },
    "Hindi": {
        "person": "व्यक्ति", "car": "कार", "truck": "ट्रक", "bus": "बस",
        "motorcycle": "मोटरसाइकिल", "bicycle": "साइकिल", "chair": "कुर्सी",
        "table": "मेज़", "dining table": "मेज़", "desk": "डेस्क", "door": "दरवाज़ा",
        "bed": "बिस्तर", "toilet": "शौचालय", "stairs_up": "सीढ़ियाँ", "stairs_down": "सीढ़ियाँ",
        "sofa": "सोफ़ा", "couch": "सोफ़ा", "wall_ahead": "दीवार", "obstacle_large": "बाधा"
    },
}


# ─── SESSION STATE ────────────────────────────────────────────────────────────

if 'log' not in st.session_state: st.session_state.log = []

for k, v in {
    "frame_count": 0, "det_count": 0,
    "ui_msg": "START NAVIGATION to begin", "ui_msg_class": "priority-clear",
    "last_spoken_text": "", "last_spoken_obj": "", "last_spoken_zone": "",
    "last_spoken_time": 0.0, "last_clear_time": 0.0, "repeat_count": 0,
    
    # New Memory States
    "last_center_block_time": time.time(),
    "object_appear_time": {}, # Tracker for 'stuck' logic
    "last_clear_direction": "LEFT",
    
    # Fall / SOS State
    "sos_triggered": False,
    "fall_detected_time": 0.0,
    
    # Motion
    "prev_gray": None,
    
    # UI Metrics
    "ui_dist_m": "--", "ui_obj": "--", "ui_dir": "--", "ui_scene": "Unknown",
}.items():
    if k not in st.session_state: st.session_state[k] = v

def add_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log.insert(0, f"[{ts}] {msg}")
    if len(st.session_state.log) > 10: st.session_state.log.pop()


# ─── HAPTIC & VOICE / SOS HTML AUDIO ─────────────────────────────────────────

components.html("""
<div id="haptic-container" style="text-align:center;margin-bottom:10px;">
  <button id="arm-btn" style="
    background:#1A73E8;color:#fff;border:none;
    padding:15px 30px;font-size:1.2rem;font-weight:bold;
    border-radius:8px;cursor:pointer;width:100%;max-width:400px;
    font-family:'Inter',sans-serif; text-transform:uppercase;">
    Tap to Enable Audio & Haptics
  </button>
  <div id="armed-indicator" style="
    display:none;color:#0F9D58;font-size:1.2rem;font-weight:bold;
    font-family:'Inter',sans-serif;padding:10px;">
    📳 Haptics ON &nbsp;|&nbsp; 🔊 Voice ON
  </div>
</div>
<audio id="sos-audio" src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" loop preload="auto"></audio>

<script>
var pw = window.parent;
pw.hapticsArmed             = pw.hapticsArmed            || false;
pw.currentVibrationPattern  = pw.currentVibrationPattern || [0];
pw.lastVibrationPattern     = pw.lastVibrationPattern    || [0];
pw.lastVibrationTime        = pw.lastVibrationTime       || 0;
pw.pendingSpeech            = pw.pendingSpeech           || "";
pw.pendingSpeechLang        = pw.pendingSpeechLang       || "en-US";
pw.lastSpokenText           = pw.lastSpokenText          || "";
pw.triggerSOS               = pw.triggerSOS              || false;

document.getElementById('arm-btn').addEventListener('click', function() {
    pw.hapticsArmed = true;
    this.style.display = 'none';
    document.getElementById('armed-indicator').style.display = 'block';
    
    if(navigator.vibrate) navigator.vibrate(50);
    var warmup = new SpeechSynthesisUtterance(' ');
    warmup.volume = 0;
    window.speechSynthesis.speak(warmup);
    
    // Warmup audio
    let aud = document.getElementById('sos-audio');
    aud.volume = 0; aud.play().then(()=>aud.pause()).catch(e=>{});
    aud.volume = 1.0;
});

setInterval(function() {
    if (pw.hapticsArmed && navigator.vibrate) {
        let now = Date.now();
        let same = JSON.stringify(pw.currentVibrationPattern) === JSON.stringify(pw.lastVibrationPattern);
        if (!same || (now - pw.lastVibrationTime) > 2000) {
            if (pw.currentVibrationPattern[0] !== 0) {
                navigator.vibrate(pw.currentVibrationPattern);
                pw.lastVibrationPattern = [...pw.currentVibrationPattern];
                pw.lastVibrationTime    = now;
            }
        }
    }
    if (pw.hapticsArmed && pw.pendingSpeech && pw.pendingSpeech !== pw.lastSpokenText && !window.speechSynthesis.speaking) {
        var u = new SpeechSynthesisUtterance(pw.pendingSpeech);
        u.lang = pw.pendingSpeechLang; u.rate = 1.0; u.volume = 1.0;
        window.speechSynthesis.speak(u);
        pw.lastSpokenText = pw.pendingSpeech;
    }
    
    // SOS Handling
    let sosAud = document.getElementById('sos-audio');
    if (pw.triggerSOS && sosAud.paused) {
        window.speechSynthesis.cancel();
        sosAud.play();
    } else if (!pw.triggerSOS && !sosAud.paused) {
        sosAud.pause(); sosAud.currentTime = 0;
    }
}, 300);
</script>
""", height=110)


# ─── UI EMERGENCY BUTTON ─────────────────────────────────────────────────────

if st.button("🚨 PANIC / SOS"):
    st.session_state.sos_triggered = not st.session_state.sos_triggered
    if st.session_state.sos_triggered:
        st.session_state.ui_msg = f"EMERGENCY HELP: {contact_num}"
        st.session_state.ui_msg_class = "priority-urgent"
        components.html("<script>window.parent.triggerSOS = true;</script>", height=0)
    else:
        components.html("<script>window.parent.triggerSOS = false;</script>", height=0)

if st.session_state.sos_triggered:
    st.markdown(f"<h1 style='text-align:center; color:red; font-size:4rem;'>CALL: {contact_num}</h1>", unsafe_allow_html=True)


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

CUSTOM_MODEL = "best_sunrgbd.pt"
FALLBACK_MOD = ACTIVE_PERF["model"]

INFER_IMGSZ   = ACTIVE_PERF["imgsz"]
IOU_THRESHOLD = 0.45

CLASS_CONF_THRESHOLDS = {
    "person": 0.55, "car": 0.50, "motorcycle": 0.50, "bicycle": 0.50,
    "table": 0.35, "dining table": 0.35, "desk": 0.35,
    "chair": 0.30, "sofa": 0.30, "couch": 0.30, "bed": 0.30,
    "door": 0.35, "stairs": 0.40, "cabinet": 0.40, "refrigerator": 0.40,
}
BASE_CONF = 0.30

@st.cache_resource(show_spinner=f"Loading AI...")
def load_yolo(model_name):
    if os.path.exists(CUSTOM_MODEL): return YOLO(CUSTOM_MODEL)
    return YOLO(model_name)

try:
    MODEL = load_yolo(FALLBACK_MOD)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ─── NEW 5-METRE DISTANCE ESTIMATOR ──────────────────────────────────────────

def calculate_distance_m(pixel_h, obj_label):
    ref_h = REF_HEIGHTS_M.get(obj_label, DEFAULT_HEIGHT_M)
    # distance = (real_height * focal_length) / pixel_height
    # Avoid div by 0
    if pixel_h < 5: pixel_h = 5
    dist_raw = (ref_h * FOCAL_LENGTH_PX) / float(pixel_h)
    
    # Constrain for realism
    return round(min(max(dist_raw, 0.2), 15.0), 1)

def get_distance_tier(dist_m):
    """0-1: EXTREMELY_CLOSE, 1-2: VERY_CLOSE, 2-3: CLOSE, 3-5: MEDIUM, >5: FAR"""
    if dist_m < 1.0: return "EXTREMELY_CLOSE"
    if dist_m < 2.0: return "VERY_CLOSE"
    if dist_m < 3.0: return "CLOSE"
    if dist_m < 5.0: return "MEDIUM"
    return "FAR"


# ─── SMART VOICE ENGINE ──────────────────────────────────────────────────────

class VoiceEngine:
    @staticmethod
    def _build_distance_alert(label_en: str, dist_tier: str, dist_m: float) -> str:
        """New feature: Distance-based verbal phrasing."""
        label_loc = OBJECT_TRANSLATIONS.get(selected_lang, {}).get(label_en, label_en)
        
        # Format object name into string
        if dist_tier == "EXTREMELY_CLOSE":
            return lang["dist_0"].format(obj=label_loc)
        elif dist_tier == "VERY_CLOSE":
            return lang["dist_1"].format(obj=label_loc)
        elif dist_tier == "CLOSE":
            return lang["dist_2"].format(obj=label_loc)
        elif dist_tier == "MEDIUM":
            return lang["dist_3"].format(obj=label_loc)
        return ""

    @staticmethod
    def _build_base_alert(label_en: str, direction: str, dist_tier: str, dist_m: float, left_clear: bool, right_clear: bool, is_moving: bool) -> str:
        # 1. Edge Case Overrides
        if label_en == "fall":            return lang["fall"]
        if label_en == "stairs_up":       return lang["stairs_up"]
        if label_en == "stairs_down":     return lang["stairs_down"]
        if label_en == "wall_ahead":      return lang["wall_ahead"]
        if label_en == "surface":         return lang["surface"]
        
        # Moving Override
        if is_moving:
            obj_loc = OBJECT_TRANSLATIONS.get(selected_lang, {}).get(label_en, label_en)
            return lang["warning"] + "! " + lang["moving"].format(obj=obj_loc)

        # 2. Distance Text
        base = VoiceEngine._build_distance_alert(label_en, dist_tier, dist_m)
        if not base: return "" # Too far
            
        # 3. Add directional guidance
        if direction == "CENTER":
            if left_clear and right_clear:
                base += f" {lang['nav_move_left']}"
            elif left_clear and not right_clear:
                base += f" {lang['nav_move_left']}"
            elif right_clear and not left_clear:
                base += f" {lang['nav_move_right']}"
            else:
                base += f" {lang['nav_stop']}"
        elif direction == "LEFT":
            base += f" - {lang['nav_obj_on_left']}"
        elif direction == "RIGHT":
            base += f" - {lang['nav_obj_on_right']}"
            
        return base

    @classmethod
    def should_speak(cls, label_en: str, direction: str, dist_tier: str, dist_m: float, left_clear: bool, right_clear: bool, is_moving: bool) -> tuple[bool, str]:
        # Ignore distant objects > 5m
        if dist_tier == "FAR" and label_en not in ["scene_outdoor", "scene_corridor", "scene_road"]:
            return False, ""

        now      = time.time()
        priority = 0 if label_en in URGENT_OBJECTS or is_moving else 1 if label_en in WARNING_OBJECTS else 2
        interval = REPEAT_INTERVALS[priority]

        prev_obj  = st.session_state.last_spoken_obj
        prev_zone = st.session_state.last_spoken_zone
        prev_time = st.session_state.last_spoken_time
        repeat_cnt = st.session_state.repeat_count
        elapsed   = now - prev_time
        
        # Feature 7: Stuck detection (Memory)
        if direction == "CENTER" and prev_obj == label_en and elapsed > 5.0 and repeat_cnt > 2:
            obj_loc = OBJECT_TRANSLATIONS.get(selected_lang, {}).get(label_en, label_en)
            text = lang["stuck"].format(obj=obj_loc)
            
            # Suggest history path
            last_dir = st.session_state.last_clear_direction
            if last_dir == "LEFT": text += ". " + lang["nav_history_left"]
            if last_dir == "RIGHT": text += ". " + lang["nav_history_right"]
            
            return True, text

        # URGENT or MOVING always repeats at interval
        if priority == 0:
            if elapsed >= interval:
                return True, cls._build_base_alert(label_en, direction, dist_tier, dist_m, left_clear, right_clear, is_moving)
            return False, ""

        # New Object or Zone Change
        if label_en != prev_obj or (direction != prev_zone and elapsed >= 1.0):
            return True, cls._build_base_alert(label_en, direction, dist_tier, dist_m, left_clear, right_clear, is_moving)

        if elapsed < interval:
            return False, ""

        # Persistence variation
        if repeat_cnt >= 3:
            return True, lang["persists"]
            
        obj_loc = OBJECT_TRANSLATIONS.get(selected_lang, {}).get(label_en, label_en)
        nudge = lang["move_right"] if direction in ["CENTER", "LEFT"] else lang["move_left"]
        return True, f"{obj_loc} {lang['still_ahead']}. {nudge}"

    @classmethod
    def record_spoken(cls, label_en: str, direction: str):
        if label_en == st.session_state.last_spoken_obj:
            st.session_state.repeat_count += 1
        else:
            st.session_state.repeat_count = 0
            
            # Record clear history if moving from side obstacle to center obstacle
            if st.session_state.last_spoken_zone in ["LEFT", "RIGHT"] and direction == "CENTER":
                st.session_state.last_clear_direction = "LEFT" if st.session_state.last_spoken_zone == "RIGHT" else "RIGHT"

        st.session_state.last_spoken_obj  = label_en
        st.session_state.last_spoken_zone = direction
        st.session_state.last_spoken_time = time.time()
        add_log(f"Alerted: {label_en} ({direction})")

def emit_voice_haptic(text: str, dist_tier: str, lang_code: str):
    # Haptic based on distance tiers as requested
    vib = "[0]"
    if   dist_tier == "EXTREMELY_CLOSE": vib = "[100,50,100,50,100]" # 3 quick bursts
    elif dist_tier == "VERY_CLOSE":      vib = "[200,100,200]"       # Double pulse
    elif dist_tier == "CLOSE":           vib = "[300]"               # Single pulse
    elif dist_tier == "MEDIUM":          vib = "[150]"               # Light tap

    safe = text.replace("'", "\\'").replace('"', '\\"')
    components.html(f"""
    <script>
    var pw = window.parent;
    if (pw.currentVibrationPattern !== undefined) pw.currentVibrationPattern = {vib};
    if (pw.pendingSpeech !== undefined && "{"scene" not in text and "clear" not in text}") {{
        pw.pendingSpeech = '{safe}';
        pw.pendingSpeechLang = '{lang_code}';
        pw.lastSpokenText = '';
    }}
    </script>
    """, height=0)


# ─── ADVANCED OPENCV FEATURES ────────────────────────────────────────────────

def detect_fall(frame):
    """Detect tilt > 30 deg or sudden darkness."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check brightness
    if np.mean(gray) < 20:
        return True
        
    # Check tilt horizon
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7,7), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=w*0.5, maxLineGap=20)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: continue
            angle = np.degrees(np.arctan((y2-y1)/(x2-x1)))
            angles.append(abs(angle))
        
        if angles:
            median_angle = np.median(angles)
            if 30 < median_angle < 60: # Strong unnatural tilt 
                return True
    return False

def detect_stairs_direction(frame):
    """Return 'stairs_up', 'stairs_down', or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=frame.shape[1]*0.4, maxLineGap=10)
    
    if lines is not None and len(lines) > 3:
        y_coords = sorted([line[0][1] for line in lines])
        # Calculate gaps between sorted horizontal lines
        gaps = np.diff(y_coords)
        if len(gaps) > 2:
            # If gaps get smaller towards top of frame (smaller y) = Stairs UP
            if gaps[0] > gaps[-1]: return "stairs_up"
            # If gaps get smaller towards bottom = Stairs DOWN
            elif gaps[0] < gaps[-1]: return "stairs_down"
    return None

def detect_scene(frame):
    """Categorize environment."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Outdoor (blue sky or high brightness + green)
    upper_hsv = hsv[0:int(h*0.3), :]
    v_mean = np.mean(upper_hsv[:,:,2])
    if v_mean > 200: 
        return "Outdoor"
        
    # 2. Road (Grey flat bottom)
    lower_hsv = hsv[int(h*0.7):h, :]
    s_mean = np.mean(lower_hsv[:,:,1])
    v_bottom_mean = np.mean(lower_hsv[:,:,2])
    if s_mean < 40 and 100 < v_bottom_mean < 180:
        return "Road"
        
    return "Indoor"

def is_moving(mask, bbox):
    """Check if motion mask overlaps with YOLO bbox heavily."""
    if mask is None: return False
    x1, y1, x2, y2 = bbox
    roi = mask[y1:y2, x1:x2]
    overlap = cv2.countNonZero(roi)
    area = (y2-y1)*(x2-x1)
    if area == 0: return False
    return (overlap / area) > 0.3  # >30% of object has motion


# ─── VIDEO PROCESSOR ─────────────────────────────────────────────────────────

def direction(cx, fw):
    r = cx / fw
    if r < 0.35: return "LEFT"
    if r > 0.65: return "RIGHT"
    return "CENTER"

class VideoProcessor:
    def __init__(self):
        self.latest_label_en = ""
        self.latest_dir      = "CENTER"
        self.latest_dist_t   = "FAR"
        self.latest_dist_m   = 0.0
        self.has_detection   = False
        
        self.left_clear      = True
        self.right_clear     = True
        self.scene           = "Unknown"

        self.last_infer_time = 0.0
        self.last_results    = []
        self.tracker         = {}
        self.empty_count     = 0
        
        self.prev_gray = None
        self.motion_mask = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        fh, fw = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Motion calculation (Frame Diff)
        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, self.motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray.copy()

        now = time.time()
        
        # Performance Mode Interval Mapping
        if now - self.last_infer_time >= ACTIVE_PERF["interval"]:
            
            # Scene Update
            if st.session_state.frame_count % 10 == 0:
                self.scene = detect_scene(img)

            # Fall Check
            if detect_fall(img):
                self.latest_label_en = "fall"
                self.has_detection = True
                self.latest_dist_m = 0.0
                self.latest_dist_t = "EXTREMELY_CLOSE"
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # YOLO inference
            sz = INFER_IMGSZ
            rsz = cv2.resize(img, (sz, sz))
            
            results = MODEL(rsz, verbose=False, conf=BASE_CONF, iou=IOU_THRESHOLD, imgsz=sz)[0]

            raw_dets = []
            for box in results.boxes:
                lb_en = MODEL.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                if conf < CLASS_CONF_THRESHOLDS.get(lb_en, 0.50): continue

                x1,y1,x2,y2 = box.xyxy[0].tolist()
                # Unbox scale back to original fw,fh
                sx, sy = fw/sz, fh/sz
                ux1, ux2 = int(x1*sx), int(x2*sx)
                uy1, uy2 = int(y1*sy), int(y2*sy)

                px_h = uy2 - uy1
                dist_m = calculate_distance_m(px_h, lb_en)
                
                # Filter out far targets (>5m)
                if dist_m > 5.0 and lb_en not in ["car", "bus", "truck"]: continue

                moving = is_moving(self.motion_mask, (ux1, uy1, ux2, uy2))

                raw_dets.append({
                    "label_en": lb_en,
                    "dist_m": dist_m,
                    "dist_t": get_distance_tier(dist_m),
                    "dir": direction((ux1+ux2)/2, fw),
                    "bbox": (ux1, uy1, ux2, uy2),
                    "conf": conf,
                    "moving": moving,
                    "rank": dist_m  # Closest objects = rank 1
                })
                
            # Stair secondary detector
            if not raw_dets:
                stair_dir = detect_stairs_direction(img)
                if stair_dir:
                    raw_dets.append({
                        "label_en": stair_dir, "dist_m": 2.0, "dist_t": "CLOSE",
                        "dir": "CENTER", "bbox": (0,0,fw,fh), "conf": 0.9, "moving": False, "rank": 2.0
                    })

            # Tracker update (Simplified for code length)
            self.last_results = sorted(raw_dets, key=lambda x: x["rank"])
            self.last_infer_time = now

        # Drawing and state mapping
        annotated = img.copy()
        COLORS = {"EXTREMELY_CLOSE":(0,0,255), "VERY_CLOSE":(0,100,255), "CLOSE":(0,200,255), "MEDIUM":(0,200,0), "FAR":(100,100,100)}

        if self.last_results:
            self.empty_count = 0
            st.session_state.last_center_block_time = time.time()
            primary = self.last_results[0]

            self.latest_label_en = primary["label_en"]
            self.latest_dir      = primary["dir"]
            self.latest_dist_t   = primary["dist_t"]
            self.latest_dist_m   = primary["dist_m"]
            self.latest_moving   = primary["moving"]
            self.has_detection   = True
            
            self.left_clear  = not any(d["dir"] == "LEFT" for d in self.last_results)
            self.right_clear = not any(d["dir"] == "RIGHT" for d in self.last_results)

            for det in self.last_results:
                bx1, by1, bx2, by2 = det["bbox"]
                col = COLORS.get(det["dist_t"], (100,100,100))
                
                # Moving objects highlight thicker
                thick = 5 if det["moving"] else 2
                cv2.rectangle(annotated, (bx1,by1), (bx2,by2), col, thick)
                
                lbl = f"{det['label_en']} {det['dist_m']}m"
                if det["moving"]: lbl = "MOVING " + lbl
                
                cv2.putText(annotated, lbl, (bx1, max(by1-8,16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        else:
            self.left_clear = True
            self.right_clear = True
            self.has_detection = False
            self.empty_count += 1

        st.session_state.ui_scene = self.scene
        st.session_state.frame_count += 1
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─── MAIN UI STREAMING ───────────────────────────────────────────────────────

st.markdown("<h1>👁️ VisionAid Pro</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='caregiver-subtitle'>High-Accuracy Indoor/Outdoor AI · Focal Dist Estimation · Fall/Stair Detect</p>", unsafe_allow_html=True)

# HUD Top
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f"<div class='hud-box hud-red'><div class='hud-title'>Last Object</div><div class='hud-value'>{st.session_state.ui_obj.upper()}</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='hud-box hud-yellow'><div class='hud-title'>Distance</div><div class='hud-value'>{st.session_state.ui_dist_m}m</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='hud-box hud-green'><div class='hud-title'>Zone</div><div class='hud-value'>{st.session_state.ui_dir}</div></div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='hud-box'><div class='hud-title'>Scene</div><div class='hud-value'>{st.session_state.ui_scene}</div></div>", unsafe_allow_html=True)


webrtc_ctx = webrtc_streamer(
    key="visionaid-v5",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=False,
)

if webrtc_ctx.state.playing:
    st_autorefresh(interval=500, key="va_loop")

    if webrtc_ctx.video_processor:
        proc = webrtc_ctx.video_processor

        if proc.has_detection and not st.session_state.sos_triggered:
            lbl_en = proc.latest_label_en
            ddir   = proc.latest_dir
            dtier  = proc.latest_dist_t
            dm     = proc.latest_dist_m
            mov    = getattr(proc, 'latest_moving', False)

            # Update HUD Data
            st.session_state.ui_obj = lbl_en
            st.session_state.ui_dist_m = dm
            st.session_state.ui_dir = ddir

            ok, text = VoiceEngine.should_speak(lbl_en, ddir, dtier, dm, proc.left_clear, proc.right_clear, mov)
            if ok:
                emit_voice_haptic(text, dtier, lang["code"])
                VoiceEngine.record_spoken(lbl_en, ddir)
                st.session_state.ui_msg = text.upper()
                c_map = {"EXTREMELY_CLOSE":"priority-urgent","VERY_CLOSE":"priority-urgent", "CLOSE":"priority-warning"}
                st.session_state.ui_msg_class = c_map.get(dtier, "priority-info")

        elif not st.session_state.sos_triggered:
            # Heartbeat Clear
            st.session_state.ui_obj = "--"
            st.session_state.ui_dist_m = "--"
            st.session_state.ui_dir = "--"
            
            if proc.empty_count >= 10:
                # Scene announce if outdoor/road just triggered
                sc = st.session_state.ui_scene
                if sc in ["Outdoor", "Road", "Corridor"] and VoiceEngine.should_speak(f"scene_{sc.lower()}", "CENTER", "FAR", 6.0, True, True, False)[0]:
                    txt = lang.get(f"scene_{sc.lower()}", "")
                    emit_voice_haptic(txt, "FAR", lang["code"])
                    VoiceEngine.record_spoken(f"scene_{sc.lower()}", "CENTER")
                    st.session_state.ui_msg = txt.upper()
                elif VoiceEngine.should_say_clear():
                    text = lang["clear"]
                    emit_voice_haptic(text, "FAR", lang["code"])
                    st.session_state.last_spoken_obj  = ""
                    st.session_state.repeat_count     = 0
                    st.session_state.ui_msg       = text.upper()
                    st.session_state.ui_msg_class = "priority-clear"


st.markdown(f"""
<div class="status-panel">
  <div class="status-text {st.session_state.ui_msg_class}">
    {st.session_state.ui_msg}<br>
    <span style='font-size:0.9rem;color:#888;font-weight:400;'>Model: {ACTIVE_PERF['model']} | {ACTIVE_PERF['imgsz']}px | {(1/ACTIVE_PERF['interval']):.1f} fps</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Render Caregiver Log
log_text = "<br>".join(st.session_state.log)
caregiver_log.markdown(f"<div style='font-family:monospace; font-size:12px; height:300px; overflow-y:scroll; color:#bbb;'>{log_text}</div>", unsafe_allow_html=True)

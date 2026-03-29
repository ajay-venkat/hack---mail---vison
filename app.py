import streamlit as st
import cv2
import numpy as np
import time
import os
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import av

# --- HIGH CONTRAST ACCESSIBILITY UI CONFIG ---
st.set_page_config(page_title="VisionAid", page_icon="👁️", layout="centered")

st.markdown("""
    <style>
    /* High Contrast Accessibility Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    .main { background-color: #000000 !important; color: #FFFFFF !important; font-family: 'Inter', sans-serif; }
    /* Hide top menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { max-width: 100%; padding: 0; background-color: #000000 !important; }
    
    p, li, span, div { font-size: 1.1rem !important; color: #FFFFFF !important; }
    
    .stExpander {
        background: #0D1117 !important;
        border: 2px solid #1A73E8 !important;
        border-radius: 10px;
        margin: 10px;
    }
    
    .stExpander summary { color: #FFFFFF !important; font-size: 1.2rem !important; }
    
    .element-container img { border-radius: 8px; width: 100% !important; border: 2px solid #1A73E8; }
    
    /* Status Panel underneath camera */
    .status-panel {
        background: #0D1117;
        padding: 20px;
        border: 3px solid #1A73E8;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
    }
    
    .status-text {
        font-size: 20px !important;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .model-badge {
        display: inline-block;
        background: #1A73E8;
        color: #fff !important;
        font-size: 0.75rem !important;
        padding: 2px 10px;
        border-radius: 20px;
        margin-top: 8px;
    }
    
    .status-clear { color: #888888 !important; }
    
    /* Distance Colors for Text */
    .dist-very-close { color: #DB4437 !important; }
    .dist-close { color: #F4B400 !important; }
    .dist-medium { color: #0F9D58 !important; }
    .dist-far { color: #888888 !important; }
    
    /* Caregiver text */
    .caregiver-subtitle {
        color: #888888 !important;
        font-size: 1rem !important;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── LANGUAGE CONFIG ─────────────────────────────────────────────────────────

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
        "clear": "Path appears clear",
        "dark": "Environment too dark",
        "started": "VisionAid navigation started",
        "obstacle": "Obstacle ahead",
        # Indoor-specific alerts
        "chair_ahead":   "Chair ahead",
        "door_left":     "Door on your left",
        "door_right":    "Door on your right",
        "door_ahead":    "Door ahead",
        "table_right":   "Table on your right",
        "table_left":    "Table on your left",
        "table_ahead":   "Table ahead",
        "bed_detected":  "Bed detected",
        "stairs_ahead":  "Stairs ahead, be careful",
        "sofa_ahead":    "Sofa ahead",
        "toilet_ahead":  "Toilet ahead",
        "sink_ahead":    "Sink ahead",
        "cabinet_ahead": "Cabinet ahead",
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
        # Indoor-specific alerts
        "chair_ahead":   "நாற்காலி முன்னால் உள்ளது",
        "door_left":     "கதவு இடதுபுறம் உள்ளது",
        "door_right":    "கதவு வலதுபுறம் உள்ளது",
        "door_ahead":    "கதவு முன்னால் உள்ளது",
        "table_right":   "மேசை வலதுபுறம் உள்ளது",
        "table_left":    "மேசை இடதுபுறம் உள்ளது",
        "table_ahead":   "மேசை முன்னால் உள்ளது",
        "bed_detected":  "படுக்கை கண்டறியப்பட்டது",
        "stairs_ahead":  "படிகள் முன்னால், கவனமாக இருங்கள்",
        "sofa_ahead":    "சோபா முன்னால் உள்ளது",
        "toilet_ahead":  "கழிவறை முன்னால் உள்ளது",
        "sink_ahead":    "கழுவுதொட்டி முன்னால் உள்ளது",
        "cabinet_ahead": "அலமாரி முன்னால் உள்ளது",
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
        # Indoor-specific alerts
        "chair_ahead":   "आगे कुर्सी है",
        "door_left":     "बाईं तरफ दरवाज़ा है",
        "door_right":    "दाईं तरफ दरवाज़ा है",
        "door_ahead":    "आगे दरवाज़ा है",
        "table_right":   "दाईं तरफ मेज़ है",
        "table_left":    "बाईं तरफ मेज़ है",
        "table_ahead":   "आगे मेज़ है",
        "bed_detected":  "बिस्तर मिला",
        "stairs_ahead":  "आगे सीढ़ियाँ हैं, सावधान रहें",
        "sofa_ahead":    "आगे सोफ़ा है",
        "toilet_ahead":  "आगे शौचालय है",
        "sink_ahead":    "आगे नल है",
        "cabinet_ahead": "आगे अलमारी है",
    }
}

OBJECT_TRANSLATIONS = {
    "Tamil": {
        "person": "நபர்", "car": "கார்", "truck": "லாரி",
        "bus": "பேருந்து", "motorcycle": "மோட்டார் சைக்கிள்",
        "bicycle": "சைக்கிள்", "chair": "நாற்காலி",
        "table": "மேசை", "dining table": "மேசை",
        "bottle": "பாட்டில்", "dog": "நாய்", "cat": "பூனை",
        "door": "கதவு", "bed": "படுக்கை", "toilet": "கழிவறை",
        "tv": "தொலைக்காட்சி", "monitor": "மானிட்டர்",
        "laptop": "மடிக்கணினி", "cell phone": "கைப்பேசி",
        "book": "புத்தகம்", "clock": "கடிகாரம்", "cup": "கோப்பை",
        "traffic light": "போக்குவரத்து விளக்கு",
        "fire hydrant": "தீயணைப்பு குழாய்",
        "stop sign": "நிறுத்த அடையாளம்",
        "bench": "இருக்கை", "backpack": "பை",
        "umbrella": "குடை", "handbag": "கைப்பை",
        "couch": "சோபா", "sofa": "சோபா",
        "potted plant": "தாவரம்", "sink": "கழுவுதொட்டி",
        "refrigerator": "குளிர்சாதனப்பெட்டி",
        "scissors": "கத்தரிக்கோல்", "vase": "பூச்சட்டி",
        "stairs": "படிகள்", "cabinet": "அலமாரி",
        "desk": "மேஜை", "lamp": "விளக்கு",
        "pillow": "தலையணை", "bookshelf": "புத்தக அலமாரி",
        "wall": "சுவர்", "floor": "தரை",
    },
    "Hindi": {
        "person": "व्यक्ति", "car": "कार", "truck": "ट्रक",
        "bus": "बस", "motorcycle": "मोटरसाइकिल",
        "bicycle": "साइकिल", "chair": "कुर्सी",
        "table": "मेज़", "dining table": "मेज़",
        "bottle": "बोतल", "dog": "कुत्ता", "cat": "बिल्ली",
        "door": "दरवाज़ा", "bed": "बिस्तर", "toilet": "शौचालय",
        "tv": "टीवी", "monitor": "मॉनिटर",
        "laptop": "लैपटॉप", "cell phone": "मोबाइल फ़ोन",
        "book": "किताब", "clock": "घड़ी", "cup": "कप",
        "traffic light": "ट्रैफ़िक लाइट",
        "fire hydrant": "अग्निशमन यंत्र",
        "stop sign": "रुकने का संकेत",
        "bench": "बेंच", "backpack": "बैग",
        "umbrella": "छाता", "handbag": "पर्स",
        "couch": "सोफ़ा", "sofa": "सोफ़ा",
        "potted plant": "पौधा", "sink": "नल",
        "refrigerator": "फ्रिज",
        "scissors": "कैंची", "vase": "फूलदान",
        "stairs": "सीढ़ियाँ", "cabinet": "अलमारी",
        "desk": "डेस्क", "lamp": "लैंप",
        "pillow": "तकिया", "bookshelf": "किताबों की अलमारी",
        "wall": "दीवार", "floor": "फर्श",
    }
}

PRIORITY_OBJECTS = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "chair", "stairs", "door", "bed", "toilet"
}

# Indoor objects that get specific voice alerts (mapped to alert key)
INDOOR_ALERT_MAP = {
    "chair":         {"CENTER": "chair_ahead",  "LEFT": "chair_ahead",  "RIGHT": "chair_ahead"},
    "door":          {"CENTER": "door_ahead",   "LEFT": "door_left",    "RIGHT": "door_right"},
    "table":         {"CENTER": "table_ahead",  "LEFT": "table_left",   "RIGHT": "table_right"},
    "dining table":  {"CENTER": "table_ahead",  "LEFT": "table_left",   "RIGHT": "table_right"},
    "desk":          {"CENTER": "table_ahead",  "LEFT": "table_left",   "RIGHT": "table_right"},
    "bed":           {"CENTER": "bed_detected", "LEFT": "bed_detected", "RIGHT": "bed_detected"},
    "stairs":        {"CENTER": "stairs_ahead", "LEFT": "stairs_ahead", "RIGHT": "stairs_ahead"},
    "sofa":          {"CENTER": "sofa_ahead",   "LEFT": "sofa_ahead",   "RIGHT": "sofa_ahead"},
    "couch":         {"CENTER": "sofa_ahead",   "LEFT": "sofa_ahead",   "RIGHT": "sofa_ahead"},
    "toilet":        {"CENTER": "toilet_ahead", "LEFT": "toilet_ahead", "RIGHT": "toilet_ahead"},
    "sink":          {"CENTER": "sink_ahead",   "LEFT": "sink_ahead",   "RIGHT": "sink_ahead"},
    "cabinet":       {"CENTER": "cabinet_ahead","LEFT": "cabinet_ahead","RIGHT": "cabinet_ahead"},
}


# ─── SESSION STATE ───────────────────────────────────────────────────────────

if "last_spoken"    not in st.session_state: st.session_state.last_spoken    = ""
if "last_speak_time" not in st.session_state: st.session_state.last_speak_time = 0
if "last_dark_warn" not in st.session_state: st.session_state.last_dark_warn  = 0
if "frame_count"    not in st.session_state: st.session_state.frame_count     = 0
if "det_count"      not in st.session_state: st.session_state.det_count       = 0
if "ui_msg"         not in st.session_state: st.session_state.ui_msg          = "START NAVIGATION to begin"
if "ui_msg_class"   not in st.session_state: st.session_state.ui_msg_class    = "status-clear"


# ─── HAPTIC + VOICE INJECTION ────────────────────────────────────────────────

components.html(f"""
    <div id="haptic-container" style="text-align: center; margin-bottom: 10px;">
        <button id="arm-btn" style="
            background-color: #DB4437; color: white; border: none; 
            padding: 15px 30px; font-size: 1.2rem; font-weight: bold; 
            border-radius: 8px; cursor: pointer; width: 100%; max-width: 400px;
            font-family: 'Inter', sans-serif;">
            TAP TO ENABLE HAPTICS
        </button>
        <div id="armed-indicator" style="
            display: none; color: #0F9D58; font-size: 1.2rem; 
            font-weight: bold; font-family: 'Inter', sans-serif;
            padding: 10px;">
            📳 Haptics ON
        </div>
    </div>
    <script>
    window.hapticsArmed = false;
    window.currentVibrationPattern = [0];
    window.lastVibrationPattern = [0];
    window.lastVibrationTime = 0;
    document.getElementById('arm-btn').addEventListener('click', function() {{
        window.hapticsArmed = true;
        this.style.display = 'none';
        document.getElementById('armed-indicator').style.display = 'block';
        if(navigator.vibrate) navigator.vibrate(50);
    }});
    setInterval(function() {{
        if(!window.hapticsArmed || !navigator.vibrate) return;
        let now = Date.now();
        if (JSON.stringify(window.currentVibrationPattern) !== JSON.stringify(window.lastVibrationPattern) || (now - window.lastVibrationTime) > 2000) {{
             if (window.currentVibrationPattern.length > 0 && window.currentVibrationPattern[0] !== 0) {{
                 navigator.vibrate(window.currentVibrationPattern);
                 window.lastVibrationPattern = [...window.currentVibrationPattern];
                 window.lastVibrationTime = now;
             }}
        }}
    }}, 300);
    </script>
""", height=100)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

st.sidebar.title("Accessibility Settings")
selected_lang_name = st.sidebar.selectbox("Language / மொழி / भाषा", options=list(LANGUAGES.keys()))
lang_cfg = LANGUAGES[selected_lang_name]

st.sidebar.markdown("### About This View")
st.sidebar.markdown("""
<div style="color: #bbb; font-size: 0.95rem;">
This screen is for caregivers and developers.<br>
Real-time navigation running with high-accuracy indoor model.
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
stats_placeholder = st.sidebar.empty()


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

CUSTOM_MODEL = "best_sunrgbd.pt"
FALLBACK_MODEL = "yolov8s.pt"

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_yolo():
    if os.path.exists(CUSTOM_MODEL):
        model = YOLO(CUSTOM_MODEL)
        model_name = f"SUN RGB-D Fine-tuned ({CUSTOM_MODEL})"
    else:
        model = YOLO(FALLBACK_MODEL)
        model_name = f"YOLOv8s COCO ({FALLBACK_MODEL})"
    return model, model_name

try:
    MODEL, MODEL_NAME = load_yolo()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Inference thresholds
CONF_THRESHOLD = 0.45
IOU_THRESHOLD  = 0.5
INFERENCE_IMGSZ = 640


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def translate_label(label, language):
    if language == "English":
        return label
    return OBJECT_TRANSLATIONS.get(language, {}).get(label, label)


def build_alert_text(label_en, direction, dist_level, lang):
    """
    Build contextual voice alert. Returns:
      - Indoor-specific alert if label has a special mapping
      - Generic directional alert otherwise
    """
    cfg = LANGUAGES[lang]

    # --- Indoor-specific alerts (priority for key objects) ---
    alert_key = INDOOR_ALERT_MAP.get(label_en, {}).get(direction)
    if alert_key and alert_key in cfg:
        base = cfg[alert_key]
        # Add urgency prefix for very close objects
        if dist_level == "VERY_CLOSE":
            return f"{cfg['warning']}! {base}"
        return base

    # --- Generic directional alert ---
    label_local = translate_label(label_en, lang)
    dir_str  = cfg["left"]  if direction == "LEFT"  else \
               cfg["right"] if direction == "RIGHT" else cfg["ahead"]
    dist_str = cfg["very_close"] if dist_level == "VERY_CLOSE" else \
               cfg["close"]      if dist_level == "CLOSE"      else \
               cfg["nearby"]     if dist_level == "MEDIUM"     else ""

    if dist_level == "VERY_CLOSE" and direction == "CENTER":
        return f"{cfg['warning']}! {label_local} {cfg['ahead']}, {cfg['very_close']}"

    parts = [label_local, dir_str]
    if dist_str:
        parts.append(dist_str)
    return ", ".join(parts)


def trigger_voice_and_haptic(text, dist_level="FAR"):
    """Debounced voice + haptic trigger."""
    DEBOUNCE_MS = 1500
    urgent_debounce = 800
    now = time.time() * 1000

    current_debounce = urgent_debounce if dist_level == "VERY_CLOSE" else DEBOUNCE_MS
    if text == st.session_state.last_spoken and (now - st.session_state.last_speak_time) < current_debounce:
        return

    st.session_state.last_spoken    = text
    st.session_state.last_speak_time = now

    # Haptic pattern
    vib_pattern = "[150]"
    if   dist_level == "VERY_CLOSE": vib_pattern = "[100, 50, 100, 50, 100]"
    elif dist_level == "CLOSE":      vib_pattern = "[200, 100, 200]"
    elif dist_level == "MEDIUM":     vib_pattern = "[300]"

    safe_text = text.replace("'", "\\'").replace('"', '\\"')
    components.html(f"""
        <script>
        if (window.parent.window.currentVibrationPattern !== undefined) {{
            window.parent.window.currentVibrationPattern = {vib_pattern};
        }}
        var msg = new SpeechSynthesisUtterance('{safe_text}');
        msg.lang = '{lang_cfg["code"]}';
        msg.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)


def detect_large_obstacle(frame_bgr):
    """Fast CPU-based obstacle detection using edge contours (no NN)."""
    h, w = frame_bgr.shape[:2]
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    edges   = cv2.Canny(blurred, 30, 100)

    cx1, cx2 = int(w*0.25), int(w*0.75)
    cy1, cy2 = int(h*0.15), int(h*0.85)
    center_edges = edges[cy1:cy2, cx1:cx2]

    contours, _ = cv2.findContours(center_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_area  = (cx2-cx1) * (cy2-cy1)
    total_area   = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 500)
    ratio = total_area / center_area

    if ratio > 0.45:   return "VERY_CLOSE"
    elif ratio > 0.25: return "CLOSE"
    return None


def letterbox(image_bgr, target_size=640):
    h, w  = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized  = cv2.resize(image_bgr, (new_w, new_h))
    canvas   = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas, scale, pad_x, pad_y


def unletterbox_bbox(x1, y1, x2, y2, scale, pad_x, pad_y):
    return (x1-pad_x)/scale, (y1-pad_y)/scale, (x2-pad_x)/scale, (y2-pad_y)/scale


def estimate_distance(bbox_height, frame_height):
    ratio = bbox_height / frame_height
    if   ratio > 0.60: return "VERY_CLOSE"
    elif ratio > 0.40: return "CLOSE"
    elif ratio > 0.20: return "MEDIUM"
    return "FAR"


def classify_direction(bbox_center_x, frame_width):
    ratio = bbox_center_x / frame_width
    if   ratio < 0.35: return "LEFT"
    elif ratio > 0.65: return "RIGHT"
    return "CENTER"


def get_dist_class(dist_lvl):
    mapping = {
        "VERY_CLOSE": "dist-very-close",
        "CLOSE":      "dist-close",
        "MEDIUM":     "dist-medium",
    }
    return mapping.get(dist_lvl, "dist-far")


def get_dir_icon(direction):
    if direction == "LEFT":  return "◀"
    if direction == "RIGHT": return "▶"
    return "▲"


# ─── WEBRTC VIDEO PROCESSOR ──────────────────────────────────────────────────

class VideoProcessor:
    def __init__(self):
        self.latest_announce = ""
        self.latest_dist     = "FAR"
        self.total_frames    = 0
        self.total_dets      = 0
        self.last_inference_time = 0
        self.last_results    = []
        self.empty_count     = 0

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = frame_bgr.shape[:2]
        self.total_frames += 1

        now = time.time()
        INFERENCE_INTERVAL = 0.4  # 400ms between inferences

        if now - self.last_inference_time >= INFERENCE_INTERVAL:
            lb_img, scale, pad_x, pad_y = letterbox(frame_bgr, target_size=INFERENCE_IMGSZ)
            results = MODEL(
                lb_img,
                verbose=False,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=INFERENCE_IMGSZ
            )[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                ux1, uy1, ux2, uy2 = unletterbox_bbox(x1, y1, x2, y2, scale, pad_x, pad_y)
                cls_id   = int(box.cls[0])
                label_en = MODEL.names[cls_id]
                dist     = estimate_distance(uy2 - uy1, orig_h)
                direction = classify_direction((ux1 + ux2) / 2, orig_w)

                detections.append({
                    "label":    translate_label(label_en, selected_lang_name),
                    "label_en": label_en,
                    "dist":     dist,
                    "dir":      direction,
                    "bbox":     (int(ux1), int(uy1), int(ux2), int(uy2)),
                    "conf":     float(box.conf[0]),
                    "rank": (
                        0 if label_en in PRIORITY_OBJECTS else 1,
                        0 if dist == "VERY_CLOSE" else (1 if dist == "CLOSE" else 2)
                    )
                })

            self.last_results        = detections
            self.last_inference_time = now
            self.total_dets         += len(detections)

        # ── Draw + announce using cached results ──────────────────────────────
        annotated_frame = frame_bgr.copy()

        if len(self.last_results) > 0:
            self.empty_count = 0
            self.last_results.sort(key=lambda x: x["rank"])
            primary = self.last_results[0]
            plabel_en = primary["label_en"]
            plabel    = primary["label"]
            pdist     = primary["dist"]
            pdir      = primary["dir"]

            px1, py1, px2, py2 = primary["bbox"]
            
            # Color-coded box (red = very close, yellow = close, green = medium)
            colors = {
                "VERY_CLOSE": (0, 0, 255),
                "CLOSE":      (0, 165, 255),
                "MEDIUM":     (0, 255, 0),
                "FAR":        (255, 255, 255),
            }
            box_color = colors.get(pdist, (255, 255, 255))
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), box_color, 4)
            
            conf_pct = int(primary["conf"] * 100)
            label_txt = f"{plabel.upper()} {pdist} {conf_pct}%"
            cv2.putText(annotated_frame, label_txt, (px1, max(py1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 3)

            # Direction icon on frame
            icon = get_dir_icon(pdir)
            cv2.putText(annotated_frame, icon, (orig_w // 2 - 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

            # Build contextual announce
            self.latest_announce = build_alert_text(plabel_en, pdir, pdist, selected_lang_name)
            self.latest_dist     = pdist

        else:
            # No YOLO detections → fast CPU obstacle check
            obs_dist = detect_large_obstacle(frame_bgr)
            if obs_dist:
                self.empty_count = 0
                dist_str = lang_cfg["very_close"] if obs_dist == "VERY_CLOSE" else lang_cfg["close"]
                self.latest_announce = f"{lang_cfg['warning']}! {lang_cfg['obstacle']}, {dist_str}"
                self.latest_dist     = obs_dist
            else:
                self.empty_count += 1
                if self.empty_count >= 10:
                    self.latest_announce = lang_cfg["clear"]
                    self.latest_dist     = "FAR"

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ─── MAIN UI ─────────────────────────────────────────────────────────────────

st.markdown("<h1>VisionAid</h1>", unsafe_allow_html=True)
st.markdown("<p class='caregiver-subtitle'>👁 CAREGIVER VIEW — High Accuracy Indoor Navigation</p>",
            unsafe_allow_html=True)
st.markdown(f"<div class='model-badge'>🤖 Model: {MODEL_NAME}</div>", unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="visionaid-v2",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=False,
)

if webrtc_ctx.state.playing:
    st_autorefresh(interval=500, key="voice_trigger_loop")
    if webrtc_ctx.video_processor:
        proc = webrtc_ctx.video_processor
        if proc.latest_announce:
            st.session_state.frame_count = proc.total_frames
            st.session_state.det_count   = proc.total_dets
            trigger_voice_and_haptic(proc.latest_announce, proc.latest_dist)
            st.session_state.ui_msg      = proc.latest_announce.upper()
            st.session_state.ui_msg_class = get_dist_class(proc.latest_dist)

stats_placeholder.markdown(
    f"**Frames:** {st.session_state.frame_count} | "
    f"**Detections:** {st.session_state.det_count} | "
    f"**conf:** {CONF_THRESHOLD} | **iou:** {IOU_THRESHOLD}"
)

st.markdown(f"""
    <div class="status-panel">
        <div class="status-text {st.session_state.ui_msg_class}">
            {st.session_state.ui_msg}<br>
            <span style='font-size: 0.85rem; color: #888888; font-weight: 400;'>Voice Engine Active</span>
        </div>
    </div>
""", unsafe_allow_html=True)

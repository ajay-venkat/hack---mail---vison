import streamlit as st
import cv2
import numpy as np
import time
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from ultralytics import YOLO
import torch
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
        font-size: 2rem !important;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .status-clear { color: #888888 !important; } /* FAR mapped color */
    
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


# --- CONFIGURATIONS & STATE ---
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
    }
}

OBJECT_TRANSLATIONS = {
    "Tamil": {
        "person": "நபர்", "car": "கார்", "truck": "லாரி",
        "bus": "பேருந்து", "motorcycle": "மோட்டார் சைக்கிள்",
        "bicycle": "சைக்கிள்", "chair": "நாற்காலி",
        "dining table": "மேசை", "bottle": "பாட்டில்",
        "dog": "நாய்", "cat": "பூனை", "door": "கதவு",
        "bed": "படுக்கை", "toilet": "கழிவறை",
        "tv": "தொலைக்காட்சி", "laptop": "மடிக்கணினி",
        "cell phone": "கைப்பேசி", "book": "புத்தகம்",
        "clock": "கடிகாரம்", "cup": "கோப்பை",
        "traffic light": "போக்குவரத்து விளக்கு",
        "fire hydrant": "தீயணைப்பு குழாய்",
        "stop sign": "நிறுத்த அடையாளம்",
        "bench": "இருக்கை", "backpack": "பை",
        "umbrella": "குடை", "handbag": "கைப்பை",
        "suitcase": "பெட்டி", "sports ball": "பந்து",
        "couch": "சோபா", "potted plant": "தாவரம்",
        "sink": "கழுவுதொட்டி", "refrigerator": "குளிர்சாதனப்பெட்டி",
        "scissors": "கத்தரிக்கோல்", "vase": "பூச்சட்டி",
    },
    "Hindi": {
        "person": "व्यक्ति", "car": "कार", "truck": "ट्रक",
        "bus": "बस", "motorcycle": "मोटरसाइकिल",
        "bicycle": "साइकिल", "chair": "कुर्सी",
        "dining table": "मेज़", "bottle": "बोतल",
        "dog": "कुत्ता", "cat": "बिल्ली", "door": "दरवाज़ा",
        "bed": "बिस्तर", "toilet": "शौचालय",
        "tv": "टीवी", "laptop": "लैपटॉप",
        "cell phone": "मोबाइल फ़ोन", "book": "किताब",
        "clock": "घड़ी", "cup": "कप",
        "traffic light": "ट्रैफ़िक लाइट",
        "fire hydrant": "अग्निशमन यंत्र",
        "stop sign": "रुकने का संकेत",
        "bench": "बेंच", "backpack": "बैग",
        "umbrella": "छाता", "handbag": "पर्स",
        "suitcase": "सूटकेस", "sports ball": "गेंद",
        "couch": "सोफ़ा", "potted plant": "पौधा",
        "sink": "नल", "refrigerator": "फ्रिज",
        "scissors": "कैंची", "vase": "फूलदान",
    }
}

PRIORITY_OBJECTS = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "chair", "stairs", "door"}

if "last_spoken" not in st.session_state: st.session_state.last_spoken = ""
if "last_speak_time" not in st.session_state: st.session_state.last_speak_time = 0
if "last_dark_warn" not in st.session_state: st.session_state.last_dark_warn = 0
if "empty_frames" not in st.session_state: st.session_state.empty_frames = 0
if "frame_count" not in st.session_state: st.session_state.frame_count = 0
if "det_count" not in st.session_state: st.session_state.det_count = 0
if "ui_msg" not in st.session_state: st.session_state.ui_msg = "START NAVIGATION to begin"
if "ui_msg_class" not in st.session_state: st.session_state.ui_msg_class = "status-clear"
if "snapshot" not in st.session_state: st.session_state.snapshot = None

# Custom Pattern Component
if "vib_pattern" not in st.session_state: st.session_state.vib_pattern = "[150]"

# --- INJECT HAPTIC ARM BUTTON & POLLING LOOP ---
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
    // Global state
    window.hapticsArmed = false;
    window.currentVibrationPattern = [0];
    window.lastVibrationPattern = [0];
    window.lastVibrationTime = 0;
    
    document.getElementById('arm-btn').addEventListener('click', function() {{
        window.hapticsArmed = true;
        this.style.display = 'none';
        document.getElementById('armed-indicator').style.display = 'block';
        
        // Initial tiny vibrate to confirm
        if(navigator.vibrate) navigator.vibrate(50);
    }});
    
    // Polling Loop
    setInterval(function() {{
        if(!window.hapticsArmed || !navigator.vibrate) return;
        
        // Check if pattern changed and it's been at least 2000ms since last vibrate (debounce)
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

# Sidebar Config
st.sidebar.title("Accessibility Settings")
selected_lang_name = st.sidebar.selectbox("Language / மொழி / भाषा", options=list(LANGUAGES.keys()))
lang_cfg = LANGUAGES[selected_lang_name]

# Stats
st.sidebar.markdown("### About This View")
st.sidebar.markdown("""
<div style="color: #bbb; font-size: 0.95rem;">
This screen is for caregivers, demo observers, and developers.<br>
The visually impaired user receives all feedback via voice and haptics only.<br>
No screen interaction is needed during navigation.
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
# Placeholder for stats updated dynamically
stats_placeholder = st.sidebar.empty()

# --- ML MODELS ---
@st.cache_resource(show_spinner="Loading YOLOv8s model...")
def load_yolo():
    return YOLO("yolov8s.pt")

@st.cache_resource(show_spinner="Loading MiDaS depth model...")
def load_midas():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return model, transform

try:
    MODEL = load_yolo()
    MIDAS_MODEL, MIDAS_TRANSFORM = load_midas()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()
    
# --- HELPER FUNCTIONS ---
def translate_label(label, language):
    if language == "English":
        return label
    return OBJECT_TRANSLATIONS.get(language, {}).get(label, label)

def trigger_voice_and_haptic(text, dist_level="FAR"):
    # Debounce check
    now = time.time() * 1000
    if text == st.session_state.last_spoken and (now - st.session_state.last_speak_time) < 2000:
        return
        
    st.session_state.last_spoken = text
    st.session_state.last_speak_time = now
    
    # Haptic Pattern for JS
    vib_pattern_arr = "[150]"
    if dist_level == "VERY_CLOSE": vib_pattern_arr = "[100, 50, 100, 50, 100]"
    elif dist_level == "CLOSE": vib_pattern_arr = "[200, 100, 200]"
    elif dist_level == "MEDIUM": vib_pattern_arr = "[300]"
    
    # Inject voice and update haptic pattern
    components.html(f"""
        <script>
        // Update global window pattern for polling mechanism
        if (window.parent.window.currentVibrationPattern !== undefined) {{
            window.parent.window.currentVibrationPattern = {vib_pattern_arr};
        }}
        
        // Voice Feedback
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.lang = '{lang_cfg["code"]}';
        msg.rate = 0.9;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)


def estimate_depth_obstacle(frame_bgr, midas_model, midas_transform):
    """
    Returns: "VERY_CLOSE" | "CLOSE" | "MEDIUM" | "CLEAR"
    based on the minimum depth value in the CENTER zone of the frame.
    Center zone = middle 40% width, middle 60% height.
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(frame_rgb)

    with torch.no_grad():
        depth = midas_model(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(h, w),
            mode="bicubic", align_corners=False
        ).squeeze()

    depth_np = depth.numpy()

    # Center zone only — ignore edges
    cx1 = int(w * 0.30)
    cx2 = int(w * 0.70)
    cy1 = int(h * 0.20)
    cy2 = int(h * 0.80)
    center_depth = depth_np[cy1:cy2, cx1:cx2]

    # MiDaS outputs INVERSE depth — higher value = closer object
    max_depth = float(center_depth.max())
    depth_range = float(depth_np.max() - depth_np.min()) + 1e-6
    normalized = max_depth / depth_range  # 0.0 to 1.0

    if normalized > 0.85:   return "VERY_CLOSE"
    elif normalized > 0.70: return "CLOSE"
    elif normalized > 0.55: return "MEDIUM"
    else:                   return "CLEAR"


def letterbox(image_bgr, target_size=640):
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas, scale, pad_x, pad_y

def unletterbox_bbox(x1, y1, x2, y2, scale, pad_x, pad_y):
    return (x1-pad_x)/scale, (y1-pad_y)/scale, (x2-pad_x)/scale, (y2-pad_y)/scale
    
def estimate_distance(bbox_height, frame_height):
    ratio = bbox_height / frame_height
    if ratio > 0.60:   return "VERY_CLOSE"
    elif ratio > 0.40: return "CLOSE"
    elif ratio > 0.20: return "MEDIUM"
    else:              return "FAR"
    
def classify_direction(bbox_center_x, frame_width):
    ratio = bbox_center_x / frame_width
    if ratio < 0.35:   return "LEFT"
    elif ratio > 0.65: return "RIGHT"
    else:              return "CENTER"

def get_dist_class(dist_lvl):
    if dist_lvl == "VERY_CLOSE": return "dist-very-close"
    if dist_lvl == "CLOSE": return "dist-close"
    if dist_lvl == "MEDIUM": return "dist-medium"
    return "dist-far"

def get_dir_icon(direction):
    if direction == "LEFT": return "◀"
    if direction == "RIGHT": return "▶"
    return "▲"

# --- WEBRTC PROCESSOR ---
class VideoProcessor:
    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # Inference with Letterbox (YOLOv8s)
        lb_img, scale, pad_x, pad_y = letterbox(frame_bgr)
        results = MODEL(lb_img, verbose=False, conf=0.70)[0]
        
        # Inference MiDaS
        midas_dist = estimate_depth_obstacle(frame_bgr, MIDAS_MODEL, MIDAS_TRANSFORM)
        
        detections = []
        yolo_active_dists = set()
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.70: continue
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Unletterbox back to original dims
            ux1, uy1, ux2, uy2 = unletterbox_bbox(x1, y1, x2, y2, scale, pad_x, pad_y)
            
            # Clamp to frame
            ux1, uy1 = max(0, ux1), max(0, uy1)
            ux2, uy2 = min(orig_w, ux2), min(orig_h, uy2)
            
            cls_id = int(box.cls[0])
            label_en = MODEL.names[cls_id]
            label_trans = translate_label(label_en, selected_lang_name)
            
            bbox_h = uy2 - uy1
            bbox_cx = (ux1 + ux2) / 2
            
            dist = estimate_distance(bbox_h, orig_h)
            direction = classify_direction(bbox_cx, orig_w)
            
            yolo_active_dists.add(dist)
            
            # Distance rank for sorting (lower is closer)
            dist_rank = 0 if dist == "VERY_CLOSE" else (1 if dist == "CLOSE" else (2 if dist == "MEDIUM" else 3))
            is_prio = 0 if label_en in PRIORITY_OBJECTS else 1
            
            detections.append({
                "label": label_trans,
                "dist": dist,
                "dir": direction,
                "conf": conf,
                "bbox": (int(ux1), int(uy1), int(ux2), int(uy2)),
                "rank": (is_prio, dist_rank)
            })
            
        # Draw on frame and prepare global messages
        annotated_frame = frame_bgr.copy()
        
        if len(detections) > 0:
            # Sort: Priority first, then closest
            detections.sort(key=lambda x: x["rank"])
            
            primary = detections[0]
            plabel, pdist, pdir = primary["label"], primary["dist"], primary["dir"]
            
            # Draw primary bbox thick
            px1, py1, px2, py2 = primary["bbox"]
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 4)
            # Label
            icon = get_dir_icon(pdir)
            cv2.putText(annotated_frame, f"{plabel.upper()} {pdist}", (px1, py1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# --- MAIN UI ---
st.markdown("<h1>VisionAid</h1>", unsafe_allow_html=True)
st.markdown("<p class='caregiver-subtitle'>👁 CAREGIVER VIEW — Live Detection Monitor</p>", unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="visionaid",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    # Important: False means the recv runs synchronously in the main thread space 
    # so we can draw annotations inline safely.
    async_processing=False,
)

stats_placeholder.markdown(f"**Frames Processed:** {st.session_state.frame_count}\n\n**Objects Detected:** {st.session_state.det_count}")

# Status panel below the video component
st.markdown(f"""
    <div class="status-panel">
        <div class="status-text {st.session_state.ui_msg_class}">
            {st.session_state.ui_msg}<br>
            <span style='font-size: 0.9rem; color: #888888;'>🔊 Announcements managed via voice</span>
        </div>
    </div>
""", unsafe_allow_html=True)

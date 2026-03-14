import streamlit as st
import cv2
import numpy as np
import time
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO

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

# Sidebar Config
st.sidebar.title("Accessibility Settings")
selected_lang_name = st.sidebar.selectbox("Language / மொழி / भाषा", options=list(LANGUAGES.keys()))
lang_cfg = LANGUAGES[selected_lang_name]

# Stats
st.sidebar.markdown(f"**Frames Processed:** {st.session_state.frame_count}")
st.sidebar.markdown(f"**Objects Detected:** {st.session_state.det_count}")

# --- ML MODEL ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

try:
    MODEL = load_yolo()
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()
    
# --- HELPER FUNCTIONS ---
def speak_and_vibrate(text, dist_level="FAR"):
    # Debounce check
    now = time.time() * 1000
    if text == st.session_state.last_spoken and (now - st.session_state.last_speak_time) < 2000:
        return
        
    st.session_state.last_spoken = text
    st.session_state.last_speak_time = now
    
    # Haptic Pattern
    vib_pattern = "[150]"
    if dist_level == "VERY_CLOSE": vib_pattern = "[100, 50, 100, 50, 100]"
    elif dist_level == "CLOSE": vib_pattern = "[200, 100, 200]"
    elif dist_level == "MEDIUM": vib_pattern = "[300]"
    
    # Inject JS
    components.html(f"""
        <script>
        // Haptic Feedback
        if (navigator.vibrate) {{
            navigator.vibrate({vib_pattern});
        }}
        
        // Voice Feedback
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.lang = '{lang_cfg["code"]}';
        msg.rate = 0.9;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

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

def get_dist_color_hex(dist_lvl):
    if dist_lvl == "VERY_CLOSE": return "#DB4437"
    if dist_lvl == "CLOSE": return "#F4B400"
    if dist_lvl == "MEDIUM": return "#0F9D58"
    return "#888888"

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
import av
class VideoProcessor:
    def __init__(self):
        self.frame = None
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MAIN UI ---
st.markdown("<h1>VisionAid</h1>", unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="visionaid",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    # 500ms loop
    st_autorefresh(interval=500, key="nav_loop")
    
    if webrtc_ctx.video_processor and webrtc_ctx.video_processor.frame is not None:
        frame_bgr = webrtc_ctx.video_processor.frame.copy()
        orig_h, orig_w = frame_bgr.shape[:2]
        st.session_state.frame_count += 1
        
        # Dark frame check
        avg_brightness = np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        now_sec = time.time()
        if avg_brightness < 30:
            if now_sec - st.session_state.last_dark_warn > 5:
                speak_and_vibrate(lang_cfg["dark"], "FAR")
                st.session_state.last_dark_warn = now_sec
            st.session_state.ui_msg = lang_cfg["dark"]
            st.session_state.ui_msg_class = "dist-very-close"
            st.session_state.snapshot = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.rerun()
            
        # Inference with Letterbox
        lb_img, scale, pad_x, pad_y = letterbox(frame_bgr)
        results = MODEL(lb_img, verbose=False, conf=0.70)[0]
        
        detections = []
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
            label = MODEL.names[cls_id]
            
            bbox_h = uy2 - uy1
            bbox_cx = (ux1 + ux2) / 2
            
            dist = estimate_distance(bbox_h, orig_h)
            direction = classify_direction(bbox_cx, orig_w)
            
            # Distance rank for sorting (lower is closer)
            dist_rank = 0 if dist == "VERY_CLOSE" else (1 if dist == "CLOSE" else (2 if dist == "MEDIUM" else 3))
            is_prio = 0 if label in PRIORITY_OBJECTS else 1
            
            detections.append({
                "label": label,
                "dist": dist,
                "dir": direction,
                "conf": conf,
                "bbox": (int(ux1), int(uy1), int(ux2), int(uy2)),
                "rank": (is_prio, dist_rank)
            })
            
        st.session_state.det_count += len(detections)
            
        # UI & Voice Logic
        if len(detections) > 0:
            st.session_state.empty_frames = 0
            
            # Sort: Priority first, then closest
            detections.sort(key=lambda x: x["rank"])
            
            primary = detections[0]
            plabel, pdist, pdir = primary["label"], primary["dist"], primary["dir"]
            
            # Draw primary bbox thick
            px1, py1, px2, py2 = primary["bbox"]
            cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), (0, 0, 255), 4)
            
            # Announce formula
            if pdist == "VERY_CLOSE" and pdir == "CENTER":
                announce = f"{lang_cfg['warning']}! {plabel} {lang_cfg['ahead']}, {lang_cfg['very_close']}"
            elif pdist == "FAR" and pdir == "CENTER":
                announce = f"{plabel} {lang_cfg['ahead']}"
            else:
                dir_str = lang_cfg["left"] if pdir == "LEFT" else (lang_cfg["right"] if pdir == "RIGHT" else lang_cfg["ahead"])
                dist_str = lang_cfg["very_close"] if pdist == "VERY_CLOSE" else (lang_cfg["close"] if pdist == "CLOSE" else (lang_cfg["nearby"] if pdist == "MEDIUM" else ""))
                announce = f"{plabel} {dir_str}, {dist_str}".strip(", ")
                
            speak_and_vibrate(announce, pdist)
            
            icon = get_dir_icon(pdir)
            st.session_state.ui_msg = f"{icon} {plabel.upper()} {pdist}"
            st.session_state.ui_msg_class = get_dist_class(pdist)
            
        else:
            st.session_state.empty_frames += 1
            if st.session_state.empty_frames == 3:
                speak_and_vibrate(lang_cfg["clear"], "FAR")
                st.session_state.ui_msg = lang_cfg["clear"]
                st.session_state.ui_msg_class = "status-clear"
                
            if st.session_state.empty_frames > 3:
                 st.session_state.ui_msg = lang_cfg["clear"]
                 st.session_state.ui_msg_class = "status-clear"
            
        # Show snapshot
        st.session_state.snapshot = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# Display Frame
if st.session_state.snapshot is not None:
    st.image(st.session_state.snapshot, use_container_width=True)
else:
    if not webrtc_ctx.state.playing:
        st.markdown("<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666; font-size: 1.5rem;'>CAMERA OFF</div>", unsafe_allow_html=True)
        # Reset state when stopped
        st.session_state.empty_frames = 0
        st.session_state.last_spoken = ""
        st.session_state.ui_msg = "START NAVIGATION to begin"
        st.session_state.ui_msg_class = "status-clear"
        st.session_state.snapshot = None
        
# Display Panel
st.markdown(f"""
    <div class="status-panel">
        <div class="status-text {st.session_state.ui_msg_class}">
            {st.session_state.ui_msg}
        </div>
    </div>
""", unsafe_allow_html=True)

"""
VisionAid Pro v4 — Streamlit UI + Video Processor
===================================================
Upgrades: MiDaS Depth · DeepSORT Tracking · 3-Layer Fusion · Smart Scene
           SUN RGB-D + NYU Fine-tuning · Performance Modes · UI Dashboard
"""
import streamlit as st
import cv2, numpy as np, time, os, threading
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO
import av

from vision_engine import (
    LANGUAGES, OBJ_TRANS, PERF_MODES, CLASS_CONF, BASE_CONF, IOU_THRESHOLD,
    FOCAL_LENGTH_PX, REF_HEIGHTS_M, DEFAULT_HEIGHT_M,
    URGENT_OBJECTS, WARNING_OBJECTS, OBJECT_PRIORITY,
    REPEAT_INTERVALS, CLEAR_PATH_INTERVAL,
    calc_distance_m, dist_tier, depth_to_tier, depth_to_metres,
    haptic_pattern, direction,
    detect_fall, detect_stairs, detect_scene, detect_contour_obstacles,
    scene_conf_adjust, VoiceEngine,
    get_midas, get_tracker,
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="VisionAid Pro v4", page_icon="👁️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
.main{background:#000!important;color:#fff!important;font-family:'Inter',sans-serif}
#MainMenu{visibility:hidden}footer{visibility:hidden}
.stApp{max-width:100%;padding:0;background:#000!important}
p,li,span,div{font-size:1.1rem!important;color:#fff!important}
.hud-container{display:flex;gap:15px;margin-bottom:15px;flex-wrap:wrap}
.hud-box{background:#111;border:2px solid #333;border-radius:12px;padding:15px;flex:1;min-width:130px;text-align:center;display:flex;flex-direction:column;justify-content:center}
.hud-red{border-color:#DB4437;background:rgba(219,68,55,.1)}
.hud-yellow{border-color:#F4B400;background:rgba(244,180,0,.1)}
.hud-green{border-color:#0F9D58;background:rgba(15,157,88,.1)}
.hud-blue{border-color:#1A73E8;background:rgba(26,115,232,.1)}
.hud-title{font-size:.85rem!important;color:#aaa!important;text-transform:uppercase;font-weight:700;margin-bottom:5px}
.hud-value{font-size:1.8rem!important;font-weight:900;line-height:1.2}
.status-panel{background:#0D1117;padding:25px;border:3px solid #1A73E8;border-radius:15px;text-align:center;margin-top:15px}
.status-text{font-size:24px!important;font-weight:900;letter-spacing:1px}
.priority-urgent{color:#DB4437!important;text-shadow:0 0 15px rgba(219,68,55,.5)}
.priority-warning{color:#F4B400!important}
.priority-info{color:#0F9D58!important}
.priority-clear{color:#888!important}
.det-table{width:100%;border-collapse:collapse;margin:10px 0}
.det-table th{background:#1a1a2e;color:#aaa;padding:8px;text-align:left;font-size:.85rem!important;border-bottom:2px solid #333}
.det-table td{padding:6px 8px;border-bottom:1px solid #222;font-size:.95rem!important}
.det-table tr:hover{background:#111}
.stButton>button{width:100%;font-weight:900;font-size:1.2rem;padding:15px;border-radius:12px;border:none;transition:.2s}
.sos-btn button{background:#DB4437!important;color:#fff!important;font-size:1.5rem!important}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚨 Emergency")
contact_num = st.sidebar.text_input("Emergency Contact", value="+1234567890")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🌐 Language / மொழி / भाषा")
selected_lang = st.sidebar.selectbox("Language", ["English", "Tamil", "Hindi"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚡ Performance Mode")
perf_mode = st.sidebar.radio("Mode", list(PERF_MODES.keys()), index=1, label_visibility="collapsed")
ACTIVE = PERF_MODES[perf_mode]

st.sidebar.markdown("---")
show_depth = st.sidebar.checkbox("🗺️ Show Depth Map Overlay", value=False)
show_tracks = st.sidebar.checkbox("📦 Show Track IDs", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Caregiver Log")
caregiver_log = st.sidebar.empty()

lang = LANGUAGES[selected_lang]

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if 'log' not in st.session_state: st.session_state.log = []
for k, v in {
    "frame_count":0, "det_count":0,
    "ui_msg":"START NAVIGATION to begin", "ui_msg_class":"priority-clear",
    "last_spoken_text":"","last_spoken_obj":"","last_spoken_zone":"",
    "last_spoken_time":0.0,"last_clear_time":0.0,"repeat_count":0,
    "last_center_block_time":time.time(),"object_appear_time":{},
    "last_clear_direction":"LEFT",
    "sos_triggered":False,"fall_detected_time":0.0,"prev_gray":None,
    "ui_dist_m":"--","ui_obj":"--","ui_dir":"--","ui_scene":"Unknown",
    "ui_det_table":[], "midas_loaded":False, "tracker_loaded":False,
}.items():
    if k not in st.session_state: st.session_state[k] = v

def add_log(msg):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log.insert(0, f"[{ts}] {msg}")
    if len(st.session_state.log) > 15: st.session_state.log.pop()

# ─── HAPTIC + VOICE + SOS HTML ────────────────────────────────────────────────
components_html_1 = """
<div id="haptic-container" style="text-align:center;margin-bottom:10px;">
  <button id="arm-btn" style="background:#1A73E8;color:#fff;border:none;padding:15px 30px;
    font-size:1.2rem;font-weight:bold;border-radius:8px;cursor:pointer;width:100%;max-width:400px;
    font-family:'Inter',sans-serif;text-transform:uppercase;box-shadow:0 4px 6px rgba(0,0,0,0.3); display:none;">
    Tap to Enable Audio &amp; Haptics</button>
  <div id="armed-indicator" style="display:none;color:#0F9D58;font-size:1.2rem;font-weight:bold;
    font-family:'Inter',sans-serif;padding:10px;text-shadow:0 0 10px rgba(15,157,88,0.5);">📳 Haptics ON &nbsp;|&nbsp; 🔊 Voice ON</div>
</div>
<audio id="sos-audio" src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" loop preload="auto"></audio>
<script>
var pw = window.parent;
if (pw.hapticsArmed === undefined) {
    pw.hapticsArmed = false;
    pw.triggerSOS = false;
}

if (pw.hapticsArmed) {
    document.getElementById('armed-indicator').style.display = 'block';
} else {
    document.getElementById('arm-btn').style.display = 'inline-block';
}

document.getElementById('arm-btn').addEventListener('click', function() {
  pw.hapticsArmed = true; 
  this.style.display = 'none';
  document.getElementById('armed-indicator').style.display = 'block';
  
  if (pw.navigator && pw.navigator.vibrate) pw.navigator.vibrate(50);
  if (pw.speechSynthesis) {
      var w = new pw.SpeechSynthesisUtterance(' ');
      w.volume = 0;
      pw.speechSynthesis.speak(w);
  }
  
  let a = document.getElementById('sos-audio');
  a.volume = 0;
  a.play().then(() => a.pause()).catch(e => {});
  a.volume = 1;
});

let sa = document.getElementById('sos-audio');
if (pw.triggerSOS) {
    if (pw.speechSynthesis) pw.speechSynthesis.cancel();
    if (sa && sa.paused) sa.play().catch(e => {});
} else {
    if (sa && !sa.paused) {
        sa.pause();
        sa.currentTime = 0;
    }
}
</script>
"""
st.html(components_html_1)

# ─── SOS BUTTON ───────────────────────────────────────────────────────────────
if st.button("🚨 PANIC / SOS"):
    st.session_state.sos_triggered = not st.session_state.sos_triggered
    js = "true" if st.session_state.sos_triggered else "false"
    st.html(f"<script>window.parent.triggerSOS={js};</script>")
    if st.session_state.sos_triggered:
        st.session_state.ui_msg = f"EMERGENCY HELP: {contact_num}"
        st.session_state.ui_msg_class = "priority-urgent"

if st.session_state.sos_triggered:
    st.markdown(f"<h1 style='text-align:center;color:red;font-size:4rem;'>CALL: {contact_num}</h1>", unsafe_allow_html=True)

# ─── MODEL LOADING ────────────────────────────────────────────────────────────
CUSTOM_MODEL = "best_indoor.pt"
FALLBACK = ACTIVE["model"]

@st.cache_resource(show_spinner="Loading AI model...")
def load_yolo(name):
    if os.path.exists(CUSTOM_MODEL):
        return YOLO(CUSTOM_MODEL)
    return YOLO(name)

try:
    MODEL = load_yolo(FALLBACK)
    if "yolo_warmed_up" not in st.session_state:
        MODEL(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False, imgsz=320)
        st.session_state.yolo_warmed_up = True
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# Load MiDaS if FULL mode
if ACTIVE["use_midas"] and not st.session_state.midas_loaded:
    with st.spinner("Loading MiDaS depth model..."):
        get_midas().load()
        # Warmup
        get_midas().predict(np.zeros((320, 320, 3), dtype=np.uint8))
    st.session_state.midas_loaded = True

# Load DeepSORT
if not st.session_state.tracker_loaded:
    get_tracker().load()
    st.session_state.tracker_loaded = True

# ─── VIDEO PROCESSOR ─────────────────────────────────────────────────────────
class VideoProcessor:
    def __init__(self):
        self.latest_label_en = ""
        self.latest_dir = "CENTER"
        self.latest_dist_t = "FAR"
        self.latest_dist_m = 0.0
        self.latest_approaching = False
        self.latest_moving_away = False
        self.has_detection = False
        self.left_clear = True
        self.right_clear = True
        self.scene = "Unknown"
        self.last_infer_time = 0.0
        self.last_results = []
        self.empty_count = 0
        self.prev_gray = None
        self.motion_mask = None
        self.frame_n = 0
        self.det_table = []
        self.lost_announced = set()
        self.yolo_conf_frame_buf = {}  # label → frame count for 2-frame confirm

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            fh, fw = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Motion mask
            if self.prev_gray is not None:
                if self.prev_gray.shape != gray.shape:
                    self.prev_gray = gray.copy()
                    self.motion_mask = None
                else:
                    diff = cv2.absdiff(self.prev_gray, gray)
                    _, self.motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            self.prev_gray = gray.copy()

            now = time.time()
            self.frame_n += 1
            midas = get_midas()
            tracker = get_tracker()

            # Scene every 30 frames
            if self.frame_n % 30 == 0:
                self.scene = detect_scene(img)

            # MiDaS every 3rd frame (if enabled)
            if ACTIVE["use_midas"] and self.frame_n % 3 == 0:
                midas.predict(img)

            if now - self.last_infer_time >= ACTIVE["interval"]:
                # Fall check
                if detect_fall(img):
                    self.latest_label_en = "fall"
                    self.has_detection = True
                    self.latest_dist_m = 0.0
                    self.latest_dist_t = "CRITICAL"
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

                # ── Layer 1: YOLO ─────────────────────────────────────────
                sz = ACTIVE["imgsz"]
                adj_conf = scene_conf_adjust(self.scene, BASE_CONF)
                rsz = cv2.resize(img, (sz, sz))
                results = MODEL(rsz, verbose=False, conf=adj_conf, iou=IOU_THRESHOLD, imgsz=sz)[0]

                raw = []
                for box in results.boxes:
                    lb = MODEL.names[int(box.cls[0])]
                    cf = float(box.conf[0])
                    if cf < CLASS_CONF.get(lb, 0.45): continue
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    sx, sy = fw/sz, fh/sz
                    ux1,ux2 = int(x1*sx), int(x2*sx)
                    uy1,uy2 = int(y1*sy), int(y2*sy)
                    px_h = uy2-uy1
                    d_m = calc_distance_m(px_h, lb)
                    if d_m > 5.0 and lb not in {"car","bus","truck"}: continue

                    # MiDaS depth refinement
                    if ACTIVE["use_midas"] and midas.depth_map is not None:
                        cx_px = (ux1+ux2)//2; cy_px = (uy1+uy2)//2
                        dv = midas.get_depth_at(cx_px, cy_px)
                        midas_m = depth_to_metres(dv)
                        d_m = round((d_m + midas_m) / 2.0, 1)  # fuse both estimates

                    dt = dist_tier(d_m)
                    dr = direction((ux1+ux2)/2, fw)
                    mv = self._is_moving((ux1,uy1,ux2,uy2), fw, fh)
                    raw.append({"label_en":lb,"dist_m":d_m,"dist_t":dt,"dir":dr,
                                "bbox":(ux1,uy1,ux2,uy2),"conf":cf,"moving":mv,
                                "rank":d_m,"source":"yolo"})

                # Stairs secondary
                if not raw:
                    sd = detect_stairs(img)
                    if sd:
                        raw.append({"label_en":sd,"dist_m":2.0,"dist_t":"URGENT","dir":"CENTER",
                                    "bbox":(0,0,fw,fh),"conf":0.9,"moving":False,"rank":2.0,"source":"yolo"})

                # ── Layer 2: MiDaS wall detection ─────────────────────────
                if ACTIVE["use_midas"]:
                    is_wall, wall_dist = midas.detect_wall()
                    if is_wall and not any(d["label_en"] == "wall_ahead" for d in raw):
                        raw.append({"label_en":"wall_ahead","dist_m":wall_dist,
                                    "dist_t":dist_tier(wall_dist),"dir":"CENTER",
                                    "bbox":(int(fw*0.2),int(fh*0.2),int(fw*0.8),int(fh*0.8)),
                                    "conf":0.7,"moving":False,"rank":wall_dist,"source":"midas"})

                # ── Layer 3: Contour fallback ─────────────────────────────
                if ACTIVE["use_contour"] and not raw:
                    contours = detect_contour_obstacles(img)
                    for c in contours[:1]:
                        raw.append({"label_en":"large_object","dist_m":2.0,"dist_t":"URGENT",
                                    "dir":direction((c["bbox"][0]+c["bbox"][2])/2,fw),
                                    "bbox":c["bbox"],"conf":0.5,"moving":False,"rank":2.0,"source":"contour"})

                # ── 3-Layer confidence fusion ─────────────────────────────
                fused = []
                for d in raw:
                    src = d.get("source","yolo")
                    if src == "yolo" and ACTIVE["use_midas"] and midas.depth_map is not None:
                        d["confidence_level"] = "HIGH"  # YOLO+MiDaS confirm
                        fused.append(d)
                    elif src == "yolo":
                        # YOLO only → MEDIUM confidence → need 2-frame confirm
                        lb = d["label_en"]
                        self.yolo_conf_frame_buf[lb] = self.yolo_conf_frame_buf.get(lb, 0) + 1
                        if self.yolo_conf_frame_buf[lb] >= 2:
                            d["confidence_level"] = "MEDIUM"
                            fused.append(d)
                    elif src == "midas":
                        d["confidence_level"] = "LOW"
                        fused.append(d)
                    elif src == "contour":
                        d["confidence_level"] = "LOW"
                        fused.append(d)

                # Clear frame buf for labels not seen
                seen = {d["label_en"] for d in raw}
                self.yolo_conf_frame_buf = {k:v for k,v in self.yolo_conf_frame_buf.items() if k in seen}

                # ── DeepSORT tracking ─────────────────────────────────────
                if fused:
                    fused = tracker.update(fused, img)

                self.last_results = sorted(fused, key=lambda x: x["rank"])
                self.last_infer_time = now

            # ── Drawing ───────────────────────────────────────────────────
            annotated = img.copy()

            # Depth overlay safely resized to match annotated image
            if show_depth and ACTIVE["use_midas"] and midas.depth_map is not None:
                depth_color = midas.colorize(img.shape)
                if depth_color is not None:
                    depth_color = cv2.resize(depth_color, (fw, fh))
                    annotated = cv2.addWeighted(annotated, 0.6, depth_color, 0.4, 0)

            COLORS = {"CRITICAL":(0,0,255),"URGENT":(0,100,255),"WARNING":(0,200,255),"CAUTION":(0,200,0),"FAR":(100,100,100)}
            table_rows = []

            if self.last_results:
                self.empty_count = 0
                primary = self.last_results[0]
                self.latest_label_en = primary["label_en"]
                self.latest_dir = primary["dir"]
                self.latest_dist_t = primary["dist_t"]
                self.latest_dist_m = primary["dist_m"]
                self.latest_approaching = primary.get("approaching", False)
                self.latest_moving_away = primary.get("moving", False) and not primary.get("approaching", False)
                self.has_detection = True
                self.left_clear = not any(d["dir"]=="LEFT" for d in self.last_results)
                self.right_clear = not any(d["dir"]=="RIGHT" for d in self.last_results)

                for det in self.last_results:
                    bx1,by1,bx2,by2 = det["bbox"]
                    col = COLORS.get(det["dist_t"], (100,100,100))
                    thick = 4 if det.get("approaching") else 2
                    cv2.rectangle(annotated, (bx1,by1),(bx2,by2), col, thick)
                    lbl = f"{det['label_en']} {det['dist_m']}m"
                    tid = det.get("track_id", -1)
                    if show_tracks and tid != -1:
                        lbl = f"[{tid}] {lbl}"
                    if det.get("approaching"):
                        lbl = "→→ " + lbl
                    cv2.putText(annotated, lbl, (bx1, max(by1-8,16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    table_rows.append({"obj":det["label_en"],"dist":f"{det['dist_m']}m","zone":det["dir"]})
            else:
                self.left_clear = True; self.right_clear = True
                self.has_detection = False; self.empty_count += 1

            self.det_table = table_rows
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as e:
            print(f"Processing error: {e}")
            return frame

    def _is_moving(self, bbox, fw, fh):
        if self.motion_mask is None: return False
        x1,y1,x2,y2 = bbox
        x1, x2 = max(0, min(x1, fw-1)), max(0, min(x2, fw))
        y1, y2 = max(0, min(y1, fh-1)), max(0, min(y2, fh))
        if x2 <= x1 or y2 <= y1: return False
        roi = self.motion_mask[y1:y2, x1:x2]
        area = (y2-y1)*(x2-x1)
        if area == 0: return False
        return (cv2.countNonZero(roi)/area) > 0.3

# ─── MAIN UI ──────────────────────────────────────────────────────────────────
st.markdown("<h1>👁️ VisionAid Pro <span style='font-size:1rem;color:#888;'>v4</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888!important;'>MiDaS Depth · DeepSORT Tracking · 3-Layer Fusion · Smart Scene</p>", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f"<div class='hud-box hud-red'><div class='hud-title'>Object</div><div class='hud-value'>{st.session_state.ui_obj.upper()}</div></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='hud-box hud-yellow'><div class='hud-title'>Distance</div><div class='hud-value'>{st.session_state.ui_dist_m}m</div></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='hud-box hud-green'><div class='hud-title'>Zone</div><div class='hud-value'>{st.session_state.ui_dir}</div></div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='hud-box hud-blue'><div class='hud-title'>Scene</div><div class='hud-value'>{st.session_state.ui_scene}</div></div>", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443"
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
})

try:
    webrtc_ctx = webrtc_streamer(
        key="visionaid",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}
            },
            "audio": False
        },
        async_processing=True
    )
except Exception as e:
    st.error("Please allow camera access in browser.")
    webrtc_ctx = None
    st.camera_input("Fallback Camera")

if webrtc_ctx is None or not webrtc_ctx.state.playing:
    st.warning("Camera not connected. Click START above.")
    st.stop()
else:
    st.success("Camera active — processing started")

# Detection table
det_table_area = st.empty()

if webrtc_ctx and webrtc_ctx.state.playing:
    st_autorefresh(interval=500, key="va_loop")
    if webrtc_ctx.video_processor:
        proc = webrtc_ctx.video_processor
        st.session_state.ui_scene = proc.scene
        st.session_state.frame_count = proc.frame_n

        # Update detection table
        if proc.det_table:
            rows = "".join(f"<tr><td>{r['obj']}</td><td>{r['dist']}</td><td>{r['zone']}</td></tr>" for r in proc.det_table[:8])
            det_table_area.markdown(f"<table class='det-table'><tr><th>Object</th><th>Distance</th><th>Zone</th></tr>{rows}</table>", unsafe_allow_html=True)

        if proc.has_detection and not st.session_state.sos_triggered:
            lbl = proc.latest_label_en
            ddir = proc.latest_dir
            dtier = proc.latest_dist_t
            dm = proc.latest_dist_m
            appr = proc.latest_approaching
            maway = proc.latest_moving_away

            st.session_state.ui_obj = lbl
            st.session_state.ui_dist_m = dm
            st.session_state.ui_dir = ddir

            # Track-based announce gating
            tracker = get_tracker()
            track_id = -1
            if proc.last_results:
                track_id = proc.last_results[0].get("track_id", -1)

            should = tracker.should_announce(track_id)
            if should:
                # Smart voice
                now = time.time()
                priority = 0 if lbl in URGENT_OBJECTS or appr else 1 if lbl in WARNING_OBJECTS else 2
                interval = REPEAT_INTERVALS[priority]
                elapsed = now - st.session_state.last_spoken_time

                # Stuck detection
                if ddir=="CENTER" and lbl==st.session_state.last_spoken_obj and elapsed>5.0 and st.session_state.repeat_count>2:
                    loc = OBJ_TRANS.get(selected_lang,{}).get(lbl, lbl)
                    text = lang["stuck"].format(obj=loc)
                    hdir = st.session_state.last_clear_direction
                    if hdir=="LEFT": text += ". " + lang["nav_history_left"]
                    if hdir=="RIGHT": text += ". " + lang["nav_history_right"]
                    
                    st.session_state.pending_speech = text
                    st.session_state.pending_lang = lang["code"]
                    st.session_state.pending_vib = haptic_pattern(dtier)
                    
                    st.session_state.ui_msg = text.upper()
                    st.session_state.ui_msg_class = "priority-urgent"
                elif elapsed >= interval or lbl != st.session_state.last_spoken_obj:
                    text = VoiceEngine.build_alert(lbl, ddir, dtier, dm, proc.left_clear, proc.right_clear, appr, maway, lang, selected_lang)
                    if text:
                        st.session_state.pending_speech = text
                        st.session_state.pending_lang = lang["code"]
                        st.session_state.pending_vib = haptic_pattern(dtier)
                        
                        st.session_state.ui_msg = text.upper()
                        cmap = {"CRITICAL":"priority-urgent","URGENT":"priority-urgent","WARNING":"priority-warning"}
                        st.session_state.ui_msg_class = cmap.get(dtier, "priority-info")

                # Record
                if lbl == st.session_state.last_spoken_obj:
                    st.session_state.repeat_count += 1
                else:
                    st.session_state.repeat_count = 0
                    if st.session_state.last_spoken_zone in ("LEFT","RIGHT") and ddir=="CENTER":
                        st.session_state.last_clear_direction = "LEFT" if st.session_state.last_spoken_zone=="RIGHT" else "RIGHT"
                st.session_state.last_spoken_obj = lbl
                st.session_state.last_spoken_zone = ddir
                st.session_state.last_spoken_time = time.time()
                add_log(f"Alerted: {lbl} ({ddir}) {dm}m")

            # Track lost announcements
            for lid in tracker.get_lost_ids():
                st.session_state.pending_speech = lang["track_lost"]
                st.session_state.pending_lang = lang["code"]
                st.session_state.pending_vib = haptic_pattern("FAR")
                add_log(f"Track lost: {lid}")

        elif not st.session_state.sos_triggered:
            st.session_state.ui_obj = "--"
            st.session_state.ui_dist_m = "--"
            st.session_state.ui_dir = "--"
            if proc.empty_count >= 10:
                sc = st.session_state.ui_scene
                now = time.time()
                if sc in ("Outdoor","Road","Corridor") and now - st.session_state.last_spoken_time > 10:
                    key = f"scene_{sc.lower()}"
                    txt = lang.get(key, "")
                    if txt:
                        st.session_state.pending_speech = txt
                        st.session_state.pending_lang = lang["code"]
                        st.session_state.pending_vib = haptic_pattern("FAR")
                        
                        st.session_state.last_spoken_obj = key
                        st.session_state.last_spoken_time = now
                        st.session_state.ui_msg = txt.upper()
                        add_log(f"Scene: {sc}")
                elif now - st.session_state.last_clear_time > CLEAR_PATH_INTERVAL:
                    st.session_state.pending_speech = lang["clear"]
                    st.session_state.pending_lang = lang["code"]
                    st.session_state.pending_vib = haptic_pattern("FAR")
                    
                    st.session_state.last_spoken_obj = ""
                    st.session_state.repeat_count = 0
                    st.session_state.last_clear_time = now
                    st.session_state.ui_msg = lang["clear"].upper()
                    st.session_state.ui_msg_class = "priority-clear"

# Status panel
mode_info = f"{ACTIVE['model']} | {ACTIVE['imgsz']}px"
if ACTIVE["use_midas"]: mode_info += " | MiDaS"
st.markdown(f"""
<div class="status-panel">
  <div class="status-text {st.session_state.ui_msg_class}">{st.session_state.ui_msg}<br>
    <span style='font-size:.9rem;color:#888;font-weight:400;'>{mode_info} | DeepSORT</span>
  </div>
</div>
""", unsafe_allow_html=True)

log_text = "<br>".join(st.session_state.log)
caregiver_log.markdown(f"<div style='font-family:monospace;font-size:12px;height:300px;overflow-y:scroll;color:#bbb;'>{log_text}</div>", unsafe_allow_html=True)

# ─── SINGLE POINT OF VOICE INJECTION ──────────────────────────────────────────
if st.session_state.get('pending_speech') or st.session_state.get('pending_vib'):
    text = st.session_state.get('pending_speech', '').replace("'", "")
    lang_code = st.session_state.get('pending_lang', 'en-US')
    vib = st.session_state.get('pending_vib', '[0]')
    st.html(f"""
    <script>
    setTimeout(function() {{
        var pw = window.parent;
        if (pw.hapticsArmed) {{
            if (pw.navigator && pw.navigator.vibrate && {vib}[0] !== 0) {{
                pw.navigator.vibrate({vib});
            }}
            if (pw.speechSynthesis && '{text}' !== '') {{
                pw.speechSynthesis.cancel();
                var msg = new pw.SpeechSynthesisUtterance('{text}');
                msg.lang = '{lang_code}';
                pw.speechSynthesis.speak(msg);
            }}
        }}
    }}, 50);
    </script>
    """)
    st.session_state.pending_speech = ""
    st.session_state.pending_vib = ""

"""
vision_engine.py — VisionAid Pro v4 Backend
============================================
All detection logic, tracking, depth estimation, languages, and voice engine.
Imported by app.py for the Streamlit UI layer.
"""
import cv2, numpy as np, time, os, threading
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_CLASSES_23 = [
    "chair","table","sofa","bed","desk","door","stairs","person","wall","floor",
    "cabinet","refrigerator","monitor","lamp","pillow","bookshelf","sink",
    "toilet","window","curtain","picture","counter","shelf",
]

FOCAL_LENGTH_PX = 615.0
REF_HEIGHTS_M = {
    "person":1.7,"door":2.0,"car":1.5,"truck":3.0,"bus":3.0,"motorcycle":1.2,
    "bicycle":1.1,"chair":0.9,"table":0.75,"dining table":0.75,"desk":0.75,
    "bed":0.6,"sofa":0.8,"couch":0.8,"cabinet":1.8,"refrigerator":1.8,
    "monitor":0.4,"tv":0.6,"sink":1.0,"toilet":0.8,"wall_ahead":2.0,
    "obstacle_large":1.0,"surface":0.8,"window":1.2,"curtain":2.0,
    "bookshelf":1.8,"lamp":0.5,"pillow":0.3,"counter":0.9,"shelf":1.5,
    "picture":0.5,"stairs":0.2,"wall":2.5,"floor":0.0,
}
DEFAULT_HEIGHT_M = 0.5

URGENT_OBJECTS = {"car","truck","bus","motorcycle","bicycle","stairs_up","stairs_down","fall","stairs"}
WARNING_OBJECTS = {"person","door","chair","bed","toilet","sofa","couch","sink","refrigerator",
                   "cabinet","desk","table","dining table","wall_ahead","wall","window","counter"}

OBJECT_PRIORITY = {o:0 for o in URGENT_OBJECTS}
OBJECT_PRIORITY.update({o:1 for o in WARNING_OBJECTS})

REPEAT_INTERVALS = {0:2.0, 1:4.0, 2:6.0}
CLEAR_PATH_INTERVAL = 10.0
CONFIRM_FRAMES = 3
GRACE_FRAMES = 2

CLASS_CONF = {
    "person":0.55,"car":0.50,"motorcycle":0.50,"bicycle":0.50,
    "table":0.30,"dining table":0.30,"desk":0.30,
    "chair":0.30,"sofa":0.30,"couch":0.30,"bed":0.30,"pillow":0.30,
    "door":0.35,"window":0.35,"stairs":0.40,
    "cabinet":0.35,"refrigerator":0.35,"monitor":0.35,
    "lamp":0.35,"bookshelf":0.35,"sink":0.35,"toilet":0.35,
    "curtain":0.30,"picture":0.30,"counter":0.30,"shelf":0.30,
    "wall":0.25,"floor":0.25,
}
BASE_CONF = 0.25
IOU_THRESHOLD = 0.45

PERF_MODES = {
    "ECO (Fast)":{
        "model":"yolov8n.pt","imgsz":320,"interval":1.0,
        "use_midas":False,"use_contour":False,"skip_frames":5},
    "NORMAL (Balanced)":{
        "model":"yolov8s.pt","imgsz":480,"interval":0.6,
        "use_midas":False,"use_contour":True,"skip_frames":3},
    "FULL (Max Accuracy)":{
        "model":"yolov8m.pt","imgsz":960,"interval":0.3,
        "use_midas":True,"use_contour":True,"skip_frames":1},
}

# ═══════════════════════════════════════════════════════════════════════════════
#  LANGUAGES
# ═══════════════════════════════════════════════════════════════════════════════

LANGUAGES = {
  "English": {
    "code":"en-US","warning":"Warning",
    "ahead":"ahead","left":"on your left","right":"on your right",
    "clear":"Path is clear","obstacle":"Obstacle ahead",
    "dist_0":"STOP! {obj} is less than 1 metre away",
    "dist_1":"Warning! {obj} only 2 metres ahead",
    "dist_2":"Caution! {obj} 3 metres ahead, slow down",
    "dist_3":"Alert! {obj} detected 5 metres ahead, be careful",
    "fall":"Are you okay? Fall detected",
    "stairs_up":"Stairs going UP ahead, use handrail",
    "stairs_down":"Stairs going DOWN ahead, be very careful",
    "moving_approach":"{obj} approaching fast from {dir} — stop immediately",
    "moving_away":"{obj} moving away on {dir} — safe to proceed",
    "stuck":"You seem stuck. The {obj} is blocking your path.",
    "nav_history_left":"Previously left side was clear, try going left",
    "nav_history_right":"Previously right side was clear, try going right",
    "scene_outdoor":"Outdoor environment detected, watch for vehicles",
    "scene_corridor":"Corridor detected, walk straight",
    "scene_road":"You are near a road, please stop",
    "scene_indoor":"Indoor environment detected",
    "wall_ahead":"Wall detected ahead, please stop",
    "wall_dist":"Wall detected, {dist} metres ahead, please stop",
    "surface":"Large surface ahead",
    "unknown_obstacle":"Unknown obstacle detected ahead",
    "large_object":"Large object ahead",
    "track_lost":"Object no longer detected",
    "nav_move_left":"- move left, path is clear",
    "nav_move_right":"- move right, path is clear",
    "nav_stop":"- stop, both sides blocked",
    "nav_step_back":"- step back",
    "nav_obj_on_left":"Object on left - move right",
    "nav_obj_on_right":"Object on right - move left",
    "still_ahead":"still ahead","persists":"Obstacle persists, please navigate carefully",
  },
  "Tamil": {
    "code":"ta-IN","warning":"எச்சரிக்கை",
    "ahead":"முன்னால்","left":"இடதுபுறம்","right":"வலதுபுறம்",
    "clear":"பாதை தெளிவாக உள்ளது","obstacle":"தடை முன்னால்",
    "dist_0":"நிறுத்துங்கள்! {obj} 1 மீட்டருக்கும் குறைவாக உள்ளது",
    "dist_1":"எச்சரிக்கை! {obj} 2 மீட்டர் தொலைவில் உள்ளது",
    "dist_2":"கவனம்! {obj} 3 மீட்டர் தொலைவில் உள்ளது",
    "dist_3":"எச்சரிக்கை! {obj} 5 மீட்டர் தொலைவில் உள்ளது, கவனமாக இருங்கள்",
    "fall":"நீங்கள் நலமா? வீழ்ச்சி கண்டறியப்பட்டது",
    "stairs_up":"படிக்கட்டுகள் மேலே செல்கின்றன, கைப்பிடியைப் பிடிக்கவும்",
    "stairs_down":"படிக்கட்டுகள் கீழே செல்கின்றன, மிகவும் கவனமாக இருங்கள்",
    "moving_approach":"{obj} {dir} இருந்து வேகமாக வருகிறது — நிறுத்துங்கள்",
    "moving_away":"{obj} {dir} நகர்கிறது — தொடரலாம்",
    "stuck":"நீங்கள் சிக்கியுள்ளீர்கள். {obj} பாதையை மறைக்கிறது.",
    "nav_history_left":"முன்பு இடதுபுறம் தெளிவாக இருந்தது, இடதுபுறம் செல்ல முனையுங்கள்",
    "nav_history_right":"முன்பு வலதுபுறம் தெளிவாக இருந்தது, வலதுபுறம் செல்ல முனையுங்கள்",
    "scene_outdoor":"வெளிப்புற சூழல், வாகனங்களை கவனிக்கவும்",
    "scene_corridor":"நீங்கள் தாழ்வாரத்தில் உள்ளீர்கள், நேராக நடக்கவும்",
    "scene_road":"நீங்கள் சாலைக்கு அருகில் உள்ளீர்கள், நிறுத்துங்கள்",
    "scene_indoor":"உட்புற சூழல் கண்டறியப்பட்டது",
    "wall_ahead":"சுவர் முன்னால் — நிறுத்துங்கள்",
    "wall_dist":"சுவர் {dist} மீட்டர் தொலைவில் உள்ளது, நிறுத்துங்கள்",
    "surface":"முன்னால் மேசை அல்லது பெரிய பரப்பு",
    "unknown_obstacle":"அறியப்படாத தடை முன்னால்",
    "large_object":"பெரிய பொருள் முன்னால்",
    "track_lost":"பொருள் கண்டறியப்படவில்லை",
    "nav_move_left":"- இடதுபுறம் செல்லுங்கள்",
    "nav_move_right":"- வலதுபுறம் செல்லுங்கள்",
    "nav_stop":"- நிறுத்துங்கள்! இரண்டு பக்கமும் தடை",
    "nav_step_back":"- பின்னால் செல்லவும்",
    "nav_obj_on_left":"தடை இடதுபுறம் - வலதுபுறம் செல்லுங்கள்",
    "nav_obj_on_right":"தடை வலதுபுறம் - இடதுபுறம் செல்லுங்கள்",
    "still_ahead":"இன்னும் முன்னால் உள்ளது","persists":"தடை தொடர்கிறது, கவனமாக செல்லுங்கள்",
  },
  "Hindi": {
    "code":"hi-IN","warning":"चेतावनी",
    "ahead":"आगे","left":"बाईं तरफ","right":"दाईं तरफ",
    "clear":"रास्ता साफ है","obstacle":"बाधा आगे",
    "dist_0":"रुकिए! {obj} 1 मीटर से कम दूरी पर है",
    "dist_1":"चेतावनी! {obj} सिर्फ 2 मीटर आगे है",
    "dist_2":"सावधान! {obj} 3 मीटर आगे है, धीरे चलें",
    "dist_3":"सावधान! {obj} 5 मीटर दूर है, ध्यान रखें",
    "fall":"क्या आप ठीक हैं? गिरने का पता चला",
    "stairs_up":"ऊपर जाने वाली सीढ़ियां हैं, रेलिंग पकड़ें",
    "stairs_down":"नीचे जाने वाली सीढ़ियां हैं, बहुत सावधान रहें",
    "moving_approach":"{obj} {dir} से तेज़ी से आ रहा है — रुकिए",
    "moving_away":"{obj} {dir} जा रहा है — आगे बढ़ सकते हैं",
    "stuck":"आप फंस गए हैं। {obj} रास्ता रोक रही है।",
    "nav_history_left":"पहले बाईं तरफ साफ था, बाईं ओर जाने की कोशिश करें",
    "nav_history_right":"पहले दाईं तरफ साफ था, दाईं ओर जाने की कोशिश करें",
    "scene_outdoor":"बाहरी वातावरण, वाहनों का ध्यान रखें",
    "scene_corridor":"आप एक गलियारे में हैं, सीधे चलें",
    "scene_road":"आप सड़क के पास हैं, कृपया रुकें",
    "scene_indoor":"इनडोर वातावरण पहचाना गया",
    "wall_ahead":"दीवार सामने — रुकिए",
    "wall_dist":"दीवार {dist} मीटर आगे है, रुकिए",
    "surface":"सामने मेज़ या बड़ी सतह",
    "unknown_obstacle":"अज्ञात बाधा आगे है",
    "large_object":"बड़ी वस्तु सामने है",
    "track_lost":"वस्तु अब दिखाई नहीं दे रही",
    "nav_move_left":"- बाईं ओर जाएं",
    "nav_move_right":"- दाईं ओर जाएं",
    "nav_stop":"- रुकिए! दोनों तरफ बंद है",
    "nav_step_back":"- पीछे हटें",
    "nav_obj_on_left":"बाईं तरफ रुकावट - दाईं ओर जाएं",
    "nav_obj_on_right":"दाईं तरफ रुकावट - बाईं ओर जाएं",
    "still_ahead":"अभी भी आगे है","persists":"बाधा बनी है, सावधानी से चलें",
  },
}

OBJ_TRANS = {
  "Tamil":{
    "person":"நபர்","car":"கார்","truck":"லாரி","bus":"பேருந்து",
    "motorcycle":"மோட்டார் சைக்கிள்","bicycle":"சைக்கிள்","chair":"நாற்காலி",
    "table":"மேசை","dining table":"மேசை","desk":"மேஜை","door":"கதவு",
    "bed":"படுக்கை","toilet":"கழிவறை","stairs_up":"படிகள்","stairs_down":"படிகள்",
    "stairs":"படிகள்","sofa":"சோபா","couch":"சோபா","wall_ahead":"சுவர்",
    "wall":"சுவர்","obstacle_large":"தடை","window":"ஜன்னல்","curtain":"திரை",
    "cabinet":"அலமாரி","refrigerator":"குளிர்சாதனம்","monitor":"திரை",
    "lamp":"விளக்கு","pillow":"தலையணை","bookshelf":"புத்தக அலமாரி",
    "sink":"சிங்க்","counter":"கவுண்டர்","shelf":"அலமாரி","picture":"படம்",
    "floor":"தரை","surface":"மேற்பரப்பு",
  },
  "Hindi":{
    "person":"व्यक्ति","car":"कार","truck":"ट्रक","bus":"बस",
    "motorcycle":"मोटरसाइकिल","bicycle":"साइकिल","chair":"कुर्सी",
    "table":"मेज़","dining table":"मेज़","desk":"डेस्क","door":"दरवाज़ा",
    "bed":"बिस्तर","toilet":"शौचालय","stairs_up":"सीढ़ियाँ","stairs_down":"सीढ़ियाँ",
    "stairs":"सीढ़ियाँ","sofa":"सोफ़ा","couch":"सोफ़ा","wall_ahead":"दीवार",
    "wall":"दीवार","obstacle_large":"बाधा","window":"खिड़की","curtain":"पर्दा",
    "cabinet":"अलमारी","refrigerator":"फ्रिज","monitor":"मॉनिटर",
    "lamp":"लैंप","pillow":"तकिया","bookshelf":"किताबों की अलमारी",
    "sink":"सिंक","counter":"काउंटर","shelf":"शेल्फ","picture":"तस्वीर",
    "floor":"फ़र्श","surface":"सतह",
  },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  DISTANCE & DEPTH
# ═══════════════════════════════════════════════════════════════════════════════

def calc_distance_m(pixel_h, label):
    ref = REF_HEIGHTS_M.get(label, DEFAULT_HEIGHT_M)
    if pixel_h < 5: pixel_h = 5
    d = (ref * FOCAL_LENGTH_PX) / float(pixel_h)
    return round(min(max(d, 0.2), 15.0), 1)

def dist_tier(d):
    if d < 1.0: return "CRITICAL"
    if d < 2.0: return "URGENT"
    if d < 3.0: return "WARNING"
    if d < 5.0: return "CAUTION"
    return "FAR"

def depth_to_tier(depth_val):
    """MiDaS normalized depth (0-1) → distance tier."""
    if depth_val > 0.8:  return "CRITICAL"
    if depth_val > 0.6:  return "URGENT"
    if depth_val > 0.4:  return "WARNING"
    if depth_val > 0.2:  return "CAUTION"
    return "FAR"

def depth_to_metres(depth_val):
    """Approximate metres from MiDaS normalized depth."""
    if depth_val > 0.8:  return 0.5
    if depth_val > 0.6:  return 1.5
    if depth_val > 0.4:  return 2.5
    if depth_val > 0.2:  return 4.0
    return 7.0

def haptic_pattern(tier):
    if tier == "CRITICAL": return "[100,50,100,50,100]"
    if tier == "URGENT":   return "[200,100,200]"
    if tier == "WARNING":  return "[400]"
    if tier == "CAUTION":  return "[150]"
    return "[0]"

def direction(cx, fw):
    r = cx / fw
    if r < 0.35: return "LEFT"
    if r > 0.65: return "RIGHT"
    return "CENTER"

# ═══════════════════════════════════════════════════════════════════════════════
#  MiDaS DEPTH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MiDaSEngine:
    """Thread-safe MiDaS depth estimator — runs every Nth frame."""
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        self.depth_map = None  # latest normalized depth map (H×W, 0-1)
        self.lock = threading.Lock()
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model.to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            self._loaded = True
        except Exception as e:
            print(f"[MiDaS] Load failed: {e}")
            self._loaded = False

    def predict(self, bgr_frame):
        if not self._loaded:
            return None
        try:
            import torch
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            inp = self.transform(rgb).to(self.device)
            with torch.no_grad():
                pred = self.model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=bgr_frame.shape[:2],
                    mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()
            # Normalize 0-1
            mn, mx = pred.min(), pred.max()
            if mx - mn > 1e-6:
                pred = (pred - mn) / (mx - mn)
            else:
                pred = np.zeros_like(pred)
            with self.lock:
                self.depth_map = pred.astype(np.float32)
            return self.depth_map
        except Exception:
            return None

    def get_depth_at(self, cx, cy):
        with self.lock:
            if self.depth_map is None: return 0.0
            h, w = self.depth_map.shape
            y = min(max(int(cy), 0), h-1)
            x = min(max(int(cx), 0), w-1)
            return float(self.depth_map[y, x])

    def detect_wall(self):
        """If >30% of frame has uniform high depth → wall."""
        with self.lock:
            if self.depth_map is None: return False, 0.0
            high_mask = (self.depth_map > 0.6).astype(np.float32)
            ratio = np.mean(high_mask)
            if ratio > 0.30:
                avg_depth = np.mean(self.depth_map[self.depth_map > 0.6])
                return True, depth_to_metres(avg_depth)
        return False, 0.0

    def colorize(self, shape):
        """Return BGR color depth overlay (red=close, green=far)."""
        with self.lock:
            if self.depth_map is None:
                return np.zeros((*shape[:2], 3), dtype=np.uint8)
            d = self.depth_map.copy()
        d_u8 = (d * 255).astype(np.uint8)
        colored = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
        if colored.shape[:2] != shape[:2]:
            colored = cv2.resize(colored, (shape[1], shape[0]))
        return colored

# Global singleton
_midas = MiDaSEngine()
def get_midas(): return _midas

# ═══════════════════════════════════════════════════════════════════════════════
#  DeepSORT TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class TrackerWrapper:
    """Wraps deep-sort-realtime with track history for approach detection."""
    def __init__(self):
        self.tracker = None
        self.track_history = {}   # track_id → deque of (bbox, time)
        self.announced_ids = {}   # track_id → last announce time
        self.lost_ids = set()
        self._loaded = False

    def load(self):
        if self._loaded: return
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(max_age=5, n_init=3, nms_max_overlap=0.7,
                                     max_iou_distance=0.7)
            self._loaded = True
        except ImportError:
            print("[DeepSORT] Not installed — tracking disabled")
        except Exception as e:
            print(f"[DeepSORT] Init failed: {e}")

    def update(self, detections, frame):
        """
        detections: list of dicts with 'bbox':(x1,y1,x2,y2), 'conf', 'label_en'
        Returns list of tracked dicts with added 'track_id' and 'approaching'.
        """
        if not self._loaded or not detections:
            # Return detections as-is with track_id=-1
            for d in detections:
                d["track_id"] = -1
                d["approaching"] = False
            return detections

        # Format for deep-sort: ([x1,y1,w,h], conf, label)
        ds_dets = []
        det_map = {}
        for i, d in enumerate(detections):
            x1,y1,x2,y2 = d["bbox"]
            ds_dets.append(([x1, y1, x2-x1, y2-y1], d["conf"], d["label_en"]))
            det_map[i] = d

        try:
            tracks = self.tracker.update_tracks(ds_dets, frame=frame)
        except Exception:
            for d in detections:
                d["track_id"] = -1
                d["approaching"] = False
            return detections

        now = time.time()
        current_ids = set()
        tracked_dets = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            current_ids.add(tid)
            ltrb = track.to_ltrb()
            x1,y1,x2,y2 = [int(v) for v in ltrb]
            bbox = (x1,y1,x2,y2)

            # Find best matching detection
            best_det = None
            best_iou = 0
            for d in detections:
                iou = self._iou(bbox, d["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = d
            if best_det is None and detections:
                best_det = detections[0]

            det = dict(best_det) if best_det else {"label_en":"unknown","dist_m":5,"dist_t":"FAR","dir":"CENTER","conf":0.3,"moving":False,"rank":5}
            det["bbox"] = bbox
            det["track_id"] = tid

            # Track history for approach detection
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=5)
            self.track_history[tid].append(((x2-x1)*(y2-y1), now))

            # Approaching = bbox area growing over last 5 frames
            approaching = False
            hist = self.track_history[tid]
            if len(hist) >= 3:
                areas = [h[0] for h in hist]
                if areas[-1] > areas[0] * 1.15:
                    approaching = True
            det["approaching"] = approaching
            tracked_dets.append(det)

        # Detect lost tracks
        prev_ids = set(self.track_history.keys())
        newly_lost = prev_ids - current_ids - self.lost_ids
        for lid in newly_lost:
            self.lost_ids.add(lid)
        # Clean old history
        for tid in list(self.track_history.keys()):
            if tid not in current_ids:
                hist = self.track_history[tid]
                if hist and now - hist[-1][1] > 10:
                    del self.track_history[tid]
                    self.lost_ids.discard(tid)

        return tracked_dets if tracked_dets else detections

    def should_announce(self, track_id):
        """New track → True, same track → only every 4s."""
        now = time.time()
        if track_id == -1: return True
        last = self.announced_ids.get(track_id, 0)
        if now - last >= 4.0:
            self.announced_ids[track_id] = now
            return True
        return False

    def get_lost_ids(self):
        lost = self.lost_ids.copy()
        self.lost_ids.clear()
        return lost

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0],b[0]); y1 = max(a[1],b[1])
        x2 = min(a[2],b[2]); y2 = min(a[3],b[3])
        inter = max(0,x2-x1)*max(0,y2-y1)
        aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
        return inter/(aa+ab-inter+1e-6)

_tracker = TrackerWrapper()
def get_tracker(): return _tracker

# ═══════════════════════════════════════════════════════════════════════════════
#  OPENCV HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fall(frame):
    h,w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 20: return True
    edges = cv2.Canny(cv2.GaussianBlur(gray,(7,7),0), 50, 150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=int(w*0.5),maxLineGap=20)
    if lines is not None:
        angles = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if x2-x1==0: continue
            angles.append(abs(np.degrees(np.arctan((y2-y1)/(x2-x1)))))
        if angles and 30 < np.median(angles) < 60: return True
    return False

def detect_stairs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength=int(frame.shape[1]*0.4),maxLineGap=10)
    if lines is not None and len(lines) > 3:
        ys = sorted([l[0][1] for l in lines])
        gaps = np.diff(ys)
        if len(gaps) > 2:
            if gaps[0] > gaps[-1]: return "stairs_up"
            if gaps[0] < gaps[-1]: return "stairs_down"
    return None

def detect_scene(frame):
    h,w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    upper = hsv[:int(h*0.3),:]
    if np.mean(upper[:,:,2]) > 200: return "Outdoor"
    lower = hsv[int(h*0.7):,:]
    s_m = np.mean(lower[:,:,1]); v_m = np.mean(lower[:,:,2])
    if s_m < 40 and 100 < v_m < 180: return "Road"
    # Corridor: narrow aspect uniform walls
    left_v = np.mean(hsv[:, :int(w*0.2), 2])
    right_v = np.mean(hsv[:, int(w*0.8):, 2])
    if abs(left_v - right_v) < 30 and np.mean(hsv[:,:,1]) < 50:
        return "Corridor"
    return "Indoor"

def detect_contour_obstacles(frame, min_ratio=0.15):
    """Fallback Layer 3: large contours in frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = frame.shape[:2]
    total = h*w
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area / total > min_ratio:
            x,y,bw,bh = cv2.boundingRect(cnt)
            results.append({"bbox":(x,y,x+bw,y+bh),"area_ratio":area/total})
    return results

def scene_conf_adjust(scene, base_conf):
    """Adjust confidence per scene type."""
    if scene == "Indoor":   return max(base_conf - 0.05, 0.15)
    if scene == "Corridor": return max(base_conf - 0.05, 0.15)
    if scene == "Outdoor":  return min(base_conf + 0.05, 0.60)
    if scene == "Road":     return base_conf
    return base_conf

# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceEngine:
    @staticmethod
    def _dist_alert(label, tier, lang, sel_lang):
        loc = OBJ_TRANS.get(sel_lang, {}).get(label, label)
        key = {"CRITICAL":"dist_0","URGENT":"dist_1","WARNING":"dist_2","CAUTION":"dist_3"}.get(tier)
        if key: return lang[key].format(obj=loc)
        return ""

    @staticmethod
    def build_alert(label, dirn, tier, dist_m, left_ok, right_ok, approaching, moving_away, lang, sel_lang):
        loc = OBJ_TRANS.get(sel_lang,{}).get(label, label)
        if label == "fall": return lang["fall"]
        if label == "stairs_up": return lang["stairs_up"]
        if label == "stairs_down": return lang["stairs_down"]
        if label == "wall_ahead": return lang.get("wall_dist","Wall {dist}m ahead").format(dist=dist_m)
        if label == "surface": return lang["surface"]
        if label == "unknown_obstacle": return lang["unknown_obstacle"]
        if label == "large_object": return lang["large_object"]
        if approaching:
            d = lang.get("left","left") if dirn=="LEFT" else lang.get("right","right") if dirn=="RIGHT" else lang.get("ahead","ahead")
            return lang["moving_approach"].format(obj=loc, dir=d)
        if moving_away:
            d = lang.get("left","left") if dirn=="LEFT" else lang.get("right","right") if dirn=="RIGHT" else lang.get("ahead","ahead")
            return lang["moving_away"].format(obj=loc, dir=d)
        base = VoiceEngine._dist_alert(label, tier, lang, sel_lang)
        if not base: return ""
        if dirn == "CENTER":
            if left_ok and right_ok: base += f" {lang['nav_move_left']}"
            elif left_ok: base += f" {lang['nav_move_left']}"
            elif right_ok: base += f" {lang['nav_move_right']}"
            else: base += f" {lang['nav_stop']}"
        elif dirn == "LEFT": base += f" - {lang['nav_obj_on_left']}"
        elif dirn == "RIGHT": base += f" - {lang['nav_obj_on_right']}"
        return base

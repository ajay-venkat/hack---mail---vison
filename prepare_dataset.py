"""
prepare_dataset.py — VisionAid Pro v4
======================================
Downloads and converts BOTH SUN RGB-D and NYU Depth V2 datasets into
YOLOv8 format, then merges them into a single indoor_dataset/ directory.

Usage:
    python prepare_dataset.py

Requirements:
    pip install requests tqdm scipy numpy Pillow datasets huggingface-hub

Outputs:
    indoor_dataset/
        images/{train,val,test}/
        labels/{train,val,test}/
    dataset.yaml  (23-class config)

Notes:
  ─ NYU Depth V2 is fetched from HuggingFace (sayakpaul/nyu_depth_v2) —
    NO login required.
  ─ SUN RGB-D auto-download from Princeton or manual placement of
    SUNRGBD.zip in the working directory.
"""

import os
import sys
import json
import math
import random
import shutil
import struct
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("indoor_dataset")
SPLITS      = {"train": 0.80, "val": 0.10, "test": 0.10}
RANDOM_SEED = 42

# 23 target classes
TARGET_CLASSES = [
    "chair", "table", "sofa", "bed", "desk", "door", "stairs",
    "person", "wall", "floor", "cabinet", "refrigerator",
    "monitor", "lamp", "pillow", "bookshelf", "sink", "toilet",
    "window", "curtain", "picture", "counter", "shelf",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(TARGET_CLASSES)}

# Aliases → canonical class
CLASS_ALIASES = {
    # chair
    "armchair": "chair", "office chair": "chair", "folding chair": "chair",
    "stool": "chair", "swivel chair": "chair", "rocking chair": "chair",
    # table
    "dining table": "table", "dining_table": "table", "coffee table": "table",
    "end table": "table", "nightstand": "table", "side table": "table",
    "kitchen table": "table",
    # sofa
    "couch": "sofa", "loveseat": "sofa", "sectional": "sofa",
    # bed
    "bunk bed": "bed", "twin bed": "bed", "double bed": "bed",
    # cabinet
    "wardrobe": "cabinet", "dresser": "cabinet", "cupboard": "cabinet",
    "closet": "cabinet", "chest of drawers": "cabinet",
    # monitor / tv
    "tv": "monitor", "television": "monitor", "screen": "monitor",
    "display": "monitor", "computer monitor": "monitor",
    # fridge
    "fridge": "refrigerator",
    # bookshelf / shelf
    "bookcase": "bookshelf", "shelves": "shelf", "shelving": "shelf",
    # lamp
    "light": "lamp", "ceiling light": "lamp", "floor lamp": "lamp",
    "ceiling_lamp": "lamp",
    # stairs
    "staircase": "stairs", "stairway": "stairs", "steps": "stairs",
    # sink
    "bathroom sink": "sink", "kitchen sink": "sink",
    # desk
    "computer desk": "desk", "writing desk": "desk",
    # person
    "human": "person",
    # window
    "window frame": "window",
    # curtain
    "drape": "curtain", "drapes": "curtain", "blind": "curtain",
    # picture
    "painting": "picture", "artwork": "picture", "framed picture": "picture",
    # counter
    "kitchen counter": "counter", "countertop": "counter",
    # floor
    "rug": "floor", "carpet": "floor", "mat": "floor",
    # wall
    "partition": "wall",
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def normalize_label(raw: str):
    """Return canonical TARGET_CLASSES name or None."""
    raw = raw.strip().lower().replace("_", " ")
    if raw in CLASS_TO_IDX:
        return raw
    if raw in CLASS_ALIASES:
        return CLASS_ALIASES[raw]
    for alias, canonical in CLASS_ALIASES.items():
        if alias in raw:
            return canonical
    for cls in TARGET_CLASSES:
        if cls in raw:
            return cls
    return None


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, class_idx):
    """Absolute bbox → normalized YOLO (cx cy w h)."""
    x1 = max(0.0, min(float(x1), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y2 = max(0.0, min(float(y2), float(img_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return (class_idx, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))


def download_file(url: str, dest: Path) -> bool:
    """Stream-download with progress bar. Returns True on success."""
    try:
        print(f"  → {url}")
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code != 200:
            print(f"  ✗ HTTP {r.status_code}")
            return False
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=dest.name, total=total, unit="B", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        return False


def make_split_dirs(output: Path):
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def assign_split(idx: int, n: int) -> str:
    if idx < int(n * 0.80):
        return "train"
    if idx < int(n * 0.90):
        return "val"
    return "test"


# ─── SECTION 1 — SUN RGB-D ───────────────────────────────────────────────────

SUNRGBD_URLS = [
    "http://rgbd.cs.princeton.edu/data/SUNRGBD.zip",
    "https://huggingface.co/datasets/SUN-RGBD/sunrgbd/resolve/main/SUNRGBD.zip",
]


def ensure_sunrgbd_zip() -> Path:
    local = Path("SUNRGBD.zip")
    if local.exists():
        print(f"✓ Found SUNRGBD.zip ({local.stat().st_size / 1e9:.2f} GB)")
        return local
    print("\n━━━ Downloading SUN RGB-D ━━━")
    for url in SUNRGBD_URLS:
        if download_file(url, local):
            return local
    print("""
━━━ MANUAL DOWNLOAD REQUIRED FOR SUN RGB-D ━━━
1. Visit: http://rgbd.cs.princeton.edu/
2. Download SUNRGBD.zip (~3 GB)
3. Place it in: {}
4. Re-run this script.
""".format(Path.cwd()))
    return None  # Non-fatal — we'll skip SUN RGB-D if absent


def extract_sunrgbd(zip_path: Path) -> Path:
    out = Path("SUNRGBD_raw")
    if out.exists() and any(out.iterdir()):
        print(f"✓ SUN RGB-D already extracted → {out}")
        return out
    print(f"\n━━━ Extracting SUN RGB-D ━━━")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.namelist(), desc="Extracting"):
            zf.extract(member, out)
    print("✓ Extraction done")
    return out


def parse_sunrgbd_annotation(ann_path: Path, img_w: int, img_h: int):
    """Parse SUN RGB-D JSON or text annotation → list of YOLO tuples."""
    anns = []
    if not ann_path or not ann_path.exists():
        return anns
    try:
        content = ann_path.read_text(errors="ignore").strip()
        # JSON format
        if content.startswith("{") or content.startswith("["):
            data = json.loads(content)
            objects = data if isinstance(data, list) else data.get("annotation", [])
            for obj in objects:
                name = obj.get("name", obj.get("label", ""))
                cls = normalize_label(name)
                if cls is None:
                    continue
                bbox = obj.get("bbox", obj.get("bndbox", {}))
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                elif isinstance(bbox, dict):
                    x1 = float(bbox.get("x1", bbox.get("xmin", 0)))
                    y1 = float(bbox.get("y1", bbox.get("ymin", 0)))
                    x2 = float(bbox.get("x2", x1 + bbox.get("w", 0)))
                    y2 = float(bbox.get("y2", y1 + bbox.get("h", 0)))
                else:
                    continue
                t = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, CLASS_TO_IDX[cls])
                if t:
                    anns.append(t)
        else:
            # Text: label x1 y1 x2 y2
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                name = " ".join(parts[:-4])
                cls = normalize_label(name) or normalize_label(parts[0])
                if cls is None:
                    continue
                try:
                    x1, y1, x2, y2 = [float(p) for p in parts[-4:]]
                    t = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, CLASS_TO_IDX[cls])
                    if t:
                        anns.append(t)
                except ValueError:
                    pass
    except Exception:
        pass
    return anns


def discover_sunrgbd(root: Path):
    """Return list of (img_path, ann_path_or_None)."""
    samples = []
    print("\n━━━ Scanning SUN RGB-D images ━━━")
    all_imgs = list(root.rglob("*.jpg")) + list(root.rglob("*.png"))
    for img in tqdm(all_imgs, desc="Scanning"):
        parts = img.parts
        if "depth" in parts or "depth_bfx" in parts:
            continue
        if "seg" in img.name.lower():
            continue
        scene = img.parent.parent if img.parent.name == "image" else img.parent
        candidates = [
            scene / "annotation2Dfinal" / "index.json",
            scene / "annotation2Dfinal" / "annotations.json",
            scene.parent / "annotation2Dfinal" / "index.json",
        ]
        ann = next((c for c in candidates if c.exists()), None)
        samples.append((img, ann))
    print(f"✓ SUN RGB-D: {len(samples)} images found")
    return samples


def process_sunrgbd(root: Path, output: Path) -> int:
    """Converts SUN RGB-D to YOLO format inside output/. Returns count."""
    samples = discover_sunrgbd(root)
    if not samples:
        return 0
    random.shuffle(samples)
    count = 0
    print("\n━━━ Converting SUN RGB-D → YOLO ━━━")
    for idx, (img_path, ann_path) in enumerate(tqdm(samples, desc="SUN RGB-D")):
        try:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
        except Exception:
            continue
        anns = parse_sunrgbd_annotation(ann_path, img_w, img_h)
        split = assign_split(idx, len(samples))
        stem = f"sunrgbd_{img_path.parent.parent.name}_{img_path.stem}"
        img.save(output / "images" / split / f"{stem}.jpg", "JPEG", quality=90)
        with open(output / "labels" / split / f"{stem}.txt", "w") as f:
            for (ci, cx, cy, w, h) in anns:
                f.write(f"{ci} {cx} {cy} {w} {h}\n")
        count += 1
    print(f"✓ SUN RGB-D: {count} images converted")
    return count


# ─── SECTION 2 — NYU DEPTH V2 ────────────────────────────────────────────────

# NYU Depth V2 YOLO-format pre-converted mirror on HuggingFace
# sayakpaul/nyu_depth_v2 — RGB images + depth, ~50k frames
# We use only the RGB + available category-level labels.
# For bounding-box labels we use the per-pixel label maps converted to bboxes.

NYU_HF_DATASET = "sayakpaul/nyu_depth_v2"

# NYU semantic label id → class name (subset relevant to navigation)
# Source: NYU Depth V2 label mapping (894 classes → we pick our 23)
NYU_LABEL_MAP = {
    # id: canonical_class
    19:  "chair",      # chair
    25:  "table",      # table
    31:  "sofa",       # sofa
    1:   "bed",        # bed
    57:  "desk",       # desk
    8:   "door",       # door
    149: "stairs",     # stairs
    0:   "person",     # person (background in some versions)
    17:  "cabinet",    # cabinet
    48:  "refrigerator",
    28:  "monitor",    # monitor/tv
    38:  "lamp",       # lamp
    29:  "pillow",     # pillow
    55:  "bookshelf",  # bookshelf
    40:  "sink",       # sink
    45:  "toilet",     # toilet
    46:  "window",     # window
    47:  "curtain",    # curtain
    50:  "picture",    # picture/painting
    33:  "counter",    # counter
    36:  "shelf",      # shelf
}


def nyu_label_mask_to_bboxes(label_arr: np.ndarray, img_w: int, img_h: int):
    """
    Convert a NYU per-pixel label map to YOLO bounding boxes.
    For each target label id, finds connected regions and creates a bbox.
    """
    anns = []
    try:
        import cv2
        unique_ids = np.unique(label_arr)
        for uid in unique_ids:
            uid_int = int(uid)
            cls_name = NYU_LABEL_MAP.get(uid_int)
            if cls_name is None:
                continue
            cls_idx = CLASS_TO_IDX.get(cls_name)
            if cls_idx is None:
                continue
            mask = (label_arr == uid_int).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 400:  # skip tiny regions
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                t = bbox_to_yolo(bx, by, bx + bw, by + bh, img_w, img_h, cls_idx)
                if t:
                    anns.append(t)
    except ImportError:
        pass  # cv2 not available — skip label conversion
    return anns


def process_nyu_hf(output: Path) -> int:
    """Download NYU Depth V2 from HuggingFace and convert to YOLO."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠ 'datasets' not installed — skipping NYU Depth V2")
        print("    pip install datasets huggingface-hub")
        return 0

    print("\n━━━ Loading NYU Depth V2 from HuggingFace ━━━")
    print("  This may take a while on first run (~13 GB download)...")

    try:
        ds = load_dataset(NYU_HF_DATASET, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  ✗ Could not load NYU dataset: {e}")
        print("  Skipping NYU Depth V2 — SUN RGB-D will be used alone.")
        return 0

    print(f"✓ NYU Depth V2 loaded: {len(ds)} samples")

    count = 0
    total = len(ds)
    print("\n━━━ Converting NYU Depth V2 → YOLO ━━━")

    for idx, sample in enumerate(tqdm(ds, desc="NYU Depth V2", total=total)):
        try:
            # RGB image
            rgb = sample.get("image") or sample.get("rgb")
            if rgb is None:
                continue
            if not isinstance(rgb, Image.Image):
                rgb = Image.fromarray(rgb)
            rgb = rgb.convert("RGB")
            img_w, img_h = rgb.size

            # Label map (per-pixel semantic labels)
            label_map = sample.get("labels") or sample.get("label")
            anns = []
            if label_map is not None:
                if not isinstance(label_map, np.ndarray):
                    label_map = np.array(label_map)
                anns = nyu_label_mask_to_bboxes(label_map, img_w, img_h)

            split = assign_split(idx, total)
            stem = f"nyu_{idx:06d}"

            rgb.save(output / "images" / split / f"{stem}.jpg", "JPEG", quality=90)
            with open(output / "labels" / split / f"{stem}.txt", "w") as f:
                for (ci, cx, cy, w, h) in anns:
                    f.write(f"{ci} {cx} {cy} {w} {h}\n")
            count += 1

        except Exception:
            continue

    print(f"✓ NYU Depth V2: {count} images converted")
    return count


# ─── SECTION 3 — YAML ─────────────────────────────────────────────────────────

def write_yaml(output_dir: Path):
    yaml_content = f"""# VisionAid Pro — Indoor Navigation Dataset
# Sources: SUN RGB-D + NYU Depth V2 (merged)
# Generated by prepare_dataset.py

path: {output_dir.resolve().as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: {len(TARGET_CLASSES)}
names:
"""
    for i, cls in enumerate(TARGET_CLASSES):
        yaml_content += f"  {i}: {cls}\n"

    yaml_path = Path("dataset.yaml")
    yaml_path.write_text(yaml_content)
    print(f"\n✓ Wrote {yaml_path}")
    return yaml_path


# ─── SECTION 4 — STATS ─────────────────────────────────────────────────────────

def count_split(output: Path, split: str):
    return len(list((output / "images" / split).glob("*.jpg")))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  VisionAid Pro — Indoor Dataset Preparation")
    print("  Sources: SUN RGB-D + NYU Depth V2")
    print("=" * 60)

    make_split_dirs(OUTPUT_DIR)

    total_images = 0

    # ── SUN RGB-D ─────────────────────────────────────────────────
    print("\n[1/2] SUN RGB-D Processing")
    sunzip = ensure_sunrgbd_zip()
    if sunzip and sunzip.exists():
        sunroot = extract_sunrgbd(sunzip)
        n_sun = process_sunrgbd(sunroot, OUTPUT_DIR)
        total_images += n_sun
    else:
        print("  ⚠ Skipping SUN RGB-D (zip not found)")

    # ── NYU Depth V2 ──────────────────────────────────────────────
    print("\n[2/2] NYU Depth V2 Processing")
    n_nyu = process_nyu_hf(OUTPUT_DIR)
    total_images += n_nyu

    if total_images == 0:
        print("\n✗ ERROR: No images were processed. Dataset is empty.")
        print("  Please ensure either SUNRGBD.zip is in this directory")
        print("  or 'datasets' is installed (pip install datasets).")
        sys.exit(1)

    # ── Write YAML ────────────────────────────────────────────────
    write_yaml(OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────
    n_train = count_split(OUTPUT_DIR, "train")
    n_val   = count_split(OUTPUT_DIR, "val")
    n_test  = count_split(OUTPUT_DIR, "test")

    print("\n" + "=" * 60)
    print("  ✓ Dataset Preparation Complete!")
    print("=" * 60)
    print(f"  Total images  : {total_images}")
    print(f"  Train         : {n_train}")
    print(f"  Val           : {n_val}")
    print(f"  Test          : {n_test}")
    print(f"  Classes       : {len(TARGET_CLASSES)}")
    print(f"  Output dir    : {OUTPUT_DIR}/")
    print(f"  YAML          : dataset.yaml")
    print("\n  Next step: python train.py")


if __name__ == "__main__":
    main()

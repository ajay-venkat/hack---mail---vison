"""
prepare_dataset.py
==================
Downloads and converts the SUN RGB-D dataset into YOLOv8 format.

Usage:
    python prepare_dataset.py

Requirements:
    pip install requests tqdm scipy numpy Pillow

Note: SUN RGB-D requires a direct download from http://rgbd.cs.princeton.edu/
      Due to login-wall restrictions, this script supports two modes:
      1. AUTO: Download via direct URL if available (Princeton mirror)
      2. MANUAL: If auto-download fails, script will guide you to download manually
                 and place the zip at ./SUNRGBD.zip

The script will:
  1. Download / locate the dataset
  2. Parse 2D bounding box annotations from the .mat / text annotation files
  3. Filter to 18 indoor obstacle classes
  4. Convert to YOLO txt format (normalized cx cy w h)
  5. Split 80/10/10 train/val/test
  6. Produce sunrgbd_yolo/ directory + dataset.yaml
"""

import os
import sys
import shutil
import random
import zipfile
import struct
import json
import math
import requests
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image


# ─── CONFIG ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("sunrgbd_yolo")
SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
RANDOM_SEED = 42

# 18 target classes (mapped to lowercase for matching)
TARGET_CLASSES = [
    "chair", "table", "door", "sofa", "bed", "toilet", "sink",
    "person", "stairs", "wall", "floor", "cabinet", "desk",
    "refrigerator", "monitor", "lamp", "pillow", "bookshelf"
]

# Aliases: SUN RGB-D uses many variant names — we normalize them
CLASS_ALIASES = {
    # chair variants
    "armchair": "chair", "office chair": "chair", "folding chair": "chair",
    "stool": "chair", "swivel chair": "chair",
    # table variants
    "dining table": "table", "coffee table": "table", "end table": "table",
    "nightstand": "table", "side table": "table", "kitchen table": "table",
    "dining_table": "table",
    # sofa variants
    "couch": "sofa", "loveseat": "sofa", "sectional": "sofa",
    # bed variants
    "bunk bed": "bed", "twin bed": "bed", "double bed": "bed",
    # cabinet variants
    "wardrobe": "cabinet", "dresser": "cabinet", "cupboard": "cabinet",
    "closet": "cabinet", "chest of drawers": "cabinet",
    # monitor variants
    "tv": "monitor", "television": "monitor", "screen": "monitor",
    "display": "monitor", "computer monitor": "monitor",
    # refrigerator variants
    "fridge": "refrigerator",
    # bookshelf variants
    "bookcase": "bookshelf", "shelf": "bookshelf", "shelves": "bookshelf",
    # lamp variants
    "light": "lamp", "ceiling light": "lamp", "floor lamp": "lamp",
    # stairs variants
    "staircase": "stairs", "stairway": "stairs", "steps": "stairs",
    # sink variants
    "bathroom sink": "sink", "kitchen sink": "sink",
    # desk variants
    "computer desk": "desk", "writing desk": "desk",
    # person
    "human": "person",
}

CLASS_TO_IDX = {cls: i for i, cls in enumerate(TARGET_CLASSES)}


# ─── DOWNLOAD HELPER ─────────────────────────────────────────────────────────

SUNRGBD_URLS = [
    # Try Princeton direct (may require login)
    "http://rgbd.cs.princeton.edu/data/SUNRGBD.zip",
    # Community mirror alternative
    "https://huggingface.co/datasets/SUN-RGBD/sunrgbd/resolve/main/SUNRGBD.zip",
]

def download_file(url: str, dest: Path) -> bool:
    """Attempt to download a file with progress bar. Returns True on success."""
    try:
        print(f"  Trying: {url}")
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code != 200:
            print(f"  ✗ HTTP {r.status_code}")
            return False
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=dest.name, total=total, unit="B", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def ensure_dataset_zip() -> Path:
    """Return path to SUNRGBD.zip, downloading if necessary."""
    local_zip = Path("SUNRGBD.zip")
    if local_zip.exists():
        print(f"✓ Found existing SUNRGBD.zip ({local_zip.stat().st_size / 1e9:.2f} GB)")
        return local_zip

    print("\n━━━ Downloading SUN RGB-D Dataset ━━━")
    for url in SUNRGBD_URLS:
        if download_file(url, local_zip):
            print(f"✓ Downloaded SUNRGBD.zip")
            return local_zip
        
    # Manual fallback
    print("""
━━━ MANUAL DOWNLOAD REQUIRED ━━━
The automatic download failed. Please:

1. Visit: http://rgbd.cs.princeton.edu/
2. Download: SUNRGBD.zip  (~3GB)
3. Place it in this directory: {}
4. Re-run this script.

Alternatively, you can also use the SUN RGBD Toolbox from:
https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip
""".format(Path.cwd()))
    sys.exit(1)


# ─── EXTRACT & PARSE ─────────────────────────────────────────────────────────

def extract_dataset(zip_path: Path, extract_to: Path) -> Path:
    """Extract zip if not already done."""
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"✓ Already extracted to {extract_to}")
        return extract_to
    
    print(f"\n━━━ Extracting {zip_path.name} → {extract_to} ━━━")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, extract_to)
    print(f"✓ Extraction complete.")
    return extract_to


def normalize_label(raw: str) -> str:
    """Normalize a raw label string to one of TARGET_CLASSES or None."""
    raw = raw.strip().lower().replace("_", " ")
    if raw in CLASS_TO_IDX:
        return raw
    if raw in CLASS_ALIASES:
        return CLASS_ALIASES[raw]
    # partial match
    for alias, canonical in CLASS_ALIASES.items():
        if alias in raw:
            return canonical
    for cls in TARGET_CLASSES:
        if cls in raw:
            return cls
    return None


def parse_bb2d_file(bb2d_path: Path, img_w: int, img_h: int):
    """
    Parse a SUN RGB-D 2D bounding box annotation file.
    Files are typically 'annotation2Dfinal/index.json' or structured text.
    Returns list of (class_idx, cx_norm, cy_norm, w_norm, h_norm)
    """
    annotations = []
    
    if not bb2d_path.exists():
        return annotations

    try:
        # Try JSON format first
        with open(bb2d_path, "r", errors="ignore") as f:
            content = f.read().strip()
        
        if content.startswith("{") or content.startswith("["):
            data = json.loads(content)
            objects = data if isinstance(data, list) else data.get("annotation", [])
            for obj in objects:
                name = obj.get("name", obj.get("label", ""))
                cls_name = normalize_label(name)
                if cls_name is None:
                    continue
                bbox = obj.get("bbox", obj.get("bndbox", {}))
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                elif isinstance(bbox, dict):
                    x1 = float(bbox.get("x1", bbox.get("xmin", 0)))
                    y1 = float(bbox.get("y1", bbox.get("ymin", 0)))
                    x2 = float(bbox.get("x2", x1 + bbox.get("w", 0)))
                    y2 = float(bbox.get("y2", y1 + bbox.get("h", 0)))
                else:
                    continue
                annotations.append(_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, CLASS_TO_IDX[cls_name]))
        else:
            # Text format: label x1 y1 x2 y2 per line
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    name = " ".join(parts[:-4])
                    cls_name = normalize_label(name)
                    if cls_name is None:
                        cls_name = normalize_label(parts[0])
                    if cls_name is None:
                        continue
                    try:
                        coords = [float(x) for x in parts[-4:]]
                        x1, y1, x2, y2 = coords
                        annotations.append(_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, CLASS_TO_IDX[cls_name]))
                    except ValueError:
                        continue
    except Exception:
        pass
    
    return annotations


def _bbox_to_yolo(x1, y1, x2, y2, img_w, img_h, class_idx):
    """Convert absolute bbox to normalized YOLO format."""
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    
    return (class_idx, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))


# ─── DISCOVERY ───────────────────────────────────────────────────────────────

def discover_samples(sunrgbd_root: Path):
    """
    Walk the SUN RGB-D directory tree and discover (image_path, annotation_path) pairs.
    SUN RGB-D has ~10,335 scenes stored as depth/rgb/annotation folders.
    
    Expected structure:
        SUNRGBD/
          kv1/     (Kinect v1 scans)
          kv2/     (Kinect v2 scans)
          realsense/
          xtion/
            <scene_id>/
              image/                <- RGB images
              annotation2Dfinal/
                index.json          <- or other annotation file
    """
    samples = []
    
    # Find all RGB images
    print("\n━━━ Discovering samples ━━━")
    img_extensions = {".jpg", ".jpeg", ".png"}
    
    for img_path in tqdm(list(sunrgbd_root.rglob("*.jpg")) + 
                          list(sunrgbd_root.rglob("*.png")), 
                          desc="Scanning images"):
        # Skip depth images (usually located in 'depth' folders)
        parts = img_path.parts
        if "depth" in parts or "depth_bfx" in parts:
            continue
        if "seg" in img_path.name.lower():
            continue
            
        # Look for annotation file in sibling annotation2Dfinal dir
        scene_dir = img_path.parent.parent if img_path.parent.name == "image" else img_path.parent
        
        ann_candidates = [
            scene_dir / "annotation2Dfinal" / "index.json",
            scene_dir / "annotation2Dfinal" / "annotations.json",
            scene_dir / "bb_2d_mat" / "index.mat",  # MATLAB format
            scene_dir.parent / "annotation2Dfinal" / "index.json",
        ]
        
        ann_path = None
        for candidate in ann_candidates:
            if candidate.exists():
                ann_path = candidate
                break
        
        samples.append((img_path, ann_path))
    
    print(f"✓ Found {len(samples)} RGB images")
    return samples


# ─── BUILD OUTPUT DATASET ────────────────────────────────────────────────────

def build_yolo_dataset(samples, output_dir: Path):
    """Convert samples to YOLOv8 format and split into train/val/test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    
    n = len(samples)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])
    
    split_map = (
        [("train", s) for s in samples[:n_train]] +
        [("val",   s) for s in samples[n_train:n_train+n_val]] +
        [("test",  s) for s in samples[n_train+n_val:]]
    )
    
    stats = {"train": 0, "val": 0, "test": 0, "total_annotations": 0, "skipped": 0}
    class_counts = {cls: 0 for cls in TARGET_CLASSES}
    
    print("\n━━━ Converting to YOLOv8 format ━━━")
    for split, (img_path, ann_path) in tqdm(split_map, desc="Processing"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
        except Exception:
            stats["skipped"] += 1
            continue
        
        # Parse annotations
        annotations = []
        if ann_path is not None:
            annotations = parse_bb2d_file(ann_path, img_w, img_h)
        
        # Stem for naming
        stem = f"{img_path.parent.parent.name}_{img_path.stem}"
        
        # Save image
        out_img = output_dir / "images" / split / f"{stem}.jpg"
        img.save(out_img, "JPEG", quality=90)
        
        # Save label (empty file is valid YOLO — means "no objects")
        out_lbl = output_dir / "labels" / split / f"{stem}.txt"
        with open(out_lbl, "w") as f:
            for (cls_idx, cx, cy, w, h) in annotations:
                f.write(f"{cls_idx} {cx} {cy} {w} {h}\n")
                class_counts[TARGET_CLASSES[cls_idx]] += 1
                stats["total_annotations"] += 1
        
        stats[split] += 1
    
    return stats, class_counts


# ─── YAML ────────────────────────────────────────────────────────────────────

def write_yaml(output_dir: Path):
    yaml_content = f"""# SUN RGB-D Dataset — YOLOv8 Format
# Generated by prepare_dataset.py

path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(TARGET_CLASSES)}
names:
"""
    for i, cls in enumerate(TARGET_CLASSES):
        yaml_content += f"  {i}: {cls}\n"
    
    yaml_path = Path("dataset.yaml")
    yaml_path.write_text(yaml_content)
    print(f"\n✓ Wrote {yaml_path}")
    return yaml_path


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" SUN RGB-D → YOLOv8 Dataset Preparation")
    print("=" * 60)
    
    # Step 1: Get ZIP
    zip_path = ensure_dataset_zip()
    
    # Step 2: Extract
    extract_dir = Path("SUNRGBD_raw")
    sunrgbd_root = extract_dataset(zip_path, extract_dir)
    
    # Step 3: Discover
    samples = discover_samples(sunrgbd_root)
    if not samples:
        print("ERROR: No images found. Check the extracted directory structure.")
        sys.exit(1)
    
    # Step 4: Convert & split
    stats, class_counts = build_yolo_dataset(samples, OUTPUT_DIR)
    
    # Step 5: Write YAML
    write_yaml(OUTPUT_DIR)
    
    # Report
    print("\n" + "=" * 60)
    print(" ✓ Dataset Preparation Complete!")
    print("=" * 60)
    print(f"  Train images : {stats['train']}")
    print(f"  Val images   : {stats['val']}")
    print(f"  Test images  : {stats['test']}")
    print(f"  Total annots : {stats['total_annotations']}")
    print(f"  Skipped      : {stats['skipped']}")
    print("\n  Class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    {cls:<20} {cnt:>6}")
    print("\n  Output: ./sunrgbd_yolo/")
    print("  YAML  : ./dataset.yaml")
    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()

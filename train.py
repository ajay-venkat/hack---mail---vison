"""
train.py — VisionAid Pro v4
============================
Fine-tunes YOLOv8m on the merged SUN RGB-D + NYU Depth V2 dataset
(prepared by prepare_dataset.py).

Usage:
    python train.py

Prerequisites:
    1. Run prepare_dataset.py first → generates indoor_dataset/ + dataset.yaml
    2. pip install ultralytics

Outputs:
    runs/train/indoor_v1/weights/best.pt   (raw training artifact)
    best_indoor.pt                          (copied to project root for app.py)
"""

import os
import sys
import shutil
from pathlib import Path


# ─── PREREQUISITES CHECK ──────────────────────────────────────────────────────

def check_prerequisites():
    yaml_path = Path("dataset.yaml")
    data_dir  = Path("indoor_dataset")

    if not yaml_path.exists():
        print("ERROR: dataset.yaml not found.")
        print("       Run: python prepare_dataset.py")
        sys.exit(1)

    if not data_dir.exists():
        print("ERROR: indoor_dataset/ not found.")
        print("       Run: python prepare_dataset.py")
        sys.exit(1)

    train_imgs = list((data_dir / "images" / "train").glob("*.jpg"))
    if not train_imgs:
        print("ERROR: No training images in indoor_dataset/images/train/")
        print("       Run: python prepare_dataset.py")
        sys.exit(1)

    val_imgs  = list((data_dir / "images" / "val").glob("*.jpg"))
    test_imgs = list((data_dir / "images" / "test").glob("*.jpg"))

    print("=" * 60)
    print("  Pre-flight checks passed:")
    print(f"    ✓ dataset.yaml")
    print(f"    ✓ Train images   : {len(train_imgs):,}")
    print(f"    ✓ Val images     : {len(val_imgs):,}")
    print(f"    ✓ Test images    : {len(test_imgs):,}")
    print("=" * 60)


# ─── TRAINING CONFIGURATION ───────────────────────────────────────────────────

TRAIN_CONFIG = {
    # ── Model ─────────────────────────────────────────────────────────────────
    "model"        : "yolov8m.pt",      # YOLOv8 Medium — pretrained COCO
    "data"         : "dataset.yaml",

    # ── Training ──────────────────────────────────────────────────────────────
    "epochs"       : 100,
    "imgsz"        : 960,
    "batch"        : 8,
    "patience"     : 15,               # early stopping

    # ── Optimiser ─────────────────────────────────────────────────────────────
    "optimizer"    : "AdamW",
    "lr0"          : 0.001,
    "lrf"          : 0.01,             # final lr = lr0 * lrf
    "momentum"     : 0.937,
    "weight_decay" : 0.0005,

    # ── Augmentation (tuned for indoor) ───────────────────────────────────────
    "hsv_h"        : 0.015,
    "hsv_s"        : 0.7,
    "hsv_v"        : 0.4,
    "degrees"      : 5.0,
    "translate"    : 0.1,
    "scale"        : 0.5,
    "shear"        : 2.0,
    "flipud"       : 0.3,              # vertical flip helps for stairs/floor
    "fliplr"       : 0.5,
    "mosaic"       : 1.0,
    "mixup"        : 0.1,
    "copy_paste"   : 0.0,

    # ── Class-specific confidence thresholds (used at inference) ──────────────
    # These are stored here for reference; app.py reads them from CLASS_CONF_THRESHOLDS
    # person: 0.55 | chair,table,sofa,bed: 0.30 | door,window: 0.35
    # stairs: 0.40 | wall,floor: 0.25
    "conf"         : 0.25,             # low base during training validation
    "iou"          : 0.45,

    # ── Output ────────────────────────────────────────────────────────────────
    "project"      : "runs/train",
    "name"         : "indoor_v1",
    "save"         : True,
    "exist_ok"     : True,
    "verbose"      : True,
    "workers"      : 4,
    "device"       : "",               # auto (GPU if available, else CPU)
}

OUTPUT_MODEL = Path("best_indoor.pt")


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    import torch
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = (
        f"GPU: {torch.cuda.get_device_name(0)}"
        if device == "cuda" else "CPU (no GPU — training will be very slow)"
    )

    print("\n" + "=" * 60)
    print("  VisionAid Pro — YOLOv8m Indoor Fine-Tuning")
    print("=" * 60)
    print(f"  Base model  : {TRAIN_CONFIG['model']}")
    print(f"  Device      : {gpu_info}")
    print(f"  Epochs      : {TRAIN_CONFIG['epochs']}")
    print(f"  Image size  : {TRAIN_CONFIG['imgsz']}px")
    print(f"  Batch size  : {TRAIN_CONFIG['batch']}")
    print(f"  Early stop  : patience={TRAIN_CONFIG['patience']}")
    print(f"  Output      : {OUTPUT_MODEL}")
    print("=" * 60 + "\n")

    # Load base model
    print(f"Loading base model '{TRAIN_CONFIG['model']}'...")
    model = YOLO(TRAIN_CONFIG["model"])

    # Fine-tune
    print("Starting fine-tuning...\n")
    results = model.train(
        data         = TRAIN_CONFIG["data"],
        epochs       = TRAIN_CONFIG["epochs"],
        imgsz        = TRAIN_CONFIG["imgsz"],
        batch        = TRAIN_CONFIG["batch"],
        patience     = TRAIN_CONFIG["patience"],
        optimizer    = TRAIN_CONFIG["optimizer"],
        lr0          = TRAIN_CONFIG["lr0"],
        lrf          = TRAIN_CONFIG["lrf"],
        momentum     = TRAIN_CONFIG["momentum"],
        weight_decay = TRAIN_CONFIG["weight_decay"],
        hsv_h        = TRAIN_CONFIG["hsv_h"],
        hsv_s        = TRAIN_CONFIG["hsv_s"],
        hsv_v        = TRAIN_CONFIG["hsv_v"],
        degrees      = TRAIN_CONFIG["degrees"],
        translate    = TRAIN_CONFIG["translate"],
        scale        = TRAIN_CONFIG["scale"],
        shear        = TRAIN_CONFIG["shear"],
        flipud       = TRAIN_CONFIG["flipud"],
        fliplr       = TRAIN_CONFIG["fliplr"],
        mosaic       = TRAIN_CONFIG["mosaic"],
        mixup        = TRAIN_CONFIG["mixup"],
        project      = TRAIN_CONFIG["project"],
        name         = TRAIN_CONFIG["name"],
        save         = TRAIN_CONFIG["save"],
        exist_ok     = TRAIN_CONFIG["exist_ok"],
        device       = device,
        workers      = TRAIN_CONFIG["workers"],
        verbose      = TRAIN_CONFIG["verbose"],
        conf         = TRAIN_CONFIG["conf"],
        iou          = TRAIN_CONFIG["iou"],
    )

    # ── Copy best weights to project root ──────────────────────────
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, OUTPUT_MODEL)
        print(f"\n✓ Best model saved → {OUTPUT_MODEL}")
    else:
        print(f"\n⚠ best.pt not found at {best}")
        last = Path(results.save_dir) / "weights" / "last.pt"
        if last.exists():
            shutil.copy(last, OUTPUT_MODEL)
            print(f"  Fallback: last.pt copied → {OUTPUT_MODEL}")

    # ── Validation ─────────────────────────────────────────────────
    print("\n━━━ Final Validation ━━━")
    eval_model = YOLO(str(OUTPUT_MODEL))
    metrics = eval_model.val(
        data    = TRAIN_CONFIG["data"],
        imgsz   = TRAIN_CONFIG["imgsz"],
        conf    = TRAIN_CONFIG["conf"],
        iou     = TRAIN_CONFIG["iou"],
        split   = "val",
        verbose = True,
    )

    # ── mAP Summary ────────────────────────────────────────────────
    try:
        map50   = metrics.box.map50
        map5095 = metrics.box.map
    except AttributeError:
        rd = getattr(metrics, "results_dict", {})
        map50   = rd.get("metrics/mAP50(B)", 0)
        map5095 = rd.get("metrics/mAP50-95(B)", 0)

    print("\n" + "=" * 60)
    print("  ✓ Training Complete!")
    print("=" * 60)
    print(f"  mAP@50        : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP@50-95     : {map5095:.4f}  ({map5095*100:.1f}%)")
    print(f"  Best model    : {OUTPUT_MODEL}")
    print(f"  Training logs : {results.save_dir}")
    print("""
Next step:
  app.py will auto-load best_indoor.pt if present in project root.
  Run:  streamlit run app.py
""")
    return metrics


# ─── CLASS-SPECIFIC CONFIDENCE THRESHOLDS (for reference) ────────────────────
#
#  app.py reads these at runtime. Listed here for documentation.
#
#  person             : 0.55
#  chair, table, sofa, bed, pillow  : 0.30
#  desk               : 0.30
#  door, window       : 0.35
#  stairs             : 0.40
#  wall, floor        : 0.25   ← hard to detect, very low threshold
#  cabinet, refrigerator, monitor : 0.35
#  lamp, bookshelf, sink, toilet  : 0.35
#  curtain, picture, counter, shelf : 0.30


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_prerequisites()
    train()

"""
train.py
=========
Fine-tunes YOLOv8s on the SUN RGB-D dataset (prepared by prepare_dataset.py).

Usage:
    python train.py

Prerequisites:
    1. Run prepare_dataset.py first to generate sunrgbd_yolo/ and dataset.yaml
    2. pip install ultralytics

Outputs:
    - runs/train/sunrgbd_v1/weights/best.pt  (raw training artifact)
    - best_sunrgbd.pt                         (copied to project root for app.py)
"""

import os
import sys
import shutil
from pathlib import Path

# ─── VALIDATE PREREQUISITES ──────────────────────────────────────────────────

def check_prerequisites():
    yaml_path = Path("dataset.yaml")
    data_dir  = Path("sunrgbd_yolo")
    
    if not yaml_path.exists():
        print("ERROR: dataset.yaml not found.")
        print("       Please run: python prepare_dataset.py")
        sys.exit(1)
    
    if not data_dir.exists():
        print("ERROR: sunrgbd_yolo/ directory not found.")
        print("       Please run: python prepare_dataset.py")
        sys.exit(1)
    
    train_imgs = list((data_dir / "images" / "train").glob("*.jpg"))
    if not train_imgs:
        print("ERROR: No training images found in sunrgbd_yolo/images/train/")
        print("       Please run: python prepare_dataset.py")
        sys.exit(1)
    
    print(f"✓ dataset.yaml found")
    print(f"✓ Training images: {len(train_imgs)}")
    
    val_imgs = list((data_dir / "images" / "val").glob("*.jpg"))
    print(f"✓ Validation images: {len(val_imgs)}")


# ─── TRAINING CONFIG ─────────────────────────────────────────────────────────

TRAIN_CONFIG = {
    # Model
    "model"     : "yolov8s.pt",      # YOLOv8 Small — pretrained on COCO
    "data"      : "dataset.yaml",
    
    # Training hyperparams
    "epochs"    : 50,
    "imgsz"     : 640,
    "batch"     : 16,
    "patience"  : 10,               # Early stopping patience
    
    # Optimiser
    "optimizer" : "AdamW",
    "lr0"       : 0.001,
    "lrf"       : 0.01,            # Final LR = lr0 * lrf
    "momentum"  : 0.937,
    "weight_decay": 0.0005,
    
    # Augmentation (good for indoor scenes)
    "hsv_h"     : 0.015,
    "hsv_s"     : 0.7,
    "hsv_v"     : 0.4,
    "degrees"   : 5.0,             # small rotation for indoor
    "translate" : 0.1,
    "scale"     : 0.5,
    "shear"     : 2.0,
    "flipud"    : 0.0,             # no vertical flip (indoor)
    "fliplr"    : 0.5,
    "mosaic"    : 1.0,
    "mixup"     : 0.1,
    
    # Output
    "project"   : "runs/train",
    "name"      : "sunrgbd_v1",
    "save"      : True,
    "exist_ok"  : True,
    "device"    : "",              # auto-detect (GPU if available, else CPU)
    "workers"   : 4,
    "verbose"   : True,
    
    # Inference thresholds
    "conf"      : 0.45,
    "iou"       : 0.5,
}

OUTPUT_MODEL = Path("best_sunrgbd.pt")


# ─── TRAIN ───────────────────────────────────────────────────────────────────

def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if device == "cuda" else "CPU (no GPU found)"
    
    print("=" * 60)
    print(" VisionAid — YOLOv8s SUN RGB-D Fine-Tuning")
    print("=" * 60)
    print(f"  Base model : {TRAIN_CONFIG['model']}")
    print(f"  Device     : {gpu_info}")
    print(f"  Epochs     : {TRAIN_CONFIG['epochs']}")
    print(f"  Image size : {TRAIN_CONFIG['imgsz']}")
    print(f"  Batch size : {TRAIN_CONFIG['batch']}")
    print(f"  Early stop : patience={TRAIN_CONFIG['patience']}")
    print("=" * 60)
    
    # Load base model
    print(f"\nLoading base model: {TRAIN_CONFIG['model']} ...")
    model = YOLO(TRAIN_CONFIG["model"])
    
    # Fine-tune
    print("\nStarting fine-tuning...\n")
    results = model.train(
        data       = TRAIN_CONFIG["data"],
        epochs     = TRAIN_CONFIG["epochs"],
        imgsz      = TRAIN_CONFIG["imgsz"],
        batch      = TRAIN_CONFIG["batch"],
        patience   = TRAIN_CONFIG["patience"],
        optimizer  = TRAIN_CONFIG["optimizer"],
        lr0        = TRAIN_CONFIG["lr0"],
        lrf        = TRAIN_CONFIG["lrf"],
        momentum   = TRAIN_CONFIG["momentum"],
        weight_decay = TRAIN_CONFIG["weight_decay"],
        hsv_h      = TRAIN_CONFIG["hsv_h"],
        hsv_s      = TRAIN_CONFIG["hsv_s"],
        hsv_v      = TRAIN_CONFIG["hsv_v"],
        degrees    = TRAIN_CONFIG["degrees"],
        translate  = TRAIN_CONFIG["translate"],
        scale      = TRAIN_CONFIG["scale"],
        shear      = TRAIN_CONFIG["shear"],
        flipud     = TRAIN_CONFIG["flipud"],
        fliplr     = TRAIN_CONFIG["fliplr"],
        mosaic     = TRAIN_CONFIG["mosaic"],
        mixup      = TRAIN_CONFIG["mixup"],
        project    = TRAIN_CONFIG["project"],
        name       = TRAIN_CONFIG["name"],
        save       = TRAIN_CONFIG["save"],
        exist_ok   = TRAIN_CONFIG["exist_ok"],
        device     = device,
        workers    = TRAIN_CONFIG["workers"],
        verbose    = TRAIN_CONFIG["verbose"],
        conf       = TRAIN_CONFIG["conf"],
        iou        = TRAIN_CONFIG["iou"],
    )
    
    # ─── Copy best weights to project root ───────────────────────────────────
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    
    if best_weights.exists():
        shutil.copy(best_weights, OUTPUT_MODEL)
        print(f"\n✓ Best model saved: {OUTPUT_MODEL}")
    else:
        print(f"\n⚠ best.pt not found at {best_weights} — training may have stopped early.")
        last_weights = Path(results.save_dir) / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy(last_weights, OUTPUT_MODEL)
            print(f"  Using last.pt as fallback → {OUTPUT_MODEL}")
    
    # ─── Evaluate on validation set ──────────────────────────────────────────
    print("\n━━━ Running Validation ━━━")
    eval_model = YOLO(str(OUTPUT_MODEL))
    metrics = eval_model.val(
        data  = TRAIN_CONFIG["data"],
        imgsz = TRAIN_CONFIG["imgsz"],
        conf  = TRAIN_CONFIG["conf"],
        iou   = TRAIN_CONFIG["iou"],
        split = "val",
        verbose = True,
    )
    
    # ─── Print mAP summary ───────────────────────────────────────────────────
    map50    = metrics.box.map50   if hasattr(metrics, "box") else metrics.results_dict.get("metrics/mAP50(B)", 0)
    map5095  = metrics.box.map     if hasattr(metrics, "box") else metrics.results_dict.get("metrics/mAP50-95(B)", 0)
    
    print("\n" + "=" * 60)
    print(" ✓ Training Complete!")
    print("=" * 60)
    print(f"  mAP@50        : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP@50-95     : {map5095:.4f}  ({map5095*100:.1f}%)")
    print(f"  Best model    : {OUTPUT_MODEL}")
    print(f"  Training logs : {results.save_dir}")
    print("""
Next step:
  The app.py will automatically load best_sunrgbd.pt if present.
  Run: streamlit run app.py
""")
    
    return metrics


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_prerequisites()
    train()

#!/usr/bin/env python3
"""
train_det.py -- YOLO oriented-bounding-box (OBB) detectors for ROIs and hair follicles on whole-slide exports.

    python training/annotate_boxes.py                            # annotate; quitting rebuilds datasets/roi_follicle/yolo_det
    python training/detection/train_det.py train --target roi   # roi.yaml -> runs/det/roi/train*/weights/best.pt
    python training/detection/train_det.py train --target follicle --name f5 --epochs 200
    python training/detection/train_det.py predict --target roi --source <slides>

--target picks the config next to this script (roi.yaml / follicle.yaml), which also restricts
training to that class; --config <yaml> uses another file. Command-line flags override the YAML.
The ROI dataset (yolo_det) is built by annotate_qc_boxes.py (4-corner boxes, slides downscaled so their
long side is --max-side px). Predictions are mapped back to original-pixel coordinates. Use the
same --max-side for predict as for the dataset.

The follicle dataset (yolo_follicle) holds tiles cut at a fixed um/px; --data may also be one of its
fold<k>.yaml files. Training on it stamps that scale into the weights (ckpt["tile_scale"]), and
predict / postprocess.py then tile each slide at the same scale (slide um/px from --um-per-px or
--calibration).
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for running this file as a script
from training.annotate_qc_boxes import list_images  # noqa: E402
from training.config_utils import apply_config  # noqa: E402
from training.detection.follicle_tiles import detect_tiled, load_calibration, resize, tile_scale  # noqa: E402

TARGETS = ("roi", "follicle")


# ----------------------------------------------------------------------------


def stamp_tile_scale(weights_dir: Path, scale: dict):
    """Record the dataset's tile scale in the weights so inference resamples slides the same way."""
    import torch

    for p in weights_dir.glob("*.pt"):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        ckpt["tile_scale"] = scale
        torch.save(ckpt, p)
    print(f"tile scale {scale} stamped into {weights_dir}/*.pt", flush=True)


def cmd_train(a):
    from ultralytics import YOLO

    data = Path(a.data)
    data = (data if data.suffix in (".yaml", ".yml") else data / "data.yaml").resolve()
    if not data.exists():
        raise SystemExit(f"{data} not found; build it with: python training/annotate_qc_boxes.py --prepare-only")
    meta = yaml.safe_load(data.read_text())
    if "tile" in meta and a.imgsz != meta["tile"]:
        raise SystemExit(f"--imgsz {a.imgsz} must equal the tile size of {data} ({meta['tile']})")
    classes = None
    if a.classes:
        names = meta["names"]
        ids = {v: k for k, v in names.items()}
        unknown = set(a.classes.split(",")) - set(ids)
        if unknown:
            raise SystemExit(f"unknown classes {sorted(unknown)}; dataset has {list(ids)}")
        classes = [ids[c] for c in a.classes.split(",")]
    model = YOLO(a.model)
    model.train(
        data=str(data), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
        patience=a.patience, device=a.device, workers=a.workers, seed=0,
        project=str(Path(a.project).resolve()), name=a.name,
        # histology has no canonical orientation; keep hue jitter small since stain colour is informative
        fliplr=0.5, flipud=0.5, hsv_h=0.01, degrees=a.degrees, scale=a.scale, translate=a.translate,
        shear=a.shear, hsv_s=a.hsv_s, hsv_v=a.hsv_v, classes=classes,
    )
    if "um_per_px" in meta:
        stamp_tile_scale(Path(model.trainer.save_dir) / "weights",
                         {k: meta[k] for k in ("um_per_px", "tile", "overlap")})


def cmd_predict(a):
    from ultralytics import YOLO

    model = YOLO(a.weights)
    scale = tile_scale(model)
    calibration = load_calibration(a.calibration) if a.calibration else {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, unreadable = [], []
    for p in list_images(Path(a.source)):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"SKIPPED {p.name}: could not read", flush=True)
            unreadable.append(p.name)
            continue
        if scale:
            # tiled models are single-class
            dets = detect_tiled(model, img, calibration.get(p.name, a.um_per_px), scale, a.conf, a.device)
            f = min(1.0, a.max_side / max(img.shape[:2]))
            vis = resize(img, f)
            for quad, c in dets:
                rows.append([p.name, model.names[0], *(round(float(v)) for v in quad.ravel()), round(c, 3)])
                cv2.polylines(vis, [np.round(quad * f).astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / f"{p.stem}_pred.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{p.name}: {len(dets)} detections", flush=True)
            continue
        f = min(1.0, a.max_side / max(img.shape[:2]))
        img = resize(img, f)
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        for quad, c, k in zip(r.obb.xyxyxyxy.tolist(), r.obb.conf.tolist(), r.obb.cls.tolist()):
            rows.append([p.name, r.names[int(k)], *(round(v / f) for pt in quad for v in pt), round(c, 3)])
        cv2.imwrite(str(out / f"{p.stem}_pred.jpg"), r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{p.name}: {len(r.obb)} detections", flush=True)
    with open(out / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "class", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "conf"])
        w.writerows(rows)
    print(f"\nWrote overlays and detections.csv (original-pixel coordinates) to {out}")
    if unreadable:
        print(f"unreadable, skipped: {unreadable}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a YOLO-OBB detector on the dataset built by annotate_boxes.py")
    t.add_argument("--data", default="datasets/roi_follicle/yolo_det")
    t.add_argument("--model", default="models/init/yolo11n-obb.pt", help="pretrained YOLO-OBB weights, e.g. yolo11n-obb.pt | yolo11s-obb.pt")
    t.add_argument("--imgsz", type=int, default=1024)
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--patience", type=int, default=50)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--device", default=None)
    t.add_argument("--project", default="runs/det")
    t.add_argument("--name", default="train")
    t.add_argument("--degrees", type=float, default=180.0, help="random rotation range +/- degrees (0 = off)")
    t.add_argument("--scale", type=float, default=0.5, help="random zoom gain, e.g. 0.5 = 50%%-150%%")
    t.add_argument("--translate", type=float, default=0.1, help="random shift as a fraction of image size")
    t.add_argument("--shear", type=float, default=0.0, help="random shear +/- degrees")
    t.add_argument("--hsv-s", type=float, default=0.5, help="saturation jitter (stain intensity)")
    t.add_argument("--hsv-v", type=float, default=0.3, help="brightness jitter")
    t.add_argument("--classes", help="comma-separated class names to train on, e.g. roi (default: all)")
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("predict", help="detect ROIs / follicles in a folder of slides")
    d.add_argument("--weights", required=True)
    d.add_argument("--source", required=True)
    d.add_argument("--out", default="runs/det/predict")
    d.add_argument("--max-side", type=int, default=1024,
                   help="whole-slide models: must match annotate_qc_boxes.py --max-side; tiled models: overlay size")
    d.add_argument("--imgsz", type=int, default=1024)
    d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--device", default=None)
    d.add_argument("--um-per-px", type=float, default=0.2577, help="native um/px of the slides (tiled models)")
    d.add_argument("--calibration", help="JSON {\"images\": {name: um/px}} overriding --um-per-px per slide")
    d.set_defaults(func=cmd_predict)
    for sp in (t, d):
        sp.add_argument("--target", choices=TARGETS, help="load <target>.yaml next to this script")
    return ap


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--target", choices=TARGETS)
    target = pre.parse_known_args()[0].target
    ap = build_parser()
    apply_config(ap, default=str(Path(__file__).with_name(f"{target}.yaml")) if target else None)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

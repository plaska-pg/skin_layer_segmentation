#!/usr/bin/env python3
"""
eval_seg.py -- boundary-accuracy evaluation of predict.predict_mask against
YOLO-seg ground truth, at each image's native resolution.

Per class: Dice, boundary F1 at a few um tolerances, and the signed
boundary offset (um; + = prediction edge lies outside the true region,
- = inside), measured only on predicted edge pixels within --bias-window-um
of a true edge so stray blobs don't dominate.

    python eval_seg.py
    python eval_seg.py --morph 5,15,10 --morph 3,8,5   # also score postprocess open+close settings (SC,Epi,gland um)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from config import CONFIG
from measurement import _rasterise
from postprocess import erode_dilate, kernel_px
from predict import IMAGE_EXTS, TTA_OPS, load_model, pick_device, predict_mask_safe

_PREDICT_CFG = CONFIG.get("predict", {})
TOLERANCES_UM = (1.0, 2.0, 4.0)


def _boundary(mask: np.ndarray) -> np.ndarray:
    # erosion treats outside-image as foreground, so image borders aren't counted as edges
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def _dist_to(mask: np.ndarray) -> np.ndarray:
    """Distance (px) of every pixel to the nearest True pixel of mask."""
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def class_stats(pred: np.ndarray, gt: np.ndarray, um_per_px: float, bias_window_um: float) -> dict:
    s = {"inter": int((pred & gt).sum()), "psum": int(pred.sum()), "gsum": int(gt.sum())}
    for t in TOLERANCES_UM:
        s[f"bp_{t}"] = s[f"bp_tot_{t}"] = s[f"bg_{t}"] = s[f"bg_tot_{t}"] = 0
    s["bias_sum"] = 0.0
    s["bias_n"] = 0
    if not pred.any() and not gt.any():
        return s
    pad = int(np.ceil(max(max(TOLERANCES_UM), bias_window_um) / um_per_px)) + 2
    ys, xs = np.nonzero(pred | gt)
    y0, y1 = max(0, ys.min() - pad), min(pred.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(pred.shape[1], xs.max() + pad + 1)
    pred, gt = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    bp, bg = _boundary(pred), _boundary(gt)
    d_to_bg, d_to_bp = _dist_to(bg)[bp], _dist_to(bp)[bg]
    for t in TOLERANCES_UM:
        tp = t / um_per_px
        s[f"bp_{t}"], s[f"bp_tot_{t}"] = int((d_to_bg <= tp).sum()), int(bp.sum())
        s[f"bg_{t}"], s[f"bg_tot_{t}"] = int((d_to_bp <= tp).sum()), int(bg.sum())
    near = d_to_bg <= bias_window_um / um_per_px
    if near.any():
        sign = np.where(gt[bp][near], -1.0, 1.0)
        s["bias_sum"] = float((sign * d_to_bg[near]).sum() * um_per_px)
        s["bias_n"] = int(near.sum())
    return s


def summarise(acc: dict) -> dict:
    out = {"dice": 2 * acc["inter"] / max(1, acc["psum"] + acc["gsum"])}
    for t in TOLERANCES_UM:
        p = acc[f"bp_{t}"] / max(1, acc[f"bp_tot_{t}"])
        r = acc[f"bg_{t}"] / max(1, acc[f"bg_tot_{t}"])
        out[f"bf1@{t:g}um"] = 2 * p * r / max(1e-9, p + r)
    out["bias_um"] = acc["bias_sum"] / max(1, acc["bias_n"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=_PREDICT_CFG.get("weights", "models/3cls_best_v8_e.pt"))
    ap.add_argument("--data", default="datasets/seg_finetune")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=_PREDICT_CFG.get("device", "auto"))
    ap.add_argument("--reference-um-per-px", type=float, default=_PREDICT_CFG.get("reference_um_per_px", 0.645))
    ap.add_argument("--tile", type=int, default=_PREDICT_CFG.get("tile", 1024))
    ap.add_argument("--tta", choices=list(TTA_OPS), default=_PREDICT_CFG.get("tta", "none"))
    ap.add_argument("--morph", action="append", default=[],
                    help="'SC,Epi,gland' open+close kernel diameters (um) to also score; repeatable")
    ap.add_argument("--bias-window-um", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json-out", default=None, help="also write the summary table here")
    args = ap.parse_args()

    data = Path(args.data)
    names_y = yaml.safe_load((data / "data.yaml").read_text())["names"]
    gt_names = [names_y[i] for i in sorted(names_y)] if isinstance(names_y, dict) else list(names_y)
    calib = json.loads((data / "scale_calibration.json").read_text())["images"]

    device = pick_device(args.device)
    model, names, _ = load_model(args.weights, device)
    gt_to_model = np.zeros(len(gt_names) + 1, np.uint8)
    for i, n in enumerate(gt_names):
        if n in names:
            gt_to_model[i + 1] = names.index(n) + 1

    settings = {"raw": None}
    for m in args.morph:
        settings[f"morph {m}"] = [float(v) for v in m.split(",")]
    acc = {(s, n): {} for s in settings for n in names}

    files = sorted(p for p in (data / "images" / args.split).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[:args.limit]
    for p in files:
        mpp = calib.get(p.stem)
        if mpp is None:
            print(f"{p.name}: no um_per_px in scale_calibration.json, skipping")
            continue
        img = cv2.imread(str(p))
        gt = gt_to_model[_rasterise(data / "labels" / args.split / f"{p.stem}.txt", *img.shape[:2])]
        pred = predict_mask_safe(model, img, device, mpp / args.reference_um_per_px, name=p.name, tile=args.tile,
                                 tta=args.tta)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        line = []
        for s, kernels in settings.items():
            for k, n in enumerate(names):
                pm = pred == k + 1
                if kernels is not None and k < len(kernels):
                    pm = erode_dilate(pm, kernel_px(kernels[k], mpp))
                st = class_stats(pm, gt == k + 1, mpp, args.bias_window_um)
                a = acc[(s, n)]
                for key, v in st.items():
                    a[key] = a.get(key, 0) + v
                if s == "raw":
                    r = summarise(st)
                    line.append(f"{n} dice {r['dice']:.3f} bf1@2 {r['bf1@2um']:.3f} bias {r['bias_um']:+.2f}")
        print(f"{p.name}: " + " | ".join(line), flush=True)

    rows = {f"{s} / {n}": summarise(a) for (s, n), a in acc.items() if a}
    cols = list(next(iter(rows.values())).keys())
    print("\n" + f"{'setting / class':<32}" + "".join(f"{c:>11}" for c in cols))
    for k, r in rows.items():
        print(f"{k:<32}" + "".join(f"{r[c]:>11.3f}" for c in cols))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

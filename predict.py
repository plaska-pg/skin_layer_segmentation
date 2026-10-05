#!/usr/bin/env python3
"""
predict.py -- run the trained SC/Granular Layer ResUNet model (see
pretrained_model/train_semseg.py) on a folder of images and, by default,
postprocess each prediction (postprocess.py: close -> identify ROI -> rotate
-> measure) into a labeled composite QA image plus a dataset-wide CSV.

    python predict.py
    python predict.py --source inference_images --out runs/predict_out
    python predict.py --no-postprocess   # just dump raw overlays, no CSV/composite

    python predict.py --limit 5 --out runs/predict_mini

Default weights/scale match dataset/scale_calibration.json + the
inference_images/ Motic 40X series (see plan.md "Decisions").
"""

import argparse
import builtins
import csv
import re
import sys
from datetime import datetime
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # these whole-slide TIFFs/SVS are far bigger than PIL's default decompression-bomb cap

sys.path.insert(0, str(Path(__file__).resolve().parent / "pretrained_model"))
from train_semseg import MEAN, STD, ResUNet, pick_device  # noqa: E402

from config import BASE_CONFIG_PATH, CONFIG  # noqa: E402
from postprocess import (BOTTOM_CROP_FRAC, CLOSE_RADII_UM, LATERAL_CLIP_FRAC, MIN_TOUCH_FRAC, SC_SMOOTH_RADIUS_UM,
                         TOUCH_MARGIN_UM, overlay, process_image)  # noqa: E402
from measurement import CSV_COLUMNS, _rasterise  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs"}  # .svs (Aperio) is TIFF-based - cv2 reads its
# full-resolution page directly, ignoring the pyramid/thumbnail sub-IFDs and Aperio-specific tags

_MPP_RE = re.compile(r"MPP\s*=\s*([\d.]+)")
_PREDICT_CFG = CONFIG.get("predict", {})
_ORIGINAL_PRINT = builtins.print


def _timestamped_print(*args, **kwargs):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    _ORIGINAL_PRINT(f"[{timestamp}]", *args, **kwargs)


def write_run_config(path: Path, args, src: Path, out: Path, device, names, files, effective_mpp):
    """Write the resolved settings and per-image calibration used by a run."""
    import yaml

    config = {
        "run": {
            "created_at": datetime.now().astimezone().isoformat(),
            "source": str(src.resolve()),
            "output": str(out.resolve()),
            "image_count": len(files),
            "device": str(device),
            "model_class_names": list(names),
            "base_config_path": str(BASE_CONFIG_PATH),
        },
        "parameters": vars(args),
        "resolved": {
            "weights": str(Path(args.weights).resolve()),
            "csv_output": str((Path(args.csv_out) if args.csv_out else out / "results.csv").resolve()),
            "reference_um_per_px": args.reference_um_per_px,
            "postprocess": not args.no_postprocess,
            "close_radii_um": list(CLOSE_RADII_UM),
            "touch_margin_um": TOUCH_MARGIN_UM,
            "min_touch_frac": MIN_TOUCH_FRAC,
            "sc_smooth_radius_um": SC_SMOOTH_RADIUS_UM,
            "lateral_clip_frac": LATERAL_CLIP_FRAC,
            "bottom_crop_frac": BOTTOM_CROP_FRAC,
            "input_files": [
                {
                    "file": p.name,
                    "um_per_px": effective_mpp[p.name],
                    "inference_scale": (effective_mpp[p.name] / args.reference_um_per_px)
                    if effective_mpp[p.name] else 1.0,
                }
                for p in files
            ],
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def svs_mpp(path: Path):
    """Native microns-per-pixel embedded in an Aperio .svs's ImageDescription
    tag (e.g. '...|MPP = 0.2500|...'), or None if the file isn't .svs or the
    tag is missing/unreadable - lets --um-per-px be an accurate per-scan
    value instead of one assumed default for the whole --source folder."""
    if path.suffix.lower() != ".svs":
        return None
    try:
        with Image.open(path) as im:
            desc = str(im.tag_v2.get(270, ""))
    except Exception:
        return None
    m = _MPP_RE.search(desc)
    return float(m.group(1)) if m else None


@torch.no_grad()
def predict_mask(model, orig_bgr: np.ndarray, device, scale: float, max_side: int = None) -> np.ndarray:
    """Class-index mask (0=bg, 1..N=names), at the ORIGINAL image's own
    resolution - scale is only used to normalise the model's input to its
    training resolution, not to change what gets returned. max_side (if
    given) additionally caps the longest edge fed to the model on top of
    that, purely to bound GPU memory on very large slides - the mask is
    still resized back to the image's own full resolution afterward, so
    measurement precision is unaffected; only the segmentation's own input
    detail gets coarser."""
    oh, ow = orig_bgr.shape[:2]
    th, tw = (round(oh * scale), round(ow * scale)) if scale != 1.0 else (oh, ow)
    if max_side and max(th, tw) > max_side:
        extra = max_side / max(th, tw)
        th, tw = round(th * extra), round(tw * extra)
    resized = (th, tw) != (oh, ow)
    bgr = cv2.resize(orig_bgr, (max(1, tw), max(1, th)),
                      interpolation=cv2.INTER_AREA if (tw < ow or th < oh) else cv2.INTER_LINEAR) \
        if resized else orig_bgr
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32  # pad to a multiple of 32 for the encoder strides
    rgb_p = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x = torch.from_numpy(((rgb_p.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None].to(device)
    pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)[:h, :w]
    if resized:
        pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return pred


def predict_mask_safe(model, orig_bgr: np.ndarray, device, scale: float, name: str = "",
                      max_side: int = None) -> np.ndarray:
    """predict_mask, falling back to CPU (moving the model there and back) on a CUDA
    out-of-memory error - whole-slide images vary enough in size that one image fitting
    on a small GPU (e.g. 4GB) doesn't guarantee the next one will."""
    try:
        return predict_mask(model, orig_bgr, device, scale, max_side)
    except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
        if device.type != "cuda":
            raise
        print(f"{name}: GPU out of memory ({e}); clearing cache and retrying on CPU", flush=True)
        torch.cuda.empty_cache()
        model.to("cpu")
        try:
            return predict_mask(model, orig_bgr, torch.device("cpu"), scale, max_side)
        finally:
            model.to(device)
            torch.cuda.empty_cache()


def load_model(weights: str, device):
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    names = ckpt["names"]
    class_index = {n: k + 1 for k, n in enumerate(names)}
    model = ResUNet(len(names) + 1, ckpt["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, names, class_index


def mask_to_yolo_polygons(mask: np.ndarray, min_area_px: float = 20.0):
    """Yields one (N, 2) array of (x, y) points per external contour of mask,
    normalised to [0, 1] by the mask's own width/height - same convention as
    dataset/labels/*.txt. Contours smaller than min_area_px (specks/noise)
    are dropped; holes are not represented (RETR_EXTERNAL only)."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if len(c) < 3 or cv2.contourArea(c) < min_area_px:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        yield pts


def write_yolo_labels(pred_mask: np.ndarray, class_index: dict, out_path: Path, min_area_px: float = 20.0) -> None:
    """Writes pred_mask's per-class contours as YOLO-seg polygon lines (same
    format/class-id convention as dataset/labels/*.txt): one line per polygon,
    '<class_id 0-indexed> x1 y1 x2 y2 ...' with coordinates normalised [0, 1]."""
    lines = []
    for name, idx in class_index.items():
        for pts in mask_to_yolo_polygons(pred_mask == idx, min_area_px):
            coords = " ".join(f"{v:.6f}" for v in pts.flatten())
            lines.append(f"{idx - 1} {coords}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    builtins.print = _timestamped_print
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=_PREDICT_CFG.get("weights", "runs/semseg/dataset_2cls_finetune_v2/best.pt"),
                     help="ResUNet checkpoint trained on SC/Granular Layer (dataset/dataset.yaml)")
    ap.add_argument("--source", default=_PREDICT_CFG.get("source", "inference_images"),
                     help="a single image file OR a folder of images to predict on")
    ap.add_argument("--out", default=None,
                     help="default: '<source folder>/predicted_<source folder name>' (for a single image: "
                          "'<image's folder>/predicted_<image stem>')")
    ap.add_argument("--device", default=_PREDICT_CFG.get("device", "auto"))
    ap.add_argument("--um-per-px", type=float, default=_PREDICT_CFG.get("um_per_px", 0.26),
                     help="native um/px of --source images (default matches inference_images' Motic 40X "
                          "calibration in calibrations/Motic 40X scan calibration.IQC). 0 disables rescaling. "
                          "For .svs files, the scanner's own embedded MPP tag is used instead when present.")
    ap.add_argument("--reference-um-per-px", type=float, default=_PREDICT_CFG.get("reference_um_per_px", 0.645),
                     help="um/px the model was trained at (default matches dataset/scale_calibration.json)")
    ap.add_argument("--max-side", type=int, default=_PREDICT_CFG.get("max_side", None),
                     help="cap the longest edge fed to the model to at most this many px, on top of the "
                          "--um-per-px normalisation - bounds GPU memory on very large slides. Masks are "
                          "resized back to the image's full native resolution before measuring, so this only "
                          "coarsens the segmentation's own input detail, not measurement precision. Try e.g. "
                          "3000 if you hit CUDA out-of-memory errors.")
    ap.add_argument("--no-postprocess", action="store_true",
                     help="skip close/ROI/rotate/measure - just write raw prediction overlays")
    ap.add_argument("--no-labels", action="store_true",
                     help="skip writing <out>/labels/<stem>.txt YOLO-seg polygon labels for the raw "
                          "(un-rotated, un-closed) per-image prediction mask")
    ap.add_argument("--resume", action="store_true",
                     help="if <out>/labels/<stem>.txt already exists from a previous (e.g. crashed) run, "
                          "reuse it instead of re-running GPU inference for that image - postprocess/measure "
                          "still re-runs (cheap, CPU-only) so results.csv comes out complete either way")
    ap.add_argument("--csv-out", default=None, help="default: <out>/results.csv")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N images (for quick test runs)")
    ap.add_argument("--workers", type=int, default=_PREDICT_CFG.get("workers", 1),
                     help="run the CPU-bound postprocess/measure step for up to N images concurrently in "
                          "background threads, overlapping it with the next image's (serial, single-device) "
                          "model inference - cv2/scipy/skimage release the GIL so this gets real parallelism. "
                          "Ignored with --no-postprocess. Keep this <= your CPU core count.")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, names, class_index = load_model(args.weights, device)
    if not args.no_postprocess and not {"SC", "Granular Layer"} <= set(names):
        raise SystemExit(f"--weights checkpoint names {names} must include 'SC' and 'Granular Layer' "
                          f"for postprocessing; pass --no-postprocess to skip it")

    scale = (args.um_per_px / args.reference_um_per_px) if args.um_per_px else 1.0
    if scale != 1.0:
        print(f"normalising inputs: {args.um_per_px} um/px -> {args.reference_um_per_px} um/px "
              f"(resize factor {scale:.4f})")

    src = Path(args.source)
    # --source may be a single image file or a folder of images
    if src.is_file():
        files = [src] if src.suffix.lower() in IMAGE_EXTS else []
        default_out = src.parent / f"predicted_{src.stem}"
    else:
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        default_out = src / f"predicted_{src.name}"
    out = Path(args.out) if args.out else default_out
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_postprocess:
        print("postprocess hyperparameters: "
              f"close_radii_um={CLOSE_RADII_UM}; "
              f"touch_margin_um={TOUCH_MARGIN_UM}; "
              f"min_touch_frac={MIN_TOUCH_FRAC}; "
              f"sc_smooth_radius_um={SC_SMOOTH_RADIUS_UM}; "
              f"lateral_clip_frac={LATERAL_CLIP_FRAC}; "
              f"bottom_crop_frac={BOTTOM_CROP_FRAC}", flush=True)
    labels_dir = out / "labels"
    if not args.no_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        raise SystemExit(f"no images found in {src}")
    if args.limit:
        files = files[:args.limit]
    effective_mpp = {p.name: svs_mpp(p) or args.um_per_px for p in files}
    run_config_path = out / "run_config.yaml"
    write_run_config(run_config_path, args, src, out, device, names, files, effective_mpp)
    print(f"Wrote run configuration to {run_config_path}", flush=True)

    csv_path = Path(args.csv_out) if args.csv_out else out / "results.csv"
    prior_rows = {}
    if args.resume and csv_path.exists():
        with csv_path.open(newline="") as f:
            prior_rows = {r["file name"]: r for r in csv.DictReader(f)}

    rows = []

    def _collect(name, row):
        if row:
            rows.append(row)
            print(f"{name}: SC area {row['SC area']:.0f}  Epidermal area {row['Epidermal area']:.0f}", flush=True)

    use_pool = args.workers > 1 and not args.no_postprocess
    executor = ThreadPoolExecutor(max_workers=args.workers) if use_pool else None
    pending = {}  # future -> file name, capped at args.workers so raw images in flight stay bounded

    for i, p in enumerate(files, 1):
        done_marker = out / f"{p.stem}_overlay.jpg" if args.no_postprocess else out / f"{p.stem}_steps.jpg"
        measurement_marker = out / f"{p.stem}_measurement.jpg"
        label_path = labels_dir / f"{p.stem}.txt"
        if args.resume and done_marker.exists() and measurement_marker.exists() and p.stem in prior_rows:
            # fully done in a prior run (steps.jpg + a results.csv row already exist) - reuse that row
            # verbatim so we never re-read the image or re-run postprocess/measure for it.
            print(f"[{i}/{len(files)}] {p.name}: already processed, reusing its results.csv row", flush=True)
            rows.append(prior_rows[p.stem])
            continue
        if args.resume and done_marker.exists() and measurement_marker.exists() and not label_path.exists():
            # done in a prior run before --resume/label-saving existed: no saved mask to cheaply recompute
            # a CSV row from, so just skip re-running (expensive) GPU inference for it entirely.
            print(f"[{i}/{len(files)}] {p.name}: output already exists and no saved label to reuse, "
                  f"skipping (its CSV row will be missing this run)", flush=True)
            continue
        print(f"[{i}/{len(files)}] {p.name}: reading...", flush=True)
        orig_bgr = cv2.imread(str(p))
        if orig_bgr is None:
            print(f"WARNING: could not read {p}, skipping", flush=True)
            continue
        um_per_px = effective_mpp[p.name]
        if um_per_px != args.um_per_px:
            print(f"{p.name}: using embedded MPP {um_per_px:.4g} um/px (overrides --um-per-px {args.um_per_px})",
                  flush=True)
        scale = (um_per_px / args.reference_um_per_px) if um_per_px else 1.0
        if args.resume and label_path.exists():
            print(f"[{i}/{len(files)}] {p.name}: reusing saved labels, skipping GPU inference...", flush=True)
            pred_mask = _rasterise(label_path, *orig_bgr.shape[:2])
        else:
            print(f"[{i}/{len(files)}] {p.name}: running inference ({orig_bgr.shape[1]}x{orig_bgr.shape[0]})...",
                  flush=True)
            pred_mask = predict_mask_safe(model, orig_bgr, device, scale, name=p.name, max_side=args.max_side)
            if device.type == "cuda":
                torch.cuda.empty_cache()  # defragment cache between images so one large image doesn't starve the next
            if not args.no_labels:
                write_yolo_labels(pred_mask, class_index, label_path)

        if args.no_postprocess:
            masks = {n: pred_mask == (k + 1) for k, n in enumerate(names)}
            cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), overlay(orig_bgr, masks), [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{p.name}: wrote raw overlay", flush=True)
            continue

        if executor is None:
            row = process_image(p.stem, src.name, orig_bgr, pred_mask, class_index,
                                 um_per_px or 1.0, out, source_path=p)
            _collect(p.name, row)
            continue

        if len(pending) >= args.workers:  # wait for a slot so at most `workers` raw images are held in memory
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                _collect(pending.pop(fut), fut.result())
        fut = executor.submit(process_image, p.stem, src.name, orig_bgr, pred_mask, class_index,
                               um_per_px or 1.0, out, source_path=p)
        pending[fut] = p.name

    if pending:
        done, _ = wait(pending, return_when=ALL_COMPLETED)
        for fut in done:
            _collect(pending.pop(fut), fut.result())
    if executor:
        executor.shutdown(wait=True)

    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")
    print(f"Processed {len(files)} images -> {out}")


if __name__ == "__main__":
    main()
    # python .\predict.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Desktop\yolo_skin_seg\difficult_test_images\s_ex_9_24_d2_l9c2a.svs" 
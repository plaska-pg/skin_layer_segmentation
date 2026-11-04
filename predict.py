#!/usr/bin/env python3
"""
predict.py -- run the trained SC/Epidermis ResUNet model (see
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
from postprocess import (EPIDERMAL_ERODE_DILATE_UM, FOLLICLE_CONF, FOLLICLE_ENABLED, FOLLICLE_MASK_FRAC,
                         FOLLICLE_PADDING_UM, FOLLICLE_WEIGHTS, GLAND_ERODE_DILATE_UM, LATERAL_CLIP_FRAC, ROI_CONF, ROI_ENABLED, ROI_WEIGHTS,
                         QUALITY_ENABLED, QUALITY_SECTIONS, QUALITY_WEIGHTS,
                         SC_EPIDERMIS_CONTACT_ENABLED, SC_EPIDERMIS_CONTACT_GAP_UM,
                         SC_ERODE_DILATE_UM, imwrite_jpg, long_path, overlay, process_image,
                         write_yolo_labels)  # noqa: E402
from measurement import CSV_COLUMNS, _rasterise  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs"}  # .svs (Aperio) is TIFF-based - cv2 reads its
# full-resolution page directly, ignoring the pyramid/thumbnail sub-IFDs and Aperio-specific tags

_MPP_RE = re.compile(r"MPP\s*=\s*([\d.]+)")
TILE_OVERLAP = 0.25
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
            "csv_output": str((Path(args.csv_out) if args.csv_out else out / "_results.csv").resolve()),
            "reference_um_per_px": args.reference_um_per_px,
            "postprocess": not args.no_postprocess,
            "sc_erod_dil_kernel_size_um": SC_ERODE_DILATE_UM,
            "epidermal_erod_dil_kernel_size_um": EPIDERMAL_ERODE_DILATE_UM,
            "gland_erod_dil_kernel_size_um": GLAND_ERODE_DILATE_UM,
            "sc_epidermis_contact_enabled": SC_EPIDERMIS_CONTACT_ENABLED,
            "sc_epidermis_contact_gap_um": SC_EPIDERMIS_CONTACT_GAP_UM,
            "lateral_clip_frac": LATERAL_CLIP_FRAC,
            "follicle_enabled": FOLLICLE_ENABLED,
            "follicle_weights": str(Path(FOLLICLE_WEIGHTS).resolve()),
            "follicle_conf": FOLLICLE_CONF,
            "follicle_padding_um": FOLLICLE_PADDING_UM,
            "follicle_mask_frac": FOLLICLE_MASK_FRAC,
            "roi_enabled": ROI_ENABLED,
            "roi_weights": str(Path(ROI_WEIGHTS).resolve()),
            "roi_conf": ROI_CONF,
            "quality_enabled": QUALITY_ENABLED,
            "quality_weights": str(Path(QUALITY_WEIGHTS).resolve()),
            "quality_sections": QUALITY_SECTIONS,
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


def _save_svs_tiff(p: Path, tiff_dir: Path, bgr: np.ndarray = None) -> None:
    """Write .svs input p as a lossless full-resolution <tiff_dir>/<stem>.tif, unless it already exists."""
    if p.suffix.lower() != ".svs":
        return
    dst = tiff_dir / f"{p.stem}.tif"
    if long_path(dst).exists():
        return
    if bgr is None:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"WARNING: could not read {p}, no TIFF written", flush=True)
            return
    ok, buf = cv2.imencode(".tif", bgr)
    if not ok:
        raise RuntimeError(f"could not encode {dst}")
    long_path(dst).write_bytes(buf.tobytes())
    print(f"{p.name}: wrote TIFF {dst}", flush=True)


# (quarter turns, horizontal flip) applied to each tile; predictions are mapped back and averaged
TTA_OPS = {
    "none": [(0, False)],
    "flip": [(0, False), (0, True), (2, False), (2, True)],
    "d4": [(k, f) for k in range(4) for f in (False, True)],
}


def _tile_probs(model, x: torch.Tensor, tta: str) -> torch.Tensor:
    ops = TTA_OPS[tta]
    acc = 0
    for k, flip in ops:
        xt = torch.rot90(x, k, (2, 3))
        p = model(xt.flip(3) if flip else xt).softmax(1)
        acc = acc + torch.rot90(p.flip(3) if flip else p, -k, (2, 3))
    return acc / len(ops)


def _ramp(n: int, ramp: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    return np.clip(np.minimum(i + 0.5, n - i - 0.5) / max(ramp, 1), 1e-3, 1.0)


@torch.no_grad()
def _tiled_probs(model, rgb: np.ndarray, device, tile: int, overlap: float = TILE_OVERLAP,
                 tta: str = "none") -> np.ndarray:
    """Softmax probabilities (C, H, W) from overlapping tiles, cross-faded with a
    linear ramp across each overlap so tile edges (poor context) carry ~no weight."""
    h, w = rgb.shape[:2]
    m = int(tile * overlap / 2)
    stride = tile - 2 * m
    ys = sorted(set(range(0, max(1, h - tile), stride)) | {max(0, h - tile)})
    xs = sorted(set(range(0, max(1, w - tile), stride)) | {max(0, w - tile)})
    probs, wsum = None, np.zeros((h, w), np.float32)
    for y0 in ys:
        for x0 in xs:
            patch = rgb[y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[:2]
            patch = cv2.copyMakeBorder(patch, 0, (32 - ph % 32) % 32, 0, (32 - pw % 32) % 32, cv2.BORDER_REFLECT)
            x = torch.from_numpy(((patch.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None].to(device)
            p = _tile_probs(model, x, tta)[0, :, :ph, :pw].float().cpu().numpy()
            if probs is None:
                probs = np.zeros((p.shape[0], h, w), np.float32)
            wt = np.outer(_ramp(ph, 2 * m), _ramp(pw, 2 * m))
            probs[:, y0:y0 + ph, x0:x0 + pw] += p * wt
            wsum[y0:y0 + ph, x0:x0 + pw] += wt
    return probs / wsum


def _upsample_argmax(probs: np.ndarray, oh: int, ow: int, band: int = 512) -> np.ndarray:
    """argmax of probs bilinearly resampled to (oh, ow), in row bands to bound memory.
    Uses cv2.resize's pixel-centre convention, so it inverts the input resize exactly."""
    _, h, w = probs.shape
    map_x = np.broadcast_to(((np.arange(ow, dtype=np.float32) + 0.5) * (w / ow) - 0.5)[None], (band, ow))
    out = np.empty((oh, ow), np.uint8)
    for r0 in range(0, oh, band):
        r1 = min(oh, r0 + band)
        sy = ((np.arange(r0, r1, dtype=np.float32) + 0.5) * (h / oh) - 0.5)[:, None]
        map_y = np.broadcast_to(sy, (r1 - r0, ow)).astype(np.float32)
        mx = np.ascontiguousarray(map_x[:r1 - r0])
        best = np.full((r1 - r0, ow), -1.0, np.float32)
        for c in range(probs.shape[0]):
            pc = cv2.remap(probs[c], mx, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            better = pc > best
            best[better] = pc[better]
            out[r0:r1][better] = c
    return out


def predict_mask(model, orig_bgr: np.ndarray, device, scale: float, tile: int = 1024,
                 tta: str = "none") -> np.ndarray:
    """Class-index mask (0=bg, 1..N=names), at the ORIGINAL image's own
    resolution. The input is resized by `scale` to the model's training um/px
    and run as overlapping `tile` px tiles, so GPU memory stays bounded without
    ever downscaling a large slide below the training resolution. Class
    probabilities (not labels) are resampled back, so edges are sub-pixel smooth."""
    oh, ow = orig_bgr.shape[:2]
    th, tw = (max(1, round(oh * scale)), max(1, round(ow * scale))) if scale != 1.0 else (oh, ow)
    resized = (th, tw) != (oh, ow)
    bgr = cv2.resize(orig_bgr, (tw, th), interpolation=cv2.INTER_AREA if (tw < ow or th < oh) else cv2.INTER_LINEAR) \
        if resized else orig_bgr
    probs = _tiled_probs(model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), device, tile, tta=tta)
    if resized:
        return _upsample_argmax(probs, oh, ow)
    return probs.argmax(0).astype(np.uint8)


def predict_mask_safe(model, orig_bgr: np.ndarray, device, scale: float, name: str = "",
                      tile: int = 1024, tta: str = "none") -> np.ndarray:
    """predict_mask, falling back to CPU (moving the model there and back) on a CUDA
    out-of-memory error - whole-slide images vary enough in size that one image fitting
    on a small GPU (e.g. 4GB) doesn't guarantee the next one will."""
    try:
        return predict_mask(model, orig_bgr, device, scale, tile, tta)
    except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
        if device.type != "cuda":
            raise
        print(f"{name}: GPU out of memory ({e}); clearing cache and retrying on CPU", flush=True)
        torch.cuda.empty_cache()
        model.to("cpu")
        try:
            return predict_mask(model, orig_bgr, torch.device("cpu"), scale, tile, tta)
        finally:
            model.to(device)
            torch.cuda.empty_cache()


def load_model(weights: str, device):
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    names = ckpt["names"]
    class_index = {n: k + 1 for k, n in enumerate(names)}
    model = ResUNet(len(names) + 1, ckpt["encoder"], pretrained=False, full_res=ckpt.get("full_res", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, names, class_index


def main():
    builtins.print = _timestamped_print
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=_PREDICT_CFG.get("weights", "models/best.pt"),
                     help="ResUNet checkpoint trained on SC/Epidermis (dataset/dataset.yaml)")
    ap.add_argument("--source", default=_PREDICT_CFG.get("source", "inference_images"),
                     help="a single image file OR a folder of images to predict on")
    ap.add_argument("--out", default=None,
                     help="default: '<source folder>/predicted_<source folder name>' (for a single image: "
                          "'<image's folder>/PREDICTED_<image stem>')")
    ap.add_argument("--device", default=_PREDICT_CFG.get("device", "auto"))
    ap.add_argument("--um-per-px", type=float, default=_PREDICT_CFG.get("um_per_px", 0.2577),
                     help="native um/px of --source images (default matches inference_images' Motic 40X "
                          "calibration in calibrations/Motic 40X scan calibration.IQC). 0 disables rescaling. "
                          "For .svs files, the scanner's own embedded MPP tag is used instead when present.")
    ap.add_argument("--reference-um-per-px", type=float, default=_PREDICT_CFG.get("reference_um_per_px", 0.645),
                     help="um/px the model was trained at (default matches dataset/scale_calibration.json)")
    ap.add_argument("--tile", type=int, default=_PREDICT_CFG.get("tile", 1024),
                     help="tile size (px, at the training um/px) for sliding-window inference; bounds GPU "
                          "memory without downscaling. Lower it (e.g. 768) if you hit CUDA out-of-memory errors.")
    ap.add_argument("--tta", choices=list(TTA_OPS), default=_PREDICT_CFG.get("tta", "none"),
                     help="test-time augmentation per tile: none, flip (4 passes) or d4 (flips + rot90, 8 passes)")
    ap.add_argument("--no-postprocess", action="store_true",
                     help="skip close/ROI/rotate/measure - just write raw prediction overlays")
    ap.add_argument("--no-labels", action="store_true",
                     help="skip writing <out>/raw_segmentations/<stem>.txt (raw prediction mask) and "
                          "<out>/postprocessed_segmentations/<stem>.txt (final ROI SC/Epidermis + glands, "
                          "mapped back to the original image frame) YOLO-seg polygon labels")
    ap.add_argument("--resume", action="store_true",
                     help="continue a previous (e.g. crashed) run in the same --out: images that already have a "
                          "row in the CSV are skipped and their row is kept; otherwise, if "
                          "<out>/raw_segmentations/<stem>.txt exists it is reused instead of re-running GPU "
                          "inference, and postprocess/measure re-runs")
    ap.add_argument("--csv-out", default=None, help="default: <out>/_results.csv")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N images (for quick test runs)")
    ap.add_argument("--workers", type=int, default=_PREDICT_CFG.get("workers", 1),
                     help="run the CPU-bound postprocess/measure step for up to N images concurrently in "
                          "background threads, overlapping it with the next image's (serial, single-device) "
                          "model inference - cv2/scipy/skimage release the GIL so this gets real parallelism. "
                          "Ignored with --no-postprocess. Keep this <= your CPU core count.")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, names, class_index = load_model(args.weights, device)
    # 'Granular Layer' is the epidermis class name in older checkpoints (e.g. models/best.pt)
    if not args.no_postprocess and not ("SC" in names and {"Epidermis", "Granular Layer"} & set(names)):
        raise SystemExit(f"--weights checkpoint names {names} must include 'SC' and 'Epidermis' "
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
    out_io = long_path(out)
    out_io.mkdir(parents=True, exist_ok=True)
    if not args.no_postprocess:
        print("postprocess hyperparameters: "
              f"roi_enabled={ROI_ENABLED}; roi_conf={ROI_CONF}; "
              f"quality_enabled={QUALITY_ENABLED}; quality_sections={QUALITY_SECTIONS}; "
              f"follicle_enabled={FOLLICLE_ENABLED}; follicle_mask_frac={FOLLICLE_MASK_FRAC}; "
              f"sc_erod_dil_kernel_size_um={SC_ERODE_DILATE_UM}; "
              f"epidermal_erod_dil_kernel_size_um={EPIDERMAL_ERODE_DILATE_UM}; "
              f"gland_erod_dil_kernel_size_um={GLAND_ERODE_DILATE_UM}; "
              f"sc_epidermis_contact_enabled={SC_EPIDERMIS_CONTACT_ENABLED}; "
              f"sc_epidermis_contact_gap_um={SC_EPIDERMIS_CONTACT_GAP_UM}; "
              f"lateral_clip_frac={LATERAL_CLIP_FRAC}", flush=True)
    labels_dir = out_io / "raw_segmentations"
    post_labels_dir = out_io / "postprocessed_segmentations"
    if not args.no_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_postprocess:
            post_labels_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        raise SystemExit(f"no images found in {src}")
    if args.limit:
        files = files[:args.limit]
    src_dir = src.parent if src.is_file() else src
    tiff_dir = src_dir / f"TIFF_{src_dir.name}"
    if any(p.suffix.lower() == ".svs" for p in files):
        long_path(tiff_dir).mkdir(parents=True, exist_ok=True)
    effective_mpp = {p.name: svs_mpp(p) or args.um_per_px for p in files}
    run_config_path = out / "run_config.yaml"
    write_run_config(long_path(run_config_path), args, src, out, device, names, files, effective_mpp)
    print(f"Wrote run configuration to {run_config_path}", flush=True)

    csv_path = Path(args.csv_out) if args.csv_out else out / "_results.csv"
    csv_io = long_path(csv_path)
    prior_rows = {}
    if args.resume and csv_io.exists():
        with csv_io.open(newline="") as f:
            prior_rows = {r["file name"]: r for r in csv.DictReader(f)}

    rows = []
    # prior rows are already in memory, so it's safe to truncate the CSV and rewrite it row by row
    csv_file = csv_io.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()

    def _write_row(row):
        rows.append(row)
        writer.writerow(row)
        csv_file.flush()

    def _collect(name, row):
        if row:
            _write_row(row)
            print(f"{name}: SC area {row['SC area']:.0f}  Epidermal area {row['Epidermal area']:.0f}", flush=True)

    use_pool = args.workers > 1 and not args.no_postprocess
    executor = ThreadPoolExecutor(max_workers=args.workers) if use_pool else None
    pending = {}  # future -> file name, capped at args.workers so raw images in flight stay bounded

    for i, p in enumerate(files, 1):
        label_path = labels_dir / f"{p.stem}.txt"
        post_label_path = None if args.no_labels else post_labels_dir / f"{p.stem}.txt"
        if args.resume:
            if args.no_postprocess and (out_io / f"{p.stem}_overlay.jpg").exists():
                print(f"[{i}/{len(files)}] {p.name}: overlay already exists, skipping", flush=True)
                _save_svs_tiff(p, tiff_dir)
                continue
            if not args.no_postprocess and p.stem in prior_rows:
                print(f"[{i}/{len(files)}] {p.name}: already processed, reusing its CSV row", flush=True)
                _write_row(prior_rows[p.stem])
                _save_svs_tiff(p, tiff_dir)
                continue
        print(f"[{i}/{len(files)}] {p.name}: reading...", flush=True)
        orig_bgr = cv2.imread(str(p))
        if orig_bgr is None:
            print(f"WARNING: could not read {p}, skipping", flush=True)
            continue
        _save_svs_tiff(p, tiff_dir, orig_bgr)
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
            pred_mask = predict_mask_safe(model, orig_bgr, device, scale, name=p.name, tile=args.tile, tta=args.tta)
            if device.type == "cuda":
                torch.cuda.empty_cache()  # defragment cache between images so one large image doesn't starve the next
            if not args.no_labels:
                write_yolo_labels(pred_mask, class_index, label_path)

        if args.no_postprocess:
            masks = {n: pred_mask == (k + 1) for k, n in enumerate(names)}
            imwrite_jpg(out_io / f"{p.stem}_overlay.jpg", overlay(orig_bgr, masks), 85)
            print(f"{p.name}: wrote raw overlay", flush=True)
            continue

        if executor is None:
            row = process_image(p.stem, src.name, orig_bgr, pred_mask, class_index,
                                 um_per_px or 1.0, out_io, source_path=p, seg_path=post_label_path)
            _collect(p.name, row)
            continue

        if len(pending) >= args.workers:  # wait for a slot so at most `workers` raw images are held in memory
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                _collect(pending.pop(fut), fut.result())
        fut = executor.submit(process_image, p.stem, src.name, orig_bgr, pred_mask, class_index,
                               um_per_px or 1.0, out_io, source_path=p, seg_path=post_label_path)
        pending[fut] = p.name

    if pending:
        done, _ = wait(pending, return_when=ALL_COMPLETED)
        for fut in done:
            _collect(pending.pop(fut), fut.result())
    if executor:
        executor.shutdown(wait=True)

    csv_file.close()
    print(f"\nWrote {len(rows)} rows to {csv_path}")
    print(f"Processed {len(files)} images -> {out}")


if __name__ == "__main__":
    main()
    # python .\predict.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Desktop\yolo_skin_seg\difficult_test_images\s_ex_9_24_d2_l9c2a.svs" 
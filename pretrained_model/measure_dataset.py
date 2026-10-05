#!/usr/bin/env python3
"""
measure_dataset.py -- batch SC/Granular Layer segmentation + geometry measurements.

Walks --source for subfolders of images, runs the trained semseg model on each
image (same scale-correction + small-blob/adjacency cleanup as
train_semseg.py's `predict`), rotates+flips each prediction so the SC/Granular
Layer band is horizontal with SC on top (canonicalize_orientation), then
measures, per class:
  - height (thickness): perpendicular-to-tangent thickness (see
    surface_utils.layer_height_stats/layer_thickness_samples) - smooths the
    class's top/bottom boundary contours, then at points along the top
    contour casts a ray along its local normal to the bottom contour. This
    measures true cross-band thickness even where the band is locally tilted
    or curved, unlike a plain vertical column span.
  - width (length): each connected component's own horizontal (x) extent.
  - area: total pixel count for that class.
All three are converted to real micrometres via --um-per-px if given.

Writes one row per image to <out>/stats.csv, and one canonicalized-orientation
overlay per image to <out>/<folder>/<stem>_overlay.jpg (mirroring --source's
subfolder structure).

    python for_transfer/measure_dataset.py --weights runs/semseg/dataset_2cls_finetune_v2/best.pt \\
        --source "C:\\path\\to\\Raw images" --out runs/measurements --um-per-px 0.26
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_semseg import (  # noqa: E402
    MEAN, STD, ResUNet, canonicalize_orientation, colourise,
    filter_by_mutual_touch, filter_small_components, filter_unadjacent_components, pick_device,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from surface_utils import layer_height_stats  # noqa: E402

TOP_CLASS = "SC"
OTHER_CLASS = "Granular Layer"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def class_stats(mask: np.ndarray, class_id: int, px_to_um: float):
    """height stats from perpendicular-to-tangent thickness (robust to a wavy
    band - see surface_utils.layer_height_stats); width stats from each
    connected component's own x-extent; area = pixel count."""
    binary = (mask == class_id).astype(np.uint8)
    if not binary.any():
        return None
    h_stats = layer_height_stats(mask, class_id)  # {abs_min, abs_max, avg, area} in px
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    widths_px = np.array([stats[i, cv2.CC_STAT_WIDTH] for i in range(1, n)])
    return {
        "mean_height": h_stats["avg"] * px_to_um, "min_height": h_stats["abs_min"] * px_to_um,
        "max_height": h_stats["abs_max"] * px_to_um, "mean_width": widths_px.mean() * px_to_um,
        "min_width": widths_px.min() * px_to_um, "max_width": widths_px.max() * px_to_um,
        "area": h_stats["area"] * (px_to_um ** 2),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="root folder containing per-sample subfolders of images")
    ap.add_argument("--out", required=True, help="output root: <out>/stats.csv + <out>/<folder>/<stem>_overlay.jpg")
    ap.add_argument("--um-per-px", type=float, default=None, help="native um/pixel of --source images")
    ap.add_argument("--reference-um-per-px", type=float, default=0.645, help="um/pixel the model was trained at")
    ap.add_argument("--min-blob-px", type=int, default=200)
    ap.add_argument("--adjacency-margin-px", type=int, default=15)
    ap.add_argument("--min-touch-frac", type=float, default=0.4,
                    help="drop SC+Granular Layer entirely for an image unless at least this fraction of "
                         "EACH class's own pixels touch the other (within --adjacency-margin-px). 0 = off")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.weights, map_location=device)
    names = ckpt["names"]
    class_index = {n: k + 1 for k, n in enumerate(names)}
    for needed in (TOP_CLASS, OTHER_CLASS):
        if needed not in class_index:
            raise ValueError(f"{args.weights} has no {needed!r} class (has {names})")
    model = ResUNet(len(names) + 1, ckpt["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scale = (args.um_per_px / args.reference_um_per_px) if args.um_per_px else 1.0
    px_to_um = args.um_per_px if args.um_per_px else 1.0  # measured post scale-restore, i.e. at native resolution

    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = ["folder", "file"]
    for cls in (TOP_CLASS, OTHER_CLASS):
        header += [f"{cls} mean height", f"{cls} mean width", f"{cls} min height", f"{cls} max height",
                   f"{cls} min width", f"{cls} max width", f"{cls} area"]

    folders = sorted({p.parent for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS})
    rows_written = 0
    with (out / "stats.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for folder in folders:
            rel_folder = folder.relative_to(src)
            out_folder = out / rel_folder
            out_folder.mkdir(parents=True, exist_ok=True)
            for p in sorted(q for q in folder.iterdir() if q.suffix.lower() in IMG_EXTS):
                orig_bgr = cv2.imread(str(p))
                if orig_bgr is None:
                    print(f"WARNING: could not read {p}, skipping")
                    continue
                oh, ow = orig_bgr.shape[:2]
                bgr = cv2.resize(orig_bgr, (max(1, round(ow * scale)), max(1, round(oh * scale))),
                                  interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR) \
                    if scale != 1.0 else orig_bgr
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
                rgb_p = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                x = torch.from_numpy(((rgb_p.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None].to(device)
                pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)[:h, :w]
                if scale != 1.0:
                    pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)

                pred = filter_small_components(pred, args.min_blob_px)
                pred = filter_unadjacent_components(pred, [(TOP_CLASS, OTHER_CLASS)], args.adjacency_margin_px, class_index)
                if args.min_touch_frac > 0:
                    pred = filter_by_mutual_touch(pred, class_index[TOP_CLASS], class_index[OTHER_CLASS],
                                                   args.min_touch_frac, args.adjacency_margin_px)
                img_r, mask_r = canonicalize_orientation(orig_bgr, pred, class_index[TOP_CLASS], class_index[OTHER_CLASS])

                row = [str(rel_folder), p.name]
                for cls in (TOP_CLASS, OTHER_CLASS):
                    s = class_stats(mask_r, class_index[cls], px_to_um)
                    row += [round(s["mean_height"], 2), round(s["mean_width"], 2), round(s["min_height"], 2),
                            round(s["max_height"], 2), round(s["min_width"], 2), round(s["max_width"], 2),
                            round(s["area"], 2)] if s else ["", "", "", "", "", "", 0]
                writer.writerow(row)
                rows_written += 1

                cv2.imwrite(str(out_folder / f"{p.stem}_overlay.jpg"), colourise(mask_r, img_r), [cv2.IMWRITE_JPEG_QUALITY, 85])
                print(f"{rel_folder}/{p.name}: done")

    print(f"\nWrote {rows_written} rows to {out / 'stats.csv'}")


if __name__ == "__main__":
    main()

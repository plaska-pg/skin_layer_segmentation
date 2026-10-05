#!/usr/bin/env python3
"""
show_train_samples.py -- montage of augmented training crops (image | label
overlay) from a train_semseg.py dataset, so you can eyeball what the model
actually trains on (augmentations + rasterised labels, pseudolabels included).

    python show_train_samples.py                       # all train images
    python show_train_samples.py --pseudo-only         # only S_EX_9_24_D2_* pseudolabels
    python show_train_samples.py --count 40 --cols 5

Writes a single montage JPG (default runs/train_samples_preview.jpg). Each panel
is one augmented crop on the left and the same crop with its SC/Granular-Layer
label overlaid on the right, captioned with the source filename.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "pretrained_model"))
from train_semseg import MEAN, STD, PALETTE, TileDataset, colourise  # noqa: E402

PSEUDO_PREFIX = "S_EX_9_24_D2_"  # pseudolabeled D2_H_E images added by build_pseudolabel_finetune.py


def denorm(x: np.ndarray) -> np.ndarray:
    """model-normalised CHW tensor -> uint8 BGR image."""
    img = ((x.transpose(1, 2, 0) * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def caption(panel: np.ndarray, text: str, is_pseudo: bool) -> np.ndarray:
    bar = np.full((28, panel.shape[1], 3), (40, 40, 40), np.uint8)
    colour = (60, 220, 255) if is_pseudo else (230, 230, 230)
    tag = "[PSEUDO] " if is_pseudo else ""
    cv2.putText(bar, tag + text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_pseudolabel_finetune")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--count", type=int, default=40, help="number of augmented sample panels")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--panel-h", type=int, default=300, help="height each crop is shown at in the montage")
    ap.add_argument("--pseudo-only", action="store_true", help="sample only the pseudolabeled images")
    ap.add_argument("--out", default="runs/train_samples_preview.jpg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    ds = TileDataset(Path(args.data), "train", crop=args.crop, augment=True)
    idxs = list(range(len(ds)))
    if args.pseudo_only:
        idxs = [i for i in idxs if ds.images[i].stem.startswith(PSEUDO_PREFIX)]
    if not idxs:
        raise SystemExit("no matching train images")
    print(f"{len(idxs)} source images, drawing {args.count} augmented panels")

    # cycle through the eligible images so every one appears, re-sampling (new random
    # augmentation each __getitem__) until we have `count` panels
    order = (idxs * (args.count // len(idxs) + 1))[:args.count]
    random.shuffle(order)

    panels = []
    for i in order:
        x, y, name = ds[i]
        img = denorm(x.numpy())
        ov = colourise(y.numpy().astype(np.int32), img)
        pair = np.hstack([img, ov])
        h, w = pair.shape[:2]
        s = args.panel_h / h
        pair = cv2.resize(pair, (int(w * s), args.panel_h))
        is_pseudo = Path(name).stem.startswith(PSEUDO_PREFIX)
        panels.append(caption(pair, Path(name).stem, is_pseudo))

    pw = max(p.shape[1] for p in panels)
    ph = panels[0].shape[0]
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, pw - p.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for p in panels]
    cols = args.cols
    rows = (len(panels) + cols - 1) // cols
    grid = np.full((rows * (ph + 6), cols * (pw + 6), 3), (0, 0, 0), np.uint8)
    for k, p in enumerate(panels):
        r, c = divmod(k, cols)
        grid[r * (ph + 6):r * (ph + 6) + ph, c * (pw + 6):c * (pw + 6) + pw] = p

    # legend strip: which colour is which class
    legend = np.full((34, grid.shape[1], 3), (30, 30, 30), np.uint8)
    for j, (cls, col) in enumerate([("SC", PALETTE[0]), ("Granular Layer", PALETTE[1])]):
        x0 = 10 + j * 260
        cv2.rectangle(legend, (x0, 8), (x0 + 24, 26), tuple(int(v) for v in col), -1)
        cv2.putText(legend, cls, (x0 + 32, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    grid = np.vstack([legend, grid])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()

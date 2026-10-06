#!/usr/bin/env python3
"""
build_pseudolabel_finetune.py -- assemble a fine-tuning dataset that combines the
existing human-labeled data with model-predicted "pseudolabels" for the D2_H_E
images flagged as having good segmentations in pseudolabeling.md.

For each good image it renders the source Aperio .svs (read full-resolution by
cv2, then downscaled to the training reference scale of 0.645 um/px so the PNG is
small and no scale_calibration.json entry is needed - the normalized YOLO-seg
polygons are scale-invariant) into images/train, and copies the predicted
<stem>.txt label into labels/train. The source dataset's own train/val splits,
data.yaml and scale_calibration.json are copied through unchanged.

    python build_pseudolabel_finetune.py
"""

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # whole-slide SVS dwarf PIL's decompression-bomb cap

PROJECT = Path(__file__).resolve().parent
SRC_DATASET = PROJECT / "dataset_for_finetuning"
OUT_DATASET = PROJECT / "dataset_pseudolabel_finetune"

RAW = Path(r"C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts"
           r"\W Cheng Section (BDT-Skin) - Histology\Raw images")
SVS_DIR = RAW / "S-EX_SEP_2024_D2_H_E"
LABEL_DIR = RAW / f"predicted_{SVS_DIR.name}" / "labels"

REFERENCE_UM_PER_PX = 0.645  # train_semseg.py reference scale (dataset_for_finetuning/scale_calibration.json)
FALLBACK_UM_PER_PX = 0.257732  # Motic 40X value recorded for this series if a .svs lacks the MPP tag

# stems flagged as good in pseudolabeling.md (all go into train per user)
GOOD_STEMS = [
    "S_EX_9_24_D2_L1C1",
    "S_EX_9_24_D2_L1C2",
    "S_EX_9_24_D2_L1C3",
    "S_EX_9_24_D2_L7C1",
    "S_EX_9_24_D2_L7C2",
    "S_EX_9_24_D2_L7C4",
    "S_EX_9_24_D2_L8C1",
    "S_EX_9_24_D2_L8C1-42Y5X84-W10",
    "S_EX_9_24_D2_L8C2",
    "S_EX_9_24_D2_L8C3",
    "S_EX_9_24_D2_L10C3",
    "S_EX_9_24_D2_L10C4",
    "S_EX_9_24_D2_L11C1",
]

_MPP_RE = re.compile(r"MPP\s*=\s*([\d.]+)")


def svs_mpp(path: Path) -> float | None:
    """Native um/px from an Aperio .svs ImageDescription tag, or None."""
    try:
        with Image.open(path) as im:
            desc = str(im.tag_v2.get(270, ""))
    except Exception:
        return None
    m = _MPP_RE.search(desc)
    return float(m.group(1)) if m else None


def main():
    if not SRC_DATASET.exists():
        raise SystemExit(f"source dataset not found: {SRC_DATASET}")

    if OUT_DATASET.exists():
        print(f"removing existing {OUT_DATASET}")
        shutil.rmtree(OUT_DATASET)
    print(f"copying {SRC_DATASET.name} -> {OUT_DATASET.name}")
    shutil.copytree(SRC_DATASET, OUT_DATASET)

    img_train = OUT_DATASET / "images" / "train"
    lab_train = OUT_DATASET / "labels" / "train"
    img_train.mkdir(parents=True, exist_ok=True)
    lab_train.mkdir(parents=True, exist_ok=True)

    added = 0
    for stem in GOOD_STEMS:
        svs = SVS_DIR / f"{stem}.svs"
        lab = LABEL_DIR / f"{stem}.txt"
        if not svs.exists():
            print(f"SKIP {stem}: missing .svs")
            continue
        if not lab.exists():
            print(f"SKIP {stem}: missing predicted label")
            continue

        img = cv2.imread(str(svs))
        if img is None:
            print(f"SKIP {stem}: cv2 could not read {svs.name}")
            continue

        mpp = svs_mpp(svs) or FALLBACK_UM_PER_PX
        scale = mpp / REFERENCE_UM_PER_PX
        h, w = img.shape[:2]
        if scale < 1.0:
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            nh, nw = h, w

        out_png = img_train / f"{stem}.png"
        cv2.imwrite(str(out_png), img)
        shutil.copyfile(lab, lab_train / f"{stem}.txt")
        added += 1
        print(f"added {stem}: {w}x{h} @ {mpp:.4g} um/px -> {nw}x{nh} @ {REFERENCE_UM_PER_PX} um/px")

    n_img = len(list(img_train.glob("*")))
    n_lab = len(list(lab_train.glob("*.txt")))
    print(f"\ndone: added {added} pseudolabeled images")
    print(f"{OUT_DATASET.name} train: {n_img} images, {n_lab} labels")


if __name__ == "__main__":
    main()

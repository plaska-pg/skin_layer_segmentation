"""Sweep postprocessing settings while running segmentation inference once.

Example:
    python hyperparam_tuning.py --source difficult_test_images

Each configuration gets a YAML file and one vertically stacked grid image in
``runs/difficult_test_images_tuning/hyper_param_configuration_N``. The x-axis
contains the processing steps and the y-axis contains the source images.
"""

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from config import CONFIG
from postprocess import (LATERAL_CLIP_FRAC, MIN_TOUCH_FRAC, SC_SMOOTH_RADIUS_UM,
                         TOUCH_MARGIN_UM, CLOSE_RADII_UM, process_image)
from predict import IMAGE_EXTS, load_model, predict_mask_safe, svs_mpp
from pretrained_model.train_semseg import pick_device


def _floats(value: str):
    return [float(item) for item in value.split(",") if item.strip()]


def _radius_schedules(value: str):
    return [_floats(schedule) for schedule in value.split(";") if schedule.strip()]


def _grid(rows, target_height=360):
    if not rows:
        return None
    resized_rows = []
    for row in rows:
        scale = target_height / row.shape[0]
        image = cv2.resize(row, (max(1, round(row.shape[1] * scale)), target_height))
        resized_rows.append(image)
    width = max(row.shape[1] for row in resized_rows)
    resized_rows = [
        cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1],
                           cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for row in resized_rows
    ]
    return np.vstack(resized_rows)


def _cache_predictions(args, source, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    names = checkpoint["names"]
    class_index = {name: index + 1 for index, name in enumerate(names)}
    if not {"SC", "Granular Layer"} <= set(names):
        raise SystemExit(f"weights must contain SC and Granular Layer, got {names}")

    files = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[:args.limit]
    cached = []
    model = None
    device = None
    for index, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"WARNING: could not read {path}; skipping")
            continue
        mpp = svs_mpp(path) or args.um_per_px
        mask_path = cache_dir / f"{path.stem}.npy"
        mask = None
        if not args.force_inference and mask_path.exists():
            try:
                candidate = np.load(mask_path)
                if candidate.shape == image.shape[:2] and candidate.ndim == 2:
                    mask = candidate.astype(np.uint8, copy=False)
                    print(f"[{index}/{len(files)}] {path.name}: reused cached prediction")
                else:
                    print(f"[{index}/{len(files)}] {path.name}: cached mask shape mismatch; rerunning inference")
            except (OSError, ValueError) as error:
                print(f"[{index}/{len(files)}] {path.name}: could not load cached mask ({error}); rerunning inference")
        if mask is None:
            if model is None:
                device = pick_device(args.device)
                model, loaded_names, class_index = load_model(args.weights, device)
                if loaded_names != names:
                    raise SystemExit("checkpoint names changed while loading model")
            scale = mpp / args.reference_um_per_px if mpp else 1.0
            print(f"[{index}/{len(files)}] {path.name}: running inference")
            mask = predict_mask_safe(model, image, device, scale, name=path.name, max_side=args.max_side)
            np.save(mask_path, mask)
        cv2.imwrite(str(cache_dir / f"{path.stem}_raw.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        cached.append((path, image, mask, mpp or 1.0))
    return cached, class_index


def main():
    predict_cfg = CONFIG.get("predict", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="difficult_test_images")
    parser.add_argument("--out", default="runs/difficult_test_images_tuning")
    parser.add_argument("--weights", default=predict_cfg.get("weights", "runs/semseg/dataset_2cls_finetune_v2/best.pt"))
    parser.add_argument("--device", default=predict_cfg.get("device", "auto"))
    parser.add_argument("--um-per-px", type=float, default=predict_cfg.get("um_per_px", 0.645))
    parser.add_argument("--reference-um-per-px", type=float, default=predict_cfg.get("reference_um_per_px", 0.645))
    parser.add_argument("--max-side", type=int, default=predict_cfg.get("max_side") or 4500,
                        help="maximum model-input edge in pixels; output masks keep native image size (0 disables)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="deprecated; cached masks are reused automatically")
    parser.add_argument("--force-inference", action="store_true",
                        help="ignore existing cached masks and regenerate them")
    parser.add_argument("--close-radii-um", default="12.9,25.8,51.6,103.2;25.8,51.6,103.2,206.4",
                        help="semicolon-separated radius schedules; commas separate radii")
    parser.add_argument("--touch-margin-um", default="3.225,6.45,12.9")
    parser.add_argument("--min-touch-frac", default="0.2,0.35,0.5")
    parser.add_argument("--sc-smooth-radius-um", default="0,3.225,6.45")
    parser.add_argument("--lateral-clip-frac", default="0,0.05,0.10")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    cache_dir = source.parent / "difficult_test_images_tuning"
    cached, class_index = _cache_predictions(args, source, cache_dir)
    if not cached:
        raise SystemExit(f"no readable images found in {source}")

    values = {
        "close_radii_um": _radius_schedules(args.close_radii_um),
        "touch_margin_um": _floats(args.touch_margin_um),
        "min_touch_frac": _floats(args.min_touch_frac),
        "sc_smooth_radius_um": _floats(args.sc_smooth_radius_um),
        "lateral_clip_frac": _floats(args.lateral_clip_frac),
    }
    keys = list(values)
    configurations = [dict(zip(keys, combination)) for combination in product(*(values[key] for key in keys))]
    for number, settings in enumerate(configurations, 1):
        config_dir = out / f"hyper_param_configuration_{number}"
        config_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        successful = []
        for path, image, mask, mpp in cached:
            row, composite = process_image(
                path.stem, source.name, image, mask, class_index, mpp, config_dir,
                source_path=path, postprocess_config=settings, return_visuals=True,
                write_outputs=False)
            if composite is not None:
                successful.append(row)
                panels.append(composite)
        grid = _grid(panels)
        if grid is not None:
            cv2.imwrite(str(config_dir / "tuning_grid.jpg"), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        resolved = {
            "created_at": datetime.now().astimezone().isoformat(),
            "source": str(source.resolve()),
            "configuration_index": number,
            "weights": str(Path(args.weights).resolve()),
            "inference_cache": str(cache_dir.resolve()),
            "parameters": settings,
            "images": [path.name for path, _, _, _ in cached],
            "successful_postprocess_images": len(successful),
        }
        (config_dir / "configuration.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        print(f"configuration {number}/{len(configurations)}: wrote {config_dir}")


if __name__ == "__main__":
    main()# tries lots of different hyperparameter combinations on difficult_test_images,
# first performs predict.py segmentation inference on the images, saves the segmentations (saved in difficult_test_images_tuning)
# then tries different posprocessing hyperparameter combinations on the saved segmentations (saved in difficult_test_images_tuning)
# saves output to runs/difficult_test_images_tuning/hyper_param_configuration_n (where n is the configuration index)
# in hyper_param_configuration_n, there should be one image with the steps along the x axis, and all the images should be piled on the y axis (with each of their postprocessing steps on the x axis) and the yaml file with the used configuration





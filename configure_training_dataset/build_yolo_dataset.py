r"""One-pass Image-Pro tif -> YOLO-segmentation dataset entry.

Combines extract_polyline_overlay.py's tag parsing with convert_to_coco.py's
per-instance polygon extraction (convert_one()) to populate the YOLO-format
dataset used by yolo_train.py directly - no separate preview step and no
intermediate COCO annotations.json:

    dataset/images/<split>/<name>.png   - tif converted to PNG
    dataset/labels/<split>/<name>.txt   - one normalized polygon per instance
    dataset/dataset.yaml                 - class registry (id assigned/added as needed)

You choose the split per run - there's no automatic train/val splitting.

Usage:
    python configure_training_dataset/build_yolo_dataset.py "training_dataset/images/GMDE 217 1 N-1 10X_seg.tif" --split train
    python configure_training_dataset/build_yolo_dataset.py "training_dataset/images/NEW_FILE.tif" --split val
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_coco import convert_one


def load_dataset_yaml(dataset_yaml_path: Path, dataset_dir: Path):
    if dataset_yaml_path.exists():
        data = yaml.safe_load(dataset_yaml_path.read_text()) or {}
    else:
        data = {}
    names = {int(k): v for k, v in (data.get("names") or {}).items()}
    data["names"] = names
    data["path"] = str(dataset_dir.resolve())
    data.setdefault("train", "images/train")
    data.setdefault("val", "images/val")
    return data


def class_id_for(names: dict, class_name: str) -> int:
    """Look up (or register) a stable class id by name, case/whitespace-
    insensitive, so tifs whose registered class casing differs (e.g.
    "Granular layer" vs "Granular Layer") still map to the same id."""
    lookup = {v.strip().lower(): k for k, v in names.items()}
    key = class_name.strip().lower()
    if key in lookup:
        return lookup[key]
    new_id = max(names.keys(), default=-1) + 1
    names[new_id] = class_name
    return new_id


def ring_to_yolo_line(ring: np.ndarray, class_id: int, width: int, height: int) -> str:
    """One YOLO-segmentation label line: `class_id x1 y1 x2 y2 ... xn yn`,
    coordinates normalized to [0, 1] (clipped - traced points can fall
    slightly outside the image bounds)."""
    xs = np.clip(ring[:, 0] / width, 0.0, 1.0)
    ys = np.clip(ring[:, 1] / height, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in np.column_stack([xs, ys]).flatten())
    return f"{class_id} {coords}"


def build_entry(tif_path: str, dataset_dir: Path, split: str, names: dict,
                points_tag: int, class_tag: int, overlay_tag: int, overwrite: bool,
                allowed_names: set[str] | None = None):
    stem = Path(tif_path).stem
    image_out = dataset_dir / "images" / split / f"{stem}.png"
    label_out = dataset_dir / "labels" / split / f"{stem}.txt"

    if not overwrite:
        for other_split in ("train", "val"):
            other_image = dataset_dir / "images" / other_split / f"{stem}.png"
            if other_image.exists() and other_image != image_out:
                print(f"WARNING: {stem} already exists in '{other_split}' - now also adding it to '{split}'")
        if image_out.exists() or label_out.exists():
            print(f"Skipping {stem} - already present in '{split}' (pass --overwrite to replace)")
            return

    width, height, rings_by_name = convert_one(tif_path, points_tag, class_tag, overlay_tag)

    lines = []
    for name, rings in rings_by_name.items():
        if allowed_names is not None and name.strip().lower() not in allowed_names:
            continue
        class_id = class_id_for(names, name)
        for ring in rings:
            if len(ring) < 3:
                continue
            lines.append(ring_to_yolo_line(ring, class_id, width, height))

    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    Image.open(tif_path).convert("RGB").save(image_out)
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"{stem}: {width}x{height}, {len(lines)} instance(s) across {list(rings_by_name)} -> '{split}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tif_paths", nargs="+")
    parser.add_argument("--split", choices=["train", "val"], required=True,
                         help="Which dataset split to write this tif's image/label into.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--overwrite", action="store_true",
                         help="Replace an existing image/label for this tif in this split instead of skipping it.")
    parser.add_argument("--points-tag", type=int, default=38124)
    parser.add_argument("--class-tag", type=int, default=38129)
    parser.add_argument("--overlay-tag", type=int, default=38123)
    parser.add_argument(
        "--classes",
        help="Comma-separated class names to keep, e.g. 'SC,Granular Layer'. "
        "Unlisted source classes are discarded.",
    )
    args = parser.parse_args()

    dataset_yaml_path = args.dataset_dir / "dataset.yaml"
    data = load_dataset_yaml(dataset_yaml_path, args.dataset_dir)
    names = data["names"]
    allowed_names = None
    if args.classes:
        requested = [name.strip() for name in args.classes.split(",") if name.strip()]
        allowed_names = {name.lower() for name in requested}
        names = {index: name for index, name in enumerate(requested)}
        data["names"] = names

    for tif_path in args.tif_paths:
        build_entry(tif_path, args.dataset_dir, args.split, names,
                    args.points_tag, args.class_tag, args.overlay_tag, args.overwrite,
                    allowed_names)

    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Saved -> {dataset_yaml_path} (classes: {names})")


if __name__ == "__main__":
    main()

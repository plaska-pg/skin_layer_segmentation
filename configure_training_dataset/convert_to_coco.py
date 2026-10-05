r"""Convert Image-Pro manual-measurement tif(s) into a COCO instance-
segmentation JSON, reusing the boundary-curve/class-registry parsing and
raster-blob extraction from extract_polyline_overlay.py (no rasterization
needed - polygons, area and bbox are all computed directly from the traced
point geometry).

Usage:
    python configure_training_dataset/convert_to_coco.py "training_dataset/images/GMDE 217 1 N-1 10X_seg.tif" --out training_dataset/annotations/annotations.json --merge

Pass multiple tif paths (or a directory via --images-dir) to build/extend one
combined dataset with shared, deduplicated categories. Use --merge to append
to an existing --out file (new images only - files already present by
file_name are skipped) instead of overwriting it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_polyline_overlay import (
    parse_polylines,
    resolve_classes,
    band_polygons_direct,
    parse_raster_polygons,
)


def polygon_area(ring: np.ndarray) -> float:
    """Shoelace formula for one Nx2 polygon ring."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ring_to_annotation(ring: np.ndarray, image_id: int, category_id: int, ann_id: int):
    """Build one COCO annotation dict from a single polygon ring - one
    annotation per instance (e.g. each Follicle blob gets its own
    annotation, matching the existing annotations.json convention)."""
    x, y = ring[:, 0], ring[:, 1]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [np.round(ring.flatten(), 3).tolist()],
        "area": round(float(polygon_area(ring)), 1),
        "bbox": [
            round(float(x.min()), 1),
            round(float(y.min()), 1),
            round(float(x.max() - x.min()), 1),
            round(float(y.max() - y.min()), 1),
        ],
        "iscrowd": 0,
    }


def convert_one(tif_path: str, points_tag: int, class_tag: int, overlay_tag: int):
    """Return (width, height, {category_name: [ring, ring, ...]}) for one tif."""
    img = Image.open(tif_path)
    width, height = img.size

    curves = parse_polylines(tif_path, tag=points_tag)
    classes = resolve_classes(tif_path, tag=class_tag)
    class_by_index = {c["index"]: c for c in classes}

    rings_by_name = {}
    if curves:
        bands_by_class, _ = band_polygons_direct(curves, classes)
        for class_id, polygons in bands_by_class.items():
            name = class_by_index[class_id]["name"]
            rings_by_name.setdefault(name, []).extend(polygons)

    raster_polygons = parse_raster_polygons(tif_path, tag=overlay_tag)
    if raster_polygons:
        # every raster CmGraphOverlay blob observed in this project is a
        # standalone Follicle instance - see extract_polyline_overlay.add_raster_blobs
        follicle = next((c for c in classes if c["name"] == "Follicle"), None)
        if follicle is not None:
            rings_by_name.setdefault(follicle["name"], []).extend(raster_polygons)

    return width, height, rings_by_name


def build_coco(tif_paths, points_tag: int, class_tag: int, overlay_tag: int,
                existing: dict | None = None):
    """Build (or extend `existing`) a COCO dict from the given tif paths."""
    images = list(existing["images"]) if existing else []
    annotations = list(existing["annotations"]) if existing else []
    categories = list(existing["categories"]) if existing else []

    # name lookup is case/whitespace-insensitive so e.g. "Granular layer" and
    # "Granular Layer" (seen across different source files) merge into one
    # category instead of silently duplicating
    name_to_id = {c["name"].strip().lower(): c["id"] for c in categories}
    next_category_id = max((c["id"] for c in categories), default=0) + 1

    existing_file_names = {img["file_name"] for img in images}
    next_image_id = max((img["id"] for img in images), default=0) + 1
    next_ann_id = max((a["id"] for a in annotations), default=0) + 1

    for tif_path in tif_paths:
        file_name = Path(tif_path).name
        if file_name in existing_file_names:
            print(f"Skipping {file_name} - already present in output")
            continue

        width, height, rings_by_name = convert_one(tif_path, points_tag, class_tag, overlay_tag)
        image_id = next_image_id
        next_image_id += 1
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        for name, rings in rings_by_name.items():
            key = name.strip().lower()
            if key not in name_to_id:
                name_to_id[key] = next_category_id
                categories.append({"id": next_category_id, "name": name})
                next_category_id += 1
            for ring in rings:
                annotations.append(ring_to_annotation(ring, image_id, name_to_id[key], next_ann_id))
                next_ann_id += 1

        print(f"{file_name}: {width}x{height}, classes {list(rings_by_name)}")

    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tif_paths", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("coco_annotations.json"))
    parser.add_argument("--merge", action="store_true",
                         help="Append to --out if it already exists, instead of overwriting it.")
    parser.add_argument("--points-tag", type=int, default=38124)
    parser.add_argument("--class-tag", type=int, default=38129)
    parser.add_argument("--overlay-tag", type=int, default=38123)
    args = parser.parse_args()

    existing = None
    if args.out.exists():
        if not args.merge:
            parser.error(f"{args.out} already exists - pass --merge to append to it, or choose a different --out")
        existing = json.loads(args.out.read_text())

    coco = build_coco(args.tif_paths, args.points_tag, args.class_tag, args.overlay_tag, existing=existing)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(coco, indent=2))
    print(f"Saved -> {args.out} ({len(coco['images'])} images, {len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories)")


if __name__ == "__main__":
    main()

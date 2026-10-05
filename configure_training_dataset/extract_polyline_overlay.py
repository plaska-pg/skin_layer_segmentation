r"""Extract MediaCy Image-Pro manual-measurement polyline segmentation + class
labels embedded in TIFF tags 38124 (points) and 38129 (class settings).

Unlike the raster CmGraphOverlay format (tag 38123), this variant stores the
segmentation as open boundary curves (polylines). The region between two
consecutive curves (sorted top-to-bottom by mean y) is filled to form one
labeled class band.

Usage:
    python configure_training_dataset/extract_polyline_overlay.py "training_dataset/images/GMDE 217 1 N-1 10X_seg.tif"
"""
import argparse
import base64
import re
from collections import defaultdict
from functools import cmp_to_key
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Class registry (McSettings tag 38129) observed across the annotated files in
# this project (e.g. "GMDE 217 1 N-1 10X_seg.tif") - used as a fallback for
# tifs whose tag 38129 wasn't embedded, so their boundary curves (tag 38124)
# can still be turned into a labeled mask via the same class_value ordering.
DEFAULT_CLASSES = [
    {"index": 1, "name": "Granular Layer", "known_color_id": None, "class_value": 4},
    {"index": 2, "name": "SC", "known_color_id": None, "class_value": 3},
    {"index": 3, "name": "Follicle", "known_color_id": None, "class_value": 5},
    {"index": 4, "name": "Gland", "known_color_id": None, "class_value": 6},
]


def parse_polylines(tif_path: str, tag: int = 38124):
    """Return list of (name, Nx2 float array) for each Type=4 polyline feature."""
    img = Image.open(tif_path)
    root = ET.fromstring(img.tag_v2[tag])
    curves = []
    for feat in root.findall("Feature"):
        if feat.findtext("Type") != "4":
            continue
        name = feat.findtext("FeatureName") or ""
        raw_pts = (feat.findtext("Points") or "").strip()
        if not raw_pts:
            continue
        vals = [float(v) for v in raw_pts.split(";") if v.strip() != ""]
        pts = np.array(vals, dtype=float).reshape(-1, 2)
        curves.append((name, pts))
    return curves


def _local_tag(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def parse_class_settings(tif_path: str, tag: int = 38129):
    """Return list of {index, name, color} sorted by class index, from the
    Hashtable-serialized McSettings XML (base64 encoded).

    Uses XML tree traversal (rather than regex) and only reads *direct*
    children of Keys/Values, since some values are themselves nested
    arrays which would otherwise throw off a flat key/value zip."""
    img = Image.open(tif_path)
    root = ET.fromstring(base64.b64decode(img.tag_v2[tag]))

    keys_el = next(el for el in root.iter() if _local_tag(el.tag) == "Keys")
    values_el = next(el for el in root.iter() if _local_tag(el.tag) == "Values")
    keys = [c.text for c in keys_el]
    values = [None if len(c) > 0 else c.text for c in values_el]
    settings = dict(zip(keys, values))

    num_classes = int(settings.get("NumClasses", 0))
    classes = []
    for i in range(1, num_classes + 1):
        name = settings.get(f"ClassDisplayName{i}", f"Class {i}")
        color_xml = settings.get(f"ClassColor{20 + i}", "") or ""
        known = re.search(r"<knownColor[^>]*>(\d+)</knownColor>", color_xml)
        class_value = settings.get(f"ClassValue{i}")
        classes.append({
            "index": i,
            "name": name,
            "known_color_id": known.group(1) if known else None,
            "class_value": int(class_value) if class_value is not None else i,
        })
    return classes


def resolve_classes(tif_path: str, tag: int = 38129):
    """parse_class_settings, but fall back to DEFAULT_CLASSES when tag 38129
    is missing entirely, or when it's present but only registers a single
    generic, unnamed placeholder class (e.g. "Class 1" with no
    ClassDisplayName set) - seen in some manual-measurement exports where
    the class registry wasn't filled in even though real SC/Granular Layer
    boundary curves were traced."""
    try:
        classes = parse_class_settings(tif_path, tag=tag)
    except KeyError:
        classes = None
    if not classes or (len(classes) == 1 and re.fullmatch(r"Class \d+", classes[0]["name"].strip())):
        return DEFAULT_CLASSES
    return classes


# A small palette to render each class distinctly (falls back if >len colors)
_PALETTE = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0), (200, 0, 200)]


def _resample_curve(pts: np.ndarray, xs_common: np.ndarray) -> np.ndarray:
    """Interpolate y=f(x) for a curve (sorted by x) at the given x samples.
    Traced curves can double back on themselves (loops/near-vertical
    stretches) producing several y's for the same x - average those instead
    of arbitrarily keeping one, otherwise the polygon fill can jump between
    unrelated points and self-intersect (visible as jagged spikes/drips)."""
    order = np.argsort(pts[:, 0])
    xs, ys = pts[order, 0], pts[order, 1]
    xs_unique, inverse = np.unique(xs, return_inverse=True)
    ys_avg = np.bincount(inverse, weights=ys) / np.bincount(inverse)
    return np.interp(xs_common, xs_unique, ys_avg)


def _sort_curves_top_to_bottom(curves):
    """Order curves using pairwise local comparisons (over each pair's own
    overlapping x-range), since curves span different x extents and a
    single global mean-y sort can misorder them."""
    def compare(a, b):
        pts_a, pts_b = a[1], b[1]
        x_lo = max(pts_a[:, 0].min(), pts_b[:, 0].min())
        x_hi = min(pts_a[:, 0].max(), pts_b[:, 0].max())
        xs = np.linspace(x_lo, x_hi, 200)
        diff = _resample_curve(pts_a, xs) - _resample_curve(pts_b, xs)
        return -1 if diff.mean() < 0 else 1

    return sorted(curves, key=cmp_to_key(compare))


def _bands_to_classes(num_bands: int, classes):
    """Map each top-to-bottom band to a class.

    - 1 class total: every band merges into that single class (some files
      trace a class with more than 2 boundary curves).
    - bands == classes: 1:1, in class_value order.
    - bands < classes: only a subset of the registered classes are present
      in this image (e.g. blob classes like Follicle/Gland with zero
      instances here) - the classes with the smallest class_value are the
      ones traced as boundary curves, so keep just those."""
    if len(classes) == 1:
        return [classes[0]["index"]] * num_bands
    ordered = sorted(classes, key=lambda c: c["class_value"])
    if num_bands > len(ordered):
        raise ValueError(
            f"Expected at most {len(ordered)} boundary curves for {len(ordered)} classes, got {num_bands + 1}"
        )
    return [c["index"] for c in ordered[:num_bands]]


def build_label_mask(curves, classes, image_size, n_samples: int = 500, smooth_sigma: float = 8.0):
    """Sort curves top-to-bottom (pairwise) and fill the band between each
    consecutive pair, assigning bands to classes via `_bands_to_classes`.

    Both curves are resampled onto a shared x grid (the union of their x
    extents - np.interp holds each curve's edge value flat beyond its own
    range, rather than the overlap only, so bands still reach the image's
    left/right edges even when the two curves don't span the same width)
    and Gaussian-smoothed (`smooth_sigma`, in samples) so the polygon edges
    don't self-intersect or zigzag."""
    from scipy.ndimage import gaussian_filter1d

    num_bands = len(curves) - 1
    if num_bands < 1:
        raise ValueError(f"Need at least 2 boundary curves, got {len(curves)}")
    band_class_ids = _bands_to_classes(num_bands, classes)
    curves_sorted = _sort_curves_top_to_bottom(curves)

    w, h = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for band_idx in range(num_bands):
        class_id = band_class_ids[band_idx]
        top_name, top_pts = curves_sorted[band_idx]
        bot_name, bot_pts = curves_sorted[band_idx + 1]

        x_lo = min(top_pts[:, 0].min(), bot_pts[:, 0].min())
        x_hi = max(top_pts[:, 0].max(), bot_pts[:, 0].max())
        xs_common = np.linspace(x_lo, x_hi, n_samples)

        top_ys = gaussian_filter1d(_resample_curve(top_pts, xs_common), sigma=smooth_sigma, mode="nearest")
        bot_ys = gaussian_filter1d(_resample_curve(bot_pts, xs_common), sigma=smooth_sigma, mode="nearest")

        polygon = np.vstack([
            np.column_stack([xs_common, top_ys]),
            np.column_stack([xs_common[::-1], bot_ys[::-1]]),
        ])
        draw.polygon([tuple(p) for p in polygon], fill=class_id)
    return np.array(mask), curves_sorted


def band_polygons_direct(curves, classes):
    """Shared by build_label_mask_direct (rasterizes) and convert_to_coco.py
    (needs the raw polygon point lists themselves): sort curves top-to-
    bottom, assign each band to a class, and return
    ({class_id: [raw polygon points, ...]}, curves_sorted) using each
    curve's original traced points - no x-grid resampling, no smoothing."""
    num_bands = len(curves) - 1
    if num_bands < 1:
        raise ValueError(f"Need at least 2 boundary curves, got {len(curves)}")
    band_class_ids = _bands_to_classes(num_bands, classes)
    curves_sorted = _sort_curves_top_to_bottom(curves)

    bands_by_class = defaultdict(list)
    for band_idx in range(num_bands):
        class_id = band_class_ids[band_idx]
        top_pts = curves_sorted[band_idx][1]
        bot_pts = curves_sorted[band_idx + 1][1]
        bands_by_class[class_id].append(np.vstack([top_pts, bot_pts[::-1]]))
    return bands_by_class, curves_sorted


def build_label_mask_direct(curves, classes, image_size):
    """Alternative to build_label_mask: fill each band using the curves'
    raw traced points, in their original stroke order - no x-grid
    resampling, no smoothing, no forcing y=f(x). Each polygon is just the
    top curve's points followed by the bottom curve's points reversed,
    with PIL implicitly closing the loop by connecting each curve's two
    endpoints with a straight edge. Most faithful to what was actually
    drawn, but a curve that loops/backtracks a lot can still self-intersect
    since nothing is straightened out."""
    bands_by_class, curves_sorted = band_polygons_direct(curves, classes)

    w, h = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for class_id, polygons in bands_by_class.items():
        for polygon in polygons:
            draw.polygon([tuple(p) for p in polygon], fill=class_id)
    return np.array(mask), curves_sorted


def _find_point_runs(data: bytes, search_start: int = 900):
    """Locate each CmGraphObjPoly's traced-point array within the raw tag
    38123 (CmGraphOverlay) blob. There's no documented schema for this
    proprietary format, so runs are found empirically: a point array is a
    maximal sequence of 8-byte (x, y) float32 pairs where both values are
    either exactly 0.0 or between 1 and 8000 (real traced coordinates never
    hit the tiny/huge denormal-looking garbage that surrounds them).
    Multiple objects in one file share this same encoding back-to-back, so
    repeat until the blob is exhausted. Returns [(start_offset, end_offset,
    [(x, y), ...]), ...] in local (anchor-relative) coordinates."""
    import struct

    def sane(v):
        return v == 0.0 or 1.0 <= abs(v) <= 8000.0

    def decode_from(start):
        pts = []
        off = start
        while off + 8 <= len(data):
            x = struct.unpack_from("<f", data, off)[0]
            y = struct.unpack_from("<f", data, off + 4)[0]
            if not (sane(x) and sane(y)):
                break
            pts.append((x, y))
            off += 8
        return pts, off

    runs = []
    pos = search_start
    while pos < len(data) - 8:
        best = None
        for start in range(pos, min(pos + 50, len(data) - 8)):
            pts, end = decode_from(start)
            if len(pts) >= 3 and (best is None or len(pts) > len(best[0])):
                best = (pts, end, start)
        if best is None or len(best[0]) < 3:
            pos += 1
            continue
        pts, end, start = best
        runs.append((start, end, pts))
        pos = end + 1
    return runs


def _find_anchor(data: bytes, lo: int, hi: int, width: int, height: int):
    """Each traced polygon's points are stored relative to an anchor (its
    own local origin) rather than absolute image coordinates. The anchor is
    the one (x, y) float32 pair in the object's own header region (between
    the previous object's point run and this one) that plausibly falls
    inside the actual image bounds - real header/property floats are either
    tiny/huge denormals or comfortably outside the image, so this is
    unambiguous in practice. Returns (x, y) or None if not found."""
    import struct

    candidate = None
    for off in range(lo, hi):
        x = struct.unpack_from("<f", data, off)[0]
        y = struct.unpack_from("<f", data, off + 4)[0]
        if 200.0 <= x <= width and 200.0 <= y <= height:
            candidate = (x, y)  # last match wins - it's the one closest to the run
    return candidate


def parse_raster_polygons(tif_path: str, tag: int = 38123):
    """Extract each hand-traced closed shape (CmGraphObjPoly) embedded in
    the raster CmGraphOverlay tag as an absolute-image-coordinate Nx2
    array. Unlike tag 38124's polylines (open boundary curves forming
    bands), these are standalone closed blobs - used in this project for
    Follicle/Gland instances, which don't fit the top-to-bottom band model.
    Returns [] if the tif has no tag 38123."""
    img = Image.open(tif_path)
    if tag not in img.tag_v2:
        return []
    data = img.tag_v2[tag]
    width, height = img.size

    polygons = []
    prev_end = 0
    for start, end, pts in _find_point_runs(data):
        anchor = _find_anchor(data, prev_end, start - 4, width, height)
        if anchor is None:
            prev_end = end
            continue
        ax, ay = anchor
        polygons.append(np.array([(ax + x, ay + y) for x, y in pts], dtype=float))
        prev_end = end
    return polygons


def add_raster_blobs(label_mask: np.ndarray, polygons, classes):
    """Fill each raster-traced polygon (see parse_raster_polygons) directly
    into an existing label mask. All such blobs in this project are
    Follicle instances (there's no embedded metadata linking a polygon to
    a specific class, but Follicle is the only blob class actually traced
    this way - multiple follicle cross-sections can appear in one image).
    Returns the list of (class, num_pixels_filled) actually used."""
    follicle = next(c for c in classes if c["name"] == "Follicle")
    mask_img = Image.fromarray(label_mask)
    draw = ImageDraw.Draw(mask_img)
    used = []
    for polygon in polygons:
        draw.polygon([tuple(p) for p in polygon], fill=follicle["index"])
        used.append(follicle)
    return np.array(mask_img), used


def save_preview(tif_path: str, label_mask: np.ndarray, class_names, out_path: str,
                  alpha: float = 0.5, resize_width: int = 900):
    base = Image.open(tif_path).convert("RGB")
    arr = np.array(base)
    for class_idx, name in enumerate(class_names, start=1):
        color = np.array(_PALETTE[(class_idx - 1) % len(_PALETTE)], dtype=np.uint8)
        m = label_mask == class_idx
        arr[m] = (alpha * color + (1 - alpha) * arr[m]).astype(np.uint8)
    preview = Image.fromarray(arr)
    if resize_width and preview.width > resize_width:
        h = int(resize_width * preview.height / preview.width)
        preview = preview.resize((resize_width, h))
    preview.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tif_path")
    parser.add_argument("--points-tag", type=int, default=38124)
    parser.add_argument("--class-tag", type=int, default=38129)
    parser.add_argument("--overlay-tag", type=int, default=38123,
                         help="Raster CmGraphOverlay tag holding standalone traced blobs (e.g. "
                              "Follicle/Gland) - filled in on top of the band mask when present.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--smooth-sigma", type=float, default=8.0,
                         help="Gaussian smoothing (in resampled points) applied to each boundary curve "
                              "before filling - higher smooths out more jaggedness/noise. Ignored by "
                              "--method direct.")
    parser.add_argument("--method", choices=["resample", "direct"], default="resample",
                         help="'resample': fit each curve to a y=f(x) grid and Gaussian-smooth it (tunable "
                              "via --smooth-sigma). 'direct': fill using each curve's raw traced points "
                              "as-is, no resampling or smoothing.")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Where to save the mask/preview/legend - default: training_dataset/label_previews "
                              "(kept separate from training_dataset/images/, which should hold only source tifs).")
    args = parser.parse_args()

    curves = parse_polylines(args.tif_path, tag=args.points_tag)
    classes = resolve_classes(args.tif_path, tag=args.class_tag)
    print(f"Found {len(curves)} boundary curves: {[c[0] for c in curves]}")
    print(f"Found {len(classes)} classes: {[c['name'] for c in classes]}")

    img = Image.open(args.tif_path)
    if args.method == "direct":
        label_mask, curves_sorted = build_label_mask_direct(curves, classes, img.size)
    else:
        label_mask, curves_sorted = build_label_mask(curves, classes, img.size, smooth_sigma=args.smooth_sigma)
    print("Curve order top-to-bottom:", [c[0] for c in curves_sorted])

    raster_polygons = parse_raster_polygons(args.tif_path, tag=args.overlay_tag)
    if raster_polygons:
        label_mask, used_classes = add_raster_blobs(label_mask, raster_polygons, classes)
        print(f"Found {len(raster_polygons)} raster blob(s) (tag {args.overlay_tag}), "
              f"assigned to: {[c['name'] for c in used_classes]}")

    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent / "training_dataset" / "label_previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / Path(args.tif_path).stem
    out_mask = f"{stem}_label_mask.npy"
    out_preview = f"{stem}_label_preview.png"
    out_legend = f"{stem}_label_legend.txt"

    np.save(out_mask, label_mask)
    save_preview(args.tif_path, label_mask, [c["name"] for c in classes], out_preview, alpha=args.alpha)
    with open(out_legend, "w") as f:
        for c in classes:
            count = int((label_mask == c["index"]).sum())
            f.write(f"{c['index']}: {c['name']} - {count} px\n")
            print(f"  class {c['index']} = {c['name']!r}: {count} px")

    print(f"Saved mask -> {out_mask}")
    print(f"Saved preview -> {out_preview}")
    print(f"Saved legend -> {out_legend}")


if __name__ == "__main__":
    main()

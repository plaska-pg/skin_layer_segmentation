#!/usr/bin/env python3
"""
postprocess.py -- turn a raw SC/Epidermal prediction mask into a measured,
canonically-oriented region of interest, plus a side-by-side QA image.

Pipeline (see plan.md for the full spec/rationale):
  1. rotate the full image and prediction mask in 90-degree steps using the
      trained orientation classifier (see rotate.py)
  2. close small gaps per class (grouping only - never adds pixels to the
      mask actually used for measurement), then pick the SC/Epidermal pair
      with the highest mutual boundary-touch fraction as the ROI
  3. erode then dilate (morphological opening) the SC segment to smooth out
      jagged segmentation noise along its boundary
  4. measure each class - see measurement.py (perpendicular-to-tangent
      thickness on smoothed boundary curves, tortuosity, hole count)
  5. render a labeled composite: Original | Raw Prediction | Rotated
      Prediction | ROI (closed) | Epidermal Midline | Height Spine
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from config import CONFIG
from measurement import CSV_COLUMNS, _file_date, _layer_spine, draw_gl_midline_panel, draw_spine_panel, measure_layers, overlay
from rotate import build_model, pick_device, predict_orientation, rotate_k

# um, not raw px: native pixel size varies with scan resolution (e.g. 0.26 vs
# 0.645 um/px), so a fixed px count means a different physical distance on
# different scans. Values come from base_config.yaml's postprocess section
# (fallback matches the codebase's existing 10px-at-0.645um/px convention).
_PP_CFG = CONFIG.get("postprocess", {})
CLOSE_RADII_UM = tuple(_PP_CFG.get("close_radii_um", (12.9, 25.8, 51.6, 103.2)))
TOUCH_MARGIN_UM = _PP_CFG.get("touch_margin_um", 6.45)
MIN_TOUCH_FRAC = _PP_CFG.get("min_touch_frac", 0.2)
SC_SMOOTH_RADIUS_UM = _PP_CFG.get("sc_smooth_radius_um", 6.45)  # opening radius knocking jagged noise off the SC boundary
LATERAL_CLIP_FRAC = _PP_CFG.get("lateral_clip_frac", 0.10)  # fraction of the rotated ROI's horizontal extent trimmed per side
BOTTOM_CROP_FRAC = _PP_CFG.get("bottom_crop_frac", 0.40)  # fraction of the rotated image's bottom rows dropped before ROI/measurement
ROTATION_WEIGHTS = _PP_CFG.get("rotation_weights", "runs/rotation/rotate_model.pt")
ROTATION_DEVICE = _PP_CFG.get("rotation_device", "auto")


def lateral_clip_window(sc_mask: np.ndarray, gl_mask: np.ndarray, frac: float = LATERAL_CLIP_FRAC):
    """Column range [lo, hi) kept after trimming frac of the combined ROI's
    horizontal extent off each side; (0, width) when nothing is trimmed."""
    w = sc_mask.shape[1]
    combined = sc_mask | gl_mask
    cols = np.nonzero(combined.any(axis=0))[0]
    if frac <= 0 or cols.size == 0:
        return 0, w
    x0, x1 = int(cols[0]), int(cols[-1])
    clip = round(frac * (x1 - x0 + 1))
    if clip <= 0:
        return 0, w
    return x0 + clip, x1 - clip + 1


def clip_lateral_margins(sc_mask: np.ndarray, gl_mask: np.ndarray, frac: float = LATERAL_CLIP_FRAC):
    """Zero out SC/Epidermal pixels within frac of the combined ROI's
    horizontal extent from its left/right edges - trims tissue near the cut
    ends of the mounted section (often thin/unreliable) before measurement.
    Returns new (sc_mask, gl_mask); the originals are left untouched so
    already-built rotation/overlay panels aren't affected."""
    lo, hi = lateral_clip_window(sc_mask, gl_mask, frac)
    if lo <= 0 and hi >= sc_mask.shape[1]:
        return sc_mask, gl_mask
    sc_clipped, gl_clipped = sc_mask.copy(), gl_mask.copy()
    sc_clipped[:, :lo] = False
    sc_clipped[:, hi:] = False
    gl_clipped[:, :lo] = False
    gl_clipped[:, hi:] = False
    return sc_clipped, gl_clipped


# ----------------------------------------------------------------------------
# closing / grouping (gap bridging without synthetic pixels)
# ----------------------------------------------------------------------------

def morphological_group(binary: np.ndarray, radius: int) -> np.ndarray:
    """Per-pixel logical-group id (0 = background). Grouping only: closing
    decides which original fragments belong together, but the returned ids are
    masked back to the ORIGINAL foreground pixels - no filled-in pixels leak
    into anything measured/rotated downstream. Uses a rectangular kernel
    (separable/fast in OpenCV regardless of size) rather than an elliptical
    one - these images are large (multi-thousand px) and radius can grow into
    the hundreds of px, where a non-separable elliptical kernel is far too
    slow."""
    if not binary.any():
        return np.zeros(binary.shape, np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    _, labels = cv2.connectedComponents(closed, connectivity=8)
    return np.where(binary, labels, 0)


def largest_group_mask(group_labels: np.ndarray):
    """Return the binary mask of the largest-by-area group, or None if empty."""
    ids, counts = np.unique(group_labels[group_labels > 0], return_counts=True)
    if len(ids) == 0:
        return None
    return group_labels == ids[np.argmax(counts)]


def boundary_ring(comp: np.ndarray) -> np.ndarray:
    comp_u8 = comp.astype(np.uint8)
    ring = comp_u8 - cv2.erode(comp_u8, np.ones((3, 3), np.uint8))
    return ring if ring.any() else comp_u8


def smooth_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening (erode then dilate, elliptical kernel): knocks
    off small jagged protrusions/noise along the boundary without growing
    the segment or shifting its overall position, unlike a closing."""
    if radius <= 0 or not mask.any():
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate then erode a mask to bridge short gaps between segments."""
    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def mutual_touch_frac(a: np.ndarray, b: np.ndarray, margin_px: int):
    """Fraction of a's own boundary ring within margin_px of b, and vice versa."""
    kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), np.uint8) if margin_px > 0 else None
    a_u8, b_u8 = a.astype(np.uint8), b.astype(np.uint8)
    b_dil = cv2.dilate(b_u8, kernel) if kernel is not None else b_u8
    a_dil = cv2.dilate(a_u8, kernel) if kernel is not None else a_u8
    ring_a, ring_b = boundary_ring(a_u8), boundary_ring(b_u8)
    frac_a = float((ring_a & b_dil).sum()) / max(int(ring_a.sum()), 1)
    frac_b = float((ring_b & a_dil).sum()) / max(int(ring_b.sum()), 1)
    return frac_a, frac_b


def find_roi(mask: np.ndarray, sc_id: int, gl_id: int, um_per_px: float = 1.0,
             radii_um=CLOSE_RADII_UM, margin_um=TOUCH_MARGIN_UM, min_frac=MIN_TOUCH_FRAC):
    """Grow the closing radius (converted from um to this mask's own native
    px via um_per_px) until the largest SC/Epidermal logical components
    mutually touch >= min_frac on both sides. Returns a dict with
    sc_mask/gl_mask/radius/ok(/best_frac if not ok), or None if either class
    is entirely absent from the mask."""
    sc_bin, gl_bin = mask == sc_id, mask == gl_id
    if not sc_bin.any() or not gl_bin.any():
        return None
    margin_px = max(1, round(margin_um / um_per_px))
    fallback = None
    for r_um in radii_um:
        r = max(1, round(r_um / um_per_px))
        sc_mask = largest_group_mask(morphological_group(sc_bin, r))
        gl_mask = largest_group_mask(morphological_group(gl_bin, r))
        if sc_mask is None or gl_mask is None:
            continue
        gl_mask = close_mask(gl_mask, r)
        frac_sc, frac_gl = mutual_touch_frac(sc_mask, gl_mask, margin_px)
        fallback = (sc_mask, gl_mask, r, min(frac_sc, frac_gl))
        if frac_sc >= min_frac and frac_gl >= min_frac:
                return {"sc_mask": sc_mask, "gl_mask": gl_mask, "radius": r, "radius_um": r_um,
                    "touch_frac_sc": frac_sc, "touch_frac_gl": frac_gl, "ok": True}
    if fallback is None:
        return None
    sc_mask, gl_mask, r, worst = fallback
    return {"sc_mask": sc_mask, "gl_mask": gl_mask, "radius": r, "ok": False, "best_frac": worst}


# ----------------------------------------------------------------------------
# rotation
# ----------------------------------------------------------------------------

_rotation_model_cache = {}


def _get_rotation_model(weights_path: str = ROTATION_WEIGHTS, device_str: str = ROTATION_DEVICE):
    """Lazily loads+caches the rotate.py orientation classifier (one entry
    per distinct weights/device combo) so repeated process_image calls don't
    reload it from disk each time."""
    key = (weights_path, device_str)
    if key not in _rotation_model_cache:
        device = pick_device(device_str)
        ckpt = torch.load(weights_path, map_location=device)
        model = build_model(ckpt["backbone"], pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _rotation_model_cache[key] = (model, device, ckpt.get("size", 224))
    return _rotation_model_cache[key]


def rotate_to_canonical(img: np.ndarray, sc_mask: np.ndarray, gl_mask: np.ndarray):
    """Uses the trained rotate.py classifier to predict how far img was
    rotated from its upright orientation, then applies the same inverse
    quarter-turn correction rotate.py's own predict() uses."""
    model, device, size = _get_rotation_model()
    k_pred = predict_orientation(model, img, size, device)
    k = (4 - k_pred) % 4
    return rotate_k(img, k), rotate_k(sc_mask, k), rotate_k(gl_mask, k), k


# ----------------------------------------------------------------------------
# visualisation
# ----------------------------------------------------------------------------

def _label_panel(img: np.ndarray, text: str, bar_h: int = 28) -> np.ndarray:
    bar = np.full((bar_h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_composite(panels, target_h: int = 480) -> np.ndarray:
    """panels: list of (label, image_bgr). Resized to a common height and
    stacked horizontally with a caption bar above each."""
    labeled = []
    for name, panel in panels:
        h, w = panel.shape[:2]
        scale = target_h / h
        resized = cv2.resize(panel, (max(1, int(round(w * scale))), target_h))
        labeled.append(_label_panel(resized, name))
    max_h = max(p.shape[0] for p in labeled)
    labeled = [cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
               for p in labeled]
    sep = np.zeros((max_h, 4, 3), np.uint8)
    parts = []
    for i, p in enumerate(labeled):
        if i:
            parts.append(sep)
        parts.append(p)
    return np.hstack(parts)


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------

def process_image(stem: str, folder_name: str, orig_bgr: np.ndarray, pred_mask: np.ndarray,
                   class_index: dict, um_per_px: float, out_dir: Path, source_path: Path = None,
                   postprocess_config: dict = None, return_visuals: bool = False,
                   write_outputs: bool = True):
    """Runs the full rotate -> close/ROI -> measure pipeline for one image,
    writes the labeled composite QA image, and returns a CSV row dict (or
    None if no SC/Epidermal pair reached the touch threshold).
    `source_path`, if given, is the original image file - its last-modified
    date is used for the CSV's 'date' column."""
    settings = postprocess_config or {}
    close_radii_um = tuple(settings.get("close_radii_um", CLOSE_RADII_UM))
    touch_margin_um = settings.get("touch_margin_um", TOUCH_MARGIN_UM)
    min_touch_frac = settings.get("min_touch_frac", MIN_TOUCH_FRAC)
    sc_smooth_radius_um = settings.get("sc_smooth_radius_um", SC_SMOOTH_RADIUS_UM)
    lateral_clip_frac = settings.get("lateral_clip_frac", LATERAL_CLIP_FRAC)
    bottom_crop_frac = settings.get("bottom_crop_frac", BOTTOM_CROP_FRAC)

    sc_id = class_index["SC"]
    gl_id = class_index.get("Epidermal", class_index.get("Granular Layer"))
    if gl_id is None:
        raise KeyError("class_index must contain 'Epidermal' or 'Granular Layer'")

    # Preserve raw-frame QA, but rotate the complete prediction before any
    # grouping or ROI selection so all downstream postprocessing uses the
    # canonical orientation.
    rot_img, rot_sc_full, rot_gl_full, k = rotate_to_canonical(
        orig_bgr, pred_mask == sc_id, pred_mask == gl_id)
    raw_overlay = overlay(orig_bgr, {"SC": pred_mask == sc_id, "Epidermal": pred_mask == gl_id})

    # Drop the bottom rows of the rotated frame (dermis/substrate side) so ROI
    # selection and measurement only see the upper epidermal band.
    keep_rows = max(1, int(round(rot_img.shape[0] * (1.0 - bottom_crop_frac))))
    rot_img = rot_img[:keep_rows]
    rot_sc_full = rot_sc_full[:keep_rows]
    rot_gl_full = rot_gl_full[:keep_rows]

    rot_pred_mask = np.zeros(rot_sc_full.shape, dtype=pred_mask.dtype)
    rot_pred_mask[rot_sc_full] = sc_id
    rot_pred_mask[rot_gl_full] = gl_id
    rotated_overlay = overlay(rot_img, {"SC": rot_sc_full, "Epidermal": rot_gl_full})

    roi = find_roi(rot_pred_mask, sc_id, gl_id, um_per_px=um_per_px,
                   radii_um=close_radii_um, margin_um=touch_margin_um,
                   min_frac=min_touch_frac)
    if roi is None or not roi["ok"]:
        reached = roi["best_frac"] if roi else 0.0
        print(f"SKIP {stem}: no SC/Epidermal pair reached {min_touch_frac:.0%} mutual touch "
              f"(best {reached:.0%} at max closing radius {close_radii_um[-1]}um)")
        return (None, None) if return_visuals else None
    print(f"{stem}: postprocess radius {roi['radius_um']:.2f}um ({roi['radius']}px), "
          f"touch SC={roi['touch_frac_sc']:.1%} GL={roi['touch_frac_gl']:.1%}", flush=True)
    roi["sc_mask"] = smooth_mask(roi["sc_mask"], max(1, round(sc_smooth_radius_um / um_per_px)))

    roi_overlay = overlay(rot_img, {"SC": roi["sc_mask"], "Epidermal": roi["gl_mask"]})
    rot_sc, rot_gl = roi["sc_mask"], roi["gl_mask"]

    # after rotation, find the longest horizontal span of the SC la
    midline_panel = draw_gl_midline_panel(rot_img, rot_sc, rot_gl)

    # spines first, then clip: build each layer's medial-axis spine on the full
    # rotated mask so the lateral clip can't carve cut-edges into the skeleton,
    # then keep only the clip-window portion as the mean-height denominator.
    lo, hi = lateral_clip_window(rot_sc, rot_gl, lateral_clip_frac)
    sc_spine, sc_len = _layer_spine(rot_sc.astype(bool), col_window=(lo, hi))
    gl_spine, gl_len = _layer_spine(rot_gl.astype(bool), col_window=(lo, hi))
    meas_sc, meas_gl = clip_lateral_margins(rot_sc, rot_gl, lateral_clip_frac)
    row_extra, measurement_panel, spine_cache = measure_layers(
        rot_img, meas_sc, meas_gl, um_per_px, date=_file_date(source_path),
        spines={"SC": sc_spine, "Epidermal": gl_spine},
        lengths_px={"SC": sc_len, "Epidermal": gl_len})

    sc_stats = {"mean_height": row_extra["SC mean height"], "area": row_extra["SC area"]}
    gl_stats = {"mean_height": row_extra["Epidermal mean height"], "area": row_extra["Epidermal area"]}
    spine_panel = draw_spine_panel(rot_img, meas_sc, meas_gl, sc_stats, gl_stats, um_per_px,
                                   spines=spine_cache["spines"], lengths_px=spine_cache["lengths_px"])

    composite = make_composite([
        ("Original", orig_bgr),
        ("Raw Prediction", raw_overlay),
        (f"Rotated Prediction ({k * 90} deg)", rotated_overlay),
        ("ROI (closed)", roi_overlay),
        ("Epidermal Midline", midline_panel),
        ("Height Spine", spine_panel),
    ])
    out_dir = Path(out_dir)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}_measurement.jpg"), measurement_panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(out_dir / f"{stem}_steps.jpg"), composite, [cv2.IMWRITE_JPEG_QUALITY, 90])

    row = {"folder name": folder_name, "file name": stem, **row_extra}
    return (row, composite) if return_visuals else row

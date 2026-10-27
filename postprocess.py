#!/usr/bin/env python3
"""
postprocess.py -- turn a raw SC/Epidermal prediction mask into a measured,
canonically-oriented region of interest, plus a side-by-side QA image.

Pipeline:
  1. rotate the full image and prediction mask in 90-degree steps using the
      trained orientation classifier (see rotate.py)
  2. detect follicles (YOLO-OBB) on the rotated image
  3. detect ROI boxes (YOLO-OBB roi_det) on the rotated image and score
      quality_sections even sections of each with the train_qual.py model
      (CSV 'quality' = mean section score, 1 worst - 3 best)
  4. keep only SC/Epidermal pixels inside the union of ROI boxes (full frame
      if none detected) and outside the bottom follicle_mask_frac of each
      follicle box (cut parallel to the box's top edge)
  5. per-class elliptical opening then closing (sc_/epidermal_/gland_erod_dil_kernel_size_um)
  6. measure each class - see measurement.py (perpendicular-to-tangent
      thickness on smoothed boundary curves, tortuosity)
  7. render a labeled composite: Original | Raw Prediction | Rotated
      Prediction | Follicles | ROI Boxes | Quality | ROI (masked + morph) |
      Epidermal Midline | Height Spine

Gland predictions are rotated and smoothed - never ROI/follicle masked or clipped.
"""

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CONFIG
from measurement import (CSV_COLUMNS, _file_date, _layer_spine, draw_gl_midline_panel, draw_spine_panel,
                         measure_layers, overlay, tissue_area_um2)
from rotate import build_model, pick_device, predict_orientation, rotate_k

sys.path.append(str(Path(__file__).resolve().parent / "pretrained_model"))
from annotate_boxes import ROI_CLASS, even_sections, section_quad  # noqa: E402
from train_qual import load_model, mode_of, predict_tiles, prepare, score_color, warp_section  # noqa: E402

# um, not raw px: native pixel size varies with scan resolution (e.g. 0.26 vs
# 0.645 um/px), so a fixed px count means a different physical distance on
# different scans. Values come from base_config.yaml's postprocess section
# (fallback matches the codebase's existing 10px-at-0.645um/px convention).
_PP_CFG = CONFIG.get("postprocess", {})
SC_ERODE_DILATE_UM = _PP_CFG.get("sc_erod_dil_kernel_size_um", 5.0)
EPIDERMAL_ERODE_DILATE_UM = _PP_CFG.get("epidermal_erod_dil_kernel_size_um", 10.0)
GLAND_ERODE_DILATE_UM = _PP_CFG.get("gland_erod_dil_kernel_size_um", 0.0)
LATERAL_CLIP_FRAC = _PP_CFG.get("lateral_clip_frac", 0.10)  # fraction of the rotated ROI's horizontal extent trimmed per side
ROTATION_WEIGHTS = _PP_CFG.get("rotation_weights", "runs/rotation/rotate_model.pt")
ROTATION_DEVICE = _PP_CFG.get("rotation_device", "auto")
FOLLICLE_ENABLED = _PP_CFG.get("follicle_enabled", True)
FOLLICLE_WEIGHTS = _PP_CFG.get("follicle_weights", "models/follicle_det.pt")
FOLLICLE_DEVICE = _PP_CFG.get("follicle_device", "auto")
FOLLICLE_CONF = _PP_CFG.get("follicle_conf", 0.25)
FOLLICLE_MAX_SIDE = _PP_CFG.get("follicle_max_side", 1024)
FOLLICLE_IMGSZ = _PP_CFG.get("follicle_imgsz", 1024)
FOLLICLE_PADDING_UM = _PP_CFG.get("follicle_padding_um", 0.0)
FOLLICLE_MASK_FRAC = _PP_CFG.get("follicle_mask_frac", 0.8)
ROI_ENABLED = _PP_CFG.get("roi_enabled", True)
ROI_WEIGHTS = _PP_CFG.get("roi_weights", "models/roi_det.pt")
ROI_DEVICE = _PP_CFG.get("roi_device", "auto")
ROI_CONF = _PP_CFG.get("roi_conf", 0.25)
ROI_MAX_SIDE = _PP_CFG.get("roi_max_side", 1024)
ROI_IMGSZ = _PP_CFG.get("roi_imgsz", 1024)
QUALITY_ENABLED = _PP_CFG.get("quality_enabled", True)
QUALITY_WEIGHTS = _PP_CFG.get("quality_weights", "models/qual_v1.pt")
QUALITY_DEVICE = _PP_CFG.get("quality_device", "auto")
QUALITY_SECTIONS = _PP_CFG.get("quality_sections", 5)


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
# morphology
# ----------------------------------------------------------------------------

def kernel_px(size_um: float, um_per_px: float) -> int:
    """Kernel diameter in native px (forced odd); <= 1 means no-op."""
    size = int(round(size_um / um_per_px)) if size_um > 0 else 0
    return size + 1 if size > 1 and size % 2 == 0 else size


def erode_dilate(mask: np.ndarray, size_px: int) -> np.ndarray:
    """Elliptical opening (removes specks/jagged edges) then closing (bridges small gaps)."""
    if size_px <= 1 or not mask.any():
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_px, size_px))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel).astype(bool)


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
# YOLO-OBB detection (follicles, ROI)
# ----------------------------------------------------------------------------

_obb_model_cache = {}
# predict.py runs process_image in a ThreadPoolExecutor; ultralytics predict isn't thread-safe
_obb_lock = threading.Lock()


def _get_obb_model(weights_path: str, device_str: str, name: str):
    """Lazily loads+caches a YOLO-OBB detector. Returns (model, device) or
    None (warning printed once) if the weights or ultralytics are unavailable,
    so the pipeline still runs without that detector."""
    key = (weights_path, device_str)
    if key not in _obb_model_cache:
        model = None
        if not Path(weights_path).exists():
            print(f"WARNING: {name} weights not found at {weights_path}; skipping {name} detection", flush=True)
        else:
            try:
                from ultralytics import YOLO
                model = (YOLO(weights_path), str(pick_device(device_str)))
            except ImportError:
                print(f"WARNING: ultralytics not installed; skipping {name} detection", flush=True)
        _obb_model_cache[key] = model
    return _obb_model_cache[key]


def detect_obb(img_bgr: np.ndarray, weights_path: str, device_str: str, conf: float,
               max_side: int, imgsz: int, name: str = "OBB"):
    """Returns a list of (quad, conf) with quad a 4x2 float array of OBB
    corners in img_bgr's pixel coordinates, or None if no detector is available."""
    with _obb_lock:
        loaded = _get_obb_model(weights_path, device_str, name)
        if loaded is None:
            return None
        model, device = loaded
        # same downscale as annotate_boxes.load_scaled, which built the training set
        h, w = img_bgr.shape[:2]
        f = min(1.0, max_side / max(h, w))
        small = cv2.resize(img_bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else img_bgr
        r = model.predict(small, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    quads = r.obb.xyxyxyxy.cpu().numpy() / f
    return list(zip(quads, r.obb.conf.cpu().numpy().tolist()))


def roi_mask(shape, boxes) -> np.ndarray:
    """Bool mask of the union of all ROI oriented boxes."""
    mask = np.zeros(shape[:2], np.uint8)
    for quad, _ in boxes:
        cv2.fillPoly(mask, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def pad_obb(quad: np.ndarray, padding_px: float) -> np.ndarray:
    """Expand every edge of a cyclic oriented rectangle by padding_px."""
    quad = np.asarray(quad, np.float64)
    if padding_px <= 0:
        return quad
    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1, keepdims=True)
    directions = np.divide(edges, lengths, out=np.zeros_like(edges), where=lengths > 0)
    return quad + padding_px * (np.roll(directions, 1, axis=0) - directions)


def follicle_mask(shape, follicles, frac: float = FOLLICLE_MASK_FRAC) -> np.ndarray:
    """Bool mask of the bottom frac of each follicle's oriented box, cut by a
    line parallel to its top edge, in the frame the quads were detected in."""
    h, w = shape[:2]
    mask = np.zeros((h, w), bool)
    if not follicles or frac <= 0:
        return mask
    frac = min(frac, 1.0)
    for quad, _ in follicles:
        quad = np.asarray(quad, np.float64)
        # OBB corners are cyclic: the top edge is the one with the highest midpoint
        i = int(np.argmin([quad[j, 1] + quad[(j + 1) % 4, 1] for j in range(4)]))
        p_a, p_b, q_b, q_a = quad[[i, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4]]
        t = 1.0 - frac
        region = np.array([p_a + t * (q_a - p_a), p_b + t * (q_b - p_b), q_b, q_a])
        x0 = max(0, int(np.floor(region[:, 0].min())))
        x1 = min(w, int(np.ceil(region[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(region[:, 1].min())))
        y1 = min(h, int(np.ceil(region[:, 1].max())) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts = np.round(region - [x0, y0]).astype(np.int32)
        cv2.fillPoly(local, [pts.reshape(-1, 1, 2)], 1)
        mask[y0:y1, x0:x1] |= local.astype(bool)
    return mask


def draw_obb_panel(img: np.ndarray, boxes, shaded: np.ndarray = None, color=(0, 255, 255)) -> np.ndarray:
    panel = img.copy()
    if shaded is not None and shaded.any():
        shade = np.zeros_like(panel)
        shade[shaded] = color
        panel[shaded] = cv2.addWeighted(panel, 0.6, shade, 0.4, 0)[shaded]
    # composite downsizes panels to ~480px tall, so scale strokes to survive that
    s = max(1.0, max(panel.shape[:2]) / 1000)
    thickness, font_scale = max(2, int(round(3 * s))), 0.6 * s
    for quad, c in boxes:
        pts = np.round(quad).astype(np.int32)
        cv2.polylines(panel, [pts.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)
        x, y = pts[:, 0].min(), max(int(pts[:, 1].min()) - thickness, int(20 * s))
        cv2.putText(panel, f"{c:.2f}", (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    max(1, thickness // 2), cv2.LINE_AA)
    return panel


# ----------------------------------------------------------------------------
# section quality (pretrained_model/train_qual.py)
# ----------------------------------------------------------------------------

_quality_model_cache = {}
_quality_lock = threading.Lock()


def _get_quality_model(weights_path: str, device_str: str):
    """Lazily loads+caches the section quality model; (model, ckpt, device) or
    None (warning printed once) if the weights are missing."""
    key = (weights_path, device_str)
    if key not in _quality_model_cache:
        loaded = None
        if not Path(weights_path).exists():
            print(f"WARNING: quality weights not found at {weights_path}; skipping quality scoring", flush=True)
        else:
            device = pick_device(device_str)
            model, ckpt = load_model(weights_path, device)
            loaded = (model, ckpt, device)
        _quality_model_cache[key] = loaded
    return _quality_model_cache[key]


def score_sections(img_bgr: np.ndarray, rois, n_sections: int, weights_path: str, device_str: str):
    """Splits each ROI quad into n_sections even sections along its long axis and
    scores each (expected rating, 1 worst - 3 best). Returns (scores, quads) with
    quads in img_bgr's pixel coordinates, or None if no quality model is available."""
    with _quality_lock:
        loaded = _get_quality_model(weights_path, device_str)
        if loaded is None:
            return None
        model, ckpt, device = loaded
        # same downscale as annotate_boxes.load_scaled, which built the training crops
        h, w = img_bgr.shape[:2]
        f = min(1.0, ckpt["max_side"] / max(h, w))
        small = cv2.resize(img_bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else img_bgr
        src = prepare(small, mode_of(ckpt))
        cuts = even_sections(n_sections)["cuts"]
        quads = [section_quad([ROI_CLASS, *np.asarray(quad, np.float32).ravel()], cuts[k], cuts[k + 1])
                 for quad, _ in rois for k in range(n_sections)]
        tiles = [warp_section(src, q * f, ckpt["imgsz"], letterbox=ckpt.get("letterbox", True)) for q in quads]
        scores, _, _ = predict_tiles(model, tiles, device)
    return scores, quads


def draw_quality_panel(img: np.ndarray, scores, quads) -> np.ndarray:
    """Sections filled red (1) -> yellow (2) -> green (3) by score, outlined and labelled."""
    fill = img.copy()
    for s, q in zip(scores, quads):
        cv2.fillPoly(fill, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], score_color(s))
    panel = cv2.addWeighted(fill, 0.35, img, 0.65, 0)
    # composite downsizes panels to ~480px tall, so scale strokes to survive that
    sc = max(1.0, max(panel.shape[:2]) / 1000)
    thickness, font_scale = max(2, int(round(2 * sc))), 0.6 * sc
    for s, q in zip(scores, quads):
        pts = np.round(q).astype(np.int32)
        cv2.polylines(panel, [pts.reshape(-1, 1, 2)], True, (40, 40, 40), thickness, cv2.LINE_AA)
        label = f"{s:.1f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cx, cy = pts.mean(axis=0)
        org = (int(cx) - tw // 2, int(cy) + th // 2)
        cv2.putText(panel, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(panel, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return panel


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
# YOLO-seg label export
# ----------------------------------------------------------------------------

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


def long_path(p) -> Path:
    """Extended-length (\\\\?\\) form of p on Windows so paths over MAX_PATH (260) still open."""
    if os.name != "nt":
        return Path(p)
    s = os.path.abspath(str(p))  # not resolve(): don't follow OneDrive shortcut reparse points
    if s.startswith("\\\\?\\"):
        return Path(s)
    if s.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + s[2:])
    return Path("\\\\?\\" + s)


def imwrite_jpg(path, img: np.ndarray, quality: int) -> None:
    """cv2.imwrite can't open long or \\\\?\\ paths, so encode in memory and write via Python."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    Path(path).write_bytes(buf.tobytes())


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------

def process_image(stem: str, folder_name: str, orig_bgr: np.ndarray, pred_mask: np.ndarray,
                   class_index: dict, um_per_px: float, out_dir: Path, source_path: Path = None,
                   postprocess_config: dict = None, return_visuals: bool = False,
                   write_outputs: bool = True, seg_path: Path = None):
    """Runs the full rotate -> detect follicles/ROI -> mask -> morph -> measure
    pipeline for one image, writes the labeled composite QA image, and returns
    a CSV row dict (or None if SC or Epidermis is empty after masking).
    `source_path`, if given, is the original image file - its last-modified
    date is used for the CSV's 'date' column. `seg_path`, if given, receives
    the postprocessed segmentation as YOLO-seg polygons in the original frame."""
    settings = postprocess_config or {}
    sc_kernel_um = settings.get("sc_erod_dil_kernel_size_um", SC_ERODE_DILATE_UM)
    gl_kernel_um = settings.get("epidermal_erod_dil_kernel_size_um", EPIDERMAL_ERODE_DILATE_UM)
    gland_kernel_um = settings.get("gland_erod_dil_kernel_size_um", GLAND_ERODE_DILATE_UM)
    lateral_clip_frac = settings.get("lateral_clip_frac", LATERAL_CLIP_FRAC)
    follicle_enabled = settings.get("follicle_enabled", FOLLICLE_ENABLED)
    follicle_padding_um = settings.get("follicle_padding_um", FOLLICLE_PADDING_UM)
    follicle_mask_frac = settings.get("follicle_mask_frac", FOLLICLE_MASK_FRAC)
    roi_enabled = settings.get("roi_enabled", ROI_ENABLED)
    quality_enabled = settings.get("quality_enabled", QUALITY_ENABLED)

    sc_id = class_index["SC"]
    # 'Epidermal' / 'Granular Layer' are the names used by older checkpoints
    gl_id = class_index.get("Epidermis", class_index.get("Epidermal", class_index.get("Granular Layer")))
    if gl_id is None:
        raise KeyError("class_index must contain 'Epidermis' (or legacy 'Epidermal' / 'Granular Layer')")
    gland_id = class_index.get("gland", class_index.get("glands"))
    gland_full = pred_mask == gland_id if gland_id is not None else np.zeros(pred_mask.shape, bool)

    # Preserve raw-frame QA, but rotate the complete prediction before any
    # ROI selection so all downstream postprocessing uses the canonical orientation.
    rot_img, rot_sc_full, rot_gl_full, k = rotate_to_canonical(
        orig_bgr, pred_mask == sc_id, pred_mask == gl_id)
    rot_gland = rotate_k(gland_full, k)
    raw_overlay = overlay(orig_bgr, {"SC": pred_mask == sc_id, "Epidermal": pred_mask == gl_id, "gland": gland_full})
    rotated_overlay = overlay(rot_img, {"SC": rot_sc_full, "Epidermal": rot_gl_full, "gland": rot_gland})
    rot_gland = erode_dilate(rot_gland, kernel_px(gland_kernel_um, um_per_px))

    follicles = None
    if follicle_enabled:
        follicles = detect_obb(
            rot_img,
            weights_path=settings.get("follicle_weights", FOLLICLE_WEIGHTS),
            device_str=settings.get("follicle_device", FOLLICLE_DEVICE),
            conf=settings.get("follicle_conf", FOLLICLE_CONF),
            max_side=settings.get("follicle_max_side", FOLLICLE_MAX_SIDE),
            imgsz=settings.get("follicle_imgsz", FOLLICLE_IMGSZ), name="follicle")
        if follicles and follicle_padding_um > 0:
            follicles = [(pad_obb(quad, follicle_padding_um / um_per_px), confidence)
                         for quad, confidence in follicles]
    fol_removed = follicle_mask(rot_img.shape, follicles, follicle_mask_frac)
    follicle_panels = [] if follicles is None else [
        (f"Follicles ({len(follicles)})", draw_obb_panel(rotated_overlay, follicles, fol_removed))]

    rois = None
    if roi_enabled:
        rois = detect_obb(
            rot_img,
            weights_path=settings.get("roi_weights", ROI_WEIGHTS),
            device_str=settings.get("roi_device", ROI_DEVICE),
            conf=settings.get("roi_conf", ROI_CONF),
            max_side=settings.get("roi_max_side", ROI_MAX_SIDE),
            imgsz=settings.get("roi_imgsz", ROI_IMGSZ), name="ROI")
    roi_panels = [] if rois is None else [
        (f"ROI Boxes ({len(rois)})", draw_obb_panel(rotated_overlay, rois, color=(255, 0, 255)))]
    quality = ""
    if rois and quality_enabled:
        scored = score_sections(
            rot_img, rois,
            n_sections=settings.get("quality_sections", QUALITY_SECTIONS),
            weights_path=settings.get("quality_weights", QUALITY_WEIGHTS),
            device_str=settings.get("quality_device", QUALITY_DEVICE))
        if scored is not None:
            quality = round(float(np.mean(scored[0])), 3)
            roi_panels.append((f"Quality ({quality:.2f})", draw_quality_panel(rot_img, *scored)))
    if rois:
        keep = roi_mask(rot_img.shape, rois)
    else:
        if roi_enabled:
            print(f"WARNING {stem}: no ROI detected; using full frame", flush=True)
        keep = np.ones(rot_img.shape[:2], bool)
    keep &= ~fol_removed

    # re-mask after morphology so closing can't reintroduce pixels outside ROI / inside follicles
    rot_sc = erode_dilate(rot_sc_full & keep, kernel_px(sc_kernel_um, um_per_px)) & keep
    rot_gl = erode_dilate(rot_gl_full & keep, kernel_px(gl_kernel_um, um_per_px)) & keep
    if not rot_sc.any() or not rot_gl.any():
        print(f"SKIP {stem}: SC or Epidermis empty after ROI/follicle masking", flush=True)
        return (None, None) if return_visuals else None

    roi_overlay = overlay(rot_img, {"SC": rot_sc, "Epidermal": rot_gl, "gland": rot_gland})

    midline_panel = draw_gl_midline_panel(rot_img, rot_sc, rot_gl)

    # spines first, then clip: build each layer's medial-axis spine on the full
    # rotated mask so the lateral clip can't carve cut-edges into the skeleton,
    # then keep only the clip-window portion as the mean-height denominator.
    lo, hi = lateral_clip_window(rot_sc, rot_gl, lateral_clip_frac)
    sc_spine, sc_len = _layer_spine(rot_sc.astype(bool), col_window=(lo, hi))
    gl_spine, gl_len = _layer_spine(rot_gl.astype(bool), col_window=(lo, hi))
    meas_sc, meas_gl = clip_lateral_margins(rot_sc, rot_gl, lateral_clip_frac)
    cut_mask = ~keep
    cut_mask[:, :lo] = True
    cut_mask[:, hi:] = True
    row_extra, measurement_panel, spine_cache = measure_layers(
        rot_img, meas_sc, meas_gl, um_per_px, date=_file_date(source_path),
        spines={"SC": sc_spine, "Epidermal": gl_spine},
        lengths_px={"SC": sc_len, "Epidermal": gl_len},
        gland_mask=rot_gland if gland_id is not None else None,
        cut_mask=cut_mask)

    sc_stats = {"mean_height": row_extra["SC mean height"], "area": row_extra["SC area"]}
    gl_stats = {"mean_height": row_extra["Epidermal mean height"], "area": row_extra["Epidermal area"]}
    spine_panel = draw_spine_panel(rot_img, meas_sc, meas_gl, sc_stats, gl_stats, um_per_px,
                                   spines=spine_cache["spines"], lengths_px=spine_cache["lengths_px"],
                                   edj=spine_cache["edj"])

    composite = make_composite([
        ("Original", orig_bgr),
        ("Raw Prediction", raw_overlay),
        (f"Rotated Prediction ({k * 90} deg)", rotated_overlay),
        *follicle_panels,
        *roi_panels,
        ("ROI (masked + morph)", roi_overlay),
        ("Epidermal Midline", midline_panel),
        ("Height Spine", spine_panel),
    ])
    out_dir = Path(out_dir)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        imwrite_jpg(out_dir / f"{stem}_measurement.jpg", measurement_panel, 90)
        imwrite_jpg(out_dir / f"{stem}_steps.jpg", composite, 90)
    if write_outputs and seg_path is not None:
        post_mask = np.zeros(rot_img.shape[:2], dtype=pred_mask.dtype)
        if gland_id is not None:
            post_mask[rot_gland] = gland_id
        post_mask[rot_sc] = sc_id
        post_mask[rot_gl] = gl_id
        seg_path = Path(seg_path)
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        write_yolo_labels(rotate_k(post_mask, -k), class_index, seg_path)

    row = {"folder name": folder_name, "file name": stem, **row_extra, "quality": quality,
           "total_tissue_um2": round(tissue_area_um2(orig_bgr, um_per_px), 2)}
    return (row, composite) if return_visuals else row

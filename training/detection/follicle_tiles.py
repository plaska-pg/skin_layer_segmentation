"""
follicle_tiles.py -- run the follicle detector at one physical scale.

Downscaling every slide to the same long side left follicles at ~1.8-6.7 um/px depending on slide
size and scanner, so the follicle detector is trained and run on square tiles cut from slides
resampled to a fixed um/px instead. The dataset's scale is stamped into the trained weights
(ckpt["tile_scale"] = {"um_per_px", "tile", "overlap"}), so inference picks it up from the weights.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

TILE_OVERLAP = 0.25
MIN_VISIBLE = 0.2  # a box cut by a tile edge is kept (as its visible part) if this share of it is inside
MIN_TISSUE = 0.01  # tiles with less tissue than this (and no boxes) are skipped
PAD = (255, 255, 255)  # slide background


def load_calibration(path) -> dict:
    """{image file name: um/px of the file as stored} from a scale_calibration.json."""
    return json.loads(Path(path).read_text())["images"]


def tile_scale(model):
    """The tile scale stamped into YOLO weights by train_det.py, or None for whole-slide models."""
    return (getattr(model, "ckpt", None) or {}).get("tile_scale")


def resize(img: np.ndarray, f: float) -> np.ndarray:
    if f == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                      interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)


def _starts(length: int, tile: int, overlap: float):
    if length <= tile:
        return [0]
    n = math.ceil((length - tile) / (tile * (1 - overlap))) + 1
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def iter_tiles(img: np.ndarray, tile: int, overlap: float = TILE_OVERLAP):
    """(x0, y0, tile x tile crop) covering img; edge crops are padded with background."""
    h, w = img.shape[:2]
    for y0 in _starts(h, tile, overlap):
        for x0 in _starts(w, tile, overlap):
            crop = img[y0:y0 + tile, x0:x0 + tile]
            if crop.shape[:2] != (tile, tile):
                crop = cv2.copyMakeBorder(crop, 0, tile - crop.shape[0], 0, tile - crop.shape[1],
                                          cv2.BORDER_CONSTANT, value=PAD)
            yield x0, y0, crop


def has_tissue(crop: np.ndarray) -> bool:
    return (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 220).mean() >= MIN_TISSUE


def clip_quad(quad, tile: int):
    """quad (tile px) clipped to the tile as an oriented box, or None if under MIN_VISIBLE is inside."""
    q = np.float32(quad).reshape(4, 2)
    area = cv2.contourArea(q)
    if area <= 0:
        return None
    rect = np.float32([[0, 0], [tile, 0], [tile, tile], [0, tile]])
    inter, poly = cv2.intersectConvexConvex(q, rect)
    if inter < MIN_VISIBLE * area:
        return None
    if inter >= 0.999 * area:
        return q
    return np.clip(cv2.boxPoints(cv2.minAreaRect(poly)), 0, tile)


def _overlap_of_smaller(a, b) -> float:
    inter, _ = cv2.intersectConvexConvex(np.float32(a), np.float32(b))
    smaller = min(cv2.contourArea(np.float32(a)), cv2.contourArea(np.float32(b)))
    return inter / smaller if smaller > 0 else 0.0


def merge_tile_detections(dets, thr: float = 0.5):
    """Greedy NMS over (quad, conf, cut) from overlapping tiles by intersection over the smaller box;
    boxes not cut by an inner tile edge win, so a whole follicle beats its clipped copy."""
    keep = []
    for q, c, _ in sorted(dets, key=lambda d: (d[2], -d[1])):
        if all(_overlap_of_smaller(q, k) < thr for k, _ in keep):
            keep.append((q, c))
    return keep


def detect_tiled(model, img: np.ndarray, um_per_px: float, scale: dict, conf: float, device=None, batch: int = 8):
    """[(4x2 quad in img px, conf)] from tiles of img resampled to scale["um_per_px"]."""
    tile, overlap = int(scale["tile"]), float(scale.get("overlap", TILE_OVERLAP))
    f = um_per_px / float(scale["um_per_px"])
    small = resize(img, f)
    h, w = small.shape[:2]
    tiles = [t for t in iter_tiles(small, tile, overlap) if has_tissue(t[2])]
    dets = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        results = model.predict([c for _, _, c in chunk], imgsz=tile, conf=conf, device=device, verbose=False)
        for (x0, y0, _), r in zip(chunk, results):
            for q, c in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy().tolist()):
                cut = ((x0 > 0 and q[:, 0].min() < 2) or (y0 > 0 and q[:, 1].min() < 2)
                       or (x0 + tile < w and q[:, 0].max() > tile - 2) or (y0 + tile < h and q[:, 1].max() > tile - 2))
                dets.append((q + (x0, y0), c, bool(cut)))
    return [(q / f, c) for q, c in merge_tile_detections(dets)]


def detect_whole(model, img: np.ndarray, max_side: int, imgsz: int, conf: float, device=None):
    """[(4x2 quad in img px, conf)] from the whole slide downscaled to max_side (models without a tile scale)."""
    f = min(1.0, max_side / max(img.shape[:2]))
    r = model.predict(resize(img, f), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    return [(q / f, c) for q, c in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy().tolist())]

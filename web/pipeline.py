"""Browser (Pyodide) entry point: postprocess.process_image on a segmentation made by web/app.js.

The ROI / follicle detectors and the quality model run through the page's ONNX Runtime (`infer` callback).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from pyodide.ffi import to_js

sys.path.insert(0, str(Path(__file__).resolve().parent))
import obb  # noqa: E402
import postprocess  # noqa: E402
from measurement import CSV_COLUMNS  # noqa: E402
from training.qc_sections import section_tiles  # noqa: E402

OUT = Path("/out")
SRC = Path("/in")
CLASS_INDEX = {"SC": 1, "Epidermis": 2, "gland": 3}
DETECTORS = {"follicle": (postprocess.FOLLICLE_CONF, postprocess.FOLLICLE_MAX_SIDE, postprocess.FOLLICLE_IMGSZ),
             "roi": (postprocess.ROI_CONF, postprocess.ROI_MAX_SIDE, postprocess.ROI_IMGSZ)}


def defaults() -> str:
    """base_config's on/off defaults for the page's toggles."""
    return json.dumps({"roi": postprocess.ROI_ENABLED, "follicle": postprocess.FOLLICLE_ENABLED,
                       "quality": postprocess.QUALITY_ENABLED})


async def detect(infer, name: str, rot_bgr: np.ndarray, conf: float, max_side: int, imgsz: int):
    """postprocess.detect_obb equivalent; `infer(name, bytes, dims)` runs <name>.onnx and resolves to {data, dims}."""
    x, small_shape, f = obb.prepare(rot_bgr, max_side, imgsz)
    res = await infer(name, to_js(x.tobytes()), to_js(list(x.shape)))
    raw = np.frombuffer(res.data.to_py(), np.float32).reshape(res.dims.to_py())
    return obb.postprocess(raw, x.shape[2:], small_shape, f, conf, nc=raw.shape[1] - 5)


async def score(infer, rot_bgr: np.ndarray, rois, q: dict):
    """postprocess.score_sections equivalent: (scores, quads), tiles cut here, scored by quality.onnx."""
    tiles, quads = section_tiles(rot_bgr, rois, postprocess.QUALITY_SECTIONS, q["max_side"], q["imgsz"], q["mode"],
                                 q["letterbox"])
    x = np.ascontiguousarray(np.stack([t[..., ::-1] for t in tiles]).transpose(0, 3, 1, 2), dtype=np.float32)
    res = await infer("quality", to_js(x.tobytes()), to_js(list(x.shape)))
    return np.frombuffer(res.data.to_py(), np.float32).copy(), quads


def _rot_pts(p, w: float, h: float, k: int) -> np.ndarray:
    """Points (x, y) of a w x h image after postprocess.rotate_k(img, k) (k clockwise quarter turns)."""
    p = np.asarray(p, float)
    for _ in range(k % 4):
        p = np.stack([h - p[:, 1], p[:, 0]], 1)
        w, h = h, w
    return p


def _det_to_crop(quad, k: int, det_wh, region) -> np.ndarray:
    """Quad in the rotated detection image (the whole image downscaled to det_wh) -> rotated crop coordinates."""
    dw, dh = det_wh
    x0, y0, w, h, full_w, full_h = region
    rw, rh = (dh, dw) if k % 2 else (dw, dh)
    p = _rot_pts(quad, rw, rh, -k) * (full_w / dw, full_h / dh) - (x0, y0)
    return _rot_pts(p, w, h, k)


def _to_frame(path: Path, region) -> None:
    """Re-normalise a YOLO-seg label of the crop [x0, y0, w, h] of a full_w x full_h image to the full image."""
    x0, y0, w, h, full_w, full_h = region
    lines = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        xy = np.array(parts[1:], float).reshape(-1, 2) * (w, h) + (x0, y0)
        xy /= (full_w, full_h)
        lines.append(parts[0] + " " + " ".join(f"{v:.6f}" for v in xy.ravel()))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


async def run(name: str, rgb, mask, w: int, h: int, um_per_px: float, k: int, last_modified_ms: float, infer,
              quality_json: str, folder_name: str = "", options_json: str = "{}", region_json: str = "null",
              det_rgb=None, det_w: int = 0, det_h: int = 0) -> str:
    """rgb: (h, w, 3) RGB bytes; mask: (h, w) class indices; k: rotation correction; quality_json: the
    manifest's quality settings; options_json: {"roi", "follicle", "quality"} on/off (missing = base_config);
    region_json: [x0, y0, w, h, full_w, full_h] when rgb/mask are a crop of the image (labels are written in
    full-image coordinates), det_rgb: then the whole image downscaled to the detectors' max_side, which the
    detectors and quality model see instead of the crop. Returns JSON {"row", "columns", "files"}; files (paths
    relative to /out, raw/<stem>.txt = raw prediction labels) are left in /out."""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    bgr = np.frombuffer(rgb.to_py(), np.uint8).reshape(h, w, 3)[..., ::-1].copy()
    pred = np.frombuffer(mask.to_py(), np.uint8).reshape(h, w)
    on = {**json.loads(defaults()), **json.loads(options_json)}
    settings = {"rotation_k": k, "roi_enabled": on["roi"], "follicle_enabled": on["follicle"],
                "quality_enabled": on["quality"]}
    region = json.loads(region_json)
    if region and det_rgb is not None:
        rot = postprocess.rotate_k(np.frombuffer(det_rgb.to_py(), np.uint8).reshape(det_h, det_w, 3)[..., ::-1], k)
        to_crop = lambda q: _det_to_crop(q, k, (det_w, det_h), region)  # noqa: E731
    else:
        rot = postprocess.rotate_k(bgr, k)
        to_crop = None
    for det, (conf, max_side, imgsz) in DETECTORS.items():
        if on[det]:
            settings[f"{det}s"] = await detect(infer, det, rot, conf, max_side, imgsz)
            print(f"{stem}: {det} detector: {len(settings[f'{det}s'])} boxes", flush=True)
    if on["quality"] and settings.get("rois"):
        settings["quality"] = await score(infer, rot, settings["rois"], json.loads(quality_json))
    del rot
    if to_crop:
        for key in ("follicles", "rois"):
            if key in settings:
                settings[key] = [(to_crop(q), c) for q, c in settings[key]]
        if "quality" in settings:
            scores, quads = settings["quality"]
            settings["quality"] = (scores, [to_crop(q) for q in quads])
    # empty stand-in for the source file, so the CSV date column = the file's last-modified date
    SRC.mkdir(exist_ok=True)
    src = SRC / name
    src.touch()
    os.utime(src, (last_modified_ms / 1000, last_modified_ms / 1000))
    OUT.mkdir(exist_ok=True)
    for f in [p for p in OUT.rglob("*") if p.is_file()]:
        f.unlink()
    (OUT / "raw").mkdir(exist_ok=True)
    postprocess.write_yolo_labels(pred, CLASS_INDEX, OUT / "raw" / f"{stem}.txt")
    row = postprocess.process_image(stem, folder_name, bgr, pred, CLASS_INDEX, um_per_px, OUT, source_path=src,
                                    postprocess_config=settings, seg_path=OUT / f"{stem}.txt")
    del bgr, pred
    if region:
        for p in (OUT / "raw" / f"{stem}.txt", OUT / f"{stem}.txt"):
            if p.exists():
                _to_frame(p, region)
    files = sorted(str(p.relative_to(OUT)).replace("\\", "/") for p in OUT.rglob("*") if p.is_file())
    return json.dumps({"row": row, "columns": CSV_COLUMNS, "files": files},
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))

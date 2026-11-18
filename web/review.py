"""Browser (Pyodide) side of the Review page's re-measure after a save (interface/seg_editor.py remeasure_doc).
Runs in the measurement engine; the editor's polygon helpers are in review_edit.py."""
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import postprocess
from measurement import CSV_COLUMNS, _rasterise, tissue_area_um2

OUT = Path("/out_review")
SRC = Path("/in_review")


def _append_pane(steps, panel):
    """seg_editor.append_user_pane on arrays: a captioned 'User Corrected' pane after the steps image."""
    h = steps.shape[0] if steps is not None else 508  # make_composite default: 480 + 28 px caption
    pane = postprocess.make_composite([(f"User Corrected ({datetime.now():%Y-%m-%d %H:%M})", panel)],
                                      target_h=h - 28)
    return pane if steps is None else np.hstack([steps, np.zeros((h, 4, 3), np.uint8), pane])


def remeasure(name: str, rgb, w: int, h: int, um_per_px: float, k: int, label: str, names_json: str, steps,
              lateral_clip_frac, last_modified_ms: float, region_json: str = "null") -> str:
    """seg_editor.remeasure_doc: measures the saved labels on the (possibly reduced) source image.
    region_json: [x0, y0, w, h, full_w, full_h] when rgb is a crop of the image the labels are normalised to.
    Writes /out_review/measurement.jpg and steps.jpg (steps = existing <stem>_steps.jpg bytes or None)."""
    names = json.loads(names_json)
    region = json.loads(region_json)
    if region:
        x0, y0, _, _, full_w, full_h = region
        lines = []
        for line in label.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                xy = (np.array(parts[1:], float).reshape(-1, 2) * (full_w, full_h) - (x0, y0)) / (w, h)
                lines.append(parts[0] + " " + " ".join(f"{v:.6f}" for v in xy.ravel()))
        label = "\n".join(lines) + "\n"
    bgr = np.frombuffer(rgb.to_py(), np.uint8).reshape(h, w, 3)[..., ::-1].copy()
    OUT.mkdir(exist_ok=True)
    for f in OUT.iterdir():
        f.unlink()
    lbl = OUT / "label.txt"
    lbl.write_text(label)
    mask = _rasterise(lbl, h, w)
    lbl.unlink()
    # empty stand-in for the source file, so the CSV date column = the file's last-modified date
    SRC.mkdir(exist_ok=True)
    src = SRC / name
    src.touch()
    os.utime(src, (last_modified_ms / 1000, last_modified_ms / 1000))
    class_index = {n: i + 1 for i, n in enumerate(names)}
    tissue = round(tissue_area_um2(bgr, um_per_px), 2)
    res = postprocess.remeasure_labels(bgr, mask, class_index, um_per_px,
                                       lateral_clip_frac or postprocess.LATERAL_CLIP_FRAC, source_path=src, k=k)
    del mask
    if res is None:
        return json.dumps({"row": None, "tissue": tissue, "columns": CSV_COLUMNS, "outputs": []})
    row_extra, panel, corrected = res
    postprocess.imwrite_jpg(OUT / "measurement.jpg", panel, 90)
    del bgr, panel
    old = cv2.imdecode(np.frombuffer(steps.to_py(), np.uint8), cv2.IMREAD_COLOR) if steps is not None else None
    postprocess.imwrite_jpg(OUT / "steps.jpg", _append_pane(old, corrected), 90)
    return json.dumps({"row": row_extra, "tissue": tissue, "columns": CSV_COLUMNS,
                       "outputs": ["measurement.jpg", "steps.jpg"]},
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))

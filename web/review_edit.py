"""Browser (Pyodide) side of the Review editor: interface/seg_editor.py's label / polygon helpers without Streamlit.
Light on purpose (numpy, OpenCV, PyYAML only) so the editor engine starts fast; re-measuring is in review.py.
Called from web/app.js through worker.js; file I/O stays in the page (File System Access API)."""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.semseg.add_segmentations import (CLASSES, PRED_ALIASES, cut_polys, line_class, merge_polys,
                                               parse_poly, poly_line)


def run_info(text: str) -> str:
    """seg_editor.run_info from run_config.yaml's text (predict.py YAML or the browser's JSON)."""
    cfg = (yaml.safe_load(text) or {}) if text else {}
    run, params, res = cfg.get("run") or {}, cfg.get("parameters") or {}, cfg.get("resolved") or {}
    return json.dumps({
        "names": [PRED_ALIASES.get(n, n) for n in run.get("model_class_names") or CLASSES],
        "files": {Path(f["file"]).stem: f for f in res.get("input_files") or []},
        "um_default": params.get("um_per_px"),
        "lateral_clip_frac": res.get("lateral_clip_frac"),
    })


def read_polys(label: str, names_json: str, iw: int, ih: int) -> str:
    """seg_editor.read_polys: {"polys": {class: [normalised polygons]}, "other": [label lines kept verbatim]}."""
    names = json.loads(names_json)
    polys, other = {c: [] for c in CLASSES if c in names}, []
    wh = np.array([iw, ih], np.float32)
    for line in label.splitlines():
        if not line.strip():
            continue
        c = line_class(line)
        name = names[c] if c < len(names) else None
        if name not in polys:
            other.append(line)
            continue
        # predict.py writes every boundary pixel; 1 native px tolerance keeps the outline but drops the stairs
        px = cv2.approxPolyDP(parse_poly(line) * wh, 1.0, True).reshape(-1, 2)
        if len(px) >= 3:
            polys[name].append((px / wh).round(6).tolist())
    return json.dumps({"polys": polys, "other": other})


def op(kind: str, cls: str, lasso_json: str, polys_json: str, iw: int, ih: int) -> str:
    """seg_editor._on_op: merge or cut the drawn lasso into class `cls`'s polygons."""
    old = [np.array(p, np.float32) for p in json.loads(polys_json)]
    lasso = np.array(json.loads(lasso_json), np.float32) * (iw, ih)
    if kind == "cut":
        new, n_cut, n_hole = cut_polys(old, lasso, iw, ih)
        done = n_cut > 0
        msg = (f"cut {n_cut} {cls} polygon(s)" if done else f"cut does not overlap any {cls} polygon") + (
            f"; ignored inside {n_hole} polygon(s) (holes not supported, cut from the edge)" if n_hole else "")
    else:
        new, n_hit = merge_polys(old, lasso, iw, ih)
        done = True
        msg = f"merged into {n_hit} {cls} polygon(s)" if n_hit else f"added {cls} polygon"
    polys = [np.asarray(p, np.float32).round(6).tolist() for p in new] if done else None
    return json.dumps({"polys": polys, "done": done, "message": msg})


def label_text(polys_json: str, other_json: str, names_json: str) -> str:
    """seg_editor.write_labels' file content, as {"text": ...}."""
    polys, names = json.loads(polys_json), json.loads(names_json)
    lines = list(json.loads(other_json)) + [poly_line(names.index(c), np.array(p, np.float32))
                                            for c in names if c in polys for p in polys[c] if len(p) >= 3]
    return json.dumps({"text": "".join(line + "\n" for line in lines)})

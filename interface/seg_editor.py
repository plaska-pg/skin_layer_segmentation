"""Edit and review helpers: load / save predict.py segmentations and the in-browser polygon editor."""

import base64
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from training.semseg.add_segmentations import (CLASSES, PRED_ALIASES, cut_polys, line_class,  # noqa: E402
                                               merge_polys, parse_poly, poly_line)

EDITOR_MAX_SIDE = 4096  # longest side of the image sent to the browser
EDITOR_HEIGHT = 720
CLASS_COLORS = {"SC": "#00c8ff", "Epidermis": "#ffc800", "gland": "#00ff00"}  # add_segmentations.py colours
IMAGE_EXTS = (".tif", ".tiff", ".svs", ".png", ".jpg", ".jpeg")
DOC_KEY = "edit_doc"
EDITOR_KEY = "seg_editor"


def long_path(p) -> Path:
    """Extended-length path so OneDrive outputs past MAX_PATH (260 chars) still open (as postprocess.long_path)."""
    import os
    if os.name != "nt":
        return Path(p)
    s = os.path.abspath(str(p))
    if s.startswith("\\\\?\\"):
        return Path(s)
    return Path("\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s)


# ----------------------------------------------------------------------------
# prediction folder I/O
# ----------------------------------------------------------------------------

def run_info(pred_dir: Path) -> dict:
    p = long_path(pred_dir / "run_config.yaml")
    cfg = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}
    run, params, res = cfg.get("run") or {}, cfg.get("parameters") or {}, cfg.get("resolved") or {}
    return {
        "names": [PRED_ALIASES.get(n, n) for n in run.get("model_class_names") or CLASSES],
        "files": {Path(f["file"]).stem: f for f in res.get("input_files") or []},
        "um_default": params.get("um_per_px"),
        "source": run.get("source"),
        "lateral_clip_frac": res.get("lateral_clip_frac"),
    }


def results_csv(pred_dir: Path) -> Path:
    """_results.csv, or the legacy results.csv if only that exists."""
    p = pred_dir / "_results.csv"
    legacy = pred_dir / "results.csv"
    return legacy if not long_path(p).exists() and long_path(legacy).exists() else p


def read_rows(pred_dir: Path):
    p = long_path(results_csv(pred_dir))
    if not p.exists():
        return [], []
    with p.open(newline="") as f:
        rd = csv.DictReader(f)
        return list(rd), list(rd.fieldnames or [])


def update_row(pred_dir: Path, stem: str, updates: dict, new_row: dict = None) -> dict:
    """Merge `updates` into the CSV row of `stem` (appending `new_row` + updates if it has none)."""
    from measurement import CSV_COLUMNS
    rows, fields = read_rows(pred_dir)
    fields = list(dict.fromkeys(fields + CSV_COLUMNS + list(updates)))
    row = next((r for r in rows if r.get("file name") == stem), None)
    if row is None:
        row = {"file name": stem, **(new_row or {})}
        rows.append(row)
    row.update(updates)
    with long_path(results_csv(pred_dir)).open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    return row


def list_stems(pred_dir: Path) -> list[str]:
    stems = {r["file name"] for r in read_rows(pred_dir)[0] if r.get("file name")}
    for kind in ("postprocessed", "raw"):
        d = long_path(pred_dir / f"{kind}_segmentations")
        if d.is_dir():
            stems |= {p.stem for p in d.glob("*.txt")}
    return sorted(stems)


def label_path(pred_dir: Path, stem: str):
    """(path, kind) of the label to edit: postprocessed, else raw; None if neither exists."""
    for kind in ("postprocessed", "raw"):
        p = pred_dir / f"{kind}_segmentations" / f"{stem}.txt"
        if long_path(p).exists():
            return p, kind
    return None, None


def find_image(pred_dir: Path, stem: str, info: dict):
    """Source image of `stem`: from run_config's source folder, else the folder holding pred_dir."""
    dirs = []
    if info["source"]:
        src = Path(info["source"])
        dirs.append(src.parent if src.is_file() else src)
    dirs.append(pred_dir.parent)
    name = (info["files"].get(stem) or {}).get("file")
    for d in dirs:
        cands = [d / f"TIFF_{d.name}" / f"{stem}.tif"]  # predict.py's lossless copy of an .svs
        if name:
            cands.append(d / name)
        cands += [d / f"{stem}{ext}" for ext in IMAGE_EXTS]
        for c in cands:
            if long_path(c).is_file():
                return c
    return None


def read_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(long_path(path)), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not read {path}")
    return img


@st.cache_data(max_entries=4, show_spinner="Loading image...")
def display_image(path: str, _mtime: float):
    """(JPEG data URL downscaled to EDITOR_MAX_SIDE, native width, native height)."""
    img = read_image(Path(path))
    h, w = img.shape[:2]
    f = min(1.0, EDITOR_MAX_SIDE / max(h, w))
    if f < 1:
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"), w, h


def read_polys(path: Path, names: list[str], iw: int, ih: int):
    """({editable class: [normalised polygons as lists]}, other label lines kept verbatim)."""
    polys, other = {c: [] for c in CLASSES if c in names}, []
    wh = np.array([iw, ih], np.float32)
    for line in long_path(path).read_text().splitlines():
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
    return polys, other


# ----------------------------------------------------------------------------
# editor state (st.session_state[DOC_KEY])
# ----------------------------------------------------------------------------

def load_doc(pred_dir: Path, stem: str):
    info = run_info(pred_dir)
    lbl, kind = label_path(pred_dir, stem)
    img_path = find_image(pred_dir, stem, info)
    if lbl is None:
        return f"No raw or postprocessed segmentation found for {stem} in {pred_dir}"
    if img_path is None:
        return f"Source image of {stem} not found (looked in {info['source'] or '-'} and {pred_dir.parent})"
    image, iw, ih = display_image(str(img_path), long_path(img_path).stat().st_mtime)
    polys, other = read_polys(lbl, info["names"], iw, ih)
    prev = st.session_state.get(DOC_KEY) or {}
    st.session_state[DOC_KEY] = {
        "id": f"{pred_dir}|{stem}|{prev.get('loads', 0) + 1}", "loads": prev.get("loads", 0) + 1,
        "pred_dir": str(pred_dir), "stem": stem, "img_path": str(img_path), "info": info, "kind": kind,
        "iw": iw, "ih": ih, "polys": polys, "other": other, "version": 0, "dirty": False, "noop": False,
        "message": f"loaded {kind} segmentation", "pending_save": None,
    }
    return None


def _on_op():
    doc, op = st.session_state.get(DOC_KEY), st.session_state[EDITOR_KEY].get("op")
    if not doc or not op:
        return
    iw, ih, cls = doc["iw"], doc["ih"], op["cls"]
    polys = op["polys"]
    old = [np.array(p, np.float32) for p in polys.get(cls, [])]
    lasso = np.array(op["lasso"], np.float32) * (iw, ih)
    if op["kind"] == "cut":
        new, n_cut, n_hole = cut_polys(old, lasso, iw, ih)
        done = n_cut > 0
        msg = (f"cut {n_cut} {cls} polygon(s)" if done else f"cut does not overlap any {cls} polygon") + (
            f"; ignored inside {n_hole} polygon(s) (holes not supported, cut from the edge)" if n_hole else "")
    else:
        new, n_hit = merge_polys(old, lasso, iw, ih)
        done = True
        msg = f"merged into {n_hit} {cls} polygon(s)" if n_hit else f"added {cls} polygon"
    if done:
        polys[cls] = [np.asarray(p, np.float32).round(6).tolist() for p in new]
    doc.update(polys=polys, version=doc["version"] + 1, dirty=bool(op.get("dirty")) or done, noop=not done,
               message=msg)


def _on_save():
    doc, sv = st.session_state.get(DOC_KEY), st.session_state[EDITOR_KEY].get("save")
    if doc and sv:
        doc["pending_save"] = sv["polys"]


def _on_need_image():
    if st.session_state.get(DOC_KEY):
        st.session_state[DOC_KEY]["image_sent"] = False


def save_doc(doc: dict, polys: dict) -> str:
    """Overwrite postprocessed_segmentations/<stem>.txt (original kept once in original/), re-measure,
    rewrite <stem>_measurement.jpg and the CSV row. Returns a status message."""
    from measurement import _rasterise, tissue_area_um2
    from postprocess import LATERAL_CLIP_FRAC, imwrite_jpg, remeasure_labels

    pred_dir, stem, info = Path(doc["pred_dir"]), doc["stem"], doc["info"]
    names = info["names"]
    lbl = pred_dir / "postprocessed_segmentations" / f"{stem}.txt"
    backup = pred_dir / "postprocessed_segmentations" / "original" / f"{stem}.txt"
    if long_path(lbl).exists() and not long_path(backup).exists():
        long_path(backup.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(long_path(lbl), long_path(backup))
    lines = list(doc["other"]) + [poly_line(names.index(c), np.array(p, np.float32))
                                  for c in names if c in polys for p in polys[c] if len(p) >= 3]
    long_path(lbl.parent).mkdir(parents=True, exist_ok=True)
    with long_path(lbl).open("w", newline="\n") as f:
        f.writelines(l + "\n" for l in lines)

    img_path = Path(doc["img_path"])
    img = read_image(img_path)
    um = (info["files"].get(stem) or {}).get("um_per_px") or info["um_default"] or 1.0
    class_index = {n: k + 1 for k, n in enumerate(names)}
    mask = _rasterise(long_path(lbl), *img.shape[:2])
    res = remeasure_labels(img, mask, class_index, um, info["lateral_clip_frac"] or LATERAL_CLIP_FRAC,
                           source_path=long_path(img_path))
    if res is None:
        return "Saved labels; SC or Epidermis is empty, so measurements were not updated."
    row_extra, panel = res
    imwrite_jpg(long_path(pred_dir / f"{stem}_measurement.jpg"), panel, 90)
    src = Path(info["source"]) if info["source"] else pred_dir.parent
    update_row(pred_dir, stem, {**row_extra, "edited": datetime.now().isoformat(timespec="seconds")},
               new_row={"folder name": src.name, "total_tissue_um2": round(tissue_area_um2(img, um), 2)})
    return "Saved labels and updated the measurement image and CSV row."


_EDITOR = st.components.v2.component(
    "seg_editor",
    html="""
<div class="seg-ed" tabindex="0">
  <div class="bar"></div>
  <canvas></canvas>
  <div class="status"></div>
  <div class="hint"></div>
</div>
""",
    css="""
.seg-ed { outline: none; font-family: var(--st-font, sans-serif); color: var(--st-text-color); }
.seg-ed:focus canvas { box-shadow: 0 0 0 2px var(--st-primary-color); }
.bar { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
.bar button { border: 2px solid var(--st-border-color, #888); background: var(--st-secondary-background-color);
  color: var(--st-text-color); border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 0.85rem; }
.bar button.on { background: var(--st-primary-color); color: #fff; }
.bar button.primary { margin-left: auto; background: var(--st-primary-color); color: #fff;
  border-color: var(--st-primary-color); }
canvas { display: block; width: 100%; cursor: crosshair; touch-action: none; border-radius: 4px; }
.status { margin-top: 4px; font-size: 0.85rem; }
.hint { font-size: 0.75rem; opacity: 0.7; }
""",
    js=(Path(__file__).with_name("seg_editor.js")).read_text(encoding="utf-8"),
)


def editor(pred_dir: Path, stem: str):
    """Mounts the polygon editor for `stem`; saves when the editor asks to. Returns True after a save."""
    doc = st.session_state.get(DOC_KEY)
    if not doc or doc["pred_dir"] != str(pred_dir) or doc["stem"] != stem:
        err = load_doc(pred_dir, stem)
        if err:
            st.error(err)
            return False
        doc = st.session_state[DOC_KEY]
    saved = False
    if doc["pending_save"] is not None:
        polys, doc["pending_save"] = doc["pending_save"], None
        with st.spinner("Saving and re-measuring (about a minute for a whole slide)..."):
            try:
                doc["message"] = save_doc(doc, polys)
                doc.update(polys=polys, dirty=False)
                saved = True
            except Exception as e:  # noqa: BLE001 - keep the edits in the browser, show why the save failed
                doc.update(polys=polys, dirty=True, message=f"save failed: {e}")
        doc.update(version=doc["version"] + 1, noop=False)

    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.caption(f"Editing the {doc['kind']} segmentation of `{doc['img_path']}`. Unsaved edits are lost when you "
               "switch images or reload.")
    if c2.button("Reload from disk", width="stretch"):
        load_doc(pred_dir, stem)
        doc = st.session_state[DOC_KEY]
    # the multi-MB image goes out once per document; reruns for merge / cut / save only carry polygons
    image = None
    if not doc.get("image_sent"):
        image = display_image(doc["img_path"], long_path(doc["img_path"]).stat().st_mtime)[0]
        doc["image_sent"] = True
    _EDITOR(
        key=EDITOR_KEY,
        data={k: doc[k] for k in ("iw", "ih", "polys", "version", "dirty", "noop", "message")} | {
            "doc": doc["id"], "height": EDITOR_HEIGHT, "image": image,
            "classes": [{"name": c, "color": CLASS_COLORS.get(c, "#ffffff")} for c in doc["polys"]],
        },
        on_op_change=_on_op,
        on_save_change=_on_save,
        on_need_image_change=_on_need_image,
    )
    return saved

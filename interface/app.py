"""Streamlit front end for predict.py: run the pipeline, review/edit segmentations, explore results.

    streamlit run interface/app.py
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

import seg_editor
Image.MAX_IMAGE_PIXELS = None  # QA jpgs of whole-slide images exceed PIL's decompression-bomb cap

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_DIR / "base_config.yaml"
RUN_CONFIG_DIR = PROJECT_DIR / "runs" / "interface"
LAST_USED = RUN_CONFIG_DIR / "last_used.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs"}  # keep in sync with predict.py
PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]")
PREVIEW_MAX_SIDE = 2000  # downscale QA jpgs before sending them to the browser
SCANNERS = {"Motic 40X (0.2577 µm/px)": 0.2577, "BX61 10X (0.645 µm/px)": 0.645}
SCOPES = ["This folder", "Include subfolders", "One image"]
QUICK_KEYS = {("predict", "um_per_px"), ("predict", "tta"), ("postprocess", "roi_enabled"),
              ("postprocess", "follicle_enabled"), ("postprocess", "quality_enabled")}
KEY_METRICS = {"SC mean height": "SC thickness (µm)", "Epidermal mean height": "Epidermis thickness (µm)",
               "EDJ tortuosity": "EDJ tortuosity", "quality": "Quality (1-3)"}
RESULT_COLUMNS = ["folder name", "file name", *KEY_METRICS, "edited"]
SESSION = "session"  # persist_state: keep widget values across page switches

st.set_page_config(page_title="Skin segmentation", layout="wide")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_preview(path: Path) -> Image.Image:
    if os.name == "nt":  # extended-length prefix so QA jpgs past MAX_PATH (260 chars) still open
        s = os.path.abspath(str(path))
        if not s.startswith("\\\\?\\"):
            path = Path("\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s)
    with Image.open(path) as im:
        im.draft("RGB", (PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))  # fast JPEG downscale on decode
        im = im.convert("RGB")
    im.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    return im


@st.cache_data(max_entries=8, show_spinner=False)
def _cached_preview(path: str, mtime: float) -> Image.Image:
    return load_preview(Path(path))


def show_preview(path: Path, where=st, caption: str = None) -> bool:
    """Shows a downscaled `path` in `where`; False if it can't be read (missing or still being written)."""
    try:
        img = _cached_preview(str(path), seg_editor.long_path(path).stat().st_mtime)
    except OSError:
        return False
    where.image(img, caption=caption or path.name, width="stretch")
    return True


def image_folders(root: Path) -> list[Path]:
    """root and every subfolder that directly holds images, skipping predicted_* (and legacy *_predicted) output folders."""
    dirs = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
    return [d for d in dirs
            if not any(part.lower().startswith("predicted_") or "_predicted" in part.lower()
                       for part in d.relative_to(root).parts)
            and list_images(d)]


def config_help() -> dict:
    """{(section, key): trailing comment} from base_config.yaml, used as widget help text."""
    help_text, section = {}, None
    for line in BASE_CONFIG.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"(\w+):\s*(#.*)?$", line):
            section = m.group(1)
        elif section and (m := re.match(r"\s+(\w+):[^#]*#\s*(.+)", line)):
            help_text[(section, m.group(1))] = m.group(2).strip()
    return help_text


def _init_widget(wkey: str, value):
    """Seeds widget `wkey` with its base_config value, which is also what Reset restores."""
    st.session_state.setdefault("_cfg_defaults", {})[wkey] = value
    st.session_state.setdefault(wkey, value)


def cfg_widget(section: str, key: str, val, label: str = None, help: str = None):
    """One base_config value as a widget (key cfg_<section>_<key>); returns the edited value."""
    label = label or key
    wkey = f"cfg_{section}_{key}"
    kw = dict(key=wkey, help=help, persist_state=SESSION)
    if isinstance(val, bool):
        _init_widget(wkey, val)
        return st.toggle(label, **kw)
    if isinstance(val, int):
        _init_widget(wkey, val)
        return int(st.number_input(label, step=1, **kw))
    if isinstance(val, float):
        _init_widget(wkey, val)
        return float(st.number_input(label, step=0.01, format="%.4f", **kw))
    if isinstance(val, str):
        _init_widget(wkey, val)
        return st.text_input(label, **kw)
    # lists / null: edit as YAML text, e.g. "[51.6, 103.2]" or "null"
    _init_widget(wkey, yaml.safe_dump(val, default_flow_style=True).strip().removesuffix("...").strip())
    text = st.text_input(f"{label} (YAML)", **kw)
    try:
        return yaml.safe_load(text) if text.strip() else None
    except yaml.YAMLError:
        st.error(f"{section}.{key}: invalid YAML, using the base_config.yaml value")
        return val


def _reset_settings():
    for k, v in st.session_state.get("_cfg_defaults", {}).items():
        st.session_state[k] = v


def quick_settings(cfg: dict, help_text: dict):
    """The few settings most runs change; writes them into `cfg`."""
    pred, post = cfg["predict"], cfg["postprocess"]
    c1, c2 = st.columns(2)
    with c1:
        presets = [*SCANNERS, "Custom"]
        _init_widget("cfg_scanner", next((name for name, um in SCANNERS.items() if um == pred["um_per_px"]),
                                         "Custom"))
        scanner = st.selectbox("Scanner", presets, key="cfg_scanner", persist_state=SESSION,
                               help="Sets the image resolution (µm per pixel).")
        if scanner == "Custom":
            pred["um_per_px"] = cfg_widget("predict", "um_per_px", pred["um_per_px"], "µm per pixel",
                                           help_text.get(("predict", "um_per_px")))
        else:
            pred["um_per_px"] = SCANNERS[scanner]
        _init_widget("cfg_predict_tta", pred["tta"])
        pred["tta"] = st.segmented_control("Test-time augmentation", ["none", "flip", "d4"], required=True,
                                           key="cfg_predict_tta", persist_state=SESSION,
                                           help=help_text.get(("predict", "tta")))
    with c2:
        for key, label in (("roi_enabled", "Detect ROI (drop SC/Epidermis outside it)"),
                           ("follicle_enabled", "Detect and mask follicles"),
                           ("quality_enabled", "Score section quality")):
            post[key] = cfg_widget("postprocess", key, post[key], label, help_text.get(("postprocess", key)))


def advanced_settings(cfg: dict, help_text: dict):
    """Every other base_config value, three columns per section; writes them into `cfg`."""
    st.button("Reset settings to base_config.yaml", on_click=_reset_settings, icon=":material/restart_alt:")
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        keys = [k for k in values if (section, k) not in QUICK_KEYS and (section, k) != ("predict", "source")]
        st.markdown(f"**{section}**")
        cols = st.columns(3)
        for i, key in enumerate(keys):
            with cols[i % 3]:
                values[key] = cfg_widget(section, key, values[key], help=help_text.get((section, key)))


# Run in a subprocess: tkinter is unreliable on Streamlit's script thread.
_DIALOG_SCRIPT = r"""
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
kind, init = sys.argv[1], sys.argv[2] or None
if kind == "file":
    p = filedialog.askopenfilename(initialdir=init, filetypes=[
        ("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.svs"), ("All files", "*.*")])
else:
    p = filedialog.askdirectory(initialdir=init)
print(p or "")
"""


def _browse_into(key: str, kind: str):
    current = Path(st.session_state.get(key, "").strip().strip('"') or ".")
    init = current if current.is_dir() else current.parent if current.parent.is_dir() else Path.home()
    picked = subprocess.run([sys.executable, "-c", _DIALOG_SCRIPT, kind, str(init)],
                            capture_output=True, text=True).stdout.strip()
    if picked:
        st.session_state[key] = str(Path(picked))


def path_input(label: str, key: str, kind: str = "dir") -> str:
    """Text input with a Browse... button that opens a native file/folder dialog."""
    c_text, c_btn = st.columns([8, 1], vertical_alignment="bottom")
    value = c_text.text_input(label, key=key)
    c_btn.button("Browse...", key=f"{key}_browse", on_click=_browse_into, args=(key, kind), width="stretch")
    return value.strip().strip('"')


def folder_sidebar():
    """The working folder shared by all pages; remembered across app restarts in LAST_USED."""
    if "folder" not in st.session_state:
        try:
            st.session_state["folder"] = json.loads(LAST_USED.read_text(encoding="utf-8")).get("folder", "")
        except (OSError, ValueError):
            st.session_state["folder"] = ""
        st.session_state["_saved_folder"] = st.session_state["folder"]
    with st.sidebar:
        st.text_input("Working folder", key="folder",
                      help="Folder of images. Results go to predicted_* folders inside it, "
                           "and Review and Results read them from there.")
        st.button("Browse...", key="folder_browse", on_click=_browse_into, args=("folder", "dir"),
                  width="stretch")
    folder = st.session_state["folder"].strip().strip('"')
    if folder != st.session_state.get("_saved_folder") and Path(folder).is_dir():
        RUN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LAST_USED.write_text(json.dumps({"folder": folder}), encoding="utf-8")
        st.session_state["_saved_folder"] = folder
    if folder and Path(folder).is_dir():
        with st.sidebar:
            predicted_folder_select(Path(folder).resolve())


def predicted_folder_select(root: Path):
    """Picker of the predict.py output folder the Review page opens (key review_pred)."""
    target = st.session_state.get("review_target")
    if target:
        st.session_state["review_extra"] = target["dir"]  # may be outside the working folder (custom output)
        st.session_state["review_pred"] = target["dir"]
    folders = prediction_folders(str(root))
    extra = st.session_state.get("review_extra")
    if extra and extra not in folders and Path(extra).is_dir():
        folders = [extra, *folders]
    if st.session_state.get("review_pred") not in folders:
        st.session_state["review_pred"] = folders[0] if folders else None
    c_sel, c_btn = st.columns([4, 1], vertical_alignment="bottom", gap="small")
    c_sel.selectbox("Predicted folder", folders, key="review_pred", persist_state=SESSION,
                    format_func=lambda p: _rel(p, root), placeholder="None yet",
                    help="predict.py output folder opened on the Review page.")
    if c_btn.button(":material/refresh:", key="refresh_review", help="Look for new predicted folders",
                    width="stretch"):
        prediction_folders.clear()
        st.rerun()


def working_folder():
    """The sidebar folder as a Path, or None after telling the user why it can't be used."""
    folder = st.session_state.get("folder", "").strip().strip('"')
    if not folder:
        st.info("Choose a working folder in the sidebar.")
        return None
    if not Path(folder).is_dir():
        st.error(f"Folder not found: {folder}")
        return None
    return Path(folder).resolve()


def default_output(source: str, single: bool) -> str:
    if not source:
        return ""
    p = Path(source).resolve()
    return str(p.parent / f"predicted_{p.stem}" if single else p / f"predicted_{p.name}")


def newest_measurement(out_dirs: list[Path]):
    found = [p for d in out_dirs if d.exists() for p in d.glob("*_measurement.jpg")]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def run_page():
    folder = working_folder()
    if not folder:
        return
    scope = st.segmented_control("Process", SCOPES, default=SCOPES[0], required=True, key="run_scope",
                                 persist_state=SESSION)
    if scope == "One image":
        names = [p.name for p in list_images(folder)]
        if not names:
            st.warning(f"No images directly in {folder}.")
            return
        if st.session_state.get("run_image") not in names:
            st.session_state.pop("run_image", None)
        src = folder / st.selectbox("Image", names, key="run_image", persist_state=SESSION)
    else:
        src = folder
    auto_out = default_output(str(src), scope == "One image")
    # follow the source until the user types their own output folder
    if st.session_state.get("out_root", "") in ("", st.session_state.get("out_auto")):
        st.session_state["out_root"] = auto_out
    st.session_state["out_auto"] = auto_out

    help_text = config_help()
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {}
    st.subheader("Settings")
    quick_settings(cfg, help_text)
    with st.expander("Advanced settings"):
        out_root = path_input("Output folder" + (" (subfolders are mirrored beneath it; leave as the default to "
                                                  "put 'predicted_<name>' inside each subfolder)"
                                                  if scope == "Include subfolders" else ""), key="out_root")
        c1, c2 = st.columns(2)
        resume = c1.checkbox("Resume (reuse already-processed images)", value=True, key="run_resume",
                             persist_state=SESSION)
        limit = int(c2.number_input("Limit images per folder (0 = all)", min_value=0, value=0, step=1,
                                    key="run_limit", persist_state=SESSION))
        st.divider()
        advanced_settings(cfg, help_text)

    if not st.button("Run", type="primary", icon=":material/play_arrow:"):
        if st.session_state.get("last_run"):
            show_last_run(st.session_state["last_run"])
        return
    run_jobs(src, scope, out_root, auto_out, resume, limit, cfg)


def run_jobs(src: Path, scope: str, out_root: str, auto_out: str, resume: bool, limit: int, cfg: dict):
    """Runs predict.py once per image folder, streaming progress and the newest measurement image."""
    out_base = Path(out_root).resolve() if out_root else None
    if out_base == Path(auto_out):
        out_base = None

    # one predict.py call per (source, out) job
    if scope == "One image":
        jobs = [(src, out_base or src.parent / f"predicted_{src.stem}", 1)]
    else:
        folders = image_folders(src) if scope == "Include subfolders" else ([src] if list_images(src) else [])
        jobs = []
        for d in folders:
            n = len(list_images(d))
            n = min(n, limit) if limit else n
            if out_base:
                out = out_base / d.relative_to(src)
            else:
                out = d / f"predicted_{d.name}"
            jobs.append((d, out, n))
    if not jobs:
        st.error("No images found.")
        return

    RUN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = RUN_CONFIG_DIR / f"config_{datetime.now():%Y%m%d_%H%M%S}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = {**os.environ, "SKIN_SEG_CONFIG": str(cfg_path), "PYTHONIOENCODING": "utf-8"}

    total = sum(n for _, _, n in jobs)
    status = f"Processing {total} image(s) in {len(jobs)} folder(s)"
    HEADER.caption(f":material/play_circle: {status}: starting predict.py")
    st.caption(f"{len(jobs)} folder(s), {total} image(s). Config: {cfg_path}. "
               "Clicking anything (e.g. Stop) or switching pages cancels the run.")
    st.button("Stop", icon=":material/stop:")
    progress = st.progress(0.0, text="Starting...")
    preview = st.empty()
    with st.expander("Log"):
        log_box = st.empty()
    log_lines: list[str] = []
    warnings: list[str] = []
    done_before = 0
    out_dirs = [out for _, out, _ in jobs]
    shown = None

    first_src = jobs[0][0]
    first_image = first_src if first_src.is_file() else (list_images(first_src) or [None])[0]
    if first_image:
        show_preview(first_image, preview, f"Input: {first_image.name}")

    for job_i, (job_src, job_out, n) in enumerate(jobs, 1):
        cmd = [sys.executable, "-u", "predict.py", "--source", str(job_src), "--out", str(job_out)]
        if resume:
            cmd.append("--resume")
        if limit:
            cmd += ["--limit", str(limit)]
        log_lines.append(f"===== [{job_i}/{len(jobs)}] {job_src}")
        proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            with st.spinner("Computing segmentations and measurements..."):
                for line in proc.stdout:
                    line = line.rstrip()
                    log_lines.append(line)
                    log_box.code("\n".join(log_lines[-25:]), language=None)
                    m = PROGRESS_RE.search(line)
                    if m:
                        i = int(m.group(1))
                        status = f"Folder {job_i}/{len(jobs)}, image {i}/{m.group(2)}: {job_src.name}"
                        progress.progress(min((done_before + i - 1) / max(total, 1), 1.0), text=status)
                    if line.strip():
                        step = line.strip().replace("`", "'")[:160]
                        HEADER.caption(f":material/play_circle: {status} · `{step}`")
                    latest = newest_measurement(out_dirs)
                    # unreadable = still being written; picked up on a later line
                    if latest and latest != shown and show_preview(latest, preview):
                        shown = latest
                proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode != 0:
            warnings.append(f"predict.py exited with code {proc.returncode} for {job_src}")
        done_before += n

    st.session_state["last_run"] = {"out_dirs": [str(o) for o in out_dirs], "warnings": warnings}
    prediction_folders.clear()
    load_results.clear()
    st.rerun()


def show_last_run(last: dict):
    out_dirs = [Path(p) for p in last["out_dirs"]]
    st.divider()
    for w in last["warnings"]:
        st.warning(w)
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.subheader("Last run")
    if c2.button("Review results", type="primary", icon=":material/fact_check:", width="stretch"):
        st.session_state["review_target"] = {"dir": str(out_dirs[0]), "stem": None}
        st.switch_page(REVIEW_PAGE)
    latest = newest_measurement(out_dirs)
    if latest:
        show_preview(latest)
    results = [out / "_results.csv" for out in out_dirs if (out / "_results.csv").exists()]
    if results:
        st.dataframe(pd.concat([pd.read_csv(p) for p in results], ignore_index=True), hide_index=True)


# ----------------------------------------------------------------------------
# Review
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Searching for prediction folders...")
def prediction_folders(root: str) -> list[str]:
    """root and every folder below it holding predict.py output."""
    root_p = Path(root)
    found = {p.parent for name in ("run_config.yaml", "_results.csv") for p in root_p.rglob(name)}
    found |= {p.parent for p in root_p.rglob("*_segmentations") if p.is_dir()}
    return sorted(str(p) for p in found)


def _on_editor_nav(stems: list[str]):
    _step(stems, st.session_state[seg_editor.EDITOR_KEY].get("nav") or 0)
    st.session_state["_review_nav"] = True


@st.fragment
def editor_fragment(pred_dir: Path, stem: str, stems: list[str]):
    i = stems.index(stem)
    nav = {"label": f"{i + 1}/{len(stems)}", "prev": i > 0, "next": i < len(stems) - 1}
    queued = seg_editor.editor(pred_dir, stem, nav, on_nav=lambda: _on_editor_nav(stems))
    # full rerun: the header starts tracking the save; navigation refreshes the picker and metrics
    if queued or st.session_state.pop("_review_nav", False):
        st.rerun()


def _step(stems: list[str], d: int):
    i = stems.index(st.session_state["review_stem"]) + d
    st.session_state["review_stem"] = stems[max(0, min(i, len(stems) - 1))]


def _rel(p: str, root: Path) -> str:
    try:
        rel = Path(p).relative_to(root)
    except ValueError:
        return p
    return str(rel) if rel.parts else root.name


def review_page():
    root = working_folder()
    if not root:
        return
    target = st.session_state.pop("review_target", None)
    if not st.session_state.get("review_pred"):
        st.info("No predicted folders in this working folder yet. Process images on the Run page first.")
        return
    pred_dir = Path(st.session_state["review_pred"])
    stems = seg_editor.list_stems(pred_dir)
    if not stems:
        st.info("No images in this predicted folder.")
        return
    rows = {r.get("file name"): r for r in seg_editor.read_rows(pred_dir)[0]}
    if target and target["stem"]:
        stem = target["stem"] if target["stem"] in stems else Path(target["stem"]).stem
        if stem in stems:
            st.session_state["review_stem"] = stem
    if st.session_state.get("review_stem") not in stems:
        st.session_state["review_stem"] = stems[0]

    n_edited = sum(bool((rows.get(s) or {}).get("edited")) for s in stems)
    c_img, c_view = st.columns([3, 2], vertical_alignment="bottom")
    stem = c_img.selectbox(f"Image", stems, key="review_stem",
                           persist_state=SESSION,
                           format_func=lambda s: s + ("  (edited)" if (rows.get(s) or {}).get("edited") else ""))
    view = c_view.segmented_control("View", ["Edit", "Measurement", "Steps"], default="Edit", required=True,
                                    key="review_view", persist_state=SESSION)

    row = rows.get(stem) or {}

    if view == "Edit":
        editor_fragment(pred_dir, stem, stems)
    else:
        i = stems.index(stem)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.button(":material/chevron_left:", help="Previous image", disabled=i == 0, on_click=_step,
                      args=(stems, -1))
            st.caption(f"{i + 1}/{len(stems)}", width="content")
            st.button(":material/chevron_right:", help="Next image", disabled=i == len(stems) - 1,
                      on_click=_step, args=(stems, 1))
        if view == "Steps":
            st.caption("Postprocessing steps. Each save appends your corrected segmentation as a new pane.")
        if not show_preview(pred_dir / f"{stem}_{view.lower()}.jpg"):
            st.info(f"No {view.lower()} image for {stem}")
    with st.expander("All values"):
        if row:
            st.dataframe(pd.DataFrame({"value": row}), width="stretch")
        else:
            st.caption("No CSV row for this image (it may have been skipped during postprocessing).")


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Searching for _results.csv...")
def load_results(root: str) -> pd.DataFrame:
    frames = []
    paths = set(Path(root).rglob("_results.csv"))
    # legacy 'results.csv' from older runs, unless superseded by a sibling _results.csv
    paths |= {p for p in Path(root).rglob("results.csv") if not (p.parent / "_results.csv").exists()}
    for p in sorted(paths):
        try:
            df = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001 - skip unreadable/partial csvs, keep the rest
            st.warning(f"Could not read {p}: {e}")
            continue
        df["results path"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def results_page():
    root = working_folder()
    if not root:
        return
    if st.button("Refresh", key="refresh_results", icon=":material/refresh:"):
        load_results.clear()
    df = load_results(str(root))
    if df.empty:
        st.info("No _results.csv files in this folder yet. Process images on the Run page first.")
        return

    folders = sorted(df["folder name"].astype(str).unique()) if "folder name" in df else []
    c1, c2 = st.columns(2)
    selected = c1.multiselect("Folders", folders, placeholder="All folders")
    view = df[df["folder name"].astype(str).isin(selected)] if selected else df
    shown = [c for c in RESULT_COLUMNS if c in view]
    extra = c2.multiselect("More columns", [c for c in view.columns if c not in shown and c != "results path"],
                           placeholder="Add columns")
    st.caption(f"{len(view)} images from {view['results path'].nunique()} _results.csv file(s). "
               "Select a row to open it in Review.")
    event = st.dataframe(view, column_order=shown + extra, hide_index=True, on_select="rerun",
                         selection_mode="single-row", key="results_table")
    if event.selection.rows and "file name" in view:
        r = view.iloc[event.selection.rows[0]]
        if st.button(f"Open {r['file name']} in Review", type="primary", icon=":material/fact_check:"):
            st.session_state["review_target"] = {"dir": str(Path(r["results path"]).parent),
                                                 "stem": str(r["file name"])}
            st.switch_page(REVIEW_PAGE)
    st.download_button("Download combined CSV", view.to_csv(index=False), "combined_results.csv", "text/csv",
                       icon=":material/download:")

    numeric = list(view.select_dtypes("number").columns)
    if numeric and folders:
        st.subheader("By folder")
        default = next((numeric.index(k) for k in KEY_METRICS if k in numeric), 0)
        metric = st.selectbox("Metric", numeric, index=default)
        st.bar_chart(view.groupby("folder name")[metric].mean(), y_label=f"mean {metric}")
        st.dataframe(view.groupby("folder name")[numeric].agg(["mean", "std", "count"]))


# ----------------------------------------------------------------------------
# Header status
# ----------------------------------------------------------------------------

def _save_status():
    """One line on the newest background save; reruns the Review page when saves finish there."""
    jobs = st.session_state.get(seg_editor.SAVE_JOBS, [])
    active = [j for j in jobs if j["end"] is None]
    finished = [j for j in jobs if j["end"] is not None and not j.get("seen")]
    for j in finished:
        j["seen"] = True
    if finished:
        load_results.clear()
    if active:
        j = active[0]
        when = f"{time.time() - j['started']:.0f} s" if j["started"] else "queued"
        more = f" · {len(active) - 1} more queued" if len(active) > 1 else ""
        st.caption(f":material/sync: Saved **{j['stem']}**, re-measuring in the background ({when}){more}")
    elif jobs:
        j = jobs[-1]
        done = datetime.fromtimestamp(j["end"]).strftime("%H:%M:%S")
        if j["error"]:
            st.caption(f":material/error: :red[**{j['stem']}**: {j['message']}] ({done})")
        else:
            took = j["end"] - (j["started"] or j["queued"])
            st.caption(f":material/check_circle: Saved **{j['stem']}**: {j['message']} ({done}, {took:.0f} s)")
    # refresh the metrics, measurement and steps images of the image being reviewed
    if finished and not active and st.session_state.get("_page") == "Review":
        st.rerun()


@st.fragment(run_every=1)
def _save_status_live():
    _save_status()


def header_status():
    """Top-of-page line saying what the app is doing; returns the slot run_jobs fills while processing."""
    slot = st.empty()
    jobs = st.session_state.get(seg_editor.SAVE_JOBS, [])
    if any(j["end"] is None for j in jobs):
        _save_status_live()
    elif jobs:
        _save_status()
    else:
        slot.caption(":material/check_circle: Ready")
    return slot


RUN_PAGE = st.Page(run_page, title="Run", icon=":material/play_circle:", default=True)
REVIEW_PAGE = st.Page(review_page, title="Review", icon=":material/fact_check:", url_path="review")
RESULTS_PAGE = st.Page(results_page, title="Results", icon=":material/table_chart:", url_path="results")
page = st.navigation([RUN_PAGE, REVIEW_PAGE, RESULTS_PAGE], position="top")
folder_sidebar()
st.session_state["_page"] = page.title
HEADER = header_status()
page.run()

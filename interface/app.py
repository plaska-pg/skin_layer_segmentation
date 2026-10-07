"""Bare-bones Streamlit front end for predict.py.

    streamlit run interface/app.py
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # QA jpgs of whole-slide images exceed PIL's decompression-bomb cap

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_DIR / "base_config.yaml"
RUN_CONFIG_DIR = PROJECT_DIR / "runs" / "interface"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs"}  # keep in sync with predict.py
DEFAULT_EXPLORE_DIR = (r"C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts"
                       r"\W Cheng Section (BDT-Skin) - Histology\Raw images")
PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]")
PREVIEW_MAX_SIDE = 2000  # downscale QA jpgs before sending them to the browser

st.set_page_config(page_title="Skin segmentation", layout="wide")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_preview(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.draft("RGB", (PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))  # fast JPEG downscale on decode
        im = im.convert("RGB")
    im.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    return im


def image_folders(root: Path) -> list[Path]:
    """root and every subfolder that directly holds images, skipping predicted_* (and legacy *_predicted) output folders."""
    dirs = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
    return [d for d in dirs
            if not any(part.lower().startswith("predicted_") or "_predicted" in part.lower()
                       for part in d.relative_to(root).parts)
            and list_images(d)]


def config_editor(cfg: dict) -> dict:
    """Renders one widget per base_config value; returns the edited config."""
    edited = {}
    for section, values in cfg.items():
        if not isinstance(values, dict):
            edited[section] = values
            continue
        edited[section] = {}
        with st.expander(section, expanded=False):
            for key, val in values.items():
                if section == "predict" and key == "source":
                    edited[section][key] = val  # set by the input selection instead
                    continue
                wkey = f"cfg_{section}_{key}"
                if isinstance(val, bool):
                    edited[section][key] = st.checkbox(key, value=val, key=wkey)
                elif isinstance(val, int):
                    edited[section][key] = int(st.number_input(key, value=val, step=1, key=wkey))
                elif isinstance(val, float):
                    edited[section][key] = float(st.number_input(key, value=val, format="%.4f", key=wkey))
                elif isinstance(val, str):
                    edited[section][key] = st.text_input(key, value=val, key=wkey)
                else:  # lists / null: edit as YAML text, e.g. "[51.6, 103.2]" or "null"
                    text = st.text_input(f"{key} (YAML)", value=yaml.safe_dump(val, default_flow_style=True)
                                         .strip().removesuffix("...").strip(), key=wkey)
                    try:
                        edited[section][key] = yaml.safe_load(text) if text.strip() else None
                    except yaml.YAMLError:
                        st.error(f"{section}.{key}: invalid YAML, using original value")
                        edited[section][key] = val
    return edited


def newest_measurement(out_dirs: list[Path]):
    found = [p for d in out_dirs if d.exists() for p in d.glob("*_measurement.jpg")]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


# ----------------------------------------------------------------------------
# Process new
# ----------------------------------------------------------------------------

def process_tab():
    mode = st.radio("Input", ["Single image", "Folder (its own images only)", "Folder (recursive)"],
                    horizontal=True)
    source = st.text_input("Image path" if mode == "Single image" else "Folder path", key="source"
                           ).strip().strip('"')
    out_root = st.text_input("Output folder (optional - default is 'predicted_<name>' inside the source folder)"
                             ).strip().strip('"')
    c1, c2 = st.columns(2)
    resume = c1.checkbox("Resume (reuse already-processed images)", value=True)
    limit = int(c2.number_input("Limit images per folder (0 = all)", min_value=0, value=0, step=1))

    st.subheader("Configuration")
    base_cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {}
    cfg = config_editor(base_cfg)

    if not st.button("Run", type="primary"):
        return
    src = Path(source).resolve()
    if not source or not src.exists():
        st.error(f"Path not found: {source}")
        return
    out_base = Path(out_root).resolve() if out_root else None

    # one predict.py call per (source, out) job
    if mode == "Single image":
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            st.error(f"Not a supported image: {src}")
            return
        jobs = [(src, out_base or src.parent / f"predicted_{src.stem}", 1)]
    else:
        if not src.is_dir():
            st.error(f"Not a folder: {src}")
            return
        folders = image_folders(src) if mode == "Folder (recursive)" else ([src] if list_images(src) else [])
        jobs = []
        for d in folders:
            n = len(list_images(d))
            n = min(n, limit) if limit else n
            if out_base:
                rel = d.relative_to(src)
                out = out_base / rel / f"predicted_{d.name}"
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
    st.caption(f"{len(jobs)} folder(s), {total} image(s). Config: {cfg_path}. "
               "Interacting with the page (e.g. Stop) cancels the run.")
    st.button("Stop")
    progress = st.progress(0.0, text="Starting...")
    col_img, col_log = st.columns([3, 2])
    preview = col_img.empty()
    log_box = col_log.empty()
    log_lines: list[str] = []
    done_before = 0
    out_dirs = [out for _, out, _ in jobs]
    shown = None

    first_src = jobs[0][0]
    first_image = first_src if first_src.is_file() else (list_images(first_src) or [None])[0]
    if first_image:
        try:
            preview.image(load_preview(first_image), caption=f"Input: {first_image.name}", width="stretch")
        except OSError:
            pass

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
                        progress.progress(min((done_before + i - 1) / max(total, 1), 1.0),
                                          text=f"Folder {job_i}/{len(jobs)} - image {i}/{m.group(2)}: {job_src.name}")
                    latest = newest_measurement(out_dirs)
                    if latest and latest != shown:
                        try:
                            preview.image(load_preview(latest), caption=latest.name, width="stretch")
                            shown = latest
                        except OSError:
                            pass  # still being written; pick it up on the next line
                proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode != 0:
            st.warning(f"predict.py exited with code {proc.returncode} for {job_src}")
        done_before += n

    progress.progress(1.0, text="Done")
    latest = newest_measurement(out_dirs)
    if latest and latest != shown:
        try:
            preview.image(load_preview(latest), caption=latest.name, width="stretch")
        except OSError:
            pass
    results = [out / "results.csv" for out in out_dirs if (out / "results.csv").exists()]
    if results:
        st.subheader("Results")
        st.dataframe(pd.concat([pd.read_csv(p) for p in results], ignore_index=True))


# ----------------------------------------------------------------------------
# Explore data
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Searching for results.csv...")
def load_results(root: str) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(root).rglob("results.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001 - skip unreadable/partial csvs, keep the rest
            st.warning(f"Could not read {p}: {e}")
            continue
        df["results path"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def explore_tab():
    root = st.text_input("Results folder", value=DEFAULT_EXPLORE_DIR).strip().strip('"')
    if st.button("Refresh"):
        load_results.clear()
    if not Path(root).is_dir():
        st.error(f"Folder not found: {root}")
        return
    df = load_results(root)
    if df.empty:
        st.info("No results.csv files found.")
        return

    folders = sorted(df["folder name"].astype(str).unique()) if "folder name" in df else []
    selected = st.multiselect("Folders", folders, default=folders)
    view = df[df["folder name"].astype(str).isin(selected)] if folders else df
    st.caption(f"{len(view)} rows from {view['results path'].nunique()} results.csv file(s)")
    st.dataframe(view)
    st.download_button("Download combined CSV", view.to_csv(index=False), "combined_results.csv", "text/csv")

    numeric = [c for c in view.select_dtypes("number").columns]
    if numeric and folders:
        metric = st.selectbox("Metric", numeric)
        st.subheader(f"Mean {metric} by folder")
        st.bar_chart(view.groupby("folder name")[metric].mean())
        st.subheader("Summary")
        st.dataframe(view.groupby("folder name")[numeric].agg(["mean", "std", "count"]))

    st.subheader("Image viewer")
    if "file name" not in view or view.empty:
        return
    idx = st.selectbox("Image", view.index,
                       format_func=lambda i: f"{view.at[i, 'folder name']} / {view.at[i, 'file name']}")
    out_dir = Path(view.at[idx, "results path"]).parent
    name = str(view.at[idx, "file name"])
    for stem in dict.fromkeys([name, Path(name).stem]):
        imgs = [out_dir / f"{stem}_{s}.jpg" for s in ("measurement", "steps")]
        imgs = [p for p in imgs if p.exists()]
        if imgs:
            for p in imgs:
                st.image(load_preview(p), caption=p.name, width="stretch")
            break
    else:
        st.info(f"No measurement/steps images found for {name} in {out_dir}")


tab_process, tab_explore = st.tabs(["Process new", "Explore data"])
with tab_process:
    process_tab()
with tab_explore:
    explore_tab()

#!/usr/bin/env python3
"""
annotate_boxes.py -- draw oriented (4-point) bounding boxes on large images (whole-slide exports).

    python annotate_boxes.py                 # edit follicle + roi boxes in roi_follicle_bb_data/ (default)
    python annotate_boxes.py --merge-from ROI/boxes.csv,follicles/boxes.csv   # one-time merge into combined/
    python annotate_boxes.py --follicles     # old follicle-only workflow in follicles/
    python annotate_boxes.py --roi           # old ROI-only workflow in ROI/
    python annotate_boxes.py --follicles --source follicles/test_pg_imgs --classes follicle,gland

Mouse: left-click 4 corners = one box (any orientation) | right-drag = pan | wheel = zoom at cursor
       left-drag a corner of a box of the selected class = move that corner
       top-bar buttons: class buttons (multi-class) | Rate | Images v | < Prev | Undo | Next >
Keys:  n / space / Right = next image   p / Left = previous image   1-9 = choose class
       g / Images button = image list: type to filter, wheel / Up / Down / PgUp / PgDn scroll,
       click a row or Enter (top row) to jump, Esc or click outside closes
       u / Ctrl+Z = undo (last point while drawing, otherwise last add/delete/move)
       Esc = cancel the box being drawn   x = delete box of the selected class under cursor
       f = fit image to window  + / - = zoom   s = save   q = save and quit

Section rating: every roi box is split into --sections (default 5) even sections along its long axis,
numbered from its left (or top) end. r / Rate button toggles rate mode: hover a section and press
1 (worst) - 3 (best), 0 clears; left-drag a divider to move it. Saved to sections.csv next to --csv
(image,roi_idx,section,rating,t0,t1,x1..y4); on quit rated sections are cropped upright into
section_crops/{train,val}/{1,2,3}/ (split by slide) for a classifier.

Boxes are saved in original-image pixel coordinates to --csv (image,class,x1,y1,...,x4,y4)
and as YOLO-OBB labels (class_id x1 y1 ... x4 y4, normalised) in --yolo-dir, one .txt per image.
Old 2-corner CSVs (x1,y1,x2,y2) are read and converted to 4 corners on the next save.
Re-running resumes from the CSV, starting at the first image missing a class.

On quit, combined/yolo_det is rebuilt: one dataset with classes follicle=0 and roi=1, using every
image with at least one box. Slides are downscaled to --max-side px and split train/val by
slide. --follicles / --roi rebuild follicles/yolo_det or ROI/yolo_det instead. train_det.py
trains on them.
    python annotate_boxes.py --prepare-only      # rebuild the dataset without annotating
"""

import argparse
import csv
import filecmp
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # slides exceed PIL's decompression-bomb limit; only headers are read
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # scanner TIFF tags spam warnings

EXTS = {".tif", ".tiff", ".svs", ".png", ".jpg", ".jpeg"}  # cv2 reads an .svs's full-resolution first page
COLORS = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255),
          (255, 255, 0), (128, 0, 255), (0, 0, 255), (255, 0, 0)]
WIN = "annotate_boxes"
HUD = 44
FONT = cv2.FONT_HERSHEY_SIMPLEX
BUTTONS = [("Rate", "rate"), ("Images v", "menu"), ("< Prev", "prev"), ("Undo", "undo"), ("Next >", "next")]
KEY_LEFT, KEY_RIGHT, CTRL_Z = 2424832, 2555904, 26  # cv2.waitKeyEx codes on Windows
KEY_UP, KEY_DOWN, KEY_PGUP, KEY_PGDN = 2490368, 2621440, 2162688, 2228224
MENU_W, ROW = 520, 22  # image dropdown width and row height (display px)
GRAB = 10  # display px within which a click grabs a box corner
ROI_CLASS = "roi"
RATING_COLORS = {1: (0, 0, 255), 2: (0, 220, 255), 3: (0, 200, 0)}  # BGR: red, yellow, green
MIN_GAP = 0.02  # smallest section length as a fraction of the ROI length


CSV_HEADER = ["image", "class", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
SECTIONS_HEADER = ["image", "roi_idx", "section", "rating", "t0", "t1", *CSV_HEADER[2:]]


def order_quad(pts):
    """Sort 4 points by angle around their centroid so the polygon never self-intersects."""
    cx, cy = sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def load_csv(path: Path):
    """{image: [[class, x1, y1, x2, y2, x3, y3, x4, y4], ...]}; 2-corner rows become upright quads."""
    boxes = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("x4"):
                    q = [int(r[k]) for k in CSV_HEADER[2:]]
                else:
                    x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
                    q = [x1, y1, x2, y1, x2, y2, x1, y2]
                boxes.setdefault(r["image"], []).append([r["class"], *q])
    return boxes


def class_list(names: str, boxes):
    """Class order fixes YOLO IDs: --classes order first, then any other CSV classes sorted."""
    classes = names.split(",")
    return classes + sorted({b[0] for bs in boxes.values() for b in bs} - set(classes))


def load_excluded(path):
    if not path or not Path(path).exists():
        return set()
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def image_size(path: Path):
    with Image.open(path) as im:
        return im.size


def write_annotations(csv_path: Path, yolo_dir: Path, boxes, classes, size_of):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for name in sorted(boxes):
            for b in boxes[name]:
                w.writerow([name, *b])
    # rewrite every label file so none are left in the old (cx cy w h) format
    yolo_dir.mkdir(parents=True, exist_ok=True)
    for name, bs in boxes.items():
        w, h = size_of(name)
        lines = [obb_line(b, classes.index(b[0]), w, h) for b in bs]
        (yolo_dir / f"{Path(name).stem}.txt").write_text("".join(lines))
    (yolo_dir / "classes.txt").write_text("\n".join(classes) + "\n")


def merge_annotations(a):
    """Merge CSVs into a new --csv; images lacking any --classes class go to the exclude list."""
    src, out_csv, yolo_dir = Path(a.source), Path(a.csv), Path(a.yolo_dir)
    exclude = Path(a.exclude or out_csv.with_name("excluded.txt"))
    for p in (out_csv, exclude):
        if p.exists():
            raise SystemExit(f"{p} already exists; refusing to overwrite")
    if yolo_dir.exists() and any(yolo_dir.iterdir()):
        raise SystemExit(f"{yolo_dir} is not empty; refusing to overwrite")
    classes = a.classes.split(",")
    merged = defaultdict(list)
    for path in map(Path, a.merge_from.split(",")):
        if not path.exists():
            raise SystemExit(f"{path} not found")
        for name, bs in load_csv(path).items():
            img = src / name
            if not img.exists():
                raise SystemExit(f"{name} (from {path}) is not in {src}")
            # the CSV's own image copy must be the same file, or its coordinates may not apply
            copy = path.parent / "test_pg_imgs" / name
            if copy.exists() and copy.resolve() != img.resolve() and not filecmp.cmp(copy, img, shallow=False):
                raise SystemExit(f"{copy} differs from {img}")
            unknown = {b[0] for b in bs} - set(classes)
            if unknown:
                raise SystemExit(f"{path}: classes {sorted(unknown)} not in --classes {a.classes}")
            merged[name].extend(bs)
        print(f"read {path}", flush=True)

    keep = {n: bs for n, bs in merged.items() if {b[0] for b in bs} >= set(classes)}
    excluded = sorted(set(merged) - set(keep))
    write_annotations(out_csv, yolo_dir, keep, classes, lambda n: image_size(src / n))
    exclude.write_text("".join(f"{n}\n" for n in excluded))
    for c in classes:
        print(f"{c}: {sum(b[0] == c for bs in keep.values() for b in bs)} boxes")
    print(f"{len(keep)} images with all classes -> {out_csv}")
    print(f"{len(excluded)} images excluded -> {exclude}: {excluded}")


def quad_pts(box):
    return np.array(box[1:9], np.float32).reshape(4, 2)


def obb_line(box, cls_id, w, h):
    return f"{cls_id} " + " ".join(
        f"{min(max(px / w, 0), 1):.6f} {min(max(py / h, 0), 1):.6f}" for px, py in quad_pts(box)) + "\n"


def roi_axis(box):
    """(a0, a1, b0, b1): the two long edges, both running from the left (or top, if vertical) end;
    edge b lies clockwise of edge a, so crops are never mirrored."""
    q = quad_pts(box)
    e = [np.linalg.norm(q[(k + 1) % 4] - q[k]) for k in range(4)]
    a0, a1, b0, b1 = (q[0], q[1], q[3], q[2]) if e[0] + e[2] >= e[1] + e[3] else (q[1], q[2], q[0], q[3])
    axis = (a1 + b1) - (a0 + b0)
    if axis[0 if abs(axis[0]) >= abs(axis[1]) else 1] < 0:
        a0, a1, b0, b1 = a1, a0, b1, b0
    d, v = a1 - a0, b0 - a0
    if d[0] * v[1] - d[1] * v[0] < 0:
        a0, a1, b0, b1 = b0, b1, a0, a1
    return a0, a1, b0, b1


def divider_seg(box, t):
    a0, a1, b0, b1 = roi_axis(box)
    return a0 + (a1 - a0) * t, b0 + (b1 - b0) * t


def section_quad(box, ta, tb):
    """Section corners ordered top-left, top-right, bottom-right, bottom-left in the ROI's frame."""
    (p0, q0), (p1, q1) = divider_seg(box, ta), divider_seg(box, tb)
    return np.array([p0, p1, q1, q0], np.float32)


def axis_t(box, ix, iy):
    """Position of an image point along the ROI's long axis (0 = start end, 1 = far end)."""
    a0, a1, b0, b1 = roi_axis(box)
    m0, d = (a0 + b0) / 2, (a1 + b1 - a0 - b0) / 2
    return float(np.dot(np.array([ix, iy], np.float32) - m0, d) / max(float(np.dot(d, d)), 1e-9))


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def even_sections(n):
    return {"cuts": [k / n for k in range(n + 1)], "ratings": [None] * n}


def load_sections(path: Path):
    """{image: {roi_idx: {"cuts": [0, ..., 1], "ratings": [int | None, ...]}}}"""
    rows = defaultdict(list)
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["image"], int(r["roi_idx"]))].append(r)
    out = {}
    for (name, ri), rs in rows.items():
        rs.sort(key=lambda r: int(r["section"]))
        out.setdefault(name, {})[ri] = {
            "cuts": [float(r["t0"]) for r in rs] + [float(rs[-1]["t1"])],
            "ratings": [int(r["rating"]) if r["rating"] else None for r in rs]}
    return out


def write_sections(path: Path, boxes, state_of):
    """One row per section; roi_idx is the ROI's order among the image's roi boxes in --csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SECTIONS_HEADER)
        for name in sorted(boxes):
            rois = [b for b in boxes[name] if b[0] == ROI_CLASS]
            for ri, box in enumerate(rois):
                st = state_of(box)
                cuts = st["cuts"]
                for k, rating in enumerate(st["ratings"]):
                    quad = section_quad(box, cuts[k], cuts[k + 1])
                    w.writerow([name, ri, k, rating or "", f"{cuts[k]:.4f}", f"{cuts[k + 1]:.4f}",
                                *(round(float(v)) for v in quad.ravel())])


def export_section_crops(a):
    """Rebuild section_crops/{train,val}/{rating}/ (YOLO classify layout) from sections.csv."""
    path = Path(a.csv).with_name("sections.csv")
    if not path.exists():
        return
    excluded = load_excluded(a.exclude)
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["rating"] and r["image"] not in excluded]
    out = path.with_name("section_crops")
    for split in ("train", "val"):
        shutil.rmtree(out / split, ignore_errors=True)
    if not rows:
        print("no rated sections yet; section crops not built")
        return
    for split in ("train", "val"):
        for rating in RATING_COLORS:
            (out / split / str(rating)).mkdir(parents=True)

    groups = sorted({slide_key(Path(r["image"]).stem) for r in rows})
    random.Random(a.seed).shuffle(groups)
    val = set(groups[:max(1, round(len(groups) * a.val_frac))]) if len(groups) > 1 else set()
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["image"]].append(r)

    print(f"\nexporting section crops to {out} ...", flush=True)
    stats = defaultdict(int)
    for name, rs in sorted(by_image.items()):
        img = cv2.imread(str(Path(a.source) / name), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  could not read {name}; skipped {len(rs)} sections", flush=True)
            continue
        split = "val" if slide_key(Path(name).stem) in val else "train"
        for r in rs:
            q = np.array([float(r[k]) for k in CSV_HEADER[2:]], np.float32).reshape(4, 2)
            w = max(1, round((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2))
            h = max(1, round((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2))
            m = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
            crop = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR)
            dst = out / split / r["rating"] / f"{Path(name).stem}_roi{r['roi_idx']}_s{r['section']}.jpg"
            cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats[split, r["rating"]] += 1
        print(f"  {split:5s} {name}: {len(rs)} rated sections", flush=True)
    for split in ("train", "val"):
        print(f"{split}: " + "  ".join(f"rating {k}: {stats[split, str(k)]}" for k in RATING_COLORS))
    print(f"val slides: {sorted(val)}")


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def slide_key(stem: str) -> str:
    """'mde112f.svs_1' and 'mde112f.svs' are the same slide and must share a split."""
    return re.sub(r"(\.svs)?(_\d+)?$", "", stem)


def load_scaled(path: Path, max_side: int):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")
    h, w = img.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1:
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    return img, w, h, f


def prepare_dataset(a):
    src = Path(a.source)
    boxes = {k: v for k, v in load_csv(Path(a.csv)).items() if v}
    classes = class_list(a.classes, boxes)
    if not boxes:
        print("no boxes yet; dataset not built")
        return
    if a.require_all and a.include_empty:
        raise SystemExit("--include-empty is only supported with --follicles or --roi")

    excluded = load_excluded(a.exclude)
    images = [p for p in list_images(src) if p.name not in excluded]
    if a.require_all:
        files = [p for p in images if p.name in boxes]
    else:
        negatives = set()
        if a.include_empty:
            negatives = {p.stem for p in Path(a.yolo_dir).glob("*.txt")
                         if p.name != "classes.txt" and not p.read_text().strip()}
        files = [p for p in images if p.name in boxes or p.stem in negatives]
    skipped = [p.name for p in images if p not in files]
    if not files:
        print("no complete images yet; dataset not built")
        return

    pos_groups = sorted({slide_key(p.stem) for p in files if p.name in boxes})
    random.Random(a.seed).shuffle(pos_groups)
    val = set(pos_groups[:max(1, round(len(pos_groups) * a.val_frac))])

    out = Path(a.dataset)
    # combined datasets share combined/cache_* across rebuilds
    cache = (out.parent if a.require_all else out) / f"cache_{a.max_side}"
    write_dataset(out, files, val, boxes, classes, cache, a.max_side)
    print(f"val slides: {sorted(val)}")
    if skipped:
        reason = "no boxes" if a.require_all else "no boxes" + ("" if a.include_empty else " / empty labels excluded")
        print(f"skipped ({reason}): {skipped}")
    if excluded:
        print(f"excluded by {a.exclude}: {len(excluded)} images")


def write_dataset(out: Path, files, val, boxes, classes, cache: Path, max_side: int):
    if out.exists() and any(out.iterdir()) and not (out / "data.yaml").exists():
        raise SystemExit(f"{out} exists and is not a dataset made by this script; choose another --dataset")
    for d in ("images", "labels"):
        shutil.rmtree(out / d, ignore_errors=True)
    for d in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / d).mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"\nbuilding dataset in {out} ...", flush=True)
    stats = defaultdict(lambda: [0, 0])
    for p in files:
        split = "val" if slide_key(p.stem) in val else "train"
        small = cache / f"{p.stem}.jpg"
        if not small.exists() or small.stat().st_mtime < p.stat().st_mtime:
            img, _, _, _ = load_scaled(p, max_side)
            cv2.imwrite(str(small), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy2(small, out / "images" / split / small.name)
        w, h = image_size(p)
        lines = [obb_line(b, classes.index(b[0]), w, h) for b in boxes.get(p.name, [])]
        (out / "labels" / split / f"{p.stem}.txt").write_text("".join(lines))
        stats[split][0] += 1
        stats[split][1] += len(lines)
        print(f"  {split:5s} {p.name}: {len(lines)} boxes", flush=True)

    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "images/train", "val": "images/val",
        "names": dict(enumerate(classes)), "max_side": max_side}, sort_keys=False))
    for s in ("train", "val"):
        print(f"{s}: {stats[s][0]} images, {stats[s][1]} boxes")
    print(f"dataset ready: {out / 'data.yaml'}  ->  python train_det.py train --data {out.as_posix()}")


class Annotator:
    def __init__(self, args):
        excluded = load_excluded(args.exclude)
        self.files = [p for p in list_images(Path(args.source)) if p.name not in excluded]
        if not self.files:
            raise SystemExit(f"no images found in {args.source}")
        self.csv_path, self.yolo_dir = Path(args.csv), Path(args.yolo_dir)
        self.W, self.H = (int(v) for v in args.win.lower().split("x"))
        self.boxes = load_csv(self.csv_path)
        self.classes = class_list(args.classes, self.boxes)
        self.history = {}
        self.sections_path = self.csv_path.with_name("sections.csv")
        self.n_sections = args.sections
        # id(box) -> (box, state); holding the box keeps its id from being reused
        self.sec = {}
        stored = load_sections(self.sections_path)
        for name, bs in self.boxes.items():
            for ri, box in enumerate(b for b in bs if b[0] == ROI_CLASS):
                if st := stored.get(name, {}).get(ri):
                    self.sec[id(box)] = (box, st)
        self.rate = False
        self.cut_drag = None
        self.menu = None
        buttons = [b for b in BUTTONS if b[1] != "rate" or ROI_CLASS in self.classes]
        bw, gap = 90, 8
        self.buttons = [(label, action, (self.W - (len(buttons) - k) * (bw + gap), 6, bw, HUD - 12))
                        for k, (label, action) in enumerate(buttons)]
        if len(self.classes) > 1:
            x = self.buttons[0][2][0] - gap
            for k in reversed(range(min(len(self.classes), 9))):
                label = f"{k + 1} {self.classes[k]}"
                w = cv2.getTextSize(label, FONT, 0.55, 1)[0][0] + 16
                x -= w + gap
                self.buttons.insert(0, (label, k, (x, 6, w, HUD - 12)))
        self.text_w = self.buttons[0][2][0] - 16
        self.dims = {}
        self.cls = 0
        self.points = []
        self.pan = None
        self.drag = None
        self.mouse = (0, 0)
        self.dirty = True
        need = set(self.classes)
        self.i = next((k for k, p in enumerate(self.files)
                       if not need <= {b[0] for b in self.boxes.get(p.name, [])}), 0)

    # ---------------------------------------------------------------- view
    def to_img(self, x, y):
        return self.ox + x / self.s, self.oy + y / self.s

    def to_disp(self, ix, iy):
        return int(round((ix - self.ox) * self.s)), int(round((iy - self.oy) * self.s))

    def load(self):
        p = self.files[self.i]
        print(f"loading [{self.i + 1}/{len(self.files)}] {p.name} ...", flush=True)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read {p}")
        self.ih, self.iw = img.shape[:2]
        self.dims[p.name] = (self.iw, self.ih)
        self.points = []
        self.drag = None
        self.cut_drag = None
        # pyramid so zoomed-out views don't resize the full-resolution image every frame
        self.pyr = [img]
        while max(self.pyr[-1].shape[:2]) > 1024:
            self.pyr.append(cv2.pyrDown(self.pyr[-1]))
        self.fit()

    def fit(self):
        self.s = self.fit_s = min(self.W / self.iw, (self.H - HUD) / self.ih)
        self.ox = -((self.W - self.iw * self.s) / 2) / self.s
        self.oy = -(HUD + (self.H - HUD - self.ih * self.s) / 2) / self.s
        self.dirty = True

    def zoom(self, factor, x, y):
        ix, iy = self.to_img(x, y)
        self.s = float(np.clip(self.s * factor, self.fit_s / 2, 8.0))
        self.ox, self.oy = ix - x / self.s, iy - y / self.s
        self.dirty = True

    def render(self):
        canvas = np.full((self.H, self.W, 3), 40, np.uint8)
        lvl = 0
        while lvl + 1 < len(self.pyr) and self.s * 2 ** (lvl + 1) <= 1:
            lvl += 1
        f, src = 2 ** lvl, self.pyr[lvl]
        x0, y0 = max(0.0, self.ox), max(0.0, self.oy)
        x1, y1 = min(self.iw, self.ox + self.W / self.s), min(self.ih, self.oy + self.H / self.s)
        if x1 > x0 and y1 > y0:
            crop = src[int(y0 / f):math.ceil(y1 / f), int(x0 / f):math.ceil(x1 / f)]
            dx0, dy0 = self.to_disp(x0, y0)
            dx1, dy1 = self.to_disp(x1, y1)
            dx1, dy1 = min(dx1, self.W), min(dy1, self.H)
            if dx1 > dx0 and dy1 > dy0 and crop.size:
                interp = cv2.INTER_AREA if dx1 - dx0 < crop.shape[1] else cv2.INTER_LINEAR
                canvas[dy0:dy1, dx0:dx1] = cv2.resize(crop, (dx1 - dx0, dy1 - dy0), interpolation=interp)

        rois = [b for b in self.cur() if b[0] == ROI_CLASS]
        if self.rate:
            overlay = canvas.copy()
            for box in rois:
                st = self.state(box)
                for k, rating in enumerate(st["ratings"]):
                    if rating:
                        quad = section_quad(box, st["cuts"][k], st["cuts"][k + 1])
                        cv2.fillPoly(overlay, [self.disp_poly(quad)], RATING_COLORS[rating])
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        for box in self.cur():
            col = COLORS[self.classes.index(box[0]) % len(COLORS)]
            pts = np.array([self.to_disp(px, py) for px, py in quad_pts(box)], np.int32)
            active = box[0] == self.classes[self.cls]
            cv2.polylines(canvas, [pts], True, col, 3 if active else 1, cv2.LINE_AA)
            if active and not self.points:
                for k, p in enumerate(pts):
                    held = self.drag and self.drag[0] is box and self.drag[1] == k
                    cv2.circle(canvas, tuple(int(v) for v in p), 6 if held else 4, (255, 255, 255) if held else col, -1)
            if len(self.classes) > 1:
                top = pts[pts[:, 1].argmin()]
                cv2.putText(canvas, box[0], (int(top[0]), int(top[1]) - 4), FONT, 0.5, col, 1, cv2.LINE_AA)
        for box in rois:
            st = self.state(box)
            col = COLORS[self.classes.index(ROI_CLASS) % len(COLORS)]
            for t in st["cuts"][1:-1]:
                p, q = (self.to_disp(*pt) for pt in divider_seg(box, t))
                cv2.line(canvas, p, q, col, 2 if self.rate else 1, cv2.LINE_AA)
            for k, rating in enumerate(st["ratings"]):
                if rating or self.rate:
                    c = self.disp_poly(section_quad(box, st["cuts"][k], st["cuts"][k + 1])).mean(axis=0)
                    label = str(rating) if rating else "?"
                    (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
                    org = (int(c[0]) - tw // 2, int(c[1]) + th // 2)
                    cv2.putText(canvas, label, org, FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(canvas, label, org, FONT, 0.7, RATING_COLORS.get(rating, (255, 255, 255)), 2, cv2.LINE_AA)
        if self.rate and (hit := self.section_under()):
            box, k = hit
            quad = section_quad(box, self.state(box)["cuts"][k], self.state(box)["cuts"][k + 1])
            cv2.polylines(canvas, [self.disp_poly(quad)], True, (255, 255, 255), 2, cv2.LINE_AA)
        if self.points:
            pts = [self.to_disp(px, py) for px, py in self.points] + [self.mouse]
            if len(self.points) == 3:
                pts.append(pts[0])
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
            for p in pts[:len(self.points)]:
                cv2.circle(canvas, p, 4, (0, 0, 255), -1)

        ix, iy = self.to_img(*self.mouse)
        cv2.rectangle(canvas, (0, 0), (self.W, HUD), (0, 0, 0), -1)
        drawing = f"   drawing: point {len(self.points) + 1}/4 (Esc cancels)" if self.points else ""
        counts = "  ".join(f"{c}: {sum(b[0] == c for b in self.cur())}" for c in self.classes)
        if ROI_CLASS in self.classes:
            ratings = [r for b in rois for r in self.state(b)["ratings"]]
            counts += f"  rated {sum(r is not None for r in ratings)}/{len(ratings)}"
        line1 = f"[{self.i + 1}/{len(self.files)}] {self.files[self.i].name}   {counts}{drawing}"
        line2 = (f"class {self.cls + 1}: {self.classes[self.cls]}   zoom {self.s:.3f}   x={int(ix)} y={int(iy)}   "
                 "| click 4 corners, drag corner to edit, R-drag pan, wheel zoom, x delete, f fit, r rate")
        if self.rate:
            line2 = ("RATE MODE (r exits) | hover a section: 1 worst - 3 best, 0 clear | drag a divider to move it "
                     "| R-drag pan, wheel zoom")
        for text, y, scale, col in ((line1, 17, 0.5, (255, 255, 255)), (line2, 36, 0.45, (180, 180, 180))):
            while len(text) > 4 and cv2.getTextSize(text, FONT, scale, 1)[0][0] > self.text_w:
                text = text[:-4] + "..."
            cv2.putText(canvas, text, (8, y), FONT, scale, col, 1, cv2.LINE_AA)
        for label, action, (bx, by, bw, bh) in self.buttons:
            if isinstance(action, int):
                col = COLORS[action % len(COLORS)]
                fill = tuple(c // 2 for c in col) if action == self.cls else (50, 50, 50)
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), fill, -1)
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, 2 if action == self.cls else 1)
            else:
                active = (action == "rate" and self.rate) or (action == "menu" and self.menu is not None)
                fill = (60, 140, 60) if action == "next" else (0, 110, 200) if active else (90, 90, 90)
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), fill, -1)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.putText(canvas, label, (bx + (bw - tw) // 2, by + (bh + th) // 2), FONT, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        if self.menu is not None:
            self.render_menu(canvas)
        return canvas

    def render_menu(self, canvas):
        mx, my, rows = self.menu_rect()
        items = self.menu_items()
        top = self.menu["scroll"]
        shown = items[top:top + rows]
        hover = self.menu_row_under(*self.mouse)
        y1 = my + (len(shown) + 1) * ROW + 4
        cv2.rectangle(canvas, (mx, my), (mx + MENU_W, y1), (30, 30, 30), -1)
        cv2.rectangle(canvas, (mx, my), (mx + MENU_W, y1), (200, 200, 200), 1)
        head = f"filter: {self.menu['filter']}_   {len(items)}/{len(self.files)}   (type, Enter, Esc)"
        cv2.putText(canvas, head, (mx + 8, my + 16), FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        for r, k in enumerate(shown):
            ry = my + (r + 1) * ROW
            if k == hover or k == self.i:
                fill = (90, 90, 90) if k == hover else (80, 50, 0)
                cv2.rectangle(canvas, (mx + 1, ry + 2), (mx + MENU_W - 8, ry + ROW + 1), fill, -1)
            name = self.files[k].name
            bs = self.boxes.get(name, [])
            status = "  ".join(f"{c[:3]} {sum(b[0] == c for b in bs)}" for c in self.classes)
            if ROI_CLASS in self.classes:
                ratings = [x for b in bs if b[0] == ROI_CLASS for x in self.state(b)["ratings"]]
                status += f"  rated {sum(x is not None for x in ratings)}/{len(ratings)}"
            sw = cv2.getTextSize(status, FONT, 0.42, 1)[0][0]
            label = f"{k + 1:3d}  {name}"
            while len(label) > 4 and cv2.getTextSize(label, FONT, 0.45, 1)[0][0] > MENU_W - sw - 36:
                label = label[:-4] + "..."
            done = set(self.classes) <= {b[0] for b in bs}
            col = (120, 230, 120) if done else (220, 220, 220)
            cv2.putText(canvas, label, (mx + 8, ry + 16), FONT, 0.45, col, 1, cv2.LINE_AA)
            cv2.putText(canvas, status, (mx + MENU_W - sw - 14, ry + 16), FONT, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        if len(items) > rows:
            h = y1 - my - ROW
            b0 = my + ROW + h * top // len(items)
            b1 = my + ROW + h * (top + len(shown)) // len(items)
            cv2.rectangle(canvas, (mx + MENU_W - 6, b0), (mx + MENU_W - 2, b1), (200, 200, 200), -1)

    # ---------------------------------------------------------------- image menu
    def menu_items(self):
        f = self.menu["filter"].lower()
        return [k for k, p in enumerate(self.files) if f in p.name.lower()]

    def menu_rect(self):
        bx = next(r[0] for _, action, r in self.buttons if action == "menu")
        return min(bx, self.W - MENU_W - 4), HUD, max(1, (self.H - HUD - 8) // ROW - 1)

    def open_menu(self):
        self.points = []
        self.pan = None
        if self.drag:
            self.end_drag()
        if self.cut_drag:
            self.end_cut_drag()
        _, _, rows = self.menu_rect()
        self.menu = {"filter": "", "scroll": self.i - rows // 2}
        self.scroll_menu(0)

    def scroll_menu(self, d):
        _, _, rows = self.menu_rect()
        top = max(0, len(self.menu_items()) - rows)
        self.menu["scroll"] = int(np.clip(self.menu["scroll"] + d, 0, top))

    def menu_row_under(self, x, y):
        mx, my, rows = self.menu_rect()
        r = (y - my) // ROW - 1  # row 0 is the filter line
        items = self.menu_items()
        if mx <= x <= mx + MENU_W and 0 <= r < rows and self.menu["scroll"] + r < len(items):
            return items[self.menu["scroll"] + r]
        return None

    def menu_key(self, k):
        _, _, rows = self.menu_rect()
        if k == 27:
            self.menu = None
        elif k == 13:
            items = self.menu_items()[self.menu["scroll"]:]
            self.menu = None
            if items:
                self.goto(items[0])
        elif k in (KEY_UP, KEY_DOWN, KEY_PGUP, KEY_PGDN):
            self.scroll_menu({KEY_UP: -1, KEY_DOWN: 1, KEY_PGUP: -rows, KEY_PGDN: rows}[k])
        elif k == 8:
            self.menu["filter"] = self.menu["filter"][:-1]
            self.scroll_menu(-len(self.files))
        elif 32 <= k <= 126:
            self.menu["filter"] += chr(k)
            self.scroll_menu(-len(self.files))

    def menu_mouse(self, ev, x, y, flags):
        if ev == cv2.EVENT_LBUTTONDOWN:
            j = self.menu_row_under(x, y)
            mx, my, _ = self.menu_rect()
            on_filter_line = mx <= x <= mx + MENU_W and my <= y < my + ROW
            if j is not None:
                self.menu = None
                self.goto(j)
            elif not on_filter_line:
                self.menu = None
        elif ev == cv2.EVENT_MOUSEWHEEL:
            self.scroll_menu(-3 if flags > 0 else 3)

    # ---------------------------------------------------------------- boxes
    def cur(self):
        return self.boxes.setdefault(self.files[self.i].name, [])

    def clamp(self, ix, iy):
        return min(max(ix, 0), self.iw), min(max(iy, 0), self.ih)

    def hist(self):
        return self.history.setdefault(self.files[self.i].name, [])

    def add_box(self, box):
        self.cur().append(box)
        self.hist().append(("add", box))

    def add_point(self, ix, iy):
        self.points.append((ix, iy))
        if len(self.points) == 4:
            quad = order_quad(self.points)
            self.points = []
            if cv2.contourArea(np.array(quad, np.float32)) * self.s ** 2 >= 9:
                self.add_box([self.classes[self.cls], *(round(v) for p in quad for v in p)])

    def corner_under(self, x, y):
        """(box, corner index) of the nearest selected-class corner within GRAB display px, else None."""
        best, best_d = None, GRAB
        for box in self.cur():
            if box[0] != self.classes[self.cls]:
                continue
            for k, (px, py) in enumerate(quad_pts(box)):
                dx, dy = self.to_disp(px, py)
                d = math.hypot(dx - x, dy - y)
                if d <= best_d:
                    best, best_d = (box, k), d
        return best

    def end_drag(self):
        box, _, old = self.drag
        self.drag = None
        quad = order_quad([tuple(p) for p in quad_pts(box)])
        if cv2.contourArea(np.array(quad, np.float32)) * self.s ** 2 < 9:
            box[1:9] = old
            return
        box[1:9] = [round(v) for p in quad for v in p]
        if box[1:9] != old:
            self.hist().append(("move", box, old))

    def delete_under_cursor(self):
        ix, iy = self.to_img(*self.mouse)
        hits = [b for b in self.cur() if b[0] == self.classes[self.cls]
                and cv2.pointPolygonTest(quad_pts(b), (ix, iy), False) >= 0]
        if hits:
            box = min(hits, key=lambda b: cv2.contourArea(quad_pts(b)))
            idx = next(k for k, b in enumerate(self.cur()) if b is box)
            self.hist().append(("del", self.cur().pop(idx), idx))

    # ---------------------------------------------------------------- sections
    def state(self, box):
        if id(box) not in self.sec:
            self.sec[id(box)] = (box, even_sections(self.n_sections))
        return self.sec[id(box)][1]

    def disp_poly(self, quad):
        return np.array([self.to_disp(px, py) for px, py in quad], np.int32)

    def section_under(self):
        ix, iy = self.to_img(*self.mouse)
        for box in reversed([b for b in self.cur() if b[0] == ROI_CLASS]):
            cuts = self.state(box)["cuts"]
            for k in range(len(cuts) - 1):
                if cv2.pointPolygonTest(section_quad(box, cuts[k], cuts[k + 1]), (float(ix), float(iy)), False) >= 0:
                    return box, k
        return None

    def divider_under(self, x, y):
        best, best_d = None, GRAB
        p = np.array([x, y], np.float32)
        for box in self.cur():
            if box[0] != ROI_CLASS:
                continue
            cuts = self.state(box)["cuts"]
            for k in range(1, len(cuts) - 1):
                a, b = (np.array(self.to_disp(*pt), np.float32) for pt in divider_seg(box, cuts[k]))
                d = seg_dist(p, a, b)
                if d <= best_d:
                    best, best_d = (box, k), d
        return best

    def rate_section(self, rating):
        if hit := self.section_under():
            box, k = hit
            ratings = self.state(box)["ratings"]
            if ratings[k] != rating:
                self.hist().append(("rate", box, k, ratings[k]))
                ratings[k] = rating

    def end_cut_drag(self):
        box, k, old = self.cut_drag
        self.cut_drag = None
        if self.state(box)["cuts"][k] != old:
            self.hist().append(("cut", box, k, old))

    def undo(self):
        if self.points:
            self.points.pop()
        elif self.hist():
            op, box, *rest = self.hist().pop()
            if op == "add":
                self.cur().remove(box)
            elif op == "del":
                self.cur().insert(rest[0], box)
            elif op == "move":
                box[1:9] = rest[0]
            elif op == "cut":
                self.state(box)["cuts"][rest[0]] = rest[1]
            else:
                self.state(box)["ratings"][rest[0]] = rest[1]
        elif self.cur() and not self.rate:
            # boxes loaded from a previous session have no history; fall back to removing the last one
            self.cur().pop()

    def image_size(self, name):
        if name not in self.dims:
            self.dims[name] = image_size(Path(self.files[0].parent) / name)
        return self.dims[name]

    def save(self):
        write_annotations(self.csv_path, self.yolo_dir, self.boxes, self.classes, self.image_size)
        if ROI_CLASS in self.classes:
            write_sections(self.sections_path, self.boxes, self.state)
        print(f"saved {len(self.cur())} boxes for {self.files[self.i].name}", flush=True)

    def go(self, step):
        j = min(max(self.i + step, 0), len(self.files) - 1)
        if j == self.i:
            print("already at the " + ("last" if step > 0 else "first") + " image", flush=True)
            return
        self.goto(j)

    def goto(self, j):
        if j == self.i:
            return
        self.save()
        self.i = j
        canvas = self.render()
        cv2.putText(canvas, f"Loading {self.files[j].name} ...", (20, HUD + 40), FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)
        cv2.waitKey(1)
        self.load()

    def act(self, action):
        if isinstance(action, int):
            self.cls = action
        elif action == "next":
            self.go(1)
        elif action == "prev":
            self.go(-1)
        elif action == "undo":
            self.undo()
        elif action == "rate":
            self.rate = not self.rate and ROI_CLASS in self.classes
            self.points = []
        elif action == "menu":
            self.open_menu()
        self.dirty = True

    # ---------------------------------------------------------------- events
    def on_mouse(self, ev, x, y, flags, _):
        self.mouse = (x, y)
        self.dirty = True
        if self.menu is not None:
            self.menu_mouse(ev, x, y, flags)
            return
        ix, iy = self.clamp(*self.to_img(x, y))
        if ev == cv2.EVENT_LBUTTONDOWN:
            if y < HUD:
                for _, action, (bx, by, bw, bh) in self.buttons:
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.act(action)
            elif self.rate:
                if hit := self.divider_under(x, y):
                    box, k = hit
                    self.cut_drag = (box, k, self.state(box)["cuts"][k])
            elif not self.points and (hit := self.corner_under(x, y)):
                self.drag = (*hit, hit[0][1:9])
            else:
                self.add_point(ix, iy)
        elif ev == cv2.EVENT_LBUTTONUP:
            if self.drag:
                self.end_drag()
            if self.cut_drag:
                self.end_cut_drag()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self.pan = (x, y, self.ox, self.oy)
        elif ev == cv2.EVENT_MOUSEMOVE:
            if self.drag:
                box, k, _ = self.drag
                box[1 + 2 * k], box[2 + 2 * k] = round(ix), round(iy)
            if self.cut_drag:
                box, k, _ = self.cut_drag
                cuts = self.state(box)["cuts"]
                cuts[k] = float(np.clip(axis_t(box, ix, iy), cuts[k - 1] + MIN_GAP, cuts[k + 1] - MIN_GAP))
            if self.pan:
                px, py, ox, oy = self.pan
                self.ox, self.oy = ox - (x - px) / self.s, oy - (y - py) / self.s
        elif ev == cv2.EVENT_RBUTTONUP:
            self.pan = None
        elif ev == cv2.EVENT_MOUSEWHEEL:
            self.zoom(1.25 if flags > 0 else 0.8, x, y)
        self.dirty = True

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load()
        while True:
            if self.dirty:
                cv2.imshow(WIN, self.render())
                self.dirty = False
            k = cv2.waitKeyEx(15)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == -1:
                continue
            if self.menu is not None:
                self.menu_key(k)
            elif k == 27 and self.points:
                self.points = []
            elif k in (ord("q"), 27):
                break
            elif k == ord("g"):
                self.act("menu")
            elif k in (ord("n"), 32, KEY_RIGHT):
                self.act("next")
            elif k in (ord("p"), KEY_LEFT):
                self.act("prev")
            elif k == ord("r"):
                self.act("rate")
            elif self.rate and ord("0") <= k <= ord("9"):
                if k <= ord("3"):
                    self.rate_section(k - ord("0") or None)
            elif ord("1") <= k <= ord("9") and k - ord("1") < len(self.classes):
                self.cls = k - ord("1")
            elif k in (ord("u"), CTRL_Z):
                self.act("undo")
            elif k == ord("x"):
                self.delete_under_cursor()
            elif k == ord("f"):
                self.fit()
            elif k in (ord("+"), ord("=")):
                self.zoom(1.25, self.W / 2, self.H / 2)
            elif k == ord("-"):
                self.zoom(0.8, self.W / 2, self.H / 2)
            elif k == ord("s"):
                self.save()
            self.dirty = True
        self.save()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    preset = ap.add_mutually_exclusive_group()
    preset.add_argument("--combined", action="store_false",
                        help="default: edit follicle + roi boxes in combined/; builds one two-class dataset")
    preset.add_argument("--follicles", action="store_true", help="follicle-only paths in follicles/ (old workflow)")
    preset.add_argument("--roi", action="store_true", help="ROI-only paths in ROI/ (old workflow)")
    ap.add_argument("--merge-from", help="comma-separated CSVs to merge into a new --csv, then exit")
    ap.add_argument("--exclude", help="file listing image names to skip (one per line)")
    ap.add_argument("--source", help="folder of images to annotate")
    ap.add_argument("--classes", help="comma-separated class names (keys 1-9)")
    ap.add_argument("--csv", help="box list in original pixel coordinates")
    ap.add_argument("--yolo-dir", help="YOLO .txt labels, one per image")
    ap.add_argument("--win", default="1400x850", help="window size WxH; lower it for small screens")
    ap.add_argument("--dataset", help="training dataset rebuilt on quit")
    ap.add_argument("--max-side", type=int, default=1024, help="dataset images are downscaled to this long side")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of slides held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-empty", action="store_true",
                    help="use images whose label file is empty as negatives (only if you checked they have no follicles)")
    ap.add_argument("--prepare-only", action="store_true", help="rebuild the dataset without opening the annotator")
    ap.add_argument("--sections", type=int, default=5, help="number of even sections each new roi box is split into")
    args = ap.parse_args()
    if args.sections < 1:
        raise SystemExit("--sections must be >= 1")
    args.combined = not (args.follicles or args.roi)

    root = Path("roi_follicle_bb_data")
    args.source = args.source or str(root / "images")
    args.classes = args.classes or "follicle,roi"
    args.exclude = args.exclude or str(root / "excluded.txt")

    args.csv = args.csv or str(root / "boxes.csv")
    args.yolo_dir = args.yolo_dir or str(root / "labels")
    args.dataset = args.dataset or str(root / "yolo_det")
    args.require_all = args.combined
    if args.merge_from:
        merge_annotations(args)
        return
    if not args.prepare_only:
        Annotator(args).run()
    prepare_dataset(args)
    export_section_crops(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
add_segmentations.py -- draw SC / Epidermis / gland polygons into the existing YOLO-seg dataset
(dataset_for_finetuning).

    python add_segmentations.py                                     # annotate glands in the dataset
    python add_segmentations.py --cls SC                            # start with the SC class active
    python add_segmentations.py --add new_imgs --um-per-px 0.645    # copy new images in first (to --split)
    python add_segmentations.py --add a.png b.png --split val --start a

The active class is filled and editable; the other classes are shown as outlines. Lines of classes
other than SC / Epidermis / gland are written back unchanged. A class missing from data.yaml names
is appended on the first save that contains one. An image present in both train and val gets the
same polygons in both label files.

Mouse: left-click = add vertex | left-drag = trace freehand | click first vertex = close polygon
    right/middle-drag = pan | wheel = zoom at cursor | Pan toggle = left-drag to pan
Edit mode (Edit button / e): left-drag a vertex of any SC / Epidermis / gland polygon to move it,
    left-click an edge to insert a vertex, right-click / Delete / Backspace a vertex to remove it
    (only the active class is editable while other outlines are hidden with h)
Keys:  1 / 2 / 3 = SC / Epidermis / gland   Tab = next class
       Enter = close polygon   Esc = cancel polygon / leave Edit mode (else quit)   u / Ctrl+Z = undo
    g = toggle Pan mode   e = toggle Edit mode   x = delete active-class polygon under cursor
    h = hide/show other outlines   f = fit   + / - = zoom
       n / space / Right = next image   p / Left = previous image   s = save   q = save and quit
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # scanner TIFF tags spam warnings

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SPLITS = ("train", "val")
CLASSES = ("SC", "Epidermis", "gland")
CLASS_COLORS = {"SC": (255, 200, 0), "Epidermis": (0, 200, 255), "gland": (0, 255, 0)}
LAYER_COLORS = [(255, 200, 0), (0, 200, 255), (200, 120, 255), (0, 128, 255)]
WIN = "add_segmentations"
HUD = 44
FONT = cv2.FONT_HERSHEY_SIMPLEX
BUTTONS = [("Class", "class", 150), ("Pan", "pan", 90), ("Edit", "edit", 90), ("< Prev", "prev", 90),
           ("Undo", "undo", 90), ("Next >", "next", 90)]
KEY_LEFT, KEY_RIGHT, KEY_DEL, CTRL_Z, ENTER, TAB, BACKSPACE = 2424832, 2555904, 3014656, 26, 13, 9, 8  # cv2.waitKeyEx codes on Windows
CLOSE_PX = 10  # display px from the first vertex within which a click/release closes the polygon
TRACE_PX = 5  # display px between vertices added while left-dragging
VERTEX_PX = 8  # display px within which a click grabs a vertex in Edit mode
EDGE_PX = 6  # display px within which a click on an edge inserts a vertex in Edit mode


def read_names(data_yaml: Path):
    cfg = yaml.safe_load(data_yaml.read_text()) or {}
    names = cfg.get("names") or {}
    return cfg, {int(k): v for k, v in (enumerate(names) if isinstance(names, list) else names.items())}


def class_ids(data_yaml: Path):
    """{name: id} for CLASSES; classes missing from data.yaml get the next free ids (not written)."""
    _, names = read_names(data_yaml)
    ids = {v: k for k, v in names.items() if v in CLASSES}
    nxt = max(names, default=-1) + 1
    for c in CLASSES:
        if c not in ids:
            ids[c], nxt = nxt, nxt + 1
    return ids


def register_classes(data_yaml: Path, ids, used):
    """Append the used classes that data.yaml names does not list yet."""
    cfg, names = read_names(data_yaml)
    new = {ids[c]: c for c in used if c not in names.values()}
    if new:
        names.update(new)
        cfg["names"] = dict(sorted(names.items()))
        data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"added classes {new} to {data_yaml}", flush=True)


def add_images(a, root: Path):
    """Copy --add images into images/<split>, create empty label files, record --um-per-px."""
    files = []
    for s in a.add:
        p = Path(s)
        files += sorted(p.iterdir()) if p.is_dir() else [p]
    calib_path = root / "scale_calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    for p in files:
        if not p.is_file() or p.suffix.lower() not in EXTS:
            print(f"skipping {p} (not a {'/'.join(sorted(EXTS))} file)")
            continue
        dst = root / "images" / a.split / p.name
        if dst.exists():
            print(f"already in dataset: {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            print(f"added {dst}")
        lbl = root / "labels" / a.split / f"{p.stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.touch(exist_ok=True)
        if a.um_per_px:
            calib.setdefault("images", {})[p.stem] = a.um_per_px
        elif p.stem not in calib.get("images", {}):
            print(f"  warning: {p.stem} has no um_per_px in {calib_path.name}; pass --um-per-px")
    if a.um_per_px:
        calib_path.write_text(json.dumps(calib, indent=2) + "\n")


def list_entries(root: Path):
    """{stem: {"img": first image found, "labels": [label path in every split holding the image]}}."""
    entries = {}
    for split in SPLITS:
        d = root / "images" / split
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXTS:
                e = entries.setdefault(p.stem, {"img": p, "labels": []})
                lbl = root / "labels" / split / f"{p.stem}.txt"
                if lbl not in e["labels"]:
                    e["labels"].append(lbl)
    return entries


def line_class(line):
    return int(line.split(maxsplit=1)[0])


def parse_poly(line):
    """YOLO-seg line -> normalised (N, 2) polygon."""
    return np.array(line.split()[1:], np.float32).reshape(-1, 2)


def poly_line(cls_id, poly):
    return f"{cls_id} " + " ".join(f"{min(max(v, 0), 1):.6f}" for v in poly.ravel())


class Annotator:
    def __init__(self, args, root: Path):
        self.root = root
        self.entries = list_entries(root)
        self.stems = sorted(self.entries)
        if not self.stems:
            raise SystemExit(f"no images found in {root / 'images'}")
        self.ids = class_ids(root / "data.yaml")
        self.names = {v: k for k, v in self.ids.items()}
        self.active = args.cls
        self.W, self.H = (int(v) for v in args.win.lower().split("x"))
        self.polys = {}  # stem -> {class id: [normalised (N, 2) polygons]}
        self.other = {}  # label path -> lines of non-editable classes, written back verbatim
        for stem, e in self.entries.items():
            self.polys[stem] = {k: [] for k in self.names}
            for lbl in e["labels"]:
                lines = [l for l in (lbl.read_text().splitlines() if lbl.exists() else []) if l.strip()]
                self.other[lbl] = [l for l in lines if line_class(l) not in self.names]
                # editable polygons come from the first split's label file that has any
                if not any(self.polys[stem].values()):
                    for l in lines:
                        if line_class(l) in self.names:
                            self.polys[stem][line_class(l)].append(parse_poly(l))
        self.history = {}
        self.changed = set()
        gap, x = 8, self.W
        self.buttons = []
        for label, action, bw in reversed(BUTTONS):
            x -= bw + gap
            self.buttons.insert(0, (label, action, (x, 6, bw, HUD - 12)))
        self.show_layers = True
        self.points, self.strokes, self.tracing = [], [], False
        self.pan = None
        self.pan_mode = False
        self.edit_mode = False
        self.drag = None  # (class id, polygon index, vertex index) being dragged in Edit mode
        self.mouse = (0, 0)
        self.dirty = True
        if args.start:
            if args.start not in self.entries:
                raise SystemExit(f"--start {args.start!r} not found; images: {self.stems}")
            self.i = self.stems.index(args.start)
        else:
            self.i = next((k for k, s in enumerate(self.stems) if not self.polys[s][self.aid]), 0)

    @property
    def aid(self):
        return self.ids[self.active]

    def color(self, cls_id):
        name = self.names.get(cls_id)
        return CLASS_COLORS[name] if name in CLASS_COLORS else LAYER_COLORS[cls_id % len(LAYER_COLORS)]

    # ---------------------------------------------------------------- view
    def to_img(self, x, y):
        return self.ox + x / self.s, self.oy + y / self.s

    def to_disp(self, ix, iy):
        return int(round((ix - self.ox) * self.s)), int(round((iy - self.oy) * self.s))

    def disp_poly(self, poly):
        return ((poly * (self.iw, self.ih) - (self.ox, self.oy)) * self.s).astype(np.int32)

    def load(self):
        stem = self.stems[self.i]
        p = self.entries[stem]["img"]
        print(f"loading [{self.i + 1}/{len(self.stems)}] {p.name} ...", flush=True)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read {p}")
        self.ih, self.iw = img.shape[:2]
        self.layers = [(line_class(l), parse_poly(l)) for l in self.other[self.entries[stem]["labels"][0]]]
        self.points, self.strokes, self.tracing = [], [], False
        self.pan = None
        self.drag = None
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

        polys = self.polys[self.stems[self.i]]
        if self.show_layers:
            others = self.layers + [(c, p) for c, ps in polys.items() if c != self.aid for p in ps]
            for cls, poly in others:
                cv2.polylines(canvas, [self.disp_poly(poly)], True, self.color(cls), 1, cv2.LINE_AA)
        active = [self.disp_poly(g) for g in self.cur()]
        if active:
            fill = canvas.copy()
            cv2.fillPoly(fill, active, self.color(self.aid))
            cv2.addWeighted(fill, 0.25, canvas, 0.75, 0, canvas)
            cv2.polylines(canvas, active, True, self.color(self.aid), 2, cv2.LINE_AA)
        if self.points:
            pts = [self.to_disp(px, py) for px, py in self.points] + [self.mouse]
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, pts[0], CLOSE_PX, (0, 0, 255), 1, cv2.LINE_AA)
        if self.edit_mode:
            for cls, _, poly in self.editable():
                for px, py in self.disp_poly(poly):
                    if 0 <= px < self.W and HUD <= py < self.H:
                        cv2.circle(canvas, (int(px), int(py)), 3, self.color(cls), -1, cv2.LINE_AA)
            hit = self.drag or self.vertex_at(*self.mouse)
            if hit:
                cls, k, vi = hit
                px, py = self.disp_poly(polys[cls][k])[vi]
                cv2.circle(canvas, (int(px), int(py)), 6, (255, 255, 255), 2, cv2.LINE_AA)

        ix, iy = self.to_img(*self.mouse)
        cv2.rectangle(canvas, (0, 0), (self.W, HUD), (0, 0, 0), -1)
        drawing = f"   drawing: {len(self.points)} pts (Enter closes, Esc cancels)" if self.points else ""
        counts = "  ".join(f"{c}: {len(polys[self.ids[c]])}" for c in CLASSES)
        line1 = f"[{self.i + 1}/{len(self.stems)}] {self.stems[self.i]}   {counts}{drawing}"
        if self.edit_mode:
            hint = "EDIT: drag vertex, click edge = add, R-click/Del = remove, Esc exit"
        else:
            hint = "1/2/3 class, drag draw, R-drag pan, g pan, e edit, x del, h, f fit"
        line2 = f"zoom {self.s:.2f} x={int(ix)} y={int(iy)} | {hint}"
        cv2.putText(canvas, line1, (8, 17), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (8, 36), FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        for label, action, (bx, by, bw, bh) in self.buttons:
            if action == "class":
                label = f"{self.aid}: {self.active}"
                bg = tuple(c // 2 for c in self.color(self.aid))
            elif action == "pan":
                label = "Pan ON" if self.pan_mode else "Pan"
                bg = (130, 100, 40) if self.pan_mode else (90, 90, 90)
            elif action == "edit":
                label = "Edit ON" if self.edit_mode else "Edit"
                bg = (40, 100, 130) if self.edit_mode else (90, 90, 90)
            else:
                bg = (60, 140, 60) if action == "next" else (90, 90, 90)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg, -1)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.putText(canvas, label, (bx + (bw - tw) // 2, by + (bh + th) // 2), FONT, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    # ---------------------------------------------------------------- polygons
    def cur(self):
        return self.polys[self.stems[self.i]][self.aid]

    def clamp(self, ix, iy):
        return min(max(ix, 0), self.iw), min(max(iy, 0), self.ih)

    def hist(self):
        return self.history.setdefault(self.stems[self.i], [])

    def near_first(self, x, y):
        if len(self.points) < 3:
            return False
        fx, fy = self.to_disp(*self.points[0])
        return math.hypot(x - fx, y - fy) <= CLOSE_PX

    def close_poly(self):
        pts = np.array(self.points, np.float32)
        self.points, self.strokes, self.tracing = [], [], False
        if len(pts) < 3 or cv2.contourArea(pts) * self.s ** 2 < 25:
            print("polygon too small, discarded", flush=True)
            return
        self.cur().append(pts / np.array([self.iw, self.ih], np.float32))
        self.hist().append(("add", None, len(self.cur()) - 1, self.aid))
        self.changed.add(self.stems[self.i])

    def delete_under_cursor(self):
        ix, iy = self.to_img(*self.mouse)
        px = [g * np.array([self.iw, self.ih], np.float32) for g in self.cur()]
        hits = [k for k, g in enumerate(px) if cv2.pointPolygonTest(g, (float(ix), float(iy)), False) >= 0]
        if hits:
            k = min(hits, key=lambda k: cv2.contourArea(px[k]))
            self.hist().append(("del", self.cur().pop(k), k, self.aid))
            self.changed.add(self.stems[self.i])

    def editable(self):
        """(class id, polygon index, polygon) of every polygon that Edit mode may change."""
        polys = self.polys[self.stems[self.i]]
        ids = list(polys) if self.show_layers else [self.aid]
        return [(c, k, p) for c in ids for k, p in enumerate(polys[c])]

    def vertex_at(self, x, y):
        best, bd = None, VERTEX_PX
        for c, k, p in self.editable():
            d = np.hypot(*(self.disp_poly(p) - (x, y)).T)
            vi = int(d.argmin())
            if d[vi] <= bd:
                best, bd = (c, k, vi), d[vi]
        return best

    def edge_at(self, x, y):
        """(class id, polygon index, insert index) of the nearest edge, closing edge included."""
        best, bd = None, EDGE_PX
        q = np.array([x, y], np.float64)
        for c, k, p in self.editable():
            a = self.disp_poly(p).astype(np.float64)
            ab = np.roll(a, -1, axis=0) - a
            t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
            d = np.hypot(*(a + t[:, None] * ab - q).T)
            j = int(d.argmin())
            if d[j] <= bd:
                best, bd = (c, k, j + 1), d[j]
        return best

    def start_drag(self, x, y, ix, iy):
        hit = self.vertex_at(x, y)
        edge = None if hit else self.edge_at(x, y)
        if not (hit or edge):
            return
        c, k, vi = hit or edge
        lst = self.polys[self.stems[self.i]][c]
        self.hist().append(("edit", lst[k].copy(), k, c))
        if edge:
            lst[k] = np.insert(lst[k], vi, (ix / self.iw, iy / self.ih), axis=0)
        self.drag = (c, k, vi)
        self.changed.add(self.stems[self.i])

    def end_drag(self):
        c, k, _ = self.drag
        self.drag = None
        # a click on a vertex without moving it leaves nothing to undo
        if np.array_equal(self.hist()[-1][1], self.polys[self.stems[self.i]][c][k]):
            self.hist().pop()

    def delete_vertex(self, hit):
        if not hit:
            return
        c, k, vi = hit
        lst = self.polys[self.stems[self.i]][c]
        if len(lst[k]) <= 3:
            print("a polygon needs at least 3 vertices; use x to delete the whole polygon", flush=True)
            return
        self.hist().append(("edit", lst[k], k, c))
        lst[k] = np.delete(lst[k], vi, axis=0)
        self.changed.add(self.stems[self.i])

    def undo(self):
        if self.points:
            del self.points[self.strokes.pop():]
            return
        self.drag = None
        if self.hist():
            op, g, k, cls_id = self.hist().pop()
            lst = self.polys[self.stems[self.i]][cls_id]
            if op == "add":
                lst.pop(k)
            elif op == "edit":
                lst[k] = g
            else:
                lst.insert(k, g)
        elif self.cur():
            # polygons loaded from disk have no history; fall back to removing the last one
            self.cur().pop()
        else:
            return
        self.changed.add(self.stems[self.i])

    def save(self):
        for stem in sorted(self.changed):
            polys = self.polys[stem]
            register_classes(self.root / "data.yaml", self.ids, [c for c in CLASSES if polys[self.ids[c]]])
            lines = [poly_line(self.ids[c], p) for c in CLASSES for p in polys[self.ids[c]]]
            for lbl in self.entries[stem]["labels"]:
                lbl.parent.mkdir(parents=True, exist_ok=True)
                with open(lbl, "w", newline="\n") as f:
                    f.writelines(l + "\n" for l in self.other[lbl] + lines)
            counts = ", ".join(f"{len(polys[self.ids[c]])} {c}" for c in CLASSES)
            print(f"saved {counts} for {stem}", flush=True)
        self.changed.clear()

    def go(self, step):
        j = min(max(self.i + step, 0), len(self.stems) - 1)
        if j == self.i:
            print("already at the " + ("last" if step > 0 else "first") + " image", flush=True)
            return
        self.save()
        self.i = j
        canvas = self.render()
        cv2.putText(canvas, f"Loading {self.stems[j]} ...", (20, HUD + 40), FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)
        cv2.waitKey(1)
        self.load()

    def act(self, action):
        if action == "next":
            self.go(1)
        elif action == "prev":
            self.go(-1)
        elif action == "undo":
            self.undo()
        elif action == "class":
            self.active = CLASSES[(CLASSES.index(self.active) + 1) % len(CLASSES)]
        elif action == "pan":
            self.pan_mode = not self.pan_mode
            self.edit_mode = False
            self.pan = self.drag = None
            self.tracing = False
        elif action == "edit":
            self.edit_mode = not self.edit_mode
            self.pan_mode = False
            self.pan = self.drag = None
            self.points, self.strokes, self.tracing = [], [], False
        self.dirty = True

    # ---------------------------------------------------------------- events
    def on_mouse(self, ev, x, y, flags, _):
        self.mouse = (x, y)
        ix, iy = self.clamp(*self.to_img(x, y))
        if ev == cv2.EVENT_LBUTTONDOWN:
            if y < HUD:
                for _, action, (bx, by, bw, bh) in self.buttons:
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.act(action)
            elif self.pan_mode:
                self.pan = (cv2.EVENT_FLAG_LBUTTON, x, y)
                self.tracing = False
            elif self.edit_mode:
                self.start_drag(x, y, ix, iy)
            elif self.near_first(x, y):
                self.close_poly()
            else:
                self.strokes.append(len(self.points))
                self.points.append((ix, iy))
                self.tracing = True
        elif ev == cv2.EVENT_LBUTTONUP:
            if self.pan and self.pan[0] == cv2.EVENT_FLAG_LBUTTON:
                self.pan = None
            if self.drag:
                self.end_drag()
            # a freehand stroke released near the first vertex closes the polygon
            if self.tracing and len(self.points) - self.strokes[-1] > 1 and self.near_first(x, y):
                self.close_poly()
            self.tracing = False
        elif ev in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            hit = self.vertex_at(x, y) if self.edit_mode and ev == cv2.EVENT_RBUTTONDOWN and y >= HUD else None
            if hit:
                self.delete_vertex(hit)
            elif y >= HUD:
                button = cv2.EVENT_FLAG_RBUTTON if ev == cv2.EVENT_RBUTTONDOWN else cv2.EVENT_FLAG_MBUTTON
                self.pan = (button, x, y)
                self.tracing = False
        elif ev == cv2.EVENT_MOUSEMOVE:
            if self.pan:
                button, px, py = self.pan
                if flags & button:
                    self.ox -= (x - px) / self.s
                    self.oy -= (y - py) / self.s
                    self.pan = (button, x, y)
                else:
                    self.pan = None
            elif self.drag:
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    c, k, vi = self.drag
                    self.polys[self.stems[self.i]][c][k][vi] = (ix / self.iw, iy / self.ih)
                else:
                    self.end_drag()
            elif self.tracing and flags & cv2.EVENT_FLAG_LBUTTON:
                lx, ly = self.to_disp(*self.points[-1])
                if math.hypot(x - lx, y - ly) >= TRACE_PX:
                    self.points.append((ix, iy))
        elif ev in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            button = cv2.EVENT_FLAG_RBUTTON if ev == cv2.EVENT_RBUTTONUP else cv2.EVENT_FLAG_MBUTTON
            if self.pan and self.pan[0] == button:
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
            if k == 27 and self.points:
                self.points, self.strokes, self.tracing = [], [], False
            elif k == ENTER and self.points:
                self.close_poly()
            elif k == 27 and self.edit_mode:
                self.act("edit")
            elif k in (ord("q"), 27):
                break
            elif k in (ord("n"), 32, KEY_RIGHT):
                self.act("next")
            elif k in (ord("p"), KEY_LEFT):
                self.act("prev")
            elif k in (ord("u"), CTRL_Z):
                self.act("undo")
            elif ord("1") <= k < ord("1") + len(CLASSES):
                self.active = CLASSES[k - ord("1")]
            elif k == TAB:
                self.act("class")
            elif k == ord("g"):
                self.act("pan")
            elif k == ord("e"):
                self.act("edit")
            elif k in (KEY_DEL, BACKSPACE) and self.edit_mode:
                self.delete_vertex(self.vertex_at(*self.mouse))
            elif k == ord("x"):
                self.delete_under_cursor()
            elif k == ord("f"):
                self.fit()
            elif k == ord("h"):
                self.show_layers = not self.show_layers
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
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "dataset_for_finetuning"),
                    help="YOLO-seg dataset (data.yaml, images/{train,val}, labels/{train,val}) edited in place")
    ap.add_argument("--add", nargs="+", default=[], help="image files or folders to copy into the dataset first")
    ap.add_argument("--split", choices=SPLITS, default="train", help="split that --add images go to")
    ap.add_argument("--um-per-px", type=float, help="scale of the --add images, recorded in scale_calibration.json")
    ap.add_argument("--cls", choices=CLASSES, default="gland", help="class active at startup (switch with 1/2/3/Tab)")
    ap.add_argument("--start", help="image stem to open first (default: first image without --cls polygons)")
    ap.add_argument("--win", default="1400x850", help="window size WxH; lower it for small screens")
    args = ap.parse_args()
    root = Path(args.dataset)
    if not (root / "data.yaml").exists():
        raise SystemExit(f"{root / 'data.yaml'} not found")
    if args.add:
        add_images(args, root)
    Annotator(args, root).run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
measure.py -- all SC/Epidermal measurement logic: per-class height
stats (orientation-independent thickness via distance-transform + skeleton),
Epidermal boundary tortuosity, per-class hole count (number of enclosed
holes), and the Measurement panel image. postprocess.py calls
measure_layers() as its single entry point into this module.


It can also be run directly on this project's data:


   python measure.py --data yolo_dataset_4cls_2x --split val --out measure_val
   python measure.py --preds preds --images yolo_dataset_4cls_2x/images/val --out measure_preds
   python measure.py --data yolo_dataset_4cls_2x --um-per-px 1.0     # report in um instead of px


which writes measurements.csv (CSV_COLUMNS) and panels/<tile>.jpg.


Height definition: mean height uses the Area-to-Length (integral) method:
total mask Area (A, pixel count) divided by the medial-axis backbone/
centerline length (L) of the layer - Mean Height = A / L. L is the main-
spine skeleton (skeleton_spine, per connected component, longest geodesic
path through the skeleton, summed across components). layer_sides() still
splits each layer's boundary into a junction (SC/GL) side and a free-surface
/ dermal side, purely so the Measurement panel can draw those reference
lines; only area, mean height, tortuosity, and hole count are reported.


skeleton_spine is reused by _centerline_length_px for the mean-height
Area/Length denominator.
"""




import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.measure import label
from skimage.morphology import remove_small_holes, skeletonize

from config import CONFIG

CLASS_COLORS = {"SC": (58, 161, 232), "Epidermal": (192, 168, 31)}  # BGR

# values come from base_config.yaml's measurement section (fallback = previous hardcoded defaults)
_MEAS_CFG = CONFIG.get("measurement", {})
SKELETON_METHOD = _MEAS_CFG.get("skeleton_method", "lee")  # 'zhang' (skimage's 2D default) or 'lee'
HOLE_AREA_PX = _MEAS_CFG.get("hole_area_px", 64)  # small holes filled before skeletonizing, so the skeleton graph is a tree
SC_SURFACE_GL_MARGIN_FRAC = _MEAS_CFG.get("sc_surface_gl_margin_frac", 0.5)  # excludes SC boundary within this fraction of SC's typical thickness from GL
SC_SURFACE_GL_MARGIN_MIN_PX = _MEAS_CFG.get("sc_surface_gl_margin_min_px", 3)  # margin floor, so a paper-thin SC still gets some separation from GL
SC_SURFACE_GL_MARGIN_MAX_PX = _MEAS_CFG.get("sc_surface_gl_margin_max_px", 30)  # margin cap, so an unusually thick SC doesn't over-exclude its outer boundary


def _distance_transform(mask: np.ndarray) -> np.ndarray:
   mask = np.asarray(mask, dtype=bool)
   out = np.zeros(mask.shape, np.float32)
   ys, xs = np.nonzero(mask)
   if ys.size == 0:
       return out
   y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, mask.shape[0])
   x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, mask.shape[1])
   # Include adjacent in-image background without inventing background past image edges.
   crop = mask[y0:y1, x0:x1].astype(np.uint8)
   out[y0:y1, x0:x1] = cv2.distanceTransform(crop, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
   return out




CSV_COLUMNS = [
   "folder name", "file name", "date",
   "SC mean height", "SC area",
   "Epidermal mean height", "Epidermal area",
   "EDJ tortuosity",
    "SC_hole_num", "Epidermal_hole_num",
]


# ----------------------------------------------------------------------------
# visualisation primitives (shared by postprocess.py's other panels too)
# ----------------------------------------------------------------------------

def overlay(img_bgr: np.ndarray, masks: dict, alpha: float = 0.45) -> np.ndarray:
   out = img_bgr.copy()
   for name, m in masks.items():
       sel = np.asarray(m, dtype=bool)
       if sel.any():
           color = np.array(CLASS_COLORS.get(name, (0, 255, 0)))
           out[sel] = (alpha * color + (1 - alpha) * out[sel]).astype(np.uint8)
   line_px = max(1, min(4, round(min(out.shape[:2]) / 1500)))
   for name, m in masks.items():
       sel = np.asarray(m, dtype=np.uint8)
       if sel.any():
           cnts, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           cv2.drawContours(out, cnts, -1, (20, 20, 20), line_px, cv2.LINE_AA)
   return out


# ----------------------------------------------------------------------------
# skeleton + distance-transform thickness (orientation-independent)
# ----------------------------------------------------------------------------

def skeleton_spine(skel: np.ndarray, dt: np.ndarray):
   """Longest geodesic path through a (connected) skeleton - the main
   medial spine, as a graph problem: nodes = skeleton pixels, edges between
   8-connected neighbours. Two Dijkstra sweeps (by physical `length`) find
   the band's two ends (farthest node from an arbitrary start, then
   farthest from THAT - the classic two-sweep tree-diameter trick); routing
   between them uses `cost = length/dt` instead, so the spine hugs the
   thickest/most central pixels rather than an arbitrary shortest path.
   Using `length` for the end-finding sweeps matters - if cost/dt were used
   there too, thin spur tips look artificially "far away" and get picked as
   the ends instead of the band's actual extremities.
   Returns (spine_mask, subspine_mask, (path_ys, path_xs)) - the masks are
   the same shape as skel and the path arrays give the spine pixels IN
   ORDER from one end to the other; falls back to (skel, empty, unordered)
   if the skeleton is too small/degenerate to route.
   Uses scipy.sparse.csgraph (compiled C) rather than a pure-Python graph:
   the skeleton is a tree after hole-filling, so the path between the two
   diameter endpoints is unique and independent of the cost weighting."""
   ys, xs = np.nonzero(skel)
   n = len(xs)
   if n < 2:
       return skel.copy(), np.zeros_like(skel, bool), (ys, xs)
   h, w = skel.shape
   index_img = np.full((h + 2, w + 2), -1, dtype=np.int64)  # padded, so neighbour lookups never go out of bounds
   index_img[ys + 1, xs + 1] = np.arange(n)
   dt_nodes = np.maximum(dt[ys, xs].astype(np.float64), 1.0)
   src_list, dst_list, len_list, cost_list = [], [], [], []
   for dy in (-1, 0, 1):
       for dx in (-1, 0, 1):
           if dy == 0 and dx == 0:
               continue
           nbr = index_img[ys + 1 + dy, xs + 1 + dx]
           valid = nbr >= 0
           if not valid.any():
               continue
           i = np.nonzero(valid)[0]
           j = nbr[valid]
           length = float(np.hypot(dy, dx))
           src_list.append(i)
           dst_list.append(j)
           len_list.append(np.full(i.shape, length))
           # symmetric cost so the spine hugs the thickest/most-central pixels (irrelevant for a tree)
           cost_list.append(length / np.minimum(dt_nodes[i], dt_nodes[j]))
   if not src_list:
       return skel.copy(), np.zeros_like(skel, bool), (ys, xs)
   src, dst = np.concatenate(src_list), np.concatenate(dst_list)
   lens, costs = np.concatenate(len_list), np.concatenate(cost_list)
   g_len = csr_matrix((lens, (src, dst)), shape=(n, n))
   g_cost = csr_matrix((costs, (src, dst)), shape=(n, n))
   _, comp = connected_components(g_len, directed=False)
   largest = int(np.argmax(np.bincount(comp)))  # label of the biggest connected component
   start = int(np.flatnonzero(comp == largest)[0])
   # two-sweep tree-diameter trick, by physical length; nodes in other components stay at inf
   d0 = dijkstra(g_len, directed=False, indices=start)
   a = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
   d1 = dijkstra(g_len, directed=False, indices=a)
   b = int(np.argmax(np.where(np.isfinite(d1), d1, -1.0)))
   _, pred = dijkstra(g_cost, directed=False, indices=a, return_predecessors=True)
   path = []
   node = b
   while node != a and node >= 0:
       path.append(node)
       node = int(pred[node])
   if node != a:  # b unreachable from a (degenerate) - fall back to the raw skeleton
       return skel.copy(), np.zeros_like(skel, bool), (ys, xs)
   path.append(a)
   path = path[::-1]
   spine = np.zeros_like(skel, bool)
   spine[ys[path], xs[path]] = True
   return spine, skel & ~spine, (ys[path], xs[path])








# ----------------------------------------------------------------------------
# boundary-to-boundary spans (surface -> SC/GL junction -> bottom)
# ----------------------------------------------------------------------------




def _prep_layer(mask: np.ndarray, hole_area_px: int = HOLE_AREA_PX):
   """Hole-filled mask + its distance transform + medial-axis skeleton, computed
   once so layer_sides / layer_stats / the spine panel all reuse them instead of
   each recomputing the (expensive) skeletonize + distance transform."""
   clean = remove_small_holes(mask.astype(bool), max_size=hole_area_px)
   dt = _distance_transform(clean)
   skel = skeletonize(clean, method=SKELETON_METHOD)
   return clean, dt, skel


def _layer_spine(mask: np.ndarray, hole_area_px: int = HOLE_AREA_PX,
                clean: np.ndarray = None, dt: np.ndarray = None, skel: np.ndarray = None,
                col_window: tuple = None):
   """Medial-axis spine mask + total arc length (px) of a layer - per connected
   component, skeleton_spine's longest geodesic path through the skeleton,
   combined/summed across components. This IS the Area/Length denominator
   (see _centerline_length_px), exposed here so it can also be drawn.
   `clean`/`dt`/`skel` may be passed in (from _prep_layer) to skip recomputing them.
   `col_window` = (lo, hi): the skeleton is still built from the WHOLE mask (so a
   lateral clip can't carve cut-edges into it), but the returned spine mask and
   its summed length keep only the portion in columns [lo, hi)."""
   spine_mask = np.zeros_like(mask, bool)
   if not mask.any():
       return spine_mask, 0.0
   if clean is None:
       clean = remove_small_holes(mask.astype(bool), max_size=hole_area_px)
   if dt is None:
       dt = _distance_transform(clean)
   if skel is None:
       skel = skeletonize(clean, method=SKELETON_METHOD)
   comp_labels = label(clean, connectivity=2)
   total = 0.0
   for comp_id in range(1, int(comp_labels.max()) + 1):
       comp_skel = skel & (comp_labels == comp_id)
       if not comp_skel.any():
           continue
       spine, _, (pys, pxs) = skeleton_spine(comp_skel, dt)
       spine_mask |= spine
       if len(pxs) < 2:
           continue
       pys_f, pxs_f = np.asarray(pys, dtype=np.float64), np.asarray(pxs, dtype=np.float64)
       seg = np.hypot(np.diff(pxs_f), np.diff(pys_f))
       if col_window is not None:
           lo, hi = col_window
           inside = (pxs_f >= lo) & (pxs_f < hi)
           seg = seg[inside[:-1] & inside[1:]]
       total += float(seg.sum())
   if col_window is not None:
       lo, hi = col_window
       spine_mask[:, :lo] = False
       spine_mask[:, hi:] = False
   return spine_mask, total


def _centerline_length_px(mask: np.ndarray, hole_area_px: int = HOLE_AREA_PX) -> float:
   """Arc length (px) of the mask's medial-axis backbone - per connected
   component, the main spine (skeleton_spine's longest geodesic path through
   the skeleton), summed across components. Feeds layer_span_stats's
   Area-to-Length mean height (mean_height = area / this * um_per_px)."""
   return _layer_spine(mask, hole_area_px)[1]


def _inner_ring(mask: np.ndarray) -> np.ndarray:
   """Mask pixels that touch a non-mask pixel (8-connected) - the layer's own
   boundary, so it lies on the layer's contour."""
   k3 = np.ones((3, 3), np.uint8)
   return mask & (cv2.erode(mask.astype(np.uint8), k3) == 0)


def _largest_component(mask: np.ndarray) -> np.ndarray:
   """Boolean mask of just the largest-by-area (pixel count) 8-connected
   component of mask - the 'main' segment when a class has broken into
   multiple pieces (e.g. after morphological opening)."""
   labels = label(mask.astype(bool), connectivity=2)
   if labels.max() == 0:
       return np.zeros_like(mask, bool)
   ids, counts = np.unique(labels[labels > 0], return_counts=True)
   return labels == ids[np.argmax(counts)]


def _near_mask(mask: np.ndarray, radius: float) -> np.ndarray:
   """distance_transform_edt(~mask) < radius, computed only on mask's bbox padded
   by radius so huge images don't need a full-frame float64 EDT."""
   out = np.zeros_like(mask, bool)
   if not mask.any():
       return out
   rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
   r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
   c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
   pad = int(np.ceil(radius)) + 1
   r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
   r1, c1 = min(r1 + pad, mask.shape[0]), min(c1 + pad, mask.shape[1])
   out[r0:r1, c0:c1] = distance_transform_edt(~mask[r0:r1, c0:c1]) < radius
   return out


def layer_sides(sc: np.ndarray, gl: np.ndarray, hole_area_px: int = HOLE_AREA_PX,
               sc_prep: tuple = None, gl_prep: tuple = None):
   """Split each layer's boundary into the two sides a span runs between.
     SC:  'surface'  = SC boundary touching neither SC nor GL, AND farther
                       from GL than SC_SURFACE_GL_MARGIN_FRAC of SC's own
                       typical thickness (clamped to [MIN_PX, MAX_PX]) -
                       (air/glass or loose debris)
          'junction' = SC boundary touching GL
     GL:  'top'      = GL boundary touching SC
          'bottom'   = GL boundary touching tissue that is not SC (the dermal side)
   Boundary pixels closer to SC than half the GL's typical thickness are treated
   as 'top' even if the touching pixel is tissue-coloured, so unlabelled keratin
   flakes on the outside do not get mistaken for the dermal boundary.
   `sc_prep`/`gl_prep` = (clean, dt, skel) from _prep_layer; when given, the
   clean mask + distance transform + skeleton are reused (only the typical-
   thickness reference for the panel is affected, not any reported stat).
   Returns (sc_clean, gl_clean, sides) with sides = dict of boolean masks."""
   if sc_prep is not None:
       sc, sc_dt, sc_skel = sc_prep
   else:
       sc, sc_dt, sc_skel = remove_small_holes(sc.astype(bool), max_size=hole_area_px), None, None
   if gl_prep is not None:
       gl, gl_dt, gl_skel = gl_prep
   else:
       gl, gl_dt, gl_skel = remove_small_holes(gl.astype(bool), max_size=hole_area_px), None, None
   k3 = np.ones((3, 3), np.uint8)
   d_sc = cv2.dilate(sc.astype(np.uint8), k3) > 0
   d_gl = cv2.dilate(gl.astype(np.uint8), k3) > 0


   sc_ring = _inner_ring(sc)
   if sc.any():
       # typical SC thickness from its distance transform along the medial axis -
       # a flat pixel margin would swallow the whole ring on a thin SC band
       if sc_skel is None:
           sc_skel = skeletonize(sc)
       if sc_dt is None:
           sc_dt = _distance_transform(sc)
       typ_sc = 2 * np.median(sc_dt[sc_skel]) if sc_skel.any() else 0.0
   else:
       typ_sc = 0.0
   gl_margin_px = float(np.clip(SC_SURFACE_GL_MARGIN_FRAC * typ_sc,
                                SC_SURFACE_GL_MARGIN_MIN_PX, SC_SURFACE_GL_MARGIN_MAX_PX))
   near_gl = _near_mask(gl, gl_margin_px)
   sides = {"sc_junction": sc_ring & d_gl, "sc_surface": sc_ring & ~d_gl & ~near_gl}


   gl_ring = _inner_ring(gl)
   top = gl_ring & d_sc
   bottom = gl_ring & ~top
   if sc.any() and gl.any():
       # typical GL thickness from its distance transform along the medial axis
       if gl_dt is None:
           gl_dt = _distance_transform(gl)
       if gl_skel is None:
           gl_skel = skeletonize(gl)
       typ = 2 * np.median(gl_dt[gl_skel]) if gl_skel.any() else 20.0
       near_sc = _near_mask(sc, max(0.5 * typ, 5.0))
       top = top | (gl_ring & near_sc)
       bottom = gl_ring & ~top
   sides["gl_top"], sides["gl_bottom"] = top, bottom
   return sc, gl, sides




def layer_stats(layer: np.ndarray, um_per_px: float, centerline_px: float = None) -> dict:
   """mean height (um, or px when um_per_px == 1) + area (um^2) for one
   layer. mean = Area-to-Length (integral) method: mask area / medial-axis
   centerline length (see _centerline_length_px). `centerline_px` may be
   passed in (from a shared _layer_spine call) to skip recomputing the spine."""
   area = float(layer.sum()) * um_per_px ** 2
   if centerline_px is None:
       centerline_px = _centerline_length_px(layer)
   centerline_um = centerline_px * um_per_px
   mean_height = (area / centerline_um) if centerline_um > 0 else 0.0
   return {"mean_height": mean_height, "area": area}


def granular_layer_midline(gl_mask: np.ndarray):
   """Per-column centroid (mean row) of the Epidermal mask - one point
   per column that contains any GL pixel, ordered left to right. A simple
   horizontal-ish midline through the layer, distinct from the medial-axis
   spine used for mean height. Returns (xs, ys) int arrays, empty if the
   mask has no columns with any GL pixel."""
   mask = gl_mask.astype(bool)
   counts = mask.sum(axis=0)
   has_col = counts > 0
   if not has_col.any():
       return np.array([], dtype=int), np.array([], dtype=int)
   rows = np.arange(mask.shape[0])[:, None]
   sums = (mask * rows).sum(axis=0)
   xs = np.nonzero(has_col)[0]
   ys = np.round(sums[xs] / counts[xs]).astype(int)
   return xs, ys


# ----------------------------------------------------------------------------
# tortuosity + connectivity quality
# ----------------------------------------------------------------------------




def granular_layer_tortuosity(gl_mask: np.ndarray, sc_mask: np.ndarray) -> float:
   """Arc length / chord length of Epidermal's boundary, EXCLUDING the
   side touching SC (the far/dermal side) - 1.0 = flat, higher = wavier/more
   rete-ridge-like. 1.0 if there isn't enough far-side boundary to measure."""
   gl_u8 = gl_mask.astype(np.uint8)
   k3 = np.ones((3, 3), np.uint8)
   boundary = (cv2.dilate(gl_u8, k3) > 0) & ~gl_mask.astype(bool)
   near_sc = boundary & (cv2.dilate(sc_mask.astype(np.uint8), k3) > 0)
   far_ring = boundary & ~near_sc
   ys, xs = np.nonzero(far_ring)
   if len(xs) < 50:
       return 1.0
   pts = np.stack([xs, ys], axis=1).astype(np.float64)
   _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
   along = (pts - pts.mean(0)) @ vt[0]
   chord = along.max() - along.min()
   return float(len(xs) / max(chord, 1.0))








def class_hole_count(mask: np.ndarray) -> int:
   """Number of fully-enclosed holes in a layer - background regions trapped
   inside the mask. Counted on the raw mask (no hole-filling) so segmentation
   gaps are not hidden."""
   if not mask.any():
       return 0
   m = mask.astype(bool)
   holes = binary_fill_holes(m) & ~m
   return int(label(holes, connectivity=1).max())








# ----------------------------------------------------------------------------
# Measurement panel
# ----------------------------------------------------------------------------




def _line_crosses_mask(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, skip_px: int = 3) -> bool:
   """True if the segment from (x0,y0) to (x1,y1) enters mask, ignoring the
   first skip_px samples (the start point sits ON the SC boundary, so it and
   its immediate neighbourhood are expected to be inside SC and shouldn't
   count as a crossing)."""
   length = int(round(np.hypot(x1 - x0, y1 - y0)))
   if length <= skip_px:
       return False
   t = np.linspace(0.0, 1.0, length)[skip_px:]
   xs = np.round(x0 + t * (x1 - x0)).astype(int)
   ys = np.round(y0 + t * (y1 - y0)).astype(int)
   h, w = mask.shape
   valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
   return bool(mask[ys[valid], xs[valid]].any())


def _nearest_point_spans(vis: np.ndarray, src_mask: np.ndarray, dst_mask: np.ndarray,
                         sc_mask: np.ndarray = None, n_spans: int = 50,
                         color=(0, 200, 0), max_color=None, line_px: int = 1) -> None:
   """Line from n_spans points evenly spaced along src_mask's x extent to
   each point's nearest point on dst_mask - shared by the junction-to-GL-
   bottom spans and the junction-to-SC-surface spans. Targets are spaced
   evenly by x-position, then snapped to the closest actual src_mask pixel,
   so spans are evenly spaced regardless of how densely src_mask is sampled
   at each x. If sc_mask is given, spans that cross back through it (a sign
   the nearest dst_mask point requires detouring through SC) are dropped
   rather than drawn. If max_color is given, the single longest surviving
   span is drawn in max_color instead of color."""
   ys_s, xs_s = np.nonzero(src_mask)
   ys_d, xs_d = np.nonzero(dst_mask)
   if xs_s.size == 0 or xs_d.size == 0:
       return
   order = np.argsort(xs_s)
   xs_s, ys_s = xs_s[order], ys_s[order]
   n = min(n_spans, xs_s.size)
   targets = np.linspace(xs_s[0], xs_s[-1], n)
   pos = np.clip(np.searchsorted(xs_s, targets), 0, xs_s.size - 1)
   left = np.clip(pos - 1, 0, xs_s.size - 1)
   use_left = np.abs(xs_s[left] - targets) < np.abs(xs_s[pos] - targets)
   sample_idx = np.unique(np.where(use_left, left, pos))
   xs_p, ys_p = xs_s[sample_idx], ys_s[sample_idx]
   tree = cKDTree(np.stack([xs_d, ys_d], axis=1))
   _, idx = tree.query(np.stack([xs_p, ys_p], axis=1))
   segments = []
   for x0, y0, i in zip(xs_p, ys_p, idx):
       x1, y1 = xs_d[i], ys_d[i]
       if sc_mask is not None and _line_crosses_mask(sc_mask, x0, y0, x1, y1):
           continue
       segments.append((int(x0), int(y0), int(x1), int(y1)))
   if not segments:
       return
   max_i = -1
   if max_color is not None:
       lengths = [np.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in segments]
       max_i = int(np.argmax(lengths))
   for i, (x0, y0, x1, y1) in enumerate(segments):
       c = max_color if i == max_i else color
       cv2.line(vis, (x0, y0), (x1, y1), c, line_px, cv2.LINE_AA)


def draw_junction_spans(vis: np.ndarray, sides: dict, sc_mask: np.ndarray = None, n_spans: int = 50,
                        color=(0, 200, 0), max_color=(0, 0, 255), line_px: int = 1) -> None:
   """Line from n_spans points evenly spaced along the SC/GL junction's x
   extent (white) to each point's nearest point on the GL bottom boundary
   (dark red) - local junction-to-dermal spans; see _nearest_point_spans.
   The single longest surviving span is drawn in max_color instead of
   color."""
   _nearest_point_spans(vis, sides["sc_junction"], sides["gl_bottom"], sc_mask, n_spans, color, max_color, line_px)


def draw_scale_bar(vis: np.ndarray, um_per_px: float, target_frac: float = 0.15) -> None:
    """Draws a labeled scale bar (bottom-right, in place) whose length is a
    'nice' 1/2/5 x 10^n um value close to target_frac of the image width."""
    h, w = vis.shape[:2]
    target_um = w * target_frac * um_per_px
    base = 10 ** np.floor(np.log10(target_um))
    bar_um = max(m * base for m in (1, 2, 5) if m * base <= target_um)
    bar_px = int(round(bar_um / um_per_px))
    thick = max(3, round(h / 150))
    font_scale = max(0.6, w / 1500)
    font_th = max(1, round(font_scale * 2))
    label = f"{bar_um:g} um"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_th)
    margin = max(10, round(w / 50))
    x1, y = w - margin, h - margin
    x0 = x1 - bar_px
    bg_x0 = min(x0, x1 - tw) - thick
    cv2.rectangle(vis, (bg_x0, y - thick - th - 3 * thick), (x1 + thick, y + thick), (255, 255, 255), -1)
    cv2.rectangle(vis, (x0, y - thick), (x1, y), (0, 0, 0), -1)
    cv2.putText(vis, label, (x1 - bar_px // 2 - tw // 2, y - 2 * thick), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), font_th, cv2.LINE_AA)


def draw_measurement_panel(rotated_img, sc_mask, gl_mask, sc_stats, gl_stats,
                           gl_tortuosity, sc_holes, gl_holes, um_per_px: float, sides=None,
                           spines: dict = None) -> np.ndarray:
   vis = overlay(rotated_img, {"SC": sc_mask, "Epidermal": gl_mask})
   for name, m in (("SC", sc_mask), ("Epidermal", gl_mask)):
       cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       cv2.drawContours(vis, cnts, -1, (20, 20, 20), 4, cv2.LINE_AA)
       cv2.drawContours(vis, cnts, -1, CLASS_COLORS[name], 2, cv2.LINE_AA)
   line_px = max(2, min(6, round(min(vis.shape[:2]) / 1500)))
   if sides is not None:
       k = np.ones((line_px, line_px), np.uint8)
       vis[cv2.dilate(sides["sc_junction"].astype(np.uint8), k) > 0] = (255, 255, 255)
   if spines is not None:
       # the medial-axis spines whose length is the mean-height denominator
       k = np.ones((line_px, line_px), np.uint8)
       for name in ("SC", "Epidermal"):
           if name in spines:
               vis[cv2.dilate(spines[name].astype(np.uint8), k) > 0] = SPINE_COLORS[name]
   unit = "um" if um_per_px != 1.0 else "px"
   if unit == "um":
       draw_scale_bar(vis, um_per_px)

   def _fmt(name, stats, holes):
       return f"{name}: mean height {stats['mean_height']:.1f} {unit}  area {stats['area']:.0f} {unit}  holes {holes}"

   lines = [_fmt("SC", sc_stats, sc_holes), _fmt("Epidermal", gl_stats, gl_holes),
            f"EDJ tortuosity {gl_tortuosity:.2f}    "
            f"\n white = EDJ junction \n red = SC spine \n magenta = Epidermal spine"]
   h, w = vis.shape[:2]
   font_scale = max(0.5, w / 1800)          # caption size tracks the image width
   font_th = max(1, round(font_scale * 1.5))
   row_h = round(44 * font_scale)
   pad = row_h * len(lines) + round(10 * font_scale)
   canvas = np.full((h + pad, w, 3), 255, np.uint8)
   canvas[pad:] = vis
   for i, txt in enumerate(lines):
       y = round(row_h * (i + 0.8))
       cv2.putText(canvas, txt, (round(8 * font_scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (20, 20, 20), font_th, cv2.LINE_AA)
   return canvas


def draw_gl_midline_panel(rotated_img: np.ndarray, sc_mask: np.ndarray, gl_mask: np.ndarray) -> np.ndarray:
   """SC/Epidermal overlay with the Epidermal's per-column-centroid
   midline (see granular_layer_midline) drawn over it, as its own panel."""
   vis = overlay(rotated_img, {"SC": sc_mask, "Epidermal": gl_mask})
   xs, ys = granular_layer_midline(gl_mask)
   if xs.size >= 2:
       line_px = max(2, min(6, round(min(vis.shape[:2]) / 1500)))
       pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)
       cv2.polylines(vis, [pts], isClosed=False, color=(255, 255, 255), thickness=line_px + 2, lineType=cv2.LINE_AA)
       cv2.polylines(vis, [pts], isClosed=False, color=(0, 0, 255), thickness=line_px, lineType=cv2.LINE_AA)
   return vis


SPINE_COLORS = {"SC": (0, 0, 255), "Epidermal": (255, 0, 255)}  # BGR: red / magenta


def draw_spine_panel(rotated_img: np.ndarray, sc_mask: np.ndarray, gl_mask: np.ndarray,
                     sc_stats: dict, gl_stats: dict, um_per_px: float,
                     spines: dict = None, lengths_px: dict = None) -> np.ndarray:
   """SC/Epidermal overlay with each layer's ACTUAL medial-axis spine -
   the centerline mean_height is computed from (area / this length) - drawn
   in its own color, plus a label row with each spine's length/area/mean
   height. Distinct from draw_gl_midline_panel's per-column-centroid line,
   which is not what mean height uses.
   `spines`/`lengths_px` (from measure_layers) may be passed to reuse the
   already-computed spines instead of recomputing them here."""
   vis = overlay(rotated_img, {"SC": sc_mask, "Epidermal": gl_mask})
   line_px = max(2, min(6, round(min(vis.shape[:2]) / 1500)))
   k = np.ones((line_px, line_px), np.uint8)
   lens = {}
   for name, mask in (("SC", sc_mask), ("Epidermal", gl_mask)):
       if spines is not None and name in spines:
           spine, length_px = spines[name], lengths_px[name]
       else:
           spine, length_px = _layer_spine(mask.astype(bool))
       lens[name] = length_px
       vis[cv2.dilate(spine.astype(np.uint8), k) > 0] = SPINE_COLORS[name]

   unit = "um" if um_per_px != 1.0 else "px"

   def _fmt(name, stats):
       return (f"{name}: spine length {lens[name] * um_per_px:.1f} {unit}  "
               f"area {stats['area']:.0f}  mean height {stats['mean_height']:.1f} {unit}")

   lines = [_fmt("SC", sc_stats), _fmt("Epidermal", gl_stats),
            "red = SC medial-axis spine, magenta = Epidermal medial-axis spine "
            "(mean height = area / spine length)"]
   pad = 24 * len(lines) + 6
   h, w = vis.shape[:2]
   canvas = np.full((h + pad, w, 3), 255, np.uint8)
   canvas[pad:] = vis
   for i, txt in enumerate(lines):
       cv2.putText(canvas, txt, (6, 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
   return canvas


# ----------------------------------------------------------------------------
# orchestration entry point
# ----------------------------------------------------------------------------




def measure_layers(rotated_img: np.ndarray, sc_mask: np.ndarray, gl_mask: np.ndarray, um_per_px: float,
                    date: str = "", spines: dict = None, lengths_px: dict = None):
    """Runs all SC/Epidermal measurements + draws the Measurement panel.
    `date` is passed through verbatim into the row (e.g. the source image's
    last-modified date - callers decide what it means and how to format it).
    `spines`/`lengths_px` (keyed 'SC'/'Epidermal'), if given, are used as the
    medial-axis spine masks + their lengths instead of computing them from
    sc_mask/gl_mask - lets a caller build the spine from the un-clipped mask
    first and pass in the clip-restricted result (see postprocess.py).
    Returns (row_dict, measurement_panel_image, spine_cache) - row_dict has every
    CSV_COLUMNS key EXCEPT 'folder name'/'file name' (postprocess.py adds those);
    spine_cache = {'spines': {...}, 'lengths_px': {...}} so a caller can draw the
    spine panel without recomputing the (expensive) skeleton/spine."""
    sc_bin, gl_bin = sc_mask.astype(bool), gl_mask.astype(bool)
    sc_prep = _prep_layer(sc_bin)   # (clean, dt, skel) - shared by layer_sides/layer_stats/spine
    gl_prep = _prep_layer(gl_bin)
    sc_clean, sc_dt, sc_skel = sc_prep
    gl_clean, gl_dt, gl_skel = gl_prep

    _, _, sides = layer_sides(sc_bin, gl_bin, sc_prep=sc_prep, gl_prep=gl_prep)

    if spines is not None:
        sc_spine, gl_spine = spines["SC"], spines["Epidermal"]
        sc_len, gl_len = lengths_px["SC"], lengths_px["Epidermal"]
    else:
        sc_spine, sc_len = _layer_spine(sc_clean, clean=sc_clean, dt=sc_dt, skel=sc_skel)
        gl_spine, gl_len = _layer_spine(gl_clean, clean=gl_clean, dt=gl_dt, skel=gl_skel)
    sc_stats = layer_stats(sc_clean, um_per_px, centerline_px=sc_len)
    gl_stats = layer_stats(gl_clean, um_per_px, centerline_px=gl_len)
    gl_tortuosity = granular_layer_tortuosity(gl_bin, sc_bin)
    sc_holes = class_hole_count(sc_bin)   # on the original masks: hole-filling must not hide defects
    gl_holes = class_hole_count(gl_bin)

    panel = draw_measurement_panel(rotated_img, sc_clean, gl_clean, sc_stats, gl_stats,
                                   gl_tortuosity, sc_holes, gl_holes, um_per_px, sides,
                                   spines={"SC": sc_spine, "Epidermal": gl_spine})

    row = {"date": date}
    for prefix, stats in (("SC", sc_stats), ("Epidermal", gl_stats)):
        row[f"{prefix} mean height"] = round(stats["mean_height"], 2)
        row[f"{prefix} area"] = round(stats["area"], 2)
    row["EDJ tortuosity"] = round(gl_tortuosity, 2)
    row["SC_hole_num"] = sc_holes
    row["Epidermal_hole_num"] = gl_holes
    spine_cache = {"spines": {"SC": sc_spine, "Epidermal": gl_spine},
                   "lengths_px": {"SC": sc_len, "Epidermal": gl_len}}
    return row, panel, spine_cache


# ----------------------------------------------------------------------------
# command line: run on a histoseg_to_yolo.py dataset or a predictions folder
# ----------------------------------------------------------------------------


_NAME_ALIASES = {"SC": "SC", "keratin": "SC",
                "Epidermal": "Epidermal", "Granular_Layer": "Epidermal", "epidermis": "Epidermal"}




def _rasterise(label_path, h, w):
   mask = np.zeros((h, w), np.uint8)
   if not label_path.exists():
       return mask
   for line in label_path.read_text().splitlines():
       p = line.split()
       if len(p) < 7:
           continue
       pts = np.array(p[1:], np.float32).reshape(-1, 2) * [w, h]
       cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], int(p[0]) + 1)
   return mask




def _file_date(path) -> str:
   """Source image's last-modified date (YYYY-MM-DD) - the closest thing to an
   acquisition date most of these files carry (see measurement.py's date column)."""
   from datetime import datetime
   return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d") if path else ""


def _iter_inputs(args):
   """Yield (folder_name, file_name, image_bgr, sc_mask, gl_mask, date_str)."""
   from pathlib import Path
   import yaml
   if args.data:
       data = Path(args.data)
       y = yaml.safe_load((data / "data.yaml").read_text())
       names = y["names"]
       names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
       ids = {_NAME_ALIASES.get(n): i + 1 for i, n in enumerate(names) if n in _NAME_ALIASES}
       for split in ([args.split] if args.split != "all" else ["train", "val"]):
           for ip in sorted((data / "images" / split).glob("*.jpg")):
               img = cv2.imread(str(ip))
               if img is None:
                   continue
               m = _rasterise(data / "labels" / split / (ip.stem + ".txt"), *img.shape[:2])
               yield split, ip.name, img, m == ids["SC"], m == ids["Epidermal"], _file_date(ip)
   else:
       preds = Path(args.preds)
       images = Path(args.images) if args.images else None
       names = args.names.split(",")
       ids = {_NAME_ALIASES.get(n): i + 1 for i, n in enumerate(names) if n in _NAME_ALIASES}
       for mp in sorted(preds.glob("*_mask.png")):
           stem = mp.name[: -len("_mask.png")]
           m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
           img, img_path = None, None
           if images is not None:
               for ext in (".jpg", ".png", ".jpeg", ".tif"):
                   cand = images / (stem + ext)
                   if cand.exists():
                       img, img_path = cv2.imread(str(cand)), cand
                       break
           if img is None:
               img = np.full((*m.shape, 3), 255, np.uint8)
           yield preds.name, stem + ".jpg", img, m == ids["SC"], m == ids["Epidermal"], _file_date(img_path)




def main():
   import argparse
   import csv
   from pathlib import Path
   ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
   src = ap.add_mutually_exclusive_group(required=True)
   src.add_argument("--data", help="dataset folder from histoseg_to_yolo.py")
   src.add_argument("--preds", help="folder of *_mask.png from train_semseg.py predict")
   ap.add_argument("--split", default="val", help="train | val | all (with --data)")
   ap.add_argument("--images", help="images the predictions were made on (with --preds)")
   ap.add_argument("--names", default="SC,Granular_Layer,glands,follicles",
                   help="class order for --preds masks (value k = class k-1)")
   ap.add_argument("--um-per-px", type=float, default=1.0,
                   help="micrometres per pixel of the INPUT image; 1.0 = report in pixels")
   ap.add_argument("--out", default="measure_out")
   ap.add_argument("--no-panels", action="store_true")
   args = ap.parse_args()


   out = Path(args.out)
   (out / "panels").mkdir(parents=True, exist_ok=True)
   rows = []
   for folder, fname, img, sc, gl, date_str in _iter_inputs(args):
       row, panel, _ = measure_layers(img, sc, gl, args.um_per_px, date=date_str)
       row = {"folder name": folder, "file name": fname, **row}
       rows.append(row)
       if not args.no_panels:
           cv2.imwrite(str(out / "panels" / (Path(fname).stem + ".jpg")), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
       print(f"{fname}: SC mean {row['SC mean height']:.1f}  GL mean {row['Epidermal mean height']:.1f}")
   with (out / "measurements.csv").open("w", newline="") as f:
       wr = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
       wr.writeheader()
       wr.writerows(rows)
   unit = "um" if args.um_per_px != 1.0 else "px"
   print(f"\n{len(rows)} images -> {out / 'measurements.csv'} (heights in {unit})")
   for prefix in ("SC", "Epidermal"):
       means = np.array([r[f"{prefix} mean height"] for r in rows if r[f"{prefix} mean height"] > 0])
       if means.size:
           print(f"  {prefix:15s} mean height: median {np.median(means):.1f} {unit}"
                 f"  ({means.size} measurable images)")

if __name__ == "__main__":
   main()





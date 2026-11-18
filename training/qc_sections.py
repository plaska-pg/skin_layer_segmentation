"""Torch-free ROI section geometry and quality-tile preparation, shared by training (annotate_qc_boxes,
train_qual) and inference (postprocess, and the browser build in web/)."""
import cv2
import numpy as np

ROI_CLASS = "roi"
RATING_COLORS = {1: (0, 0, 255), 2: (0, 220, 255), 3: (0, 200, 0)}  # BGR: red, yellow, green


def quad_pts(box):
    return np.array(box[1:9], np.float32).reshape(4, 2)


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


def even_sections(n):
    return {"cuts": [k / n for k in range(n + 1)], "ratings": [None] * n}


def warp_section(img, quad, size, jitter=0.0, rng=None, letterbox=False):
    """Section quad (top-left, top-right, bottom-right, bottom-left) -> upright size x size BGR tile."""
    q = np.asarray(quad, np.float32)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    if jitter:
        q = q + rng.uniform(-jitter, jitter, (4, 2)).astype(np.float32) * min(w, h)
    if not letterbox:
        m = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [size, 0], [size, size], [0, size]]))
        # replicate edges so no fill colour is invented where a jittered quad leaves the image
        return cv2.warpPerspective(img, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # letterbox on white: only for checkpoints trained before stretching
    s = size / max(w, h, 1.0)
    dw, dh = max(1, round(w * s)), max(1, round(h * s))
    m = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]]))
    tile = np.full((size, size, 3), 255, np.uint8)
    x0, y0 = (size - dw) // 2, (size - dh) // 2
    tile[y0:y0 + dh, x0:x0 + dw] = cv2.warpPerspective(img, m, (dw, dh), flags=cv2.INTER_LINEAR,
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return tile


def to_gray(img):
    # kept as 3 identical channels so the ImageNet-pretrained first layer still applies
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def prepare(img, mode):
    return to_gray(img) if mode == "gray" else img


def mode_of(ckpt):
    return ckpt.get("mode") or ("gray" if ckpt.get("gray") else "color")


def score_color(score):
    """Piecewise-linear BGR colour from rating 1 (red) through 2 (yellow) to 3 (green)."""
    t = float(np.clip(score, 1, 3))
    lo = min(int(t), 2)
    c0, c1 = np.array(RATING_COLORS[lo], np.float32), np.array(RATING_COLORS[lo + 1], np.float32)
    return tuple(int(v) for v in c0 + (c1 - c0) * (t - lo))


def section_tiles(img_bgr, rois, n_sections: int, max_side: int, imgsz: int, mode: str, letterbox: bool):
    """(tiles, quads) for quality scoring: each ROI split into n_sections even sections, tiles cut from
    img_bgr downscaled to max_side; quads in img_bgr's pixel coordinates."""
    h, w = img_bgr.shape[:2]
    f = min(1.0, max_side / max(h, w))
    small = cv2.resize(img_bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else img_bgr
    src = prepare(small, mode)
    cuts = even_sections(n_sections)["cuts"]
    quads = [section_quad([ROI_CLASS, *np.asarray(quad, np.float32).ravel()], cuts[k], cuts[k + 1])
             for quad, _ in rois for k in range(n_sections)]
    return [warp_section(src, q * f, imgsz, letterbox=letterbox) for q in quads], quads

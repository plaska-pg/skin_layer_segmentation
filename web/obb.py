"""Torch-free port of Ultralytics' YOLO-OBB predict pre/postprocessing (8.4.x, rect=True as in model.predict)
for running exported ONNX detectors in the browser. Mirrors postprocess.detect_obb's downscale."""
import cv2
import numpy as np

STRIDE = 32


def prepare(img_bgr: np.ndarray, max_side: int, imgsz: int):
    """(input tensor (1, 3, H, W) float32 RGB 0-1, small image shape, downscale f) as detect_obb + LetterBox(auto=True)."""
    h, w = img_bgr.shape[:2]
    f = min(1.0, max_side / max(h, w))
    small = cv2.resize(img_bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else img_bgr
    sh, sw = small.shape[:2]
    r = min(imgsz / sh, imgsz / sw)
    nw, nh = round(sw * r), round(sh * r)
    dw, dh = (imgsz - nw) % STRIDE / 2, (imgsz - nh) % STRIDE / 2
    img = cv2.resize(small, (nw, nh), interpolation=cv2.INTER_LINEAR) if (sw, sh) != (nw, nh) else small
    img = cv2.copyMakeBorder(img, round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1),
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return x, (sh, sw), f


def _covariance(b):
    a, bb = b[:, 2] ** 2 / 12, b[:, 3] ** 2 / 12
    cos, sin = np.cos(b[:, 4]), np.sin(b[:, 4])
    return a * cos ** 2 + bb * sin ** 2, a * sin ** 2 + bb * cos ** 2, (a - bb) * cos * sin


def batch_probiou(b1, b2, eps=1e-7):
    x1, y1 = b1[:, 0:1], b1[:, 1:2]
    x2, y2 = b2[None, :, 0], b2[None, :, 1]
    a1, bb1, c1 = (v[:, None] for v in _covariance(b1))
    a2, bb2, c2 = (v[None] for v in _covariance(b2))
    den = (a1 + a2) * (bb1 + bb2) - (c1 + c2) ** 2
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (bb1 + bb2) * (x1 - x2) ** 2) / (den + eps) * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / (den + eps) * 0.5
    t3 = np.log(den / (4 * np.sqrt(np.clip(a1 * bb1 - c1 ** 2, 0, None) * np.clip(a2 * bb2 - c2 ** 2, 0, None)) + eps)
                + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    return 1 - np.sqrt(1.0 - np.exp(-bd) + eps)


def xywhr2xyxyxyxy(b):
    ctr = b[:, :2]
    cos, sin = np.cos(b[:, 4:5]), np.sin(b[:, 4:5])
    v1 = np.concatenate([b[:, 2:3] / 2 * cos, b[:, 2:3] / 2 * sin], 1)
    v2 = np.concatenate([-b[:, 3:4] / 2 * sin, b[:, 3:4] / 2 * cos], 1)
    return np.stack([ctr + v1 + v2, ctr + v1 - v2, ctr - v1 - v2, ctr - v1 + v2], -2)


def postprocess(pred: np.ndarray, in_shape, small_shape, f: float, conf: float, nc: int,
                iou: float = 0.7, max_det: int = 300, max_wh: int = 7680):
    """Raw output (1, 4 + nc + 1, N) -> [(quad 4x2 in the detect_obb input image's px, conf)]."""
    x = pred[0].T.astype(np.float64)
    scores_all = x[:, 4:4 + nc]
    x, scores_all = x[scores_all.max(1) > conf], scores_all[scores_all.max(1) > conf]
    if not len(x):
        return []
    score, cls = scores_all.max(1), scores_all.argmax(1)
    if len(x) > 30000:
        top = np.argsort(-score, kind="stable")[:30000]
        x, score, cls = x[top], score[top], cls[top]
    boxes = np.concatenate([x[:, :2] + (cls * max_wh)[:, None], x[:, 2:4], x[:, -1:]], 1)
    order = np.argsort(-score, kind="stable")
    ious = np.triu(batch_probiou(boxes[order], boxes[order]), k=1)
    keep = order[(ious > iou).sum(0) <= 0][:max_det]
    b = np.concatenate([x[keep, :4], x[keep, -1:]], 1)
    ih, iw = in_shape
    sh, sw = small_shape
    gain = min(ih / sh, iw / sw)
    nh, nw = round(sh * gain), round(sw * gain)
    gy, gx = nh / sh, nw / sw
    b[:, 0] = (b[:, 0] - round((iw - nw) / 2 - 0.1)) / gx
    b[:, 1] = (b[:, 1] - round((ih - nh) / 2 - 0.1)) / gy
    b[:, 2] /= gx
    b[:, 3] /= gy
    quads = (xywhr2xyxyxyxy(b) / f).astype(np.float32)
    return list(zip(quads, score[keep].tolist()))

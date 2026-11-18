"""Export the semseg ResUNet, rotation classifier, ROI / follicle detectors and quality model to ONNX for the
browser (GitHub Pages) build.

Segmentation graph input "image": RGB float32 0-255, shape (1, 3, H, W), H and W multiples of 32.
Segmentation graph output "probs": softmax probabilities (1, C, H, W), class 0 = background.
Rotation graph input "image": RGB float32 0-255, (1, 3, size, size); output "logits" (1, 4).
Writes <out>.onnx, <out-dir>/rot.onnx and <out>.json (classes, scale, tiling, rotation) and
checks ONNX Runtime against PyTorch on the example image.

  python web/export_onnx.py [--weights models/3cls_best_v8_e.pt] [--out runs/web/seg]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CONFIG, repo_path  # noqa: E402
from predict import TILE_OVERLAP, load_model  # noqa: E402
from training.semseg.train_semseg import MEAN, STD  # noqa: E402

PRED = CONFIG.get("predict", {})
POST = CONFIG.get("postprocess", {})


class BrowserModel(torch.nn.Module):
    def __init__(self, model, softmax: bool = True):
        super().__init__()
        self.model = model
        self.softmax = softmax
        self.register_buffer("mean", torch.from_numpy(MEAN * 255).view(1, 3, 1, 1))
        self.register_buffer("std", torch.from_numpy(STD * 255).view(1, 3, 1, 1))

    def forward(self, image):
        out = self.model((image - self.mean) / self.std)
        return out.softmax(1) if self.softmax else out


def export(wrapped, path: Path, tile: int):
    h, w = torch.export.Dim("h", min=1, max=256), torch.export.Dim("w", min=1, max=256)
    torch.onnx.export(wrapped, (torch.rand(1, 3, tile, tile) * 255,), str(path), dynamo=True,
                      input_names=["image"], output_names=["probs"], external_data=False,
                      dynamic_shapes={"image": {2: 32 * h, 3: 32 * w}})


def reference_tile(image: Path, um_per_px: float, ref_um_per_px: float, tile: int) -> np.ndarray:
    """RGB uint8 crop at the model's um/px, reflect-padded to a multiple of 32, as predict.py feeds it."""
    bgr = cv2.imread(str(image))
    s = um_per_px / ref_um_per_px
    bgr = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    patch = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[:tile, :tile]
    ph, pw = patch.shape[:2]
    return cv2.copyMakeBorder(patch, 0, (32 - ph % 32) % 32, 0, (32 - pw % 32) % 32, cv2.BORDER_REFLECT)


def check(model, onnx_path: Path, patch: np.ndarray) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        x = torch.from_numpy(((patch.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None]
        want = model(x).softmax(1).numpy()
    got = sess.run(None, {"image": patch.astype(np.float32).transpose(2, 0, 1)[None]})[0]
    return {"shape": list(got.shape), "max_abs_prob_diff": float(np.abs(got - want).max()),
            "argmax_agreement": float((got.argmax(1) == want.argmax(1)).mean())}


def export_rotation(weights: Path, path: Path, image: Path) -> dict:
    """rot.onnx from the rotate.py classifier; parity = predicted class on all four turns of `image`."""
    import onnxruntime as ort
    from training.rotation.rotate import build_model, predict_orientation, rotate_k
    ckpt = torch.load(weights, map_location="cpu")
    model = build_model(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    size = ckpt.get("size", 224)
    torch.onnx.export(BrowserModel(model, softmax=False).eval(), (torch.rand(1, 3, size, size) * 255,), str(path),
                      dynamo=True, input_names=["image"], output_names=["logits"], external_data=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    bgr, preds = cv2.imread(str(image)), []
    for k in range(4):
        img = rotate_k(bgr, k)
        square = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        got = sess.run(None, {"image": square.astype(np.float32).transpose(2, 0, 1)[None]})[0]
        preds.append([predict_orientation(model, img, size, torch.device("cpu")), int(got.argmax())])
    return {"model": path.name, "source_weights": weights.name, "size": size,
            "source_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "input": {"name": "image", "layout": "NCHW", "channels": "RGB", "range": [0, 255],
                      "resize": "INTER_AREA whole image"},
            "output": {"name": "logits", "classes_deg_cw": list(ckpt.get("angles", (0, 90, 180, 270)))},
            "parity": {"torch_vs_onnx_class_per_turn": preds, "agree": all(a == b for a, b in preds)}}


def export_obb(name: str, weights: Path, path: Path, image: Path, k: int) -> dict:
    """<name>.onnx from a YOLO-OBB detector (dynamic input, raw head output; web/obb.py does letterbox + NMS).
    Parity: web/obb.py + ONNX Runtime vs postprocess.detect_obb on `image` turned upright by k."""
    import shutil
    import onnxruntime as ort
    from ultralytics import YOLO
    import postprocess
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import obb
    exported = Path(YOLO(str(weights)).export(format="onnx", dynamic=True, imgsz=POST[f"{name}_imgsz"],
                                              simplify=False, nms=False, verbose=False))
    shutil.move(exported, path)
    conf, max_side, imgsz = POST[f"{name}_conf"], POST[f"{name}_max_side"], POST[f"{name}_imgsz"]
    img = postprocess.rotate_k(cv2.imread(str(image)), k)
    want = postprocess.detect_obb(img, str(weights), "cpu", conf, max_side, imgsz, name=name)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x, small_shape, f = obb.prepare(img, max_side, imgsz)
    raw = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    got = obb.postprocess(raw, x.shape[2:], small_shape, f, conf, nc=len(YOLO(str(weights)).names))
    corner_err = max((float(np.abs(np.asarray(a[0]) - np.asarray(b[0])).max()) for a, b in zip(want, got)), default=0.0)
    return {"model": path.name, "source_weights": weights.name, "conf": conf, "max_side": max_side, "imgsz": imgsz,
            "input": sess.get_inputs()[0].name, "nc": len(YOLO(str(weights)).names),
            "source_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "parity": {"boxes_ultralytics": len(want), "boxes_onnx": len(got), "max_corner_err_px": corner_err,
                       "max_conf_err": max((abs(a[1] - b[1]) for a, b in zip(want, got)), default=0.0)}}


def export_quality(weights: Path, path: Path, image: Path, k: int) -> dict:
    """quality.onnx: RGB 0-255 tiles (N, 3, imgsz, imgsz) -> expected rating (N,), with predict_tiles'
    flip averaging and decode inside the graph. Parity vs predict_tiles on the ROI sections of `image`."""
    import onnxruntime as ort
    import postprocess
    from training.qc_sections import mode_of, section_tiles
    from training.quality.train_qual import load_model as load_qual, predict_tiles

    class Scored(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.register_buffer("mean", torch.from_numpy(MEAN * 255).view(1, 3, 1, 1))
            self.register_buffer("std", torch.from_numpy(STD * 255).view(1, 3, 1, 1))

        def forward(self, image):
            x = (image - self.mean) / self.std
            s = torch.sigmoid((self.model(x) + self.model(x.flip(-1)) + self.model(x.flip(-2))) / 3)
            return 1 + s[:, 0] + torch.minimum(s[:, 0], s[:, 1])

    model, ckpt = load_qual(str(weights), "cpu")
    size = ckpt["imgsz"]
    n = torch.export.Dim("n", min=1, max=1024)
    torch.onnx.export(Scored(model).eval(), (torch.rand(2, 3, size, size) * 255,), str(path), dynamo=True,
                      input_names=["image"], output_names=["score"], external_data=False,
                      dynamic_shapes={"image": {0: n}})
    img = postprocess.rotate_k(cv2.imread(str(image)), k)
    rois = postprocess.detect_obb(img, POST["roi_weights"], "cpu", POST["roi_conf"], POST["roi_max_side"],
                                  POST["roi_imgsz"], name="ROI")
    tiles, _ = section_tiles(img, rois, POST.get("quality_sections", 5), ckpt["max_side"], size, mode_of(ckpt),
                             ckpt.get("letterbox", True))
    want = predict_tiles(model, tiles, "cpu")[0]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.stack([t[..., ::-1] for t in tiles]).astype(np.float32).transpose(0, 3, 1, 2)
    got = sess.run(None, {"image": np.ascontiguousarray(x)})[0]
    return {"model": path.name, "source_weights": weights.name, "input": "image",
            "source_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "imgsz": size, "max_side": ckpt["max_side"], "mode": mode_of(ckpt), "letterbox": ckpt.get("letterbox", True),
            "parity": {"tiles": len(tiles), "max_abs_score_diff": float(np.abs(got - want).max()) if len(tiles) else 0.0}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # torch.onnx prints emoji; cp1252 consoles crash on them
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=PRED.get("weights", "models/3cls_best_v8_e.pt"))
    ap.add_argument("--out", default="runs/web/seg", help="output path without extension")
    ap.add_argument("--tile", type=int, default=PRED.get("tile", 1024))
    ap.add_argument("--image", default="examples/s_ex_9_24_d2_l4c4.jpg", help="image for the parity check")
    ap.add_argument("--um-per-px", type=float, default=PRED.get("um_per_px", 0.2577), help="of --image")
    ap.add_argument("--rotation-weights", default=POST.get("rotation_weights", "models/rotate_model.pt"))
    args = ap.parse_args()

    weights = Path(repo_path(args.weights))
    out = Path(repo_path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = out.with_suffix(".onnx")
    ref_um_per_px = PRED.get("reference_um_per_px", 0.645)

    model, names, _ = load_model(str(weights), torch.device("cpu"))
    export(BrowserModel(model).eval(), onnx_path, args.tile)

    patch = reference_tile(Path(repo_path(args.image)), args.um_per_px, ref_um_per_px, args.tile)
    parity = {"tile": check(model, onnx_path, patch),
              "small_tile": check(model, onnx_path, patch[:320, :480])}
    manifest = {
        "model": onnx_path.name,
        "source_weights": weights.name,
        "source_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "classes": ["background", *names],
        "input": {"name": "image", "layout": "NCHW", "channels": "RGB", "range": [0, 255], "multiple_of": 32,
                  "pad_mode": "reflect"},
        "output": {"name": "probs", "activation": "softmax"},
        "reference_um_per_px": ref_um_per_px,
        "default_um_per_px": PRED.get("um_per_px", 0.2577),
        "tile": args.tile,
        "tile_overlap": TILE_OVERLAP,
        "parity": parity,
        "rotation": export_rotation(Path(repo_path(args.rotation_weights)), out.parent / "rot.onnx",
                                    Path(repo_path(args.image))),
    }
    for name in ("roi", "follicle"):
        manifest[name] = export_obb(name, Path(repo_path(POST[f"{name}_weights"])), out.parent / f"{name}.onnx",
                                    Path(repo_path(args.image)), k=3)
    manifest["quality"] = export_quality(Path(repo_path(POST["quality_weights"])), out.parent / "quality.onnx",
                                         Path(repo_path(args.image)), k=3)
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 2**20:.1f} MiB)")
    print(json.dumps({"seg": parity, **{n: manifest[n]["parity"] for n in ("rotation", "roi", "follicle", "quality")}},
                     indent=2))


if __name__ == "__main__":
    main()

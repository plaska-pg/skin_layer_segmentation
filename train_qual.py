#!/usr/bin/env python3
"""
train_qual.py -- score the quality of ROI sections (1 worst - 3 best) with an ordinal ResNet.

    python annotate_boxes.py                   # rate sections (r = rate mode) -> roi_follicle_bb_data/sections.csv
    python train_qual.py train --name cv1      # 5-fold CV grouped by slide + final fit -> runs/qual_cls/cv1/
    python train_qual.py predict --weights runs/qual_cls/cv1/final/best.pt \\
           --source roi_follicle_bb_data/images --rois roi_follicle_bb_data/boxes.csv
    python train_qual.py predict --weights runs/qual_cls/cv1/final/best.pt \\
           --source <slides> --rois runs/roi_det/predict/detections.csv --min-conf 0.25

Training reads the rated rows of sections.csv and cuts each section out of its --source image on the
fly (perspective warp stretched to fill an imgsz x imgsz tile, so every pixel comes from the section;
no padding). Images are
downscaled to --max-side first, the same as at predict time. --mode gray converts to grayscale
(default color); the checkpoint remembers the mode so predict converts the same way.
<run>/train_examples/rating_{1,2,3}.jpg show augmented tiles as the model sees them.

The head is ordinal (CORAL): two logits for P(rating > 1) and P(rating > 2) sharing one weight
vector; score = 1 + P(>1) + P(>2) in [1, 3] and the predicted rating counts thresholds passed.

--rois for predict may be boxes.csv (rows of class roi, split into --sections even sections),
detections.csv from train_det.py predict (also filtered by --min-conf), or sections.csv (uses the
annotated dividers). Coordinates are in the pixel space of the --source images. Images with no ROIs
there (or no --rois file at all) get ROIs from the train_det.py model in --roi-weights (saved to
<out>/detections.csv); images where it finds none are skipped:
    python train_qual.py predict --weights runs/qual_cls/cv1/final/best.pt --source <slides>
"""

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from annotate_boxes import (CSV_HEADER, RATING_COLORS, ROI_CLASS, even_sections, list_images, load_excluded,
                            load_scaled, section_quad, slide_key)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RATINGS = (1, 2, 3)
QUAD_COLS = CSV_HEADER[2:]


# ---------------------------------------------------------------------------- data
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


def augment(tile, rng, hsv_h=0.011, hsv_s=0.3, hsv_v=0.2, blur_p=0.2):
    """Flips, then hue shift +/- hsv_h (fraction of the hue circle), saturation / brightness x (1 +/- hsv_s / hsv_v)."""
    if rng.random() < 0.5:
        tile = tile[:, ::-1]
    if rng.random() < 0.5:
        tile = tile[::-1]
    hsv = cv2.cvtColor(np.ascontiguousarray(tile), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hsv_h, hsv_h) * 180) % 180
    hsv[..., 1] *= rng.uniform(1 - hsv_s, 1 + hsv_s)
    hsv[..., 2] *= rng.uniform(1 - hsv_v, 1 + hsv_v)
    tile = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < blur_p:
        tile = cv2.GaussianBlur(tile, (0, 0), rng.uniform(0.3, 1.2))
    return tile


def to_tensor(tiles):
    x = np.stack([cv2.cvtColor(np.ascontiguousarray(t), cv2.COLOR_BGR2RGB) for t in tiles]).astype(np.float32) / 255
    return torch.from_numpy(np.ascontiguousarray(((x - MEAN) / STD).transpose(0, 3, 1, 2)))


def to_gray(img):
    # kept as 3 identical channels so the ImageNet-pretrained first layer still applies
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def prepare(img, mode):
    return to_gray(img) if mode == "gray" else img


def mode_of(ckpt):
    return ckpt.get("mode") or ("gray" if ckpt.get("gray") else "color")


def load_rated(a):
    """Rated sections (quads scaled to --max-side) and the downscaled images they come from."""
    excluded = load_excluded(a.exclude)
    with open(a.sections_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["rating"] and r["image"] not in excluded]
    samples, images, scale = [], {}, {}
    for r in rows:
        name = r["image"]
        if name not in images:
            p = Path(a.source) / name
            if not p.exists():
                print(f"{p} not found; its sections are skipped", flush=True)
                images[name] = None
                continue
            images[name], _, _, scale[name] = load_scaled(p, a.max_side)
            images[name] = prepare(images[name], a.mode)
        if images[name] is None:
            continue
        quad = np.array([float(r[k]) for k in QUAD_COLS], np.float32).reshape(4, 2) * scale[name]
        samples.append({"image": name, "roi_idx": int(r["roi_idx"]), "section": int(r["section"]),
                        "rating": int(r["rating"]), "quad": quad, "group": slide_key(Path(name).stem)})
    if not samples:
        raise SystemExit(f"no rated sections in {a.sections_csv}; rate some with: python annotate_boxes.py (key r)")
    return samples, {k: v for k, v in images.items() if v is not None}


def aug_args(a):
    return {"hsv_h": a.hsv_h, "hsv_s": a.hsv_s, "hsv_v": a.hsv_v, "blur_p": a.blur_p}


class SectionDataset(Dataset):
    def __init__(self, samples, images, imgsz, jitter, aug):
        self.samples, self.images, self.imgsz, self.jitter, self.aug = samples, images, imgsz, jitter, aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        rng = np.random.default_rng()
        tile = warp_section(self.images[s["image"]], s["quad"], self.imgsz, self.jitter, rng)
        return to_tensor([augment(tile, rng, **self.aug)])[0], s["rating"]


def save_examples(samples, images, a, out: Path, n=16, cols=4):
    """rating_{1,2,3}.jpg: grids of augmented training tiles exactly as the model sees them."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for r in RATINGS:
        pool = [s for s in samples if s["rating"] == r]
        if not pool:
            continue
        tiles = []
        for i in rng.choice(len(pool), min(n, len(pool)), replace=False):
            s = pool[i]
            t = warp_section(images[s["image"]], s["quad"], a.imgsz, a.jitter, rng)
            t = np.ascontiguousarray(augment(t, rng, **aug_args(a)))
            cv2.rectangle(t, (0, 0), (a.imgsz - 1, a.imgsz - 1), RATING_COLORS[r], 3)
            label = f"{s['group'][:24]} r{s['roi_idx']}s{s['section']}"
            cv2.putText(t, label, (6, a.imgsz - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(t, label, (6, a.imgsz - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        tiles += [np.full_like(tiles[0], 40)] * (-len(tiles) % cols)
        grid = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
        cv2.imwrite(str(out / f"rating_{r}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"example training tiles -> {out}", flush=True)


def make_folds(samples, k, seed):
    """{slide group: fold}; greedy so every fold gets a similar mix of ratings."""
    groups = defaultdict(lambda: np.zeros(len(RATINGS)))
    for s in samples:
        groups[s["group"]][s["rating"] - 1] += 1
    if k > len(groups):
        raise SystemExit(f"--folds {k} but only {len(groups)} slides have rated sections")
    total = np.maximum(sum(groups.values()), 1)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda g: -groups[g].sum())
    loads = [np.zeros(len(RATINGS)) for _ in range(k)]
    fold_of = {}
    for g in names:
        f = min(range(k), key=lambda j: float((((loads[j] + groups[g]) / total) ** 2 - (loads[j] / total) ** 2).sum()))
        loads[f] += groups[g]
        fold_of[g] = f
    return fold_of


# ---------------------------------------------------------------------------- model
class QualNet(nn.Module):
    def __init__(self, arch="resnet18", pretrained=True, dropout=0.3):
        super().__init__()
        import torchvision
        r = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        c = r.fc.in_features
        r.fc = nn.Identity()
        self.backbone = r
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return self.fc(self.drop(self.backbone(x))) + self.bias


def load_hist_encoder(model, path, arch):
    """Copy the ResNet encoder of a train_semseg.py ResUNet checkpoint into the backbone."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("encoder", arch) != arch:
        raise SystemExit(f"{path} has encoder {ckpt['encoder']}; use --arch {ckpt['encoder']}")
    rename = {"stem.0.": "conv1.", "stem.1.": "bn1.", "e1.": "layer1.", "e2.": "layer2.", "e3.": "layer3.", "e4.": "layer4."}
    sd = {}
    for k, v in ckpt["model"].items():
        for old, new in rename.items():
            if k.startswith(old):
                sd[new + k[len(old):]] = v
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    if not sd or unexpected or missing:
        raise SystemExit(f"{path}: encoder weights do not match {arch} (missing {missing[:3]}, unexpected {unexpected[:3]})")
    print(f"backbone initialised from {path}", flush=True)


def build_model(a):
    model = QualNet(a.arch, pretrained=a.init == "imagenet", dropout=a.dropout)
    if a.init == "hist":
        load_hist_encoder(model, a.hist_weights, a.arch)
    return model


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = QualNet(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def decode(logits):
    """Cumulative logits -> (score in [1, 3], predicted rating, [p1, p2, p3])."""
    s = torch.sigmoid(logits)
    s1, s2 = s[:, 0], torch.minimum(s[:, 0], s[:, 1])
    probs = torch.stack([1 - s1, s1 - s2, s2], 1)
    return (1 + s1 + s2).numpy(), (1 + (s1 > 0.5).int() + (s2 > 0.5).int()).numpy(), probs.numpy()


@torch.no_grad()
def predict_tiles(model, tiles, device, batch=32):
    model.eval()
    out = []
    for i in range(0, len(tiles), batch):
        x = to_tensor(tiles[i:i + batch]).to(device)
        out.append(((model(x) + model(x.flip(-1)) + model(x.flip(-2))) / 3).float().cpu())
    return decode(torch.cat(out) if out else torch.zeros(0, 2))


# ---------------------------------------------------------------------------- metrics
def qwk(cm):
    n = cm.sum()
    if not n:
        return 0.0
    k = len(cm)
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], np.float64) / (k - 1) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / n
    denom = (w * expected).sum()
    return float(1 - (w * cm).sum() / denom) if denom else 0.0


def metrics(y, score, pred):
    y, score, pred = np.asarray(y), np.asarray(score, np.float64), np.asarray(pred)
    cm = np.zeros((len(RATINGS), len(RATINGS)), np.int64)
    for t, p in zip(y, pred):
        cm[t - 1, p - 1] += 1
    return {"n": int(len(y)), "mae": round(float(np.abs(score - y).mean()), 4), "qwk": round(qwk(cm), 4),
            "acc": round(float((y == pred).mean()), 4),
            "recall": {str(r): round(float(cm[r - 1, r - 1] / max(1, cm[r - 1].sum())), 4) for r in RATINGS},
            "confusion": cm.tolist()}


def fmt(m):
    rec = " ".join(f"r{k}={v:.2f}" for k, v in m["recall"].items())
    return f"MAE {m['mae']:.3f}  QWK {m['qwk']:.3f}  acc {m['acc']:.3f}  recall {rec}"


# ---------------------------------------------------------------------------- train
def device_of(name):
    if name and name.isdigit():
        name = f"cuda:{name}"
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def fit(train_s, val_s, images, a, out: Path, epochs, device):
    """Train one model; with val_s keep the best epoch by (QWK, -MAE), else keep the last. Returns best epoch."""
    out.mkdir(parents=True)
    model = build_model(a).to(device)
    counts = Counter(s["rating"] for s in train_s)
    sampler = WeightedRandomSampler([1.0 / counts[s["rating"]] for s in train_s], len(train_s), replacement=True)
    dl = DataLoader(SectionDataset(train_s, images, a.imgsz, a.jitter, aug_args(a)), batch_size=a.batch, sampler=sampler,
                    num_workers=a.workers, drop_last=len(train_s) > a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
    val_tiles = [warp_section(images[s["image"]], s["quad"], a.imgsz) for s in val_s]
    val_y = [s["rating"] for s in val_s]
    best_key, best_epoch, bad = None, epochs, 0
    log = open(out / "metrics.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "lr", "train_loss", "val_mae", "val_qwk", "val_acc"])
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            target = torch.stack([y > 1, y > 2], 1).float()
            loss = F.binary_cross_entropy_with_logits(model(x), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(y), n + len(y)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        ckpt = {"arch": a.arch, "imgsz": a.imgsz, "max_side": a.max_side, "mode": a.mode, "letterbox": False,
                "epoch": epoch}
        if not val_s:
            w.writerow([epoch, f"{lr:.2e}", round(total / n, 4), "", "", ""])
            print(f"epoch {epoch:3d}  loss {total / n:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            continue
        score, pred, _ = predict_tiles(model, val_tiles, device, a.batch)
        m = metrics(val_y, score, pred)
        w.writerow([epoch, f"{lr:.2e}", round(total / n, 4), m["mae"], m["qwk"], m["acc"]])
        log.flush()
        key = (m["qwk"], -m["mae"])
        mark = ""
        if best_key is None or key > best_key:
            best_key, best_epoch, bad, mark = key, epoch, 0, "  *"
            torch.save({**ckpt, "model": model.state_dict(), "metrics": m}, out / "best.pt")
        else:
            bad += 1
        print(f"epoch {epoch:3d}  loss {total / n:.4f}  val {fmt(m)}  ({time.time() - t0:.0f}s){mark}", flush=True)
        if bad >= a.patience:
            print(f"no improvement for {a.patience} epochs; stopping", flush=True)
            break
    if not val_s:
        torch.save({**ckpt, "model": model.state_dict()}, out / "best.pt")
    log.close()
    return best_epoch


def cmd_train(a):
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = device_of(a.device)
    out = Path(a.project) / a.name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists; choose another --name")
    samples, images = load_rated(a)
    counts = Counter(s["rating"] for s in samples)
    print(f"{len(samples)} rated sections from {len({s['group'] for s in samples})} slides: "
          + "  ".join(f"rating {r}: {counts[r]}" for r in RATINGS) + f"   device {device}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "func"}, indent=2))
    save_examples(samples, images, a, out / "train_examples")

    best_epochs = []
    if a.folds:
        fold_of = make_folds(samples, a.folds, a.seed)
        oof = []
        fold_metrics = []
        for k in range(a.folds):
            tr = [s for s in samples if fold_of[s["group"]] != k]
            va = [s for s in samples if fold_of[s["group"]] == k]
            vc = Counter(s["rating"] for s in va)
            print(f"\n=== fold {k + 1}/{a.folds}: train {len(tr)}  val {len(va)} sections "
                  f"({' '.join(f'r{r}={vc[r]}' for r in RATINGS)}), val slides "
                  f"{sorted({s['group'] for s in va})}", flush=True)
            torch.manual_seed(a.seed + k)
            best_epochs.append(fit(tr, va, images, a, out / f"fold{k}", a.epochs, device))
            model, _ = load_model(out / f"fold{k}" / "best.pt", device)
            tiles = [warp_section(images[s["image"]], s["quad"], a.imgsz) for s in va]
            score, pred, probs = predict_tiles(model, tiles, device, a.batch)
            fold_metrics.append(metrics([s["rating"] for s in va], score, pred))
            print(f"fold {k + 1} best epoch {best_epochs[-1]}: {fmt(fold_metrics[-1])}", flush=True)
            oof += [(s, k, sc, pr, p) for s, sc, pr, p in zip(va, score, pred, probs)]

        with open(out / "oof_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "roi_idx", "section", "slide", "fold", "rating", "score", "pred", "p1", "p2", "p3"])
            for s, k, sc, pr, p in sorted(oof, key=lambda o: (o[0]["image"], o[0]["roi_idx"], o[0]["section"])):
                w.writerow([s["image"], s["roi_idx"], s["section"], s["group"], k, s["rating"], round(float(sc), 4),
                            int(pr), *(round(float(v), 4) for v in p)])
        y = np.array([o[0]["rating"] for o in oof])
        cv = metrics(y, [o[2] for o in oof], [o[3] for o in oof])
        majority = counts.most_common(1)[0][0]
        baseline = {"majority_rating": majority, "majority_acc": round(float((y == majority).mean()), 4),
                    "majority_mae": round(float(np.abs(y - majority).mean()), 4),
                    "mean_rating_mae": round(float(np.abs(y - y.mean()).mean()), 4)}
        summary = {"cv": cv, "baseline": baseline, "best_epochs": best_epochs, "folds": fold_metrics,
                   "fold_of_slide": fold_of}
        (out / "cv_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nout-of-fold ({cv['n']} sections): {fmt(cv)}")
        print(f"confusion (rows = rating 1-3, cols = predicted): {cv['confusion']}")
        print(f"baseline: always {majority} -> acc {baseline['majority_acc']:.3f} MAE {baseline['majority_mae']:.3f}; "
              f"always mean -> MAE {baseline['mean_rating_mae']:.3f}")
        print(f"wrote {out / 'oof_predictions.csv'} and {out / 'cv_summary.json'}")

    if not a.no_final:
        epochs = a.final_epochs or (round(statistics.median(best_epochs)) if best_epochs else a.epochs)
        print(f"\n=== final model on all {len(samples)} sections for {epochs} epochs", flush=True)
        torch.manual_seed(a.seed)
        fit(samples, [], images, a, out / "final", epochs, device)
        print(f"final weights: {out / 'final' / 'best.pt'}  ->  python train_qual.py predict --weights "
              f"{(out / 'final' / 'best.pt').as_posix()}")


# ---------------------------------------------------------------------------- predict
def read_rois(path: Path, n_sections, min_conf):
    """{image: [{"roi_idx", "conf", "quads": [section quads] | None, "box": roi box | None}]}"""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rois = defaultdict(list)
    if rows and "section" in rows[0]:
        by_roi = defaultdict(list)
        for r in rows:
            by_roi[r["image"], int(r["roi_idx"])].append(r)
        for (name, ri), rs in sorted(by_roi.items()):
            rs.sort(key=lambda r: int(r["section"]))
            quads = [np.array([float(r[k]) for k in QUAD_COLS], np.float32).reshape(4, 2) for r in rs]
            rois[name].append({"roi_idx": ri, "conf": "", "quads": quads, "box": None})
        return rois
    for r in rows:
        if r["class"] != ROI_CLASS or (r.get("conf") and float(r["conf"]) < min_conf):
            continue
        box = [ROI_CLASS, *(float(r[k]) for k in QUAD_COLS)]
        rois[r["image"]].append({"roi_idx": len(rois[r["image"]]), "conf": r.get("conf", ""), "quads": None, "box": box})
    for name, rs in rois.items():
        for roi in rs:
            cuts = even_sections(n_sections)["cuts"]
            roi["quads"] = [section_quad(roi["box"], cuts[k], cuts[k + 1]) for k in range(n_sections)]
    return rois


def detect_rois(a, paths, out: Path, cache):
    """Run the ROI detector on paths; writes <out>/detections.csv like train_det.py predict."""
    from ultralytics import YOLO

    det = YOLO(a.roi_weights)
    rows = []
    for p in paths:
        img, _, _, f = load_scaled(p, a.roi_max_side)
        cache[p.name, a.roi_max_side] = (img, f)
        r = det.predict(img, imgsz=a.roi_imgsz, conf=a.min_conf, device=a.device, verbose=False)[0]
        n = 0
        for quad, c, k in zip(r.obb.xyxyxyxy.tolist(), r.obb.conf.tolist(), r.obb.cls.tolist()):
            if r.names[int(k)] == ROI_CLASS:
                rows.append([p.name, ROI_CLASS, *(round(v / f) for pt in quad for v in pt), round(c, 3)])
                n += 1
        print(f"{p.name}: {n} ROIs detected", flush=True)
    path = out / "detections.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "class", *QUAD_COLS, "conf"])
        w.writerows(rows)
    return path


def score_color(score):
    """Piecewise-linear BGR colour from rating 1 (red) through 2 (yellow) to 3 (green)."""
    t = float(np.clip(score, 1, 3))
    lo = min(int(t), 2)
    c0, c1 = np.array(RATING_COLORS[lo], np.float32), np.array(RATING_COLORS[lo + 1], np.float32)
    return tuple(int(v) for v in c0 + (c1 - c0) * (t - lo))


def cmd_predict(a):
    device = device_of(a.device)
    model, ckpt = load_model(a.weights, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    rois = {}
    if a.rois and Path(a.rois).exists():
        rois = read_rois(Path(a.rois), a.sections, a.min_conf)
    elif a.rois:
        print(f"{a.rois} not found; detecting ROIs instead", flush=True)
    todo = [p for p in list_images(Path(a.source)) if not rois.get(p.name)]
    if todo:
        print(f"{len(todo)} images without ROIs; detecting with {a.roi_weights}", flush=True)
        rois.update(read_rois(detect_rois(a, todo, out, cache), a.sections, a.min_conf))
        skipped = [p.name for p in todo if not rois.get(p.name)]
        if skipped:
            print(f"no ROIs found, skipped: {skipped}", flush=True)
    rows = []
    for name, rs in sorted(rois.items()):
        p = Path(a.source) / name
        if not p.exists():
            print(f"{p} not found; skipped", flush=True)
            continue
        rows += score_image(model, ckpt, p, rs, out, device, a.batch, cache)
    write_scores(out / "scores.csv", rows)
    print(f"\nwrote {len(rows)} section scores (original-pixel coordinates) and overlays to {out}")


SCORES_HEADER = ["image", "roi_idx", "section", "score", "pred", "p1", "p2", "p3", *QUAD_COLS, "roi_conf"]


def write_scores(path: Path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCORES_HEADER)
        w.writerows(rows)


def score_image(model, ckpt, p: Path, rs, out: Path, device, batch, cache):
    """Score every section of the ROIs rs of image p, write <stem>_qual.jpg; rows as in SCORES_HEADER."""
    name = p.name
    if (name, ckpt["max_side"]) in cache:
        img, f = cache.pop((name, ckpt["max_side"]))
    else:
        img, _, _, f = load_scaled(p, ckpt["max_side"])
    src = prepare(img, mode_of(ckpt))
    quads = [q * f for roi in rs for q in roi["quads"]]
    tiles = [warp_section(src, q, ckpt["imgsz"], letterbox=ckpt.get("letterbox", True)) for q in quads]
    score, pred, probs = predict_tiles(model, tiles, device, batch)
    rows = []
    overlay = img.copy()
    k = 0
    for roi in rs:
        for s in range(len(roi["quads"])):
            q = quads[k]
            cv2.fillPoly(overlay, [q.round().astype(np.int32)], score_color(score[k]))
            rows.append([name, roi["roi_idx"], s, round(float(score[k]), 3), int(pred[k]),
                         *(round(float(v), 3) for v in probs[k]),
                         *(round(float(v)) for v in roi["quads"][s].ravel()), roi["conf"]])
            k += 1
    canvas = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    k = 0
    for roi in rs:
        mean = float(np.mean(score[k:k + len(roi["quads"])]))
        for _ in roi["quads"]:
            q = quads[k].round().astype(np.int32)
            cv2.polylines(canvas, [q], True, (40, 40, 40), 1, cv2.LINE_AA)
            c = q.mean(axis=0).astype(int)
            label = f"{score[k]:.1f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            org = (int(c[0]) - tw // 2, int(c[1]) + th // 2)
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            k += 1
        print(f"{name} roi {roi['roi_idx']}: mean score {mean:.2f}  sections "
              + " ".join(f"{v:.1f}" for v in score[k - len(roi['quads']):k]), flush=True)
    cv2.imwrite(str(out / f"{p.stem}_qual.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rows


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    root = Path("roi_follicle_bb_data")

    t = sub.add_parser("train", help="cross-validate and train the section quality model on sections.csv")
    t.add_argument("--sections-csv", default=str(root / "sections.csv"))
    t.add_argument("--source", default=str(root / "images"), help="images the sections.csv coordinates refer to")
    t.add_argument("--exclude", default=str(root / "excluded.txt"))
    t.add_argument("--max-side", type=int, default=1024, help="images are downscaled to this long side")
    t.add_argument("--arch", default="resnet18", help="torchvision resnet18 | resnet34 | resnet50")
    t.add_argument("--init", choices=["imagenet", "hist", "none"], default="imagenet",
                   help="hist = encoder of the 3cls_hist_seg U-Net (needs --arch resnet34)")
    t.add_argument("--hist-weights", default="3cls_hist_seg/best.pt")
    t.add_argument("--imgsz", type=int, default=224)
    t.add_argument("--mode", choices=["color", "gray"], default="color")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--jitter", type=float, default=0.05, help="random corner shift as a fraction of the section size")
    t.add_argument("--hsv-h", type=float, default=0.1, help="hue shift +/- fraction of the hue circle (stain colour)")
    t.add_argument("--hsv-s", type=float, default=0.5, help="saturation jitter, x(1 +/- value) (stain intensity)")
    t.add_argument("--hsv-v", type=float, default=0.5, help="brightness jitter, x(1 +/- value)")
    t.add_argument("--blur-p", type=float, default=0.2, help="probability of a slight Gaussian blur")
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--folds", type=int, default=5, help="cross-validation folds grouped by slide (0 = skip CV)")
    t.add_argument("--no-final", action="store_true", help="skip the final model trained on all sections")
    t.add_argument("--final-epochs", type=int, help="default: median best epoch of the CV folds")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--device", default=None)
    t.add_argument("--project", default="runs/qual_cls")
    t.add_argument("--name", default="train")
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("predict", help="score the sections of ROI boxes")
    d.add_argument("--weights", required=True)
    d.add_argument("--source", default=str(root / "images"))
    d.add_argument("--rois", help="boxes.csv | detections.csv | sections.csv; images without ROIs there "
                                  "(or no file) are detected with --roi-weights, and skipped if none are found")
    d.add_argument("--roi-weights", default="runs/roi_det/train-roi-2/weights/best.pt",
                   help="train_det.py ROI model used when --rois is not given")
    d.add_argument("--roi-imgsz", type=int, default=1024)
    d.add_argument("--roi-max-side", type=int, default=1024, help="must match the ROI model's dataset --max-side")
    d.add_argument("--sections", type=int, default=5, help="even sections per ROI (not used for sections.csv)")
    d.add_argument("--min-conf", type=float, default=0.25, help="minimum ROI confidence for detections.csv")
    d.add_argument("--out", default="runs/qual_cls/predict")
    d.add_argument("--batch", type=int, default=32)
    d.add_argument("--device", default=None)
    d.set_defaults(func=cmd_predict)
    return ap


def main():
    a = build_parser().parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

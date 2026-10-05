#!/usr/bin/env python3
"""
train_semseg.py -- per-pixel (semantic) segmentation of skin layers, as an
alternative to YOLO instance segmentation. Keratin and epidermis are continuous
layers, not objects, so a pixel classifier fits them better than boxes+masks.

Reuses a dataset produced by histoseg_to_yolo.py: the polygon labels are
rasterised to a class-per-pixel mask on the fly, so no reconversion is needed.

    python train_semseg.py train --data yolo_dataset_4cls_v2 --epochs 60
    python train_semseg.py train --data yolo_dataset_4cls_v2 --encoder resnet50 --crop 768
    python train_semseg.py predict --weights runs/semseg/train/best.pt \\
           --source yolo_dataset_4cls_v2/images/val --out preds

Model: U-Net with an ImageNet-pretrained torchvision ResNet encoder.
Loss:  cross-entropy + soft Dice.   Metric: per-class IoU / Dice on val.
Training samples random square crops from the 1024 px tiles (memory-friendly);
validation and prediction run on whole tiles since the network is fully
convolutional.

Requires: torch, torchvision, opencv-python, numpy, pyyaml (all already in the
ultralytics environment). TensorBoard logging works if tensorboard is installed.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# overlay colours (BGR) for class 1..N; background is untouched
PALETTE = [(58, 161, 232), (192, 168, 31), (67, 201, 122), (61, 38, 215),
           (200, 60, 200), (0, 200, 200), (120, 120, 240), (40, 140, 40)]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def read_names(data_dir: Path):
    y = yaml.safe_load((data_dir / "data.yaml").read_text())
    names = y["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return list(names)


def read_scale_calibration(data_dir: Path):
    """Optional dataset/scale_calibration.json: {"reference_um_per_px": float,
    "images": {stem: um_per_px}}. Lets images captured at different microscope/
    scanner resolutions be normalised to a common physical scale before
    training, instead of assuming every image has the same um/pixel."""
    p = data_dir / "scale_calibration.json"
    if not p.exists():
        return None, {}
    cfg = json.loads(p.read_text())
    return cfg["reference_um_per_px"], cfg.get("images", {})


def rasterise(label_path: Path, h: int, w: int) -> np.ndarray:
    """YOLO-seg polygons -> uint8 mask, 0 = background, k = class (k-1)."""
    mask = np.zeros((h, w), np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 7:
            continue
        cid = int(p[0])
        pts = np.array(p[1:], np.float32).reshape(-1, 2) * [w, h]
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], cid + 1)
    return mask


class TileDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, crop: int = 0, augment=False, downsample: float = 1.0):
        img_dir = data_dir / "images" / split
        self.images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in img_dir.glob(ext))
        self.labels = data_dir / "labels" / split
        self.crop, self.augment, self.downsample = crop, augment, downsample
        self.ref_um_per_px, self.scale_map = read_scale_calibration(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        ip = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = rasterise(self.labels / (ip.stem + ".txt"), h, w)

        # normalise to a common um/pixel first (if calibrated), then apply --downsample
        native_um_per_px = self.scale_map.get(ip.stem) if self.ref_um_per_px else None
        scale = (native_um_per_px / self.ref_um_per_px) if native_um_per_px else 1.0
        total = scale / self.downsample
        if total != 1:
            nh, nw = max(1, round(h * total)), max(1, round(w * total))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if total < 1 else cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)
        elif self.crop:
            img, mask = self._center_crop(img, mask)

        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(mask.astype(np.int64)), ip.name

    def _center_crop(self, img, mask):
        h, w = mask.shape
        c = min(self.crop, h, w)
        y0, x0 = (h - c) // 2, (w - c) // 2
        return img[y0:y0 + c, x0:x0 + c], mask[y0:y0 + c, x0:x0 + c]

    def _crop_origin(self, mask, h, w, c, fg_bias=0.8):
        """Pick a crop's top-left corner. Whole-slide images are mostly empty
        background around the tissue, so a uniform-random crop mostly misses
        the labeled region on a dataset this small - bias most crops to be
        centered near a random labeled pixel instead, keeping a fraction
        (1-fg_bias) fully random for background exposure."""
        fg = np.flatnonzero(mask)
        if fg.size and random.random() < fg_bias:
            cy, cx = divmod(int(fg[random.randrange(fg.size)]), w)
            return min(max(0, cy - c // 2), h - c), min(max(0, cx - c // 2), w - c)
        return random.randint(0, h - c), random.randint(0, w - c)

    def _augment(self, img, mask):
        h, w = mask.shape
        # random scale (0.7-1.3x): slice a (crop/s)-sized window straight out of the
        # source image (cheap) and resize just that window to self.crop, instead of
        # resizing the whole source image first - the source can be a multi-thousand
        # pixel whole-slide image, and resizing all of it per sample is what caused
        # training to OOM/crash on such inputs.
        s = random.uniform(0.7, 1.3)
        c = min(h, w, max(1, round(self.crop / s)))
        y0, x0 = self._crop_origin(mask, h, w, c)
        img, mask = img[y0:y0 + c, x0:x0 + c], mask[y0:y0 + c, x0:x0 + c]
        if c != self.crop:
            img = cv2.resize(img, (self.crop, self.crop), interpolation=cv2.INTER_AREA if c > self.crop else cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST)
        # histology has no canonical orientation
        k = random.randint(0, 3)
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
        # mild stain jitter (hue carries class information, so keep it small)
        hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-4, 4)) % 180
        hsv[..., 1] *= random.uniform(0.8, 1.2)
        hsv[..., 2] *= random.uniform(0.85, 1.15)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)


# ----------------------------------------------------------------------------
# model: U-Net on a torchvision ResNet encoder
# ----------------------------------------------------------------------------

class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class ResUNet(nn.Module):
    def __init__(self, n_classes: int, encoder: str = "resnet34", pretrained=True):
        super().__init__()
        import torchvision
        weights = "DEFAULT" if pretrained else None
        r = getattr(torchvision.models, encoder)(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)      # /2,  64
        self.pool = r.maxpool                                   # /4
        self.e1, self.e2, self.e3, self.e4 = r.layer1, r.layer2, r.layer3, r.layer4
        # probe the channel counts per stage (resnet18/34: 64,64,128,256,512;
        # resnet50: 64,256,512,1024,2048) instead of hard-coding them
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            f = self._encode(x)
            ch = [t.shape[1] for t in f]
        c0, c1, c2, c3, c4 = ch
        self.d4 = ConvBlock(c4 + c3, 256)
        self.d3 = ConvBlock(256 + c2, 128)
        self.d2 = ConvBlock(128 + c1, 64)
        self.d1 = ConvBlock(64 + c0, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def _encode(self, x):
        s = self.stem(x)
        e1 = self.e1(self.pool(s))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        return s, e1, e2, e3, e4

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s, e1, e2, e3, e4 = self._encode(x)
        d = self.d4(torch.cat([self._up(e4, e3), e3], 1))
        d = self.d3(torch.cat([self._up(d, e2), e2], 1))
        d = self.d2(torch.cat([self._up(d, e1), e1], 1))
        d = self.d1(torch.cat([self._up(d, s), s], 1))
        return self._up(self.head(d), x)


# ----------------------------------------------------------------------------
# loss + metrics
# ----------------------------------------------------------------------------

def dice_loss(logits, target, n_classes, eps=1.0):
    prob = logits.softmax(1)
    onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    inter = (prob * onehot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


class ConfusionMatrix:
    def __init__(self, n):
        self.n = n
        self.m = np.zeros((n, n), np.int64)

    def update(self, pred, target):
        idx = target.reshape(-1) * self.n + pred.reshape(-1)
        self.m += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def scores(self):
        tp = np.diag(self.m).astype(np.float64)
        fp = self.m.sum(0) - tp
        fn = self.m.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        return iou, dice


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def colourise(mask: np.ndarray, img_bgr: np.ndarray, alpha=0.45):
    out = img_bgr.copy()
    for k in range(1, mask.max() + 1):
        sel = mask == k
        if sel.any():
            out[sel] = (alpha * np.array(PALETTE[(k - 1) % len(PALETTE)]) + (1 - alpha) * out[sel]).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------

@torch.no_grad()
def save_train_samples(loader, out_dir: Path, max_samples=8):
    """Dump a few augmented train image+ground-truth-mask panels up front, so
    you can sanity-check crops/augmentation without waiting for training to
    finish (like YOLO's train_batch*.jpg)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for x, y, names in loader:
        for b in range(x.shape[0]):
            if saved >= max_samples:
                return
            img = ((x[b].numpy().transpose(1, 2, 0) * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            panel = np.hstack([img, colourise(y[b].numpy(), img)])
            cv2.imwrite(str(out_dir / f"train_{saved:02d}_{Path(names[b]).stem}.jpg"), panel,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved += 1


VAL_MAX_SIDE = 2048  # cap the val forward-pass resolution: a whole-slide val image
# run at full res allocates tens of GB of activations, which OOMs a small GPU and, on
# the CPU fallback, can exhaust system RAM and hard-crash the machine. Scoring is still
# done at full resolution (the prediction is upsampled back).


@torch.no_grad()
def _val_predict(model, x, device, max_side=VAL_MAX_SIDE):
    """Predict class indices for one whole val image at a memory-bounded
    resolution, upsampled back to the input's own (H, W) so IoU is scored at
    full res. The forward pass runs on a copy downscaled so its longest side is
    <= max_side (on GPU, falling back to CPU only for the already-downscaled
    input if VRAM still doesn't fit) - this is what keeps validation from
    OOM-crashing the GPU or the whole machine on these large slides."""
    _, _, h, w = x.shape
    xin = x
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        xin = F.interpolate(x, size=(max(1, round(h * s)), max(1, round(w * s))),
                            mode="bilinear", align_corners=False)
    try:
        pred = model(xin.to(device)).argmax(1)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if device.type != "cuda" or "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        model.to("cpu")
        try:
            pred = model(xin.to("cpu")).argmax(1)
        finally:
            model.to(device)
    pred = pred.to(torch.uint8)
    if pred.shape[-2:] != (h, w):
        pred = F.interpolate(pred[:, None].float(), size=(h, w), mode="nearest")[:, 0].to(torch.uint8)
    return pred.cpu().numpy()


@torch.no_grad()
def evaluate(model, loader, device, n_classes, save_dir: Path = None, max_plots=8):
    model.eval()
    cm = ConfusionMatrix(n_classes)
    plotted = 0
    for x, y, names in loader:
        pred = _val_predict(model, x, device)
        tgt = y.numpy()
        cm.update(pred, tgt)
        if save_dir is not None and plotted < max_plots:
            for b in range(x.shape[0]):
                if plotted >= max_plots:
                    break
                img = ((x[b].numpy().transpose(1, 2, 0) * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                panel = np.hstack([img, colourise(tgt[b], img), colourise(pred[b], img)])
                cv2.imwrite(str(save_dir / f"val_{Path(names[b]).stem}.jpg"), panel,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                plotted += 1
    return cm.scores()


def load_pretrained(model: nn.Module, ckpt_path: str, old_class_map: list, device) -> None:
    """Warm-start from a checkpoint trained on a different (e.g. superset)
    class list: load the encoder/decoder trunk as-is, and slice the old head's
    rows named in `old_class_map` (positionally aligned to the new model's
    classes; background is always row 0) into the new (smaller) head."""
    ckpt = torch.load(ckpt_path, map_location=device)
    old_sd, old_names = ckpt["model"], ckpt["names"]
    trunk_sd = {k: v for k, v in old_sd.items() if k not in ("head.weight", "head.bias")}
    missing, unexpected = model.load_state_dict(trunk_sd, strict=False)
    assert unexpected == [] and set(missing) <= {"head.weight", "head.bias"}, \
        f"unexpected trunk mismatch: missing={missing} unexpected={unexpected}"
    rows = [0] + [old_names.index(n) + 1 for n in old_class_map]  # background + mapped classes
    with torch.no_grad():
        model.head.weight.copy_(old_sd["head.weight"][rows])
        model.head.bias.copy_(old_sd["head.bias"][rows])
    print(f"loaded pretrained trunk + head rows {list(zip(old_class_map, rows[1:]))} "
          f"(old classes={old_names}) from {ckpt_path}")


def safe_torch_save(obj, path: Path, retries: int = 15, delay: float = 2.0):
    """Save a checkpoint tolerating transient Windows sharing violations (WinError
    32): when the output dir lives inside a cloud-sync folder (OneDrive/Dropbox),
    the client briefly locks the file to upload it, so overwriting last.pt/best.pt
    every epoch intermittently fails. Write to a unique temp file first (never
    contended), then retry the atomic replace until the sync client releases the
    target; warn but keep training rather than lose a long run on a transient lock."""
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{int(time.time() * 1000) % 100000}.tmp")
    torch.save(obj, tmp)
    for attempt in range(1, retries + 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            if attempt == retries:
                print(f"WARNING: could not overwrite {path.name} after {retries} tries ({e}); "
                      f"kept {tmp.name}. Pause your cloud-sync client or move --project outside "
                      f"OneDrive. Training continues.", flush=True)
                return
            time.sleep(delay)


def cmd_train(args):
    data_dir = Path(args.data)
    names = read_names(data_dir)
    n_classes = len(names) + 1  # + background
    device = pick_device(args.device)
    out = Path(args.project) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(args).items() if k != "func"}, indent=2))
    print(f"device={device}  classes={['background'] + names}  out={out}")

    train_ds = TileDataset(data_dir, "train", crop=args.crop, augment=True, downsample=args.downsample)
    val_ds = TileDataset(data_dir, "val", downsample=args.downsample)
    print(f"train tiles={len(train_ds)}  val tiles={len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    samples_dir = out / "train_samples"
    save_train_samples(train_dl, samples_dir)
    print(f"wrote example train image|label panels to {samples_dir}")

    model = ResUNet(n_classes, args.encoder, pretrained=not args.no_pretrained).to(device)
    if args.init_weights:
        old_class_map = args.pretrained_classes.split(",") if args.pretrained_classes else names
        assert len(old_class_map) == len(names), \
            f"--pretrained-classes must list {len(names)} names (one per {names}), got {old_class_map}"
        load_pretrained(model, args.init_weights, old_class_map, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    ce = nn.CrossEntropyLoss()

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out))
    except Exception:
        pass

    csv_path = out / "results.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "time", "train_loss", "mean_fg_dice", "mean_fg_iou"]
                               + [f"dice_{n}" for n in names] + [f"iou_{n}" for n in names])

    best, bad_epochs, t0 = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y) + dice_loss(logits, y, n_classes)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        train_loss = tot / max(nb, 1)

        ckpt = {"model": model.state_dict(), "names": names, "encoder": args.encoder, "epoch": epoch}
        safe_torch_save(ckpt, out / "last.pt")

        do_val = epoch % args.val_every == 0 or epoch == args.epochs
        if not do_val:
            print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  (val skipped)  "
                  f"{(time.time() - t0) / 60:.1f} min")
            continue

        iou, dice = evaluate(model, val_dl, device, n_classes)
        fg_dice, fg_iou = dice[1:].mean(), iou[1:].mean()
        row = [epoch, round(time.time() - t0), round(train_loss, 4), round(fg_dice, 4), round(fg_iou, 4)] \
            + [round(v, 4) for v in dice[1:]] + [round(v, 4) for v in iou[1:]]
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/mean_fg_dice", fg_dice, epoch)
            writer.add_scalar("val/mean_fg_iou", fg_iou, epoch)
            for n, d, i in zip(names, dice[1:], iou[1:]):
                writer.add_scalar(f"val_dice/{n}", d, epoch)
                writer.add_scalar(f"val_iou/{n}", i, epoch)
        per_cls = "  ".join(f"{n}={d:.3f}" for n, d in zip(names, dice[1:]))
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"val dice {fg_dice:.4f}  [{per_cls}]  {(time.time() - t0) / 60:.1f} min")

        if fg_dice > best:
            best, bad_epochs = fg_dice, 0
            safe_torch_save(ckpt, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop: no val improvement for {args.patience} validation rounds")
                break

    # final report + overlay panels from the best checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    plots = out / "val_plots"
    plots.mkdir(exist_ok=True)
    iou, dice = evaluate(model, val_dl, device, n_classes, save_dir=plots)
    print(f"\nBest weights: {out / 'best.pt'}   mean foreground Dice {dice[1:].mean():.4f}")
    print(f"{'class':12s} {'Dice':>7s} {'IoU':>7s}")
    for n, d, i in zip(["background"] + names, dice, iou):
        print(f"{n:12s} {d:7.3f} {i:7.3f}")
    print(f"\nOverlay panels (image | label | prediction): {plots}")
    print(f"Predict with:\n  python train_semseg.py predict --weights {out / 'best.pt'} --source <dir> --out preds")


# ----------------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------------

def filter_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Zero out (background) any foreground connected component smaller than min_px pixels."""
    if min_px <= 0:
        return mask
    out = mask.copy()
    for cls in np.unique(mask):
        if cls == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask == cls).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_px:
                out[labels == i] = 0
    return out


def filter_unadjacent_components(mask: np.ndarray, rules: list, margin_px: int, class_index: dict) -> np.ndarray:
    """rules: list of (child_name, parent_name). Drop connected components of the
    child class that don't overlap a margin_px-dilated parent-class mask - e.g. SC
    is anatomically always adjacent to Granular Layer, so a stray SC blob elsewhere
    in the image (with no nearby Granular Layer prediction) is almost certainly noise.
    Rotation-invariant: only cares about adjacency, not "above/below"."""
    if not rules:
        return mask
    out = mask.copy()
    kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), np.uint8) if margin_px > 0 else None
    for child, parent in rules:
        if child not in class_index or parent not in class_index:
            print(f"WARNING: --require-adjacent class not found ({child!r} or {parent!r}), skipping")
            continue
        parent_mask = (mask == class_index[parent]).astype(np.uint8)
        if kernel is not None:
            parent_mask = cv2.dilate(parent_mask, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask == class_index[child]).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            comp = labels == i
            if not (comp & (parent_mask > 0)).any():
                out[comp] = 0
    return out


def estimate_band_angle(mask: np.ndarray, class_ids: list) -> float:
    """PCA on the combined foreground pixels of class_ids to find the band's
    dominant (length) axis angle in degrees, in cv2.getRotationMatrix2D's sense."""
    ys, xs = np.where(np.isin(mask, class_ids))
    if len(xs) < 2:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    _, eigvecs = cv2.PCACompute(pts, mean=None, maxComponents=1)
    vx, vy = eigvecs[0]
    return float(np.degrees(np.arctan2(vy, vx)))


def rotate_canvas(img: np.ndarray, mask: np.ndarray, angle: float):
    """Rotate img+mask together by -angle (to cancel out `angle`'s tilt), onto an
    expanded canvas so nothing gets cropped off."""
    h, w = mask.shape
    cx, cy = w / 2, h / 2
    rad = np.radians(angle)
    new_w = int(abs(w * np.cos(rad)) + abs(h * np.sin(rad)))
    new_h = int(abs(w * np.sin(rad)) + abs(h * np.cos(rad)))
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    img_r = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_r = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_r, mask_r


def canonicalize_orientation(img: np.ndarray, mask: np.ndarray, top_id: int, other_id: int):
    """Rotate so the top_id/other_id band lies horizontal, then flip vertically (if
    needed) so top_id consistently ends up above other_id - a fixed, arbitrary but
    self-consistent convention, independent of how the original slide was mounted."""
    angle = estimate_band_angle(mask, [top_id, other_id])
    img_r, mask_r = rotate_canvas(img, mask, angle)
    top_ys = np.where(mask_r == top_id)[0]
    other_ys = np.where(mask_r == other_id)[0]
    if len(top_ys) and len(other_ys) and top_ys.mean() > other_ys.mean():
        img_r, mask_r = cv2.flip(img_r, 0), cv2.flip(mask_r, 0)
    return img_r, mask_r


def filter_abnormal_components(mask: np.ndarray, class_id: int, min_h_um: float, max_h_um: float,
                                min_w_um: float, max_w_um: float, px_to_um: float) -> np.ndarray:
    """Zero out (background) any connected component of class_id whose own bounding-box
    height or width (converted to um) falls outside [min,max] - drops blobs with an
    implausible size/shape. Must be run on an already-canonicalized (roughly horizontal)
    mask, since height/width are only meaningful (thickness/length) once so oriented."""
    out = mask.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask == class_id).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        h_um = stats[i, cv2.CC_STAT_HEIGHT] * px_to_um
        w_um = stats[i, cv2.CC_STAT_WIDTH] * px_to_um
        if not (min_h_um <= h_um <= max_h_um and min_w_um <= w_um <= max_w_um):
            out[labels == i] = 0
    return out


def parse_dimension_limits(spec: str) -> dict:
    """'SC:67.4,3332.6,573.1,5557.2;Granular Layer:94.5,3198,581,5887' ->
    {"SC": (67.4, 3332.6, 573.1, 5557.2), ...} (min_h, max_h, min_w, max_w) in um."""
    limits = {}
    for block in spec.split(";"):
        cls, nums = block.split(":")
        limits[cls.strip()] = tuple(float(v) for v in nums.split(","))
    return limits


def filter_by_mutual_touch(mask: np.ndarray, class_a: int, class_b: int, min_frac: float, margin_px: int) -> np.ndarray:
    """Per-component (not whole-class) mutual touch check: drop a connected component
    of class_a (or class_b) unless at least min_frac of ITS OWN BOUNDARY pixels (its
    outer rim, not its full interior area) lie within margin_px of the other class.
    Using the boundary rather than full area matters for thick bands - SC/Granular
    Layer are each hundreds of px thick, so only a thin strip at their shared edge is
    ever near the other class; requiring most of a class's *interior* to be close to
    the other class would fail even a correctly-adjacent real pair. The boundary is
    what should actually be touching, so that's what's checked."""
    kernel = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), np.uint8) if margin_px > 0 else None
    a_full = (mask == class_a).astype(np.uint8)
    b_full = (mask == class_b).astype(np.uint8)
    b_dilated = cv2.dilate(b_full, kernel) if kernel is not None else b_full
    a_dilated = cv2.dilate(a_full, kernel) if kernel is not None else a_full
    erosion_kernel = np.ones((3, 3), np.uint8)  # 1px-thick boundary ring

    out = mask.copy()
    for cls, other_dilated in ((class_a, b_dilated), (class_b, a_dilated)):
        binary = (mask == cls).astype(np.uint8)
        n, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            comp = (labels == i).astype(np.uint8)
            boundary = comp - cv2.erode(comp, erosion_kernel)
            if not boundary.any():
                boundary = comp  # component thinner than the erosion kernel: it's all boundary
            frac = (boundary & other_dilated).sum() / boundary.sum()
            if frac < min_frac:
                out[labels == i] = 0
    return out
    return mask


@torch.no_grad()
def cmd_predict(args):
    device = pick_device(args.device)
    ckpt = torch.load(args.weights, map_location=device)
    names = ckpt["names"]
    class_index = {n: k + 1 for k, n in enumerate(names)}
    model = ResUNet(len(names) + 1, ckpt["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scale = (args.um_per_px / args.reference_um_per_px) if args.um_per_px else 1.0
    if scale != 1.0:
        print(f"normalising inputs: {args.um_per_px} um/px -> {args.reference_um_per_px} um/px "
              f"(resize factor {scale:.4f})")
    adjacency_rules = [tuple(rule.split(":")) for rule in args.require_adjacent.split(",")] if args.require_adjacent else []
    canon_ids = []
    if args.canonicalize:
        top_name, other_name = args.canonicalize.split(",")
        if top_name not in class_index or other_name not in class_index:
            raise ValueError(f"--canonicalize classes not found: {top_name!r}, {other_name!r} (have {names})")
        canon_ids = [class_index[top_name], class_index[other_name]]
        print(f"canonicalizing orientation: rotating so {top_name}/{other_name} band is horizontal, "
              f"{top_name} above {other_name}")
    dimension_limits = {}
    if args.dimension_limits:
        if not canon_ids:
            raise ValueError("--dimension-limits requires --canonicalize (height/width are only meaningful "
                              "once the mask is rotated to horizontal)")
        dimension_limits = parse_dimension_limits(args.dimension_limits)
        for cls in dimension_limits:
            if cls not in class_index:
                raise ValueError(f"--dimension-limits class not found: {cls!r} (have {names})")
    px_to_um = args.um_per_px if args.um_per_px else 1.0
    touch_ids = []
    if args.touch_classes:
        touch_a, touch_b = args.touch_classes.split(",")
        if touch_a not in class_index or touch_b not in class_index:
            raise ValueError(f"--touch-classes not found: {touch_a!r}, {touch_b!r} (have {names})")
        touch_ids = [class_index[touch_a], class_index[touch_b]]

    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    for p in files:
        orig_bgr = cv2.imread(str(p))
        oh, ow = orig_bgr.shape[:2]
        bgr = cv2.resize(orig_bgr, (max(1, round(ow * scale)), max(1, round(oh * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR) \
            if scale != 1.0 else orig_bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # pad to a multiple of 32 so the encoder strides divide evenly
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        x = torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None].to(device)
        pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)[:h, :w]
        if scale != 1.0:
            # back to the original image's own resolution, not the training-scale one
            pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)
        pred = filter_small_components(pred, args.min_blob_px)
        pred = filter_unadjacent_components(pred, adjacency_rules, args.adjacency_margin_px, class_index)
        if touch_ids:
            pred = filter_by_mutual_touch(pred, touch_ids[0], touch_ids[1], args.min_touch_frac, args.adjacency_margin_px)
        if canon_ids:
            orig_bgr, pred = canonicalize_orientation(orig_bgr, pred, canon_ids[0], canon_ids[1])
        for cls, (min_h, max_h, min_w, max_w) in dimension_limits.items():
            pred = filter_abnormal_components(pred, class_index[cls], min_h, max_h, min_w, max_w, px_to_um)
        cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), colourise(pred, orig_bgr), [cv2.IMWRITE_JPEG_QUALITY, 85])
        frac = {n: round(float((pred == k + 1).mean()), 4) for k, n in enumerate(names)}
        print(f"{p.name}: {frac}")
    print(f"\nWrote {len(files)} overlays to {out}.")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a U-Net on a histoseg_to_yolo.py dataset")
    t.add_argument("--data", default="yolo_dataset_4cls_v2", help="dataset folder with data.yaml")
    t.add_argument("--encoder", default="resnet34", help="resnet18 | resnet34 | resnet50")
    t.add_argument("--no-pretrained", action="store_true", help="skip ImageNet encoder weights")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--crop", type=int, default=512, help="training crop size taken from each tile")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--patience", type=int, default=20, help="early-stop after this many validation rounds with no improvement")
    t.add_argument("--val-every", type=int, default=1,
                   help="run validation (and update best.pt) every N epochs instead of every epoch; "
                        "last.pt is still saved every epoch, and a final validation always runs on the last epoch")
    t.add_argument("--downsample", type=float, default=1.0,
                   help="shrink images (train+val) by this factor before cropping/validation - "
                        "speeds up dev iteration on large whole-slide images")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    t.add_argument("--project", default="runs/semseg")
    t.add_argument("--name", default="train")
    t.add_argument("--init-weights", default=None,
                    help="warm-start trunk+matching head rows from another train_semseg.py best.pt "
                         "(e.g. a model trained on a superset of classes)")
    t.add_argument("--pretrained-classes", default=None,
                    help="comma-separated old checkpoint class names, positionally aligned to this "
                         "dataset's names (needed when --init-weights' class names differ from "
                         "this dataset's, e.g. 'keratin,epidermis' -> SC,Granular Layer); "
                         "defaults to this dataset's own names if omitted")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="run a trained model on a folder of images")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="folder of images (any size, multiple of 32 not required)")
    p.add_argument("--out", default="preds")
    p.add_argument("--device", default="auto")
    p.add_argument("--um-per-px", type=float, default=None,
                   help="native um/pixel of the --source images (e.g. from calibrations/*.IQC); if set, "
                        "images are resized to --reference-um-per-px before inference and predictions are "
                        "resized back, so the model sees the same physical scale it was trained on")
    p.add_argument("--reference-um-per-px", type=float, default=0.645,
                   help="um/pixel the model was trained at (default matches dataset/scale_calibration.json's "
                        "reference_um_per_px)")
    p.add_argument("--min-blob-px", type=int, default=0,
                   help="drop predicted connected components smaller than this many pixels (per class); "
                        "removes small stray/noisy blobs. 0 = off")
    p.add_argument("--require-adjacent", default=None,
                   help="comma-separated 'Child:Parent' class-name rules (e.g. 'SC:Granular Layer') - drops "
                        "connected components of Child that don't touch/overlap a (dilated) Parent-class "
                        "region, since some classes are anatomically always adjacent to another. "
                        "Rotation-invariant (adjacency-based, not orientation-based).")
    p.add_argument("--adjacency-margin-px", type=int, default=10,
                   help="how many pixels the Parent mask is dilated by before checking adjacency, to "
                        "tolerate the model's boundary being slightly off (used with --require-adjacent "
                        "and --touch-classes)")
    p.add_argument("--touch-classes", default=None,
                   help="'ClassA,ClassB' (e.g. 'SC,Granular Layer') - drops BOTH classes entirely (whole "
                        "image) unless at least --min-touch-frac of EACH class's own pixels lie within "
                        "--adjacency-margin-px of the other. Stricter than --require-adjacent's per-"
                        "component any-overlap check: catches cases where the two are mostly not touching "
                        "even if some small part technically does.")
    p.add_argument("--min-touch-frac", type=float, default=0.4,
                   help="minimum mutual-contact fraction required by --touch-classes (default 0.4 = 40%%)")
    p.add_argument("--canonicalize", default=None,
                   help="'Top,Other' class names (e.g. 'SC,Granular Layer') - rotates the image+mask so "
                        "that band lies horizontal (via PCA on their combined pixels), then flips "
                        "vertically if needed so Top consistently ends up above Other. Saved overlay is "
                        "the rotated version. A fixed, self-consistent convention, not tied to how the "
                        "original slide was mounted.")
    p.add_argument("--dimension-limits", default=None,
                   help="'Class:min_h,max_h,min_w,max_w;Class2:...' (um) - drop connected components of "
                        "Class whose own bounding-box height or width falls outside that range (implausible "
                        "size/shape = noise). Requires --canonicalize and --um-per-px (or images already at "
                        "--reference-um-per-px) so height/width are physically meaningful.")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    args.func(args)


if __name__ == "__main__":
    main()

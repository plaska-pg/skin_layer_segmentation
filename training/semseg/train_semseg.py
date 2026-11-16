#!/usr/bin/env python3
"""
train_semseg.py -- per-pixel (semantic) segmentation of skin layers, as an
alternative to YOLO instance segmentation. Keratin and epidermis are continuous
layers, not objects, so a pixel classifier fits them better than boxes+masks.


Reuses a dataset produced by histoseg_to_yolo.py: the polygon labels are
rasterised to a class-per-pixel mask on the fly, so no reconversion is needed.


    python training/semseg/train_semseg.py train --name v9          # recipe from semseg.yaml
    python training/semseg/train_semseg.py train --name v9b --lr 1e-4   # flags override the YAML
    python training/semseg/train_semseg.py predict --weights runs/semseg/v9/best.pt \\
           --source datasets/seg_finetune/images/val --out preds


Physical scale is tracked end to end: data.yaml carries the tiles' um/px (from
histoseg_to_yolo.py --source-um-per-px); --train-um-range LO,HI trains across that
band of um/px by resampling crops (log-uniform) plus blur / down-up / JPEG
degradation; the checkpoint stores um_per_px (reference = validation scale) and
train_um_range; --val-scales reports Dice per scale; predict resamples inputs
from --um-per-px to the reference and writes scale.json next to the masks.

    python train_semseg.py train --data <dataset> --train-um-range 0.7,2.8 \\
           --blur-p 0.3 --downup-p 0.3 --jpeg-p 0.3 --val-scales 0.7,1.0,1.4,2.0,2.8
    python train_semseg.py sweep --weights runs/semseg/v8_edge_2/best.pt
    python train_semseg.py predict --weights ... --source new_scans --um-per-px 0.28

Model: U-Net with an ImageNet-pretrained torchvision ResNet encoder.
Loss:  cross-entropy + soft Dice.   Metric: per-class IoU / Dice on val.
Training samples random square crops from the 1024 px tiles (memory-friendly);
validation and prediction run on whole tiles since the network is fully
convolutional.


Requires: torch, torchvision, opencv-python, numpy, pyyaml (all already in the
ultralytics environment). TensorBoard logging works if tensorboard is installed.
"""


import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for running this file as a script
from training.config_utils import apply_config  # noqa: E402


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




def read_scale(data_dir: Path):
    """um per tile pixel as recorded in data.yaml by histoseg_to_yolo.py
    (--source-um-per-px x --downsample). None for datasets that predate scale tracking."""
    y = yaml.safe_load((data_dir / "data.yaml").read_text())
    u = y.get("um_per_px")
    return (float(u) if u else None), y.get("scale_source")




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
    def __init__(self, data_dir: Path, split: str, crop: int = 0, augment=False, downsample: float = 1.0,
                 scale_aug=(0.7, 1.3), blur_p: float = 0.0, downup_p: float = 0.0, jpeg_p: float = 0.0):
        """scale_aug: (lo, hi) range of the random window factor s used in training; a crop
        of crop/s source pixels is resized to `crop`, so s < 1 produces a coarser (higher
        um/px) view and s > 1 a finer one. The trainer derives it from --train-um-range.
        blur_p / downup_p / jpeg_p: per-sample probabilities of the resolution-degradation
        augmentations (see _degrade)."""
        img_dir = data_dir / "images" / split
        self.labels = data_dir / "labels" / split
        images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff") for p in img_dir.glob(ext))
        # an image with no label file would teach its whole tissue section as background
        unlabeled = [p.name for p in images if not (self.labels / (p.stem + ".txt")).exists()]
        if unlabeled:
            print(f"WARNING: {split}: skipping {len(unlabeled)} image(s) with no label file: {unlabeled}")
        self.images = [p for p in images if p.name not in unlabeled]
        self.crop, self.augment, self.downsample = crop, augment, downsample
        self.scale_aug, self.blur_p, self.downup_p, self.jpeg_p = tuple(scale_aug), blur_p, downup_p, jpeg_p
        self.ref_um_per_px, self.scale_map = read_scale_calibration(data_dir)
        uncal = [p.name for p in self.images if self.ref_um_per_px and p.stem not in self.scale_map]
        if uncal:
            print(f"WARNING: {split}: no um_per_px in scale_calibration.json (used unscaled): {uncal}")


    def __len__(self):
        return len(self.images)


    def __getitem__(self, i):
        ip = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # rasterised at native res: fillPoly includes edge pixels, which at training scale fattens classes ~0.3 px
        mask = rasterise(self.labels / (ip.stem + ".txt"), h, w)

        # normalise to a common um/pixel first (if calibrated), then apply --downsample
        native_um_per_px = self.scale_map.get(ip.stem) if self.ref_um_per_px else None
        scale = (native_um_per_px / self.ref_um_per_px) if native_um_per_px else 1.0
        total = scale / self.downsample
        if total != 1:
            h, w = max(1, round(h * total)), max(1, round(w * total))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if total < 1 else cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)


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


    def _degrade(self, img):
        """Resolution-robustness augmentation. A lower-resolution capture is not just
        smaller: it is softer and carries resampling and JPEG artefacts, so each of
        those is simulated independently with its own probability."""
        if self.blur_p and random.random() < self.blur_p:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.5))
        if self.downup_p and random.random() < self.downup_p:
            f = random.uniform(1.2, 2.5)
            h, w = img.shape[:2]
            small = cv2.resize(img, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
        if self.jpeg_p and random.random() < self.jpeg_p:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, random.randint(40, 90)])
            if ok:
                img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img


    def _augment(self, img, mask):
        h, w = mask.shape
        # random scale (log-uniform over self.scale_aug): slice a (crop/s)-sized window straight out of the
        # source image (cheap) and resize just that window to self.crop, instead of
        # resizing the whole source image first - the source can be a multi-thousand
        # pixel whole-slide image, and resizing all of it per sample is what caused
        # training to OOM/crash on such inputs.
        lo, hi = self.scale_aug
        s = math.exp(random.uniform(math.log(lo), math.log(hi)))
        self.last_window_factor = s  # crop um/px = tile um/px / s (read by viz.py)
        c = min(h, w, max(1, round(self.crop / s)))
        y0, x0 = self._crop_origin(mask, h, w, c)
        img, mask = img[y0:y0 + c, x0:x0 + c], mask[y0:y0 + c, x0:x0 + c]
        if c != self.crop:
            img = cv2.resize(img, (self.crop, self.crop), interpolation=cv2.INTER_AREA if c > self.crop else cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST_EXACT)
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
        img = self._degrade(img)
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
    def __init__(self, n_classes: int, encoder: str = "resnet34", pretrained=True, full_res: bool = False):
        """full_res: add a stride-1 decoder stage (skip from a shallow conv on the input) so
        logits are predicted at input resolution instead of bilinearly upsampled from /2."""
        super().__init__()
        self.full_res = full_res
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
        if full_res:
            self.s0 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
            self.d0 = ConvBlock(32 + 16, 16)
        self.head = nn.Conv2d(16 if full_res else 32, n_classes, 1)


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
        if self.full_res:
            return self.head(self.d0(torch.cat([self._up(d, x), self.s0(x)], 1)))
        return self._up(self.head(d), x)




# ----------------------------------------------------------------------------
# loss + metrics
# ----------------------------------------------------------------------------


def edge_weight_map(target, width: int, gain: float):
    """1 + gain on pixels within `width` px of a class change in the label, else 1."""
    t = target.float()[:, None]
    k = 2 * width + 1
    hi = F.max_pool2d(t, k, 1, width)
    lo = -F.max_pool2d(-t, k, 1, width)
    return 1 + gain * (hi != lo)[:, 0].float()


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




VAL_TILE = 1024  # val runs tiled at training scale; downscaling whole slides erased thin SC




@torch.no_grad()
def _val_predict(model, x, device, tile=VAL_TILE, overlap=0.25):
    """Class indices (1, H, W) for one val image at its own (training) scale, run as
    overlapping tiles; each tile keeps only its centre so edges get full context."""
    _, _, h, w = x.shape
    m = int(tile * overlap / 2)
    stride = tile - 2 * m
    ys = sorted(set(list(range(0, max(1, h - tile), stride)) + [max(0, h - tile)]))
    xs = sorted(set(list(range(0, max(1, w - tile), stride)) + [max(0, w - tile)]))
    pred = np.zeros((1, h, w), np.uint8)
    for y0 in ys:
        for x0 in xs:
            patch = x[:, :, y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[-2:]
            patch = F.pad(patch, (0, (32 - pw % 32) % 32, 0, (32 - ph % 32) % 32), mode="replicate")
            p = model(patch.to(device)).argmax(1)[:, :ph, :pw].to(torch.uint8).cpu().numpy()
            ty0 = m if y0 > 0 else 0
            tx0 = m if x0 > 0 else 0
            ty1 = ph - m if y0 + ph < h else ph
            tx1 = pw - m if x0 + pw < w else pw
            pred[:, y0 + ty0:y0 + ty1, x0 + tx0:x0 + tx1] = p[:, ty0:ty1, tx0:tx1]
    return pred




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
    new_layers = {k for k in missing if k.split(".")[0] in ("s0", "d0")}  # full_res stage on a /2 checkpoint
    assert unexpected == [] and set(missing) - new_layers <= {"head.weight", "head.bias"}, \
        f"unexpected trunk mismatch: missing={missing} unexpected={unexpected}"
    if model.head.weight.shape[1] != old_sd["head.weight"].shape[1]:
        print(f"loaded pretrained trunk from {ckpt_path}; new full-res stage + head start from scratch")
        return
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




class EMA:
    """Exponential moving average of the weights (and BatchNorm buffers). Validation and
    checkpoints use the averaged model, which smooths the epoch-to-epoch swings seen with
    small batches without changing the optimisation itself."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def class_weighted_sampler(ds: "TileDataset", names, spec: str):
    """--oversample 'glands=3,follicles=2': a tile is drawn k times as often as a plain tile
    when it contains that class (max over the classes present). Returns the sampler and the
    resulting share of draws per class."""
    factors = {k: float(v) for k, v in (item.split("=") for item in spec.split(","))}
    for k in factors:
        if k not in names:
            raise SystemExit(f"--oversample: unknown class {k!r} (have {names})")
    weights, present_all = [], []
    for ip in ds.images:
        lp = ds.labels / (ip.stem + ".txt")
        present = set()
        if lp.exists():
            for line in lp.read_text().splitlines():
                if line.strip():
                    present.add(names[int(line.split()[0])])
        present_all.append(present)
        weights.append(max([factors.get(n, 1.0) for n in present], default=1.0))
    total = sum(weights)
    share = {n: round(sum(w for w, pr in zip(weights, present_all) if n in pr) / total, 3) for n in names}
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), share


def scale_sweep(model, data_dir: Path, u_yaml: float, scales, device, names, out_csv: Path, only_slides=None):
    """Validate at several physical scales. The val tiles (u_yaml um/px) are resampled
    to each target um/px before prediction, so the table shows how Dice holds up as
    resolution moves away from the training reference - the headline robustness result."""
    n_classes = len(names) + 1
    rows = []
    print(f"\nscale sweep (val tiles are {u_yaml:.3f} um/px{', slides ' + ','.join(only_slides) if only_slides else ''}):")
    print(f"{'um/px':>7s} {'mean_fg':>8s} " + " ".join(f"{n:>10s}" for n in names))
    for t in scales:
        ds = TileDataset(data_dir, "val", downsample=t / u_yaml)
        if only_slides:
            ds.images = [ip for ip in ds.images if any(ip.stem.startswith(pfx) for pfx in only_slides)]
            if not ds.images:
                raise SystemExit(f"--only-slides matched no val tiles: {only_slides}")
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        iou, dice = evaluate(model, dl, device, n_classes)
        rows.append([t, round(float(dice[1:].mean()), 4)] + [round(float(v), 4) for v in dice[1:]]
                    + [round(float(v), 4) for v in iou[1:]])
        print(f"{t:7.2f} {dice[1:].mean():8.4f} " + " ".join(f"{d:10.4f}" for d in dice[1:]), flush=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["um_per_px", "mean_fg_dice"] + [f"dice_{n}" for n in names] + [f"iou_{n}" for n in names])
        w.writerows(rows)
    print(f"-> {out_csv}")
    return rows




def resolve_scale(args, data_dir: Path):
    """Turn the dataset's um/px plus the CLI's scale options into: the window-factor
    range for training crops, the validation/reference um/px, the val downsample, and
    the metadata stored in the checkpoint."""
    u_yaml, scale_source = read_scale(data_dir)
    u_tile = u_yaml * args.downsample if u_yaml else None
    if args.train_um_range:
        if not u_tile:
            raise SystemExit("--train-um-range needs a dataset whose data.yaml records um_per_px "
                             "(rebuild it with histoseg_to_yolo.py --source-um-per-px)")
        lo, hi = (float(v) for v in args.train_um_range.split(","))
        scale_aug = (u_tile / hi, u_tile / lo)
        u_ref = args.val_um_per_px or math.sqrt(lo * hi)
        band = [lo, hi]
    else:
        lo_s, hi_s = (float(v) for v in args.scale_aug.split(","))
        scale_aug = (lo_s, hi_s)
        u_ref = args.val_um_per_px or u_tile
        band = [u_tile / hi_s, u_tile / lo_s] if u_tile else None
    val_down = args.downsample * (u_ref / u_tile) if (u_tile and u_ref) else args.downsample
    meta = {"um_per_px": u_ref, "train_um_range": band, "tile_um_per_px": u_tile,
            "dataset_um_per_px": u_yaml, "scale_source": scale_source, "dataset": str(data_dir)}
    if u_ref is None:
        print("WARNING: data.yaml has no um_per_px - this checkpoint will not know its physical scale. "
              "Rebuild the dataset with histoseg_to_yolo.py --source-um-per-px to track resolution.")
    else:
        band_txt = f"{band[0]:.2f}-{band[1]:.2f} um/px" if band else "n/a"
        print(f"scale: tiles {u_tile:.3f} um/px | train band {band_txt} (window factor "
              f"{scale_aug[0]:.3f}-{scale_aug[1]:.3f}) | val/reference {u_ref:.3f} um/px | source: {scale_source}")
    return scale_aug, u_ref, val_down, meta, u_yaml




def cmd_train(args):
    data_dir = Path(args.data)
    names = read_names(data_dir)
    n_classes = len(names) + 1  # + background
    device = pick_device(args.device)
    out = Path(args.project) / args.name
    out.mkdir(parents=True, exist_ok=True)
    scale_aug, u_ref, val_down, scale_meta, u_yaml = resolve_scale(args, data_dir)
    (out / "args.json").write_text(json.dumps({**{k: v for k, v in vars(args).items() if k != "func"},
                                               "scale": scale_meta}, indent=2))
    (out / "scale.json").write_text(json.dumps(scale_meta, indent=2))
    print(f"device={device}  classes={['background'] + names}  out={out}")


    train_ds = TileDataset(data_dir, "train", crop=args.crop, augment=True, downsample=args.downsample,
                           scale_aug=scale_aug, blur_p=args.blur_p, downup_p=args.downup_p, jpeg_p=args.jpeg_p)
    val_ds = TileDataset(data_dir, "val", downsample=val_down)
    print(f"train tiles={len(train_ds)}  val tiles={len(val_ds)}")
    leaked = {p.stem for p in train_ds.images} & {p.stem for p in val_ds.images}
    if leaked:
        raise SystemExit(f"images in both train and val (validation would be meaningless): {sorted(leaked)}")
    sampler = None
    if args.oversample:
        sampler, share = class_weighted_sampler(train_ds, names, args.oversample)
        print(f"oversampling {args.oversample}: share of draws containing each class = {share}")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                          num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)


    samples_dir = out / "train_samples"
    save_train_samples(train_dl, samples_dir)
    print(f"wrote example train image|label panels to {samples_dir}")


    model = ResUNet(n_classes, args.encoder, pretrained=not args.no_pretrained, full_res=args.full_res).to(device)
    if args.init_weights:
        old_class_map = args.pretrained_classes.split(",") if args.pretrained_classes else names
        assert len(old_class_map) == len(names), \
            f"--pretrained-classes must list {len(names)} names (one per {names}), got {old_class_map}"
        load_pretrained(model, args.init_weights, old_class_map, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, args.ema) if args.ema else None
    eval_model = ema.module if ema else model  # what gets validated and saved
    if ema:
        print(f"EMA of weights with decay {args.ema} (validation and checkpoints use the average)")
    steps = args.epochs * math.ceil(len(train_dl) / args.accumulate)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    weights = torch.ones(n_classes)
    for item in (args.class_weights.split(",") if args.class_weights else []):
        cls, w = item.split("=")
        if cls not in names:
            raise SystemExit(f"--class-weights: unknown class {cls!r} (have {names})")
        weights[names.index(cls) + 1] = float(w)
    if args.class_weights:
        print(f"class weights: {dict(zip(['background'] + names, weights.tolist()))}")
    weights = weights.to(device)
    if args.edge_weight:
        print(f"edge-weighted CE: x{1 + args.edge_weight:g} within {args.edge_width} px of label boundaries")

    def ce(logits, y):
        px = F.cross_entropy(logits, y, weight=weights, reduction="none")
        ew = edge_weight_map(y, args.edge_width, args.edge_weight) if args.edge_weight else torch.ones_like(px)
        return (px * ew).sum() / (weights[y] * ew).sum()


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
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y) + dice_loss(logits, y, n_classes)
            (loss / args.accumulate).backward()  # gradient accumulation: effective batch = batch x accumulate
            if i % args.accumulate == 0 or i == len(train_dl):
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
            tot += loss.item()
            nb += 1
        train_loss = tot / max(nb, 1)


        ckpt = {"model": eval_model.state_dict(), "names": names, "encoder": args.encoder, "epoch": epoch,
                "full_res": args.full_res, **scale_meta}
        safe_torch_save(ckpt, out / "last.pt")


        do_val = epoch % args.val_every == 0 or epoch == args.epochs
        if not do_val:
            print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  (val skipped)  "
                  f"{(time.time() - t0) / 60:.1f} min")
            continue


        iou, dice = evaluate(eval_model, val_dl, device, n_classes)
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
    try:
        import viz
        viz.make_all(out, device=device, scales=[float(v) for v in args.val_scales.split(",")] if args.val_scales else None)
    except Exception as e:  # visual summaries must never cost a finished run
        print(f"viz skipped: {e}")
    if args.val_scales:
        if not u_yaml:
            print("--val-scales skipped: dataset has no um_per_px")
        else:
            scale_sweep(model, data_dir, u_yaml, [float(v) for v in args.val_scales.split(",")],
                        device, names, out / "scale_sweep.csv")
    print(f"Predict with:\n  python train_semseg.py predict --weights {out / 'best.pt'} --source <dir> --out preds")




def cmd_sweep(args):
    """Scale sweep for an existing checkpoint."""
    device = pick_device(args.device)
    ckpt = torch.load(args.weights, map_location=device)
    names = ckpt["names"]
    model = ResUNet(len(names) + 1, ckpt["encoder"], pretrained=False, full_res=ckpt.get("full_res", False)).to(device)
    model.load_state_dict(ckpt["model"])
    data_dir = Path(args.data)
    assert read_names(data_dir) == names, f"dataset classes {read_names(data_dir)} != checkpoint {names}"
    u_yaml, _ = read_scale(data_dir)
    if not u_yaml:
        raise SystemExit("dataset has no um_per_px in data.yaml")
    ref = ckpt.get("um_per_px")
    print(f"checkpoint reference {ref} um/px, train band {ckpt.get('train_um_range')}")
    out_csv = Path(args.out) if args.out else Path(args.weights).with_name("scale_sweep.csv")
    scale_sweep(model, data_dir, u_yaml, [float(v) for v in args.scales.split(",")], device, names, out_csv,
                only_slides=args.only_slides)




# ----------------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------------


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1].astype(np.float32)
    return np.clip(np.outer(w, w), 1e-3, None)




@torch.no_grad()
def predict_probs(model, rgb: np.ndarray, device, tile: int = 1024, overlap: float = 0.25) -> np.ndarray:
    """Softmax class probabilities (C, H, W) for one RGB image already at network scale.
    Images larger than `tile` are run as overlapping tiles blended with a Hann window,
    so whole sections stitch without seams and without a full-resolution forward pass."""
    h, w = rgb.shape[:2]


    def run(patch):
        ph, pw = patch.shape[:2]
        pad_h, pad_w = (32 - ph % 32) % 32, (32 - pw % 32) % 32
        patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        x = torch.from_numpy(((patch.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None].to(device)
        return model(x).softmax(1)[0, :, :ph, :pw].cpu().numpy()


    if not tile or (h <= tile and w <= tile):
        return run(rgb)
    stride = max(1, int(tile * (1 - overlap)))
    ys = sorted(set(list(range(0, max(1, h - tile), stride)) + [max(0, h - tile)]))
    xs = sorted(set(list(range(0, max(1, w - tile), stride)) + [max(0, w - tile)]))
    win = _hann2d(tile)
    acc, wsum = None, np.zeros((h, w), np.float32)
    for y0 in ys:
        for x0 in xs:
            patch = rgb[y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[:2]
            p = run(patch)
            if acc is None:
                acc = np.zeros((p.shape[0], h, w), np.float32)
            acc[:, y0:y0 + ph, x0:x0 + pw] += p * win[:ph, :pw]
            wsum[y0:y0 + ph, x0:x0 + pw] += win[:ph, :pw]
    return acc / wsum[None]




def cmd_predict(args):
    device = pick_device(args.device)
    ckpt = torch.load(args.weights, map_location=device)
    names = ckpt["names"]
    model = ResUNet(len(names) + 1, ckpt["encoder"], pretrained=False, full_res=ckpt.get("full_res", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()


    # physical scale: the checkpoint records what it was trained at
    u_ref = args.reference_um_per_px or ckpt.get("um_per_px")
    band = ckpt.get("train_um_range")
    if args.um_per_px and not u_ref:
        raise SystemExit("--um-per-px given but the checkpoint has no reference scale; pass --reference-um-per-px")
    scale = (args.um_per_px / u_ref) if args.um_per_px else 1.0
    if args.um_per_px:
        print(f"normalising inputs: {args.um_per_px} um/px -> {u_ref} um/px (resize factor {scale:.4f})")
        if band and not (band[0] <= args.um_per_px <= band[1]):
            print(f"WARNING: {args.um_per_px} um/px is outside the band seen in training "
                  f"({band[0]:.2f}-{band[1]:.2f} um/px). Inputs are still resampled to {u_ref} um/px, which "
                  f"{'discards' if args.um_per_px < band[0] else 'invents'} detail; check the overlays.")
    elif u_ref:
        print(f"assuming --source images are already at the model's {u_ref} um/px (pass --um-per-px if not)")
    else:
        print("WARNING: checkpoint has no scale information; no resolution normalisation possible")
    tta = [float(v) for v in args.tta_scales.split(",")] if args.tta_scales else []
    if tta and not u_ref:
        raise SystemExit("--tta-scales needs a checkpoint with a reference um/px")
    if tta:
        print(f"test-time multi-scale: averaging predictions made at {tta} um/px")


    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale.json").write_text(json.dumps({
        "um_per_px": args.um_per_px or u_ref,
        "note": "um/px of the *_mask.png files (the source images' own resolution)",
        "model_um_per_px": u_ref, "train_um_range": band, "tta_scales": tta or None,
        "weights": str(args.weights), "names": names}, indent=2))
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    for p in files:
        orig_bgr = cv2.imread(str(p))
        oh, ow = orig_bgr.shape[:2]
        bgr = cv2.resize(orig_bgr, (max(1, round(ow * scale)), max(1, round(oh * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR) \
            if scale != 1.0 else orig_bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if tta:
            probs = None
            for t in tta:
                r = u_ref / t
                im = rgb if abs(r - 1) < 1e-6 else cv2.resize(
                    rgb, (max(1, round(w * r)), max(1, round(h * r))),
                    interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
                pr = predict_probs(model, im, device, args.tile, args.tile_overlap)
                if pr.shape[1:] != (h, w):
                    pr = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in pr])
                probs = pr if probs is None else probs + pr
            pred = probs.argmax(0).astype(np.uint8)
        else:
            pred = predict_probs(model, rgb, device, args.tile, args.tile_overlap).argmax(0).astype(np.uint8)
        if scale != 1.0:
            # back to the original image's own resolution, not the training-scale one
            pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / f"{p.stem}_mask.png"), pred)
        cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), colourise(pred, orig_bgr), [cv2.IMWRITE_JPEG_QUALITY, 85])
        frac = {n: round(float((pred == k + 1).mean()), 4) for k, n in enumerate(names)}
        print(f"{p.name}: {frac}")
    print(f"\nWrote {len(files)} masks (*_mask.png) + overlays to {out}; "
          f"scale.json records {args.um_per_px or u_ref} um/px for the masks.")




# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)


    t = sub.add_parser("train", help="train a U-Net on a histoseg_to_yolo.py dataset")
    t.add_argument("--data", default="datasets/seg_finetune", help="dataset folder with data.yaml")
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
    t.add_argument("--ema", type=float, default=0.0,
                   help="decay of an exponential moving average of the weights, e.g. 0.995; 0 = off. "
                        "Validation and checkpoints use the averaged weights")
    t.add_argument("--oversample", default=None,
                   help="draw tiles containing rare classes more often, e.g. 'glands=3,follicles=2'")
    t.add_argument("--accumulate", type=int, default=1,
                   help="gradient accumulation steps (effective batch = --batch x this)")
    t.add_argument("--full-res", action="store_true",
                   help="add a stride-1 decoder stage so logits are predicted at input resolution")
    t.add_argument("--edge-weight", type=float, default=0.0,
                   help="extra cross-entropy weight on pixels near label boundaries (e.g. 2 -> x3); 0 = off")
    t.add_argument("--edge-width", type=int, default=3,
                   help="half-width (px, at training scale) of the boundary band for --edge-weight")
    t.add_argument("--class-weights", default=None,
                   help="cross-entropy class weights, e.g. 'SC=2' or 'SC=2,glands=1.5' (others 1)")
    t.add_argument("--train-um-range", default=None,
                   help="LO,HI um/px band to train across (log-uniform per crop), e.g. 0.7,2.8. Needs "
                        "um_per_px in data.yaml. The reference/validation scale is sqrt(LO*HI) unless "
                        "--val-um-per-px is given")
    t.add_argument("--val-um-per-px", type=float, default=None,
                   help="um/px to validate at and to store as the checkpoint's reference scale")
    t.add_argument("--scale-aug", default="0.7,1.3",
                   help="window-factor range for the random-scale crop when --train-um-range is not used")
    t.add_argument("--blur-p", type=float, default=0.0, help="probability of Gaussian blur (sigma 0.3-1.5) per crop")
    t.add_argument("--downup-p", type=float, default=0.0,
                   help="probability of a downsample-then-upsample (1.2-2.5x) per crop, mimicking low-res captures")
    t.add_argument("--jpeg-p", type=float, default=0.0, help="probability of JPEG recompression (quality 40-90) per crop")
    t.add_argument("--val-scales", default=None,
                   help="comma-separated um/px values; after training, validate best.pt at each and write "
                        "scale_sweep.csv (e.g. 0.7,1.0,1.4,2.0,2.8)")
    t.add_argument("--project", default="runs/semseg")
    t.add_argument("--name", default="train")
    t.add_argument("--init-weights", default=None,
                    help="warm-start trunk+matching head rows from another train_semseg.py best.pt "
                         "(e.g. a model trained on a superset of classes)")
    t.add_argument("--pretrained-classes", default=None,
                    help="comma-separated old checkpoint class names, positionally aligned to this "
                         "dataset's names (needed when --init-weights' class names differ from "
                         "this dataset's, e.g. 'keratin,epidermis' -> SC,Epidermis); "
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
    p.add_argument("--reference-um-per-px", type=float, default=None,
                   help="um/pixel the model was trained at; default: read from the checkpoint (um_per_px). "
                        "Only needed for checkpoints that predate scale tracking")
    p.add_argument("--tta-scales", default=None,
                   help="comma-separated um/px values (e.g. 1.0,1.4,2.0): predict at each, average the "
                        "softmaxes. Slower but more robust when the input's true scale is uncertain")
    p.add_argument("--tile", type=int, default=1024,
                   help="run images larger than this in overlapping, Hann-blended tiles (0 = whole image)")
    p.add_argument("--tile-overlap", type=float, default=0.25)
    p.set_defaults(func=cmd_predict)


    sw = sub.add_parser("sweep", help="validate an existing checkpoint at several um/px (scale_sweep.csv)")
    sw.add_argument("--weights", required=True)
    sw.add_argument("--data", required=True, help="dataset with um_per_px in data.yaml")
    sw.add_argument("--scales", default="0.7,1.0,1.4,2.0,2.8", help="comma-separated um/px values")
    sw.add_argument("--device", default="auto")
    sw.add_argument("--out", default=None, help="csv path (default: scale_sweep.csv next to the weights)")
    sw.add_argument("--only-slides", nargs="*", default=None, metavar="PREFIX",
                    help="score only val tiles whose name starts with one of these (e.g. a specimen id)")
    sw.set_defaults(func=cmd_sweep)


    apply_config(ap, default=str(Path(__file__).with_name("semseg.yaml")))
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    args.func(args)




if __name__ == "__main__":
    main()




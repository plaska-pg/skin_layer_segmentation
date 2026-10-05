# rotate.py -- image rotation for the project
# https://d4nst.github.io/2017/01/12/image-orientation/
#
# RotNet-style orientation classifier: randomly rotates whole (resized, not
# cropped) correctly-oriented images by 0/90/180/270 degrees and trains a CNN
# to predict which rotation was applied. At inference time the predicted
# rotation is undone, auto-correcting a mis-rotated scan back to upright.
#
#   python rotate.py train --dataset dataset_rotate --output runs/rotation/rotate_model.pt --pretrained --device cuda
#   python rotate.py predict --weights runs/rotation/rotate_model.pt --source inference_images --out runs/rotation/predictions

import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ANGLES = (0, 90, 180, 270)  # class index k means "rotated k*90 deg clockwise from upright"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet stats, matches --pretrained backbones
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(requested):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def rotate_k(img, k):
    """Rotate img by k*90 degrees clockwise (k in 0..3)."""
    return np.ascontiguousarray(np.rot90(img, k=-k))


def load_square(path, size):
    """Read an image and resize (never crop) the WHOLE frame to size x size -
    rotation is the only augmentation, so no content is ever discarded."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def to_bgr(tensor):
    arr = tensor.numpy().transpose(1, 2, 0)
    arr = (arr * STD + MEAN).clip(0, 1)
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()


def make_grid(panels, cols=4, pad=4):
    h, w = panels[0].shape[:2]
    rows = math.ceil(len(panels) / cols)
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = panel
    return canvas


def build_model(backbone, pretrained):
    model = torchvision.models.get_model(backbone, weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(ANGLES))
    return model


class RotationDataset(Dataset):
    """Randomly rotates cached, whole-image-resized (never cropped) source
    images; label = rotation class applied."""

    def __init__(self, files, size, samples_per_image):
        self.files = files
        self.size = size
        self.samples_per_image = samples_per_image
        self._cache = {}

    def __len__(self):
        return len(self.files) * self.samples_per_image

    def _load(self, path):
        if path not in self._cache:
            self._cache[path] = load_square(path, self.size)
        return self._cache[path]

    def __getitem__(self, idx):
        img = self._load(self.files[idx % len(self.files)])
        label = random.randint(0, 3)
        return to_tensor(rotate_k(img, label)), label


def _worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def save_augmentation_examples(files, size, out_path, n=8, seed=0):
    rng = random.Random(seed)
    panels = []
    for _ in range(n):
        img = load_square(rng.choice(files), size)
        label = rng.randint(0, 3)
        panel = rotate_k(img, label)
        cv2.putText(panel, f"{ANGLES[label]} deg", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        panels.append(panel)
    cv2.imwrite(str(out_path), make_grid(panels))
    print(f"[rotate] wrote augmentation examples -> {out_path}")


def save_prediction_examples(model, dataset, device, out_path, n=8, seed=0):
    model.eval()
    rng = random.Random(seed + 1)
    panels = []
    with torch.no_grad():
        for _ in range(n):
            tensor, label = dataset[rng.randrange(len(dataset))]
            pred = int(model(tensor.unsqueeze(0).to(device)).argmax(1).item())
            panel = to_bgr(tensor)
            color = (0, 200, 0) if pred == label else (0, 0, 255)
            cv2.putText(panel, f"true={ANGLES[label]} pred={ANGLES[pred]}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            panels.append(panel)
    cv2.imwrite(str(out_path), make_grid(panels))
    print(f"[rotate] wrote prediction examples -> {out_path}")


def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            running_loss += criterion(logits, labels).item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return correct / total, running_loss / total


def train(args):
    device = pick_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.dataset)
    train_files = list_images(root / "images" / "train")
    val_files = list_images(root / "images" / "val") or train_files
    if not train_files:
        raise SystemExit(f"No images found under {root / 'images' / 'train'}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rotate] train images: {len(train_files)}, val images: {len(val_files)}, device={device}")
    # NOTE: dataset_rotate's images/train and images/val largely overlap (same
    # source scans) - val accuracy here mainly tracks augmentation-robustness,
    # not held-out generalization to unseen slides.

    save_augmentation_examples(train_files, args.size, out_path.parent / "augmentation_examples.jpg")

    train_ds = RotationDataset(train_files, args.size, args.samples_per_image)
    val_ds = RotationDataset(val_files, args.size, max(20, args.samples_per_image // 4))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, worker_init_fn=_worker_init_fn)

    model = build_model(args.backbone, args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        print(f"[rotate] epoch {epoch}/{args.epochs} train_loss={running_loss / total:.4f} "
              f"train_acc={correct / total:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "backbone": args.backbone,
                        "angles": ANGLES, "size": args.size}, out_path)
            print(f"[rotate]   saved best checkpoint (val_acc={best_acc:.3f}) -> {out_path}")

    save_prediction_examples(model, val_ds, device, out_path.parent / "prediction_examples.jpg")
    print(f"[rotate] done. best val_acc={best_acc:.3f}. model -> {out_path}")


def predict_orientation(model, img, size, device):
    """Classify on a whole-image resize (matches training - no crop), never on a patch."""
    square = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    tensor = to_tensor(square).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    return int(logits.argmax(1).item())


def predict(args):
    device = pick_device(args.device)
    ckpt = torch.load(args.weights, map_location=device)
    model = build_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    size = ckpt.get("size", args.size)

    src = Path(args.source)
    files = list_images(src) if src.is_dir() else [src]
    if not files:
        raise SystemExit(f"No images found at {src}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [("file", "predicted_rotation_deg")]
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # full resolution - corrected output preserves it
        if img is None:
            print(f"[rotate] skip (unreadable): {path}")
            continue
        k = predict_orientation(model, img, size, device)
        corrected = rotate_k(img, (4 - k) % 4)
        out_path = out_dir / f"{path.stem}_corrected{path.suffix}"
        cv2.imwrite(str(out_path), corrected)
        rows.append((path.name, ANGLES[k]))
        print(f"[rotate] {path.name}: predicted rotation={ANGLES[k]} deg -> {out_path.name}")

    csv_path = out_dir / "rotation_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[rotate] wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="RotNet-style orientation classifier training/inference.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the rotation classifier.")
    p_train.add_argument("--dataset", required=True, help="Root folder with images/train and images/val.")
    p_train.add_argument("--output", default="runs/rotation/rotate_model.pt")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--size", type=int, default=224, help="Whole-image resize (no crop) fed to the model.")
    p_train.add_argument("--samples-per-image", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    p_train.add_argument("--pretrained", action="store_true")
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.set_defaults(func=train)

    p_predict = sub.add_parser("predict", help="Predict + correct orientation for a folder of images.")
    p_predict.add_argument("--weights", default="runs/rotation/rotate_model.pt")
    p_predict.add_argument("--source", required=True)
    p_predict.add_argument("--out", default="runs/rotation/predictions")
    p_predict.add_argument("--device", default="auto")
    p_predict.add_argument("--size", type=int, default=224, help="Fallback resize size if not stored in the checkpoint.")
    p_predict.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()



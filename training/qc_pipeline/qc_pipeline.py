#!/usr/bin/env python3
"""
qc_pipeline.py -- ROI + follicle detection and section quality scoring as one train / eval / predict workflow.

    python training/qc_pipeline/qc_pipeline.py train --name p4    # ROI + follicle detectors, quality model -> runs/qc_pipeline/p4/
    python training/qc_pipeline/qc_pipeline.py eval --run runs/qc_pipeline/p4
    python training/qc_pipeline/qc_pipeline.py predict --run runs/qc_pipeline/p4 --source <slides>
    python training/qc_pipeline/qc_pipeline.py cv --name fcv1     # follicle detector, k-fold by slide

Shared data paths and eval thresholds come from qc_pipeline.yaml (--config); each model's training
options from its own config (detection/roi.yaml, detection/follicle.yaml, quality/qual.yaml, train:
section). --set section.key=value overrides one, e.g.
--set roi.epochs=2. The ROI detector trains on the train/val split of yolo_det (annotate_qc_boxes.py), the
follicle detector on the same split of yolo_follicle (fixed-um/px tiles); the quality model is
cross-validated by slide on all rated sections, then fitted on all of them.

eval reports, on the yolo_det val slides:
  roi       YOLO-OBB metrics of the roi class and detections matched to annotated ROIs
            (oriented-box IoU >= eval.iou, greedy) -> roi_matches.csv
  follicle  the same for follicles (eval.follicle_iou), detected on whole slides -> follicle_matches.csv,
            overlays/*_follicles.jpg; plus mask areas -> follicle_areas.csv: the ROI area that follicle
            masking (boxes padded by eval.follicle_padding_um) removes with the detected vs annotated
            boxes (wrongly removed / wrongly kept mm2, mask IoU)
  oracle    quality on the annotated section shapes -> how good the scorer is on its own
  e2e       detected ROIs split into eval.sections even sections and scored; each rated section is
            compared with the detected section containing its centre (inside the matched ROI);
            coverage = share of rated sections that land in a matched detected ROI
Each slide is scored by the quality CV fold model that did not train on it.
-> <run>/eval/metrics.json, section_eval.csv, overlays/
train --roi-weights / --follicle-weights / --qual-run reuse existing models instead of training.
Freshly trained weights are also copied to models/<name>_{roi_det,follicle_det,qual}_best.pt (overwritten).
predict also writes follicle detections to <out>/follicles/ when the run has a follicle model.

cv trains the follicle detector once per fold<k>.yaml of yolo_follicle (slides balanced by follicle
count; rerunning skips folds that have weights/best.pt) and evaluates each fold model on its held-out
slides as eval does -> <project>/<name>/cv_metrics.json (per fold + pooled out-of-fold), follicle_matches.csv,
follicle_areas.csv. Compare training changes with cv rather than the single val split.
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for running this file as a script
from training.annotate_qc_boxes import (CSV_HEADER, ROI_CLASS, axis_t, list_images, load_csv,  # noqa: E402
                                     load_excluded, load_scaled, slide_key)
from training.config_utils import load_yaml  # noqa: E402
from training.detection import train_det  # noqa: E402
from training.detection.follicle_tiles import detect_tiled, detect_whole, load_calibration, resize, tile_scale  # noqa: E402
from training.quality import train_qual  # noqa: E402
from postprocess import pad_obb  # noqa: E402

QUAD_COLS = CSV_HEADER[2:]
FOLLICLE_CLASS = "follicle"
DETECTORS = (("roi", ROI_CLASS), ("follicle", FOLLICLE_CLASS))
TRAINING = Path(__file__).resolve().parents[1]
MODELS = TRAINING.parent / "models"
MODEL_CONFIGS = {"roi": TRAINING / "detection" / "roi.yaml", "follicle": TRAINING / "detection" / "follicle.yaml",
                 "qual": TRAINING / "quality" / "qual.yaml"}


# ---------------------------------------------------------------------------- helpers
def load_config(a):
    cfg = load_yaml(a.config)
    for kind, path in MODEL_CONFIGS.items():
        cfg[kind] = {**(load_yaml(path).get("train") or {}), **(cfg.get(kind) or {})}
    for item in a.set or []:
        key, _, value = item.partition("=")
        section, _, name = key.partition(".")
        if not name and section in cfg and not isinstance(cfg[section], dict):
            cfg[section] = yaml.safe_load(value)
            continue
        if not name or section not in cfg or not isinstance(cfg[section], dict):
            raise SystemExit(f"--set {item}: expected section.key=value with section in {list(cfg)}")
        cfg[section][name] = yaml.safe_load(value)
    return cfg


def tool_args(parser, argv, overrides, section):
    """Parse a train_det.py / train_qual.py command line, then apply the config values."""
    a = parser.parse_args(argv)
    unknown = set(overrides or {}) - set(vars(a))
    if unknown:
        raise SystemExit(f"config {section}: unknown keys {sorted(unknown)}")
    vars(a).update(overrides or {})
    return a


def val_slides(cfg):
    folder = Path(cfg["data"]["dataset"]) / "images" / "val"
    if not folder.is_dir():
        raise SystemExit(f"{folder} not found; build it with: python training/annotate_boxes.py --prepare-only")
    return sorted({slide_key(p.stem) for p in folder.iterdir()})


def rated_rows(cfg):
    excluded = load_excluded(cfg["data"]["exclude"])
    path = Path(cfg["data"]["sections"])
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["rating"] and r["image"] not in excluded]


def class_boxes(cfg, cls):
    """{image: [cls box, ...]} in boxes.csv order (roi: = roi_idx in sections.csv), excluded images dropped."""
    excluded = load_excluded(cfg["data"]["exclude"])
    boxes = {}
    for name, bs in load_csv(Path(cfg["data"]["boxes"])).items():
        sel = [b for b in bs if b[0] == cls]
        if sel and name not in excluded:
            boxes[name] = sel
    return boxes


def val_slides_key(kind):
    return "val_slides" if kind == "roi" else f"{kind}_val_slides"


def read_manifest(run: Path):
    path = run / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; train first: python training/qc_pipeline/qc_pipeline.py train --name {run.name}")
    return json.loads(path.read_text())


def quad_iou(a, b):
    a, b = np.float32(a).reshape(4, 2), np.float32(b).reshape(4, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def match_rois(gts, dets, thr):
    """Greedy one-to-one matching by IoU: {gt index: (det index, iou)}."""
    pairs = sorted(((quad_iou(g[1:9], d["box"][1:9]), gi, di) for gi, g in enumerate(gts)
                    for di, d in enumerate(dets)), reverse=True)
    match, used = {}, set()
    for iou, gi, di in pairs:
        if iou < thr:
            break
        if gi not in match and di not in used:
            match[gi] = (di, iou)
            used.add(di)
    return match


def section_metrics(y, score, pred):
    return train_qual.metrics(y, score, pred) if len(y) else {"n": 0}


def match_detections(gt, dets, thr, min_conf):
    """Match dets to gt per image (in gt order): ({image: {gt: (det, iou)}}, csv rows, summary)."""
    matches, rows = {}, []
    for name, gs in gt.items():
        ds = dets.get(name, [])
        matches[name] = match_rois(gs, ds, thr)
        hit = {di for di, _ in matches[name].values()}
        for gi in range(len(gs)):
            di, iou = matches[name].get(gi, ("", 0.0))
            rows.append([name, gi, di, round(iou, 4), ds[di]["conf"] if di != "" else ""])
        rows += [[name, "", di, 0.0, ds[di]["conf"]] for di in range(len(ds)) if di not in hit]
    tp = sum(len(v) for v in matches.values())
    n_gt, n_det = sum(len(v) for v in gt.values()), sum(len(dets.get(n, [])) for n in gt)
    ious = [iou for v in matches.values() for _, iou in v.values()]
    summary = {"gt": n_gt, "detected": n_det, "tp": tp, "fp": n_det - tp, "fn": n_gt - tp,
               "precision": round(tp / max(1, n_det), 4), "recall": round(tp / max(1, n_gt), 4),
               "f1": round(2 * tp / max(1, n_det + n_gt), 4),
               "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0, "iou_thr": thr, "min_conf": min_conf}
    return matches, rows, summary


def write_matches(path: Path, kind, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", f"gt_{kind}", f"det_{kind}", "iou", "det_conf"])
        w.writerows(rows)


def yolo_class_metrics(model, cls, det, data: Path, out: Path, name):
    """YOLO-OBB precision / recall / mAP of class cls on the val split of data ({} if not available)."""
    names = yaml.safe_load(data.read_text())["names"]
    if dict(model.names) != names:
        print(f"model classes {model.names} differ from {data} {names}; {cls} YOLO metrics skipped", flush=True)
        return {}
    m = model.val(data=str(data), split="val", imgsz=det.imgsz, batch=det.batch, device=det.device,
                  project=str(out.resolve()), name=name, exist_ok=True, verbose=False)
    cid = next(k for k, v in names.items() if v == cls)
    if cid not in m.box.ap_class_index:
        return {}
    res = m.box.class_result(list(m.box.ap_class_index).index(cid))
    return {k: round(float(v), 4) for k, v in zip(("precision", "recall", "mAP50", "mAP50-95"), res)}


def print_detection(title, n_images, yolo, m, area=None):
    print(f"\n== {title} ({n_images} val images)")
    print("  YOLO   " + ("  ".join(f"{k} {v:.3f}" for k, v in yolo.items()) or "-"))
    print(f"  match  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
          f"mean IoU {m['mean_iou']:.3f}  (tp {m['tp']} fp {m['fp']} fn {m['fn']})")
    if area:
        print(f"  mask   IoU {area['mask_iou']:.3f}  wrongly removed {area['wrongly_removed_mm2']:.3f} mm2  "
              f"wrongly kept {area['wrongly_kept_mm2']:.3f} mm2  = {100 * area['error_share_of_roi']:.2f}% "
              f"of {area['roi_mm2']:.2f} mm2 ROI")


# ---------------------------------------------------------------------------- train
def copy_to_models(src: Path, name):
    MODELS.mkdir(exist_ok=True)
    dst = MODELS / name
    shutil.copy2(src, dst)
    print(f"copied {src.as_posix()} -> {dst.as_posix()}", flush=True)


def cmd_train(cfg, a):
    d = cfg["data"]
    out = Path(cfg["project"]) / a.name
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    out.mkdir(parents=True, exist_ok=True)

    def save(**kw):
        man.update(kw, config=cfg)
        man_path.write_text(json.dumps(man, indent=2))

    for kind, cls in DETECTORS:
        if a.only not in (None, kind):
            continue
        vs = val_slides(cfg)
        given = getattr(a, f"{kind}_weights")
        if given:
            if not Path(given).exists():
                raise SystemExit(f"{given} not found")
            weights = Path(given)
        else:
            if (out / f"{kind}_det").exists():
                raise SystemExit(f"{out / f'{kind}_det'} already exists; use --only <other part>, "
                                 f"--{kind}-weights or a new --name")
            t = tool_args(train_det.build_parser(), ["train"], cfg[kind], kind)
            t.data = d["follicle_dataset"] if kind == "follicle" else d["dataset"]
            t.project, t.name, t.classes = str(out), f"{kind}_det", cls
            train_det.cmd_train(t)
            weights = out / f"{kind}_det" / "weights" / "best.pt"
            copy_to_models(weights, f"{a.name}_{kind}_det_best.pt")
        save(**{f"{kind}_weights": weights.as_posix(), val_slides_key(kind): vs})
    if a.only in (None, "qual"):
        if a.qual_run:
            if not (Path(a.qual_run) / "final" / "best.pt").exists():
                raise SystemExit(f"{a.qual_run}/final/best.pt not found")
            save(qual_run=Path(a.qual_run).as_posix())
        else:
            t = tool_args(train_qual.build_parser(), ["train"], cfg["qual"], "qual")
            if t.no_final or not t.folds:
                raise SystemExit("the pipeline needs qual.folds > 0 (eval uses the fold models) and the final model")
            t.sections_csv, t.source, t.exclude, t.max_side = d["sections"], d["images"], d["exclude"], d["max_side"]
            t.project, t.name = str(out), "qual"
            train_qual.cmd_train(t)
            copy_to_models(out / "qual" / "final" / "best.pt", f"{a.name}_qual_best.pt")
            save(qual_run=(out / "qual").as_posix())
    print(f"\nmanifest: {man_path}  ->  python training/qc_pipeline/qc_pipeline.py eval --run {out.as_posix()}")


# ---------------------------------------------------------------------------- eval
def predict_args(cfg, man, **kw):
    """train_qual.py predict arguments for the run's two models."""
    det = tool_args(train_det.build_parser(), ["train"], cfg["roi"], "roi")
    weights = str(Path(man["qual_run"]) / "final" / "best.pt")
    p = tool_args(train_qual.build_parser(), ["predict", "--weights", weights], {}, "")
    p.roi_weights, p.roi_imgsz, p.roi_max_side = man["roi_weights"], det.imgsz, cfg["data"]["max_side"]
    p.min_conf, p.sections, p.device = cfg["eval"]["min_conf"], cfg["eval"]["sections"], cfg["qual"].get("device")
    vars(p).update(kw)
    return p, det


def cmd_eval(cfg, a):
    from ultralytics import YOLO

    run = Path(a.run)
    man = read_manifest(run)
    for k in ("roi_weights", "qual_run"):
        if k not in man:
            raise SystemExit(f"{run} has no {k}; finish training first")
    qual_run = Path(man["qual_run"])
    cv_path = qual_run / "cv_summary.json"
    if not cv_path.exists():
        raise SystemExit(f"{cv_path} not found; eval needs a quality run trained with folds > 0")
    fold_of = json.loads(cv_path.read_text())["fold_of_slide"]
    vs = set(val_slides(cfg))
    if set(man.get("val_slides", [])) != vs:
        print(f"WARNING: the val split of {cfg['data']['dataset']} changed since the ROI model was registered "
              "(annotate_boxes.py re-splits when slides are added); ROI metrics may include its training slides")
    ev = cfg["eval"]
    out = run / a.out_name
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    gt = {n: bs for n, bs in class_boxes(cfg, ROI_CLASS).items() if slide_key(Path(n).stem) in vs}
    paths = [p for p in list_images(Path(cfg["data"]["images"])) if p.name in gt]
    if not paths:
        raise SystemExit("no val images with annotated ROIs")
    gt = {p_.name: gt[p_.name] for p_ in paths}
    print(f"{len(paths)} val images from {len({slide_key(p.stem) for p in paths})} slides", flush=True)

    p, det = predict_args(cfg, man, device=cfg["roi"].get("device"))
    yolo = yolo_class_metrics(YOLO(man["roi_weights"]), ROI_CLASS, det, Path(cfg["data"]["dataset"]) / "data.yaml",
                              out, "roi_val")

    # detect, then match to annotated ROIs
    cache = {}
    dets = train_qual.read_rois(train_qual.detect_rois(p, paths, out, cache), p.sections, p.min_conf)
    matches, match_rows, matching = match_detections(gt, dets, ev["iou"], ev["min_conf"])
    write_matches(out / "roi_matches.csv", "roi", match_rows)

    follicle = eval_follicles(cfg, man, vs, out)

    # quality: oracle on annotated sections, end-to-end on detected sections, each by its out-of-fold model
    device = train_qual.device_of(cfg["qual"].get("device"))
    models = {}
    rated = defaultdict(list)
    for r in rated_rows(cfg):
        if r["image"] in gt:
            rated[r["image"]].append(r)
    rows = []
    for path in paths:
        name, rs = path.name, rated.get(path.name, [])
        fold = fold_of.get(slide_key(path.stem))
        if fold is None or not rs:
            continue
        if fold not in models:
            models[fold] = train_qual.load_model(qual_run / f"fold{fold}" / "best.pt", device)
        model, ckpt = models[fold]
        if (name, ckpt["max_side"]) not in cache:
            img, _, _, f = load_scaled(path, ckpt["max_side"])
            cache[name, ckpt["max_side"]] = (img, f)
        img, f = cache[name, ckpt["max_side"]]
        src = train_qual.prepare(img, train_qual.mode_of(ckpt))
        gt_quads = [np.array([float(r[k]) for k in QUAD_COLS], np.float32).reshape(4, 2) for r in rs]
        tiles = [train_qual.warp_section(src, q * f, ckpt["imgsz"], letterbox=ckpt.get("letterbox", True))
                 for q in gt_quads]
        o_score, o_pred, _ = train_qual.predict_tiles(model, tiles, device, p.batch)
        e2e = {}
        if dets.get(name):
            for row in train_qual.score_image(model, ckpt, path, dets[name], out / "overlays", device, p.batch, cache):
                e2e[row[1], row[2]] = (row[3], row[4])
        cache.pop((name, ckpt["max_side"]), None)
        for r, q, os_, op in zip(rs, gt_quads, o_score, o_pred):
            gi = int(r["roi_idx"])
            di, iou = matches[name].get(gi, ("", 0.0))
            ds, es, ep = "", "", ""
            if di != "":
                t = axis_t(dets[name][di]["box"], *q.mean(axis=0))
                if 0 <= t <= 1:
                    ds = min(int(t * p.sections), p.sections - 1)
                    es, ep = e2e[di, ds]
            rows.append([name, gi, int(r["section"]), int(r["rating"]), round(float(os_), 3), int(op),
                         di, round(iou, 4), ds, es, ep, fold])
    with open(out / "section_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt_roi", "section", "rating", "oracle_score", "oracle_pred",
                    "det_roi", "iou", "det_section", "e2e_score", "e2e_pred", "qual_fold"])
        w.writerows(rows)

    y = [r[3] for r in rows]
    covered = [r for r in rows if r[9] != ""]
    oracle = section_metrics(y, [r[4] for r in rows], [r[5] for r in rows])
    end2end = section_metrics([r[3] for r in covered], [r[9] for r in covered], [r[10] for r in covered])
    end2end["coverage"] = round(len(covered) / max(1, len(rows)), 4)
    oracle_cov = section_metrics([r[3] for r in covered], [r[4] for r in covered], [r[5] for r in covered])
    summary = {"run": run.as_posix(), "val_slides": sorted({slide_key(p_.stem) for p_ in paths}),
               "roi": {"yolo": yolo, "matching": matching}, "oracle": oracle,
               "oracle_on_covered": oracle_cov, "e2e": end2end}
    if follicle:
        summary["follicle"] = follicle
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    print_detection("ROI detection", len(paths), yolo, matching)
    if follicle:
        print_detection("follicle detection", follicle["images"], follicle["yolo"], follicle["matching"],
                        follicle["area"])
    print(f"== section quality ({len(rows)} rated val sections, out-of-fold models)")
    for label, mt in (("oracle", oracle), ("oracle*", oracle_cov), ("e2e*", end2end)):
        print(f"  {label:8s} n={mt['n']:3d}  " + (train_qual.fmt(mt) if mt["n"] else "-"))
    print(f"  * = rated sections inside a matched detected ROI (coverage {end2end['coverage']:.2f})")
    print(f"\nwrote {out / 'metrics.json'}, section_eval.csv, roi_matches.csv"
          + (", follicle_matches.csv" if follicle else "") + ", overlays/")


def annotated_images(cfg):
    """Slides in data.images with at least one box in data.boxes, excluded ones dropped."""
    excluded = load_excluded(cfg["data"]["exclude"])
    boxes = {n for n, bs in load_csv(Path(cfg["data"]["boxes"])).items() if bs and n not in excluded}
    return [p for p in list_images(Path(cfg["data"]["images"])) if p.name in boxes]


def follicle_data(cfg, model):
    """data.yaml the follicle model's YOLO metrics are computed on: the tile dataset for tiled weights."""
    return Path(cfg["data"]["follicle_dataset" if tile_scale(model) else "dataset"]) / "data.yaml"


def mask_areas(shape, um_per_px, gt, found, rois, pad_um, res_um):
    """Area (mm2) inside the annotated ROIs (whole slide if none) that follicle masking removes, as
    postprocess.py does with follicle_mask_frac 1: union of the padded boxes, rasterised at res_um um/px."""
    s = um_per_px / res_um
    h, w = max(1, round(shape[0] * s)), max(1, round(shape[1] * s))

    def fill(quads, pad):
        m = np.zeros((h, w), np.uint8)
        for q in quads:
            q = pad_obb(np.float64(q).reshape(4, 2) * s, pad)
            cv2.fillPoly(m, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 1)
        return m.astype(bool)

    roi = fill(rois, 0) if rois else np.ones((h, w), bool)
    g, p = fill(gt, pad_um / res_um) & roi, fill(found, pad_um / res_um) & roi
    mm2 = (res_um / 1000) ** 2
    return {"roi_mm2": roi.sum() * mm2, "gt_mm2": g.sum() * mm2, "removed_mm2": p.sum() * mm2,
            "correctly_removed_mm2": (g & p).sum() * mm2, "wrongly_removed_mm2": (p & ~g).sum() * mm2,
            "wrongly_kept_mm2": (g & ~p).sum() * mm2}


def area_summary(rows):
    tot = {k: float(sum(r[k] for r in rows)) for k in ("roi_mm2", "gt_mm2", "removed_mm2", "correctly_removed_mm2",
                                                      "wrongly_removed_mm2", "wrongly_kept_mm2")}
    union = tot["correctly_removed_mm2"] + tot["wrongly_removed_mm2"] + tot["wrongly_kept_mm2"]
    return {"slides": len(rows), **{k: round(v, 4) for k, v in tot.items()},
            "mask_iou": round(tot["correctly_removed_mm2"] / union, 4) if union else 1.0,
            "error_share_of_roi": round((tot["wrongly_removed_mm2"] + tot["wrongly_kept_mm2"]) / max(tot["roi_mm2"], 1e-9), 5)}


def detect_follicles_on(cfg, model, det, paths, out: Path):
    """Detect follicles on whole slides (tiled at the weights' um/px, else downscaled to data.max_side).
    Returns ({image: annotated boxes}, {image: [{box, conf}]}, per-slide mask-area rows)."""
    ev, calibration, scale = cfg["eval"], load_calibration(cfg["data"]["calibration"]), tile_scale(model)
    fol, rois = class_boxes(cfg, FOLLICLE_CLASS), class_boxes(cfg, ROI_CLASS)
    gt, dets, area_rows = {}, {}, []
    for path in paths:
        upp = calibration.get(path.name)
        if scale and upp is None:
            print(f"  {path.name}: no um/px in {cfg['data']['calibration']}; skipped", flush=True)
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if scale:
            found = detect_tiled(model, img, upp, scale, ev["follicle_conf"], det.device)
        else:
            found = detect_whole(model, img, cfg["data"]["max_side"], det.imgsz, ev["follicle_conf"], det.device)
        gt[path.name] = fol.get(path.name, [])
        dets[path.name] = [{"box": [FOLLICLE_CLASS, *(float(v) for v in q.ravel())], "conf": round(c, 3)}
                           for q, c in found]
        if upp:
            area_rows.append({"image": path.name, **mask_areas(
                img.shape, upp, [b[1:9] for b in gt[path.name]], [q for q, _ in found],
                [b[1:9] for b in rois.get(path.name, [])], ev["follicle_padding_um"], ev["mask_um_per_px"])})
        f = min(1.0, cfg["data"]["max_side"] / max(img.shape[:2]))
        canvas = resize(img, f)
        # annotated green, detected red
        for b, col in [(g, (0, 200, 0)) for g in gt[path.name]] + [(d["box"], (0, 0, 255)) for d in dets[path.name]]:
            q = (np.float32(b[1:9]).reshape(4, 2) * f).round().astype(np.int32)
            cv2.polylines(canvas, [q], True, col, 2, cv2.LINE_AA)
        cv2.imwrite(str(out / "overlays" / f"{path.stem}_follicles.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {path.name}: {len(gt[path.name])} annotated, {len(found)} detected", flush=True)
    return gt, dets, area_rows


def write_area_rows(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
        w.writeheader()
        w.writerows([{k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()} for r in rows])


def eval_follicles(cfg, man, vs, out: Path):
    """Follicle YOLO metrics, IoU matching and mask areas on the val slides; None if the run has no follicle model."""
    from ultralytics import YOLO

    if "follicle_weights" not in man:
        print("no follicle model in this run (train --only follicle); follicle eval skipped", flush=True)
        return None
    if set(man.get(val_slides_key("follicle"), [])) != vs:
        print("WARNING: the val split changed since the follicle model was registered; "
              "follicle metrics may include its training slides")
    ev = cfg["eval"]
    det = tool_args(train_det.build_parser(), ["train"], cfg["follicle"], "follicle")
    model = YOLO(man["follicle_weights"])
    yolo = yolo_class_metrics(model, FOLLICLE_CLASS, det, follicle_data(cfg, model), out, "follicle_val")

    paths = [p for p in annotated_images(cfg) if slide_key(p.stem) in vs]
    gt, dets, area_rows = detect_follicles_on(cfg, model, det, paths, out)
    _, rows, matching = match_detections(gt, dets, ev["follicle_iou"], ev["follicle_conf"])
    write_matches(out / "follicle_matches.csv", "follicle", rows)
    write_area_rows(out / "follicle_areas.csv", area_rows)
    return {"images": len(gt), "yolo": yolo, "matching": matching, "area": area_summary(area_rows)}


# ---------------------------------------------------------------------------- cv
def cmd_cv(cfg, a):
    """Slide-level k-fold cross-validation of the follicle detector on the folds of data.follicle_dataset."""
    from ultralytics import YOLO

    ds, ev = Path(cfg["data"]["follicle_dataset"]), cfg["eval"]
    folds_path = ds / "folds.json"
    if not folds_path.exists():
        raise SystemExit(f"{folds_path} not found; build it with: python training/annotate_qc_boxes.py --prepare-only")
    fold_of = json.loads(folds_path.read_text())
    folds = sorted(set(fold_of.values()))
    if a.folds:
        folds = [k for k in folds if k in a.folds]
    out = Path(cfg["project"]) / a.name
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    det = tool_args(train_det.build_parser(), ["train"], cfg["follicle"], "follicle")
    images = annotated_images(cfg)

    gt_all, dets_all, area_all, per_fold = {}, {}, [], {}
    for k in folds:
        run = out / f"fold{k}"
        weights = run / "weights" / "best.pt"
        if not weights.exists():
            if run.exists():
                raise SystemExit(f"{run} has no weights/best.pt (interrupted?); delete it to retrain fold {k}")
            print(f"\n==== fold {k}: training", flush=True)
            t = tool_args(train_det.build_parser(), ["train"], cfg["follicle"], "follicle")
            t.data, t.project, t.name, t.classes = str(ds / f"fold{k}.yaml"), str(out), run.name, FOLLICLE_CLASS
            train_det.cmd_train(t)
        print(f"\n==== fold {k}: evaluating {weights}", flush=True)
        model = YOLO(weights)
        yolo = yolo_class_metrics(model, FOLLICLE_CLASS, det, ds / f"fold{k}.yaml", out, f"fold{k}_val")
        paths = [p for p in images if fold_of.get(slide_key(p.stem)) == k]
        gt, dets, area_rows = detect_follicles_on(cfg, model, det, paths, out)
        _, _, matching = match_detections(gt, dets, ev["follicle_iou"], ev["follicle_conf"])
        per_fold[k] = {"images": len(gt), "yolo": yolo, "matching": matching, "area": area_summary(area_rows)}
        gt_all.update(gt)
        dets_all.update(dets)
        area_all += [{"fold": k, **r} for r in area_rows]

    _, rows, pooled = match_detections(gt_all, dets_all, ev["follicle_iou"], ev["follicle_conf"])
    write_matches(out / "follicle_matches.csv", "follicle", rows)
    write_area_rows(out / "follicle_areas.csv", area_all)
    yolo_keys = sorted({key for f in per_fold.values() for key in f["yolo"]})
    yolo_mean = {key: round(float(np.mean([f["yolo"][key] for f in per_fold.values() if key in f["yolo"]])), 4)
                 for key in yolo_keys}
    summary = {"dataset": yaml.safe_load((ds / "data.yaml").read_text()), "folds": per_fold,
               "pooled": {"images": len(gt_all), "yolo_mean": yolo_mean, "matching": pooled,
                          "area": area_summary(area_all)}}
    (out / "cv_metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    for k, f in per_fold.items():
        print_detection(f"fold {k}", f["images"], f["yolo"], f["matching"], f["area"])
    print_detection("pooled out-of-fold", len(gt_all), yolo_mean, pooled, summary["pooled"]["area"])
    print(f"\nwrote {out / 'cv_metrics.json'}, follicle_matches.csv, follicle_areas.csv, overlays/")


# ---------------------------------------------------------------------------- predict
def cmd_predict(cfg, a):
    run = Path(a.run)
    man = read_manifest(run)
    out = Path(a.out or run / "predict")
    p, _ = predict_args(cfg, man, source=a.source, rois=a.rois, out=str(out))
    train_qual.cmd_predict(p)
    if "follicle_weights" in man:
        det = tool_args(train_det.build_parser(), ["train"], cfg["follicle"], "follicle")
        f = train_det.build_parser().parse_args(["predict", "--weights", man["follicle_weights"],
                                                 "--source", a.source, "--out", str(out / "follicles")])
        f.max_side, f.imgsz, f.conf, f.device = cfg["data"]["max_side"], det.imgsz, cfg["eval"]["follicle_conf"], det.device
        train_det.cmd_predict(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("qc_pipeline.yaml")))
    ap.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                    help="override a config value (sections: project, data, eval, roi, follicle, qual)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the ROI and follicle detectors and the quality model")
    t.add_argument("--name", required=True)
    t.add_argument("--only", choices=["roi", "follicle", "qual"])
    t.add_argument("--roi-weights", help="use these ROI weights instead of training")
    t.add_argument("--follicle-weights", help="use these follicle weights instead of training")
    t.add_argument("--qual-run", help="use this train_qual.py run (with fold*/ and final/) instead of training")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="evaluate a run on the yolo_det val slides")
    e.add_argument("--run", required=True)
    e.add_argument("--out-name", default="eval", help="output folder inside the run")
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("cv", help="slide-level k-fold cross-validation of the follicle detector")
    c.add_argument("--name", required=True, help="output: <project>/<name>/fold<k>/, cv_metrics.json")
    c.add_argument("--folds", type=lambda s: [int(v) for v in s.split(",")], help="only these folds, e.g. 0,1")
    c.set_defaults(func=cmd_cv)

    d = sub.add_parser("predict", help="detect ROIs + follicles and score the ROI sections on new slides")
    d.add_argument("--run", required=True)
    d.add_argument("--source", required=True)
    d.add_argument("--rois", help="optional boxes.csv | detections.csv | sections.csv instead of detecting")
    d.add_argument("--out", help="default: <run>/predict")
    d.set_defaults(func=cmd_predict)

    a = ap.parse_args()
    a.func(load_config(a), a)


if __name__ == "__main__":
    main()

# Training

Run everything from the repo root. Each trainer loads the YAML next to it; `--config <yaml>` uses another
file, command-line flags override both, and each run records the values it used (`args.json` / `args.yaml`).

| Model | Trainer | Config | Data | Output |
|---|---|---|---|---|
| SC / Epidermis / gland segmentation | `semseg/train_semseg.py` | `semseg/semseg.yaml` | `datasets/seg_finetune` | `runs/semseg/<name>` |
| ROI detector | `detection/train_det.py --target roi` | `detection/roi.yaml` | `datasets/roi_follicle/yolo_det` | `runs/det/roi/<name>` |
| Follicle detector | `detection/train_det.py --target follicle` | `detection/follicle.yaml` | `datasets/roi_follicle/yolo_follicle` | `runs/det/follicle/<name>` |
| Section quality | `quality/train_qual.py` | `quality/qual.yaml` | `datasets/roi_follicle/sections.csv` | `runs/qual/<name>` |
| Orientation | `rotation/rotate.py` | `rotation/rotate.yaml` | `datasets/rotate` | `runs/rotation/` |
| ROI + follicle + quality (QC) | `qc_pipeline/qc_pipeline.py` | `qc_pipeline/qc_pipeline.yaml` + the three above | `datasets/roi_follicle` | `runs/qc_pipeline/<name>` |

Copy a finished model into `models/` and point `base_config.yaml` at it to use it in `predict.py`.

## Segmentation
    python training/semseg/add_segmentations.py               # draw / correct polygons in datasets/seg_finetune
    python training/semseg/train_semseg.py train --name v9
    python training/semseg/show_train_samples.py              # montage of augmented training crops

## ROI & follicle boxes, section ratings
    python training/annotate_boxes.py   (g / "Images v" = searchable image list; r = rate mode: hover a section + 1/2/3, 0 clears, drag dividers)
        -> datasets/roi_follicle/{boxes.csv, labels/, sections.csv, section_crops/} and rebuilds yolo_det + yolo_follicle on quit
    python training/annotate_boxes.py --prepare-only           # rebuild yolo_det, yolo_follicle + crops without annotating
    python training/annotate_qc_boxes.py --pred-source <slides> --um-per-px 0.2577 --preds <roi detections.csv>,<follicle detections.csv>
        pseudolabels: review train_det.py predict boxes on new slides (>= --min-conf 0.25); a / Accept copies the slide
        into images/ with its boxes, ratings and um/px; unaccepted edits wait in datasets/roi_follicle/pending/
        yolo_follicle = 1024 px follicle tiles of each slide resampled to 2.5 um/px (--follicle-um-per-px), same val
        slides as yolo_det, plus fold<k>.yaml for 5-fold CV by slide. The um/px of every image in images/ must be in
        datasets/roi_follicle/scale_calibration.json (downscaled .jpg = original um/px x original long side / jpg long side);
        images missing there are skipped with a warning.

    python training/detection/train_det.py train --target roi --name r4
    python training/detection/train_det.py train --target follicle --name f5   # stamps the tile um/px into the weights
    python training/detection/train_det.py predict --target roi --source <slides>
    python training/detection/train_det.py predict --target follicle --weights <best.pt> --source <slides> --um-per-px 0.2577

    python training/quality/train_qual.py train --name cv3     # 5-fold CV by slide + final model (cv_summary.json, oof_predictions.csv, final/best.pt)
    python training/quality/train_qual.py predict --weights runs/qual/cv3/final/best.pt --rois datasets/roi_follicle/boxes.csv
        --rois also takes sections.csv or a train_det.py detections.csv (--source = the images it refers to, --min-conf);
        without --rois, ROIs are detected with --roi-weights and saved to <out>/detections.csv

## QC pipeline (ROI + follicle detectors + quality model)
    python training/qc_pipeline/qc_pipeline.py train --name p4
        --only roi|follicle|qual trains one part; --roi-weights / --follicle-weights <best.pt> / --qual-run <run dir> reuse models
    python training/qc_pipeline/qc_pipeline.py eval --run runs/qc_pipeline/p4
        val slides: ROI + follicle mAP and IoU matching, follicle mask areas (ROI area wrongly removed / kept by
        follicle masking vs the annotated boxes), quality on annotated (oracle) and detected (e2e) sections,
        each slide scored by its out-of-fold CV model -> <run>/eval/metrics.json, section_eval.csv, *_matches.csv,
        follicle_areas.csv, overlays/
    python training/qc_pipeline/qc_pipeline.py cv --name fcv1   # follicle detector: train + eval each fold<k>.yaml
        -> runs/qc_pipeline/fcv1/cv_metrics.json (per fold + pooled out-of-fold); rerun resumes, --folds 0,1 runs a subset.
        Use this, not the single val split, to compare follicle training changes.
    python training/qc_pipeline/qc_pipeline.py predict --run runs/qc_pipeline/p4 --source <slides>
    python training/qc_pipeline/qc_pipeline.py --set roi.epochs=2 --set qual.folds=2 train --name quick

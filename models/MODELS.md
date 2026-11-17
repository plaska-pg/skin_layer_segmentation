this is for only the good save moels. these are pushed to github. and are in prod hopefully
Active ones are set in base_config.yaml. Trainers + configs: training/<model>/.
- `3cls_best_v8_e.pt` (active) - SC/Epidermis/gland segmentation, v8_edge (v7 fine-tune, --edge-weight 2 --edge-width 3; recipe in training/semseg/semseg.yaml). Val (10 imgs) Dice SC .897 / Epi .944 vs v7 .873 / .917
- `rotate_model.pt` (active) - orientation classifier (training/rotation/rotate.py)
- `follicle_det_v4.pt` (active) - YOLO11n-OBB follicle detector (training/detection/train_det.py --target follicle)
- `roi_det_v3.pt` (active) - YOLO11n-OBB ROI detector (class `roi`, --target roi)
- `qual_v2.pt` (active) - section quality scorer (training/quality/train_qual.py QualNet, rating 1-3)
- `<run>_roi_det_best.pt`, `<run>_follicle_det_best.pt`, `<run>_qual_best.pt` - automatic copies from `qc_pipeline.py train --name <run>`; not active unless set in base_config.yaml
- `init/` - starting checkpoints for training, not used at inference (gitignored): `3cls_best_v7.pt` (previous segmentation model, init for v8_edge), `yolo11n-obb.pt` (detector init), `hist_seg_3cls.pt` (PMC Histo-Seg 3-class U-Net, + its args/scale json); `yolo26n.pt` is only Ultralytics' AMP-check download (it re-downloads it to weights/ when training)
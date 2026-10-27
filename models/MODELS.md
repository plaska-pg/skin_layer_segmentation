this is for only the good save moels. these are pushed to github. and are in prod hopefully
- `2cls_best.pt` - segmentation
- `rotate_model.pt` - orientation classifier (rotate.py)
- `follicle_det.pt` - YOLO11n-OBB follicle detector, copied from pretrained_model/runs/follicle_det/rot1-2/weights/best.pt
- `roi_det.pt` - YOLO11n-OBB ROI detector (class `roi`), copied from pretrained_model/runs/roi_det/train/weights/best.pt
- `qual_v1.pt` - section quality scorer (train_qual.py QualNet, rating 1-3), copied from pretrained_model/runs/pipeline/p1/qual/final/best.pt
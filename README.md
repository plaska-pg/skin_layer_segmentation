 # Skin layer segmentation and quantification
 Measures the Epidermis, Stratum Corneum layers, and Glands on H&E skin images.

Train (on PMC data), finetune on (P&G data), pseudolabeling (on generated segmented P&G data) and run inference on H&E stained skin 

## Quick start

Run the app interface
```
streamlit run interface/app.py
```

## For pipeline Dev

### Download the repo from github (https://github.com/plaska-pg/skin_layer_segmentation)

```powershell
git clone https://github.com/plaska-pg/skin_layer_segmentation.git "$HOME\skin_layer_segmentation"
cd "$HOME\skin_layer_segmentation"
```
### Environment setup

Install python3.14. You can do this by typing python in the powershell terminal (if on windows). This'll pull up an install window. The commands above clone the repo into your home folder (for example, `C:\Users\plas.ka\skin_layer_segmentation` on this computer), keeping the virtual environment off OneDrive.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins the exact packages. The background `run_predict_background*.ps1` scripts resolve the interpreter from `.venv\Scripts\python.exe` under `$ProjectDir`, so keep the venv there (or update `$ProjectDir`).



### Repo layout
- root: inference pipeline (`predict.py` -> `postprocess.py` -> `measurement.py`, settings in `base_config.yaml`), `eval_seg.py`, `interface/`
- `training/<model>/`: each model's trainer + its config (`semseg/semseg.yaml`, `detection/roi.yaml` + `follicle.yaml`, `quality/qual.yaml`, `rotation/rotate.yaml`); `training/qc_pipeline/` trains/evaluates the ROI + follicle + quality models together. Trainers load their YAML automatically; `--config <yaml>` swaps it, command-line flags override it. Run them from the repo root.
- `datasets/`: `seg_finetune` (segmentation), `roi_follicle` (ROI/follicle boxes + section ratings), `rotate`
- `models/`: production weights (see `models/MODELS.md`); `models/init/`: starting checkpoints for training
- `runs/`: training/eval outputs; `archive/`: retired code and runs

### Pre-train a skin segmentation model using this data: https://pmc.ncbi.nlm.nih.gov/articles/PMC11803237/
How to train:
Download the data, convert to yolo format, and run UNET model to segment the "Keratin", "Epidermis", and "gland" classes. 
OG dataset was 0.28um/px (20x), but published at 0.7 um/px (8x) So the pretrained model was trained at 0.7um/px
pretrained 3 class model: models\init\hist_seg_3cls.pt (+ hist_seg_3cls_args.json, hist_seg_3cls_scale.json)

### Add or adjust P&G segmentations (pre-finetuning)

```
python training\semseg\add_segmentations.py
```

### then finetune on our segmented images in datasets\seg_finetune

The recipe (currently v8_edge) is in training\semseg\semseg.yaml; only the run name is needed:
```
.\.venv\Scripts\python.exe -u training\semseg\train_semseg.py train --name v9
```



### then pseudolabeling on nicely segmented images
  use .txt segmentation cooredinates in the labels folder (corresponding to an image with the same prefix) 
python training\semseg\add_segmentations.py --source "C:\path\to\image_folder"

### then for interence on data in yolo_skin_seg\inference_images

python predict.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S_EX 8_24 TIFF_H_E"

python predict.py --source "difficult_test_images" --limit 5

python predict.py --weights "runs\semseg\dataset_3cls_gland_v2\best.pt" --source "difficult_test_images" --limit 2

### to know:

BX61 10X → 0.645 µm/px
Motic 40X scan → 0.260 µm/px


## Other Models

### Rotation model: 
Rotates the image so that the SC and Epidermis layers are at the top. 
------------- train (settings in training\rotation\rotate.yaml):--------------
python training\rotation\rotate.py train
----------- inference:---------------
python training\rotation\rotate.py predict --source difficult_test_images --out runs/rotation/difficult_test_predictions --device cuda


### Follicle detection model:
YOLO11n-OBB (class `follicle`), models\follicle_det_v4.pt. Train: `python training\detection\train_det.py train --target follicle --name <run>` (training\detection\follicle.yaml).
New models train on `datasets\roi_follicle\yolo_follicle` (1024 px tiles at 2.5 um/px); the scale is stamped into the weights and
postprocess.py tiles each rotated image at that scale using the slide's um/px (v4 still runs on the whole slide at `follicle_max_side`).
Compare follicle models with `python training\qc_pipeline\qc_pipeline.py cv --name <run>` (5-fold by slide).
In postprocess.py it runs on the rotated image and adds a "Follicles (N)" panel
to the steps image. Configure via the `follicle_*` keys in base_config.yaml (`follicle_enabled: false` to turn off).

### ROI detection model:
YOLO11n-OBB (class `roi`), models\roi_det_v3.pt. Train: `python training\detection\train_det.py train --target roi --name <run>` (training\detection\roi.yaml).
Runs on the rotated image and adds a "ROI Boxes (N)" panel to the steps image. Configure via the `roi_*` keys
in base_config.yaml; if no ROI is detected (or `roi_enabled: false`) the full frame is used.

### Section quality model:
models\qual_v2.pt. Annotate/rate with `python training\annotate_boxes.py`, train with `python training\quality\train_qual.py train --name <run>` (training\quality\qual.yaml).
ROI + follicle + quality together: `python training\qc_pipeline\qc_pipeline.py train --name <run>`, then `eval --run runs\qc_pipeline\<run>` (see training\README.md).

### Postprocessing order (postprocess.py)
1. Rotate
2. Detect follicles
3. Detect ROI boxes
4. Remove SC/Epidermis outside the union of ROI boxes and inside the bottom `follicle_mask_frac` of each follicle box
5. Per-class elliptical opening then closing (`sc_erod_dil_kernel_size_um`, `epidermal_erod_dil_kernel_size_um`, `gland_erod_dil_kernel_size_um`)
6. Drop SC/Epidermis components not within `sc_epidermis_contact_gap_um` of the other class (`sc_epidermis_contact_enabled`)
7. Trim `lateral_clip_frac` per side, then measure


# adding more training examples as corrected pseudolabels

```
python training\semseg\add_segmentations.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S_Exp_SEP_2024_D3_H_E"

python training\semseg\add_segmentations.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\Skin Biology Technology Development - Fundamental_MDE (Multi Decade & Ethnicity study)\MDE-TIFF"

# review and modify current training segmentations? like using interface in add_segmentations.py
python training\semseg\add_segmentations.py 

# to do

Validate epidermal thickness on "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\Skin Biology Technology Development - Fundamental_MDE (Multi Decade & Ethnicity study)\Results-Learnings\Makio Analysis\AntiAging\MDE credentialling\Histology\Epi_Thickness_CA_face.csv"

  python predict.py --source "difficult_test_images"

just take tortuosity measure over 200um window sections of EDJ line. Like the length of the black EDJ line/ the euclidean distance between the two ends of the black line in the window.. then average. 


```
.\.venv\Scripts\python.exe -u training\semseg\train_semseg.py train `
   --data datasets\seg_finetune `
  --init-weights models\3cls_best_v6.pt `
  --encoder resnet34 --epochs 200 --batch 2 --accumulate 1 --crop 512 --lr 2e-4 `
  --class-weights "SC=3" `
   --blur-p 0.3 --downup-p 0.3 --jpeg-p 0.3 `
   --val-every 5 --patience 15 --workers 0 --device auto `
  --project runs\semseg --name dataset_3cls_gland_v7

  
Loads each image with its `predicted_MDE-TIFF\raw_segmentations` labels (`--labels postprocessed` for the postprocessed ones, `--preds` for another predict.py output folder). Correct any class (1/2/3; draw to add, merged with overlapping polygons of that class; Edit mode; `c` / Cut mode to outline and remove a large part of the active class's polygons), then press `a` / Accept: the TIFF, its labels and um_per_px are added to datasets\seg_finetune (train, or val for ~20% of subjects). Images already in the dataset are skipped. The trainer now reads .tif/.tiff too.


quality, follicle detection, roi detection, and rotation should all be under qc_pipeline. 


for the qc_pipeline, to add/annotate data:
```
python training\annotate_qc_boxes.py
```

# For current 3cls segs dataset: 
  add/adjust seg_finetune dataset: 
  ```
  python training\semseg\add_segmentations.py
  ```
  to add pseudolabels (adjust the predicted segmentations)
  ```
  training\semseg\add_segmentations.py --source "<image folder>" 
  ```

`qc_pipeline.py train --name <run>` also copies the freshly trained weights to `models\<run>_roi_det_best.pt`, `models\<run>_follicle_det_best.pt` and `models\<run>_qual_best.pt` (overwritten on retrain; not copied when reusing via `--roi-weights` / `--follicle-weights` / `--qual-run`). Activate one by pointing base_config.yaml at it. 


# can i use pseudolabels to add to the follicle/roi/quality training dataset?
Yes: predict boxes on new slides, review them in the annotator, and accept the good ones (`a`):
```
python training\detection\train_det.py predict --target roi --source "<slides>" --out runs\det\roi\pseudo
python training\detection\train_det.py predict --target follicle --weights models\follicle_det_v4.pt --source "<slides>" --out runs\det\follicle\pseudo
python training\annotate_qc_boxes.py --pred-source "<slides>" --um-per-px 0.2577 --preds runs\det\roi\pseudo\detections.csv,runs\det\follicle\pseudo\detections.csv
```
Fix every box (add missed follicles: unboxed follicles teach the model to miss them) and rate sections (`r`). Unaccepted edits stay in `datasets\roi_follicle\pending\`; Accept copies the slide into `datasets\roi_follicle\images` with its boxes, ratings and um/px. 

# the streamlit app design sucks and is ways too complicaed. 


how can i add the images from here to the  roi_follicle training dataset ?
 "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\Skin Biology Technology Development - Fundamental_MDE (Multi Decade & Ethnicity study)\MDE-TIFF"

tyring to add pseudolabels to box annotation 

$src = "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\Skin Biology Technology Development - Fundamental_MDE (Multi Decade & Ethnicity study)\MDE-TIFF"
python training\annotate_qc_boxes.py --pred-source $src --um-per-px 0.2577 --preds runs\det\roi\pseudo_mde\detections.csv,runs\det\follicle\pseudo_mde\detections.csv


11/17 FINISH THIS: 
$src = "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\Skin Biology Technology Development - Fundamental_MDE (Multi Decade & Ethnicity study)\MDE-TIFF"
python training\annotate_qc_boxes.py --pred-source $src --um-per-px 0.2577 --preds runs\det\roi\pseudo_mde\detections.csv,runs\det\follicle\pseudo_mde\detections.csv

  python predict.py --source "difficult_test_images"
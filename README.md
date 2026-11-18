 # Skin layer segmentation and quantification
 Measures the Epidermis, Stratum Corneum layers, and Glands on H&E skin images.

Train (on PMC data), finetune on (P&G data), pseudolabeling (on generated segmented P&G data) and run inference on H&E stained skin 

## Use the app (Windows)

1. Open the shared OneDrive folder of the app and choose **Add shortcut to My files** (or **Sync**), so it shows up in File Explorer.
2. Double-click **`run_app.cmd`** in that folder. The app opens in your browser; keep the black window open while you use it (closing it stops the app).

The first start takes a while: it offers to install Python 3.14 if it's missing, downloads the packages (PyTorch with GPU support if the computer has an NVIDIA GPU, otherwise the CPU version) and the models. Later starts take seconds. Everything runs on your own computer; installed files go to `%LOCALAPPDATA%\skin_layer_segmentation` (delete that folder to reinstall from scratch, e.g. after getting a GPU), and results are written next to your images.

## Dev install

1. Install Python 3.14 (on Windows, typing `python` in PowerShell opens the installer).
2. Clone the repo outside OneDrive (the trained weights in `models/` come with it):

```powershell
git clone https://github.com/plaska-pg/skin_layer_segmentation.git "$HOME\skin_layer_segmentation"
cd "$HOME\skin_layer_segmentation"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install PyTorch for your hardware, then the rest:

```powershell
# NVIDIA GPU
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# or CPU only (much slower)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

`.\run_app.ps1` in a clone does steps 1-3 itself (in `.venv`), runs `git pull`, then starts the app.

### Publishing a new version to the OneDrive folder

Commit, then mirror HEAD into the shared folder (give users view-only access):
```powershell
.\publish.ps1 -Destination "C:\Users\<you>\OneDrive - Procter and Gamble\<shared app folder>"
```
Users get it as soon as OneDrive syncs; `run_app.cmd` prints the published `VERSION` at start.

### Browser version (GitHub Pages)

`web/` is the same app running entirely in the browser (Chrome or Edge): models via ONNX Runtime Web (WebGPU), measuring and editing via `postprocess.py` & co. in Pyodide. Images never leave the computer; results are written into the chosen folder in the same layout as `predict.py`.

- `.svs` (Aperio tiled JPEG; not JPEG 2000) is read tile by tile at the µm/px in its header, and a lossless `TIFF_<folder>/<stem>.tif` copy is written unless a TIFF of that name already exists. Images over 45 MP are measured at 0.4 µm/px, cropped to the tissue (ROI / follicle / quality still see the whole slide).
- Publish: `.github/workflows/pages.yml` runs on every push to `main` (Settings > Pages > Source: **GitHub Actions**). It exports `models/` to ONNX (`web/export_onnx.py`, cached until the weights or code change) and deploys the site.
- Local: `pip install onnx==1.23.2 onnxruntime==1.32.0 onnxscript==0.7.2`, `python web/export_onnx.py`, then `python -m http.server 8765 --bind 127.0.0.1` in the repo root and open http://127.0.0.1:8765/web/index.html.

## Run inference

App (Run / Review / Results pages):
```powershell
streamlit run interface/app.py
```

Command line, on the bundled example (a Motic 40X scan at 0.2577 um/px):
```powershell
python predict.py --source examples --um-per-px 0.2577
```

- `--source`: an image file or a folder (.jpg/.png/.tif/.tiff/.svs). `.svs` files use their embedded µm/px; for other formats pass `--um-per-px` (BX61 10X = 0.645, Motic 40X = 0.2577) or set `predict.um_per_px` in `base_config.yaml`.
- Output goes to `<source folder>/predicted_<folder name>/` (or `--out <dir>`): `_results.csv` (one row of measurements per image), `raw_segmentations/` and `postprocessed_segmentations/` (YOLO-seg polygon `.txt`), plus `<image>_measurement.jpg` / `<image>_steps.jpg` QA images and `run_config.yaml`.
- All pipeline settings live in `base_config.yaml`; `--device cpu|cuda|auto` picks the device.

## For pipeline Dev

The background `run_predict_background*.ps1` scripts resolve the interpreter from `.venv\Scripts\python.exe` under `$ProjectDir`, so keep the venv there (or update `$ProjectDir`); their source folders are hard-coded personal OneDrive paths, edit them before use.



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

Re-run only postprocessing/measurement (e.g. after changing base_config.yaml) on one image or a whole folder, reusing the saved raw segmentations. It overwrites that image's `_measurement`/`_steps` images and `postprocessed_segmentations` label and replaces its `_results.csv` row; other rows and `run_config.yaml` are kept:

python predict.py --reprocess --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Lisa Image Pro analyzed images\D28 Topical GFF\D28 L10C1.svs"

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
7. Trim `lateral_clip_frac` of the band's arc length (along its medial-axis spine) off each end, then measure


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

how can i add the images from here to the  roi_follicle training dataset ?
 "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\Skin Biology Technology Development - Fundamental_MDE (Multi Decade & Ethnicity study)\MDE-TIFF"

tyring to add pseudolabels to box annotation 

11/17 FINISH THIS: ( to kate: the box annotations were saved from a prior run. now we add them to the training dataset. )
$src = "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\Skin Biology Technology Development - Fundamental_MDE (Multi Decade & Ethnicity study)\MDE-TIFF"
python training\annotate_qc_boxes.py --pred-source $src --um-per-px 0.2577 --preds runs\det\roi\pseudo_mde\detections.csv,runs\det\follicle\pseudo_mde\detections.csv

  python predict.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Lisa Image Pro analyzed images\D28 Topical GFF" --resume

Enhancement of the interface:
During a segmentation run, just show the raw image.while is processing, show a percent of that image done (as like a pie chart) over the image with updates. When it is done, replace with _measurement image. then continue . Also in the run tab, if there is already a predicted_ folder, then it should pull up a window and warn that this overwrites that folder. while it is running, the run parameters pane should close. 

after pressing run, i want a pane on the right side with a green checkmark next to each image that has been analyzed so a user can clearly see the progress. 

the wheel on the image being processed stays on "Measuring" too long. 
Done should be a popup window with run summary. It should include a button to review segmentation results. 

 is it possible to detect if there is another user using/writing to the same onedrive folder? I'd want to bring this up as an error so there are no conflicts. 

write user error tests to test functionality 
the TIFF_ folder should go in the predicted_ folder. 
 if a tiff file already exists with the same name, do not remake the tiff file in TIFF_ folder.  
 Rename the predicted_ folder to analysis_outputs_<parent_dir_name>

  On the edit page: 

  - inspect ways to speed up the saving of the user modified segmentations
  - any errors should be popup window: like if unable to write to _results file, show popup error. 
  - allow user to edit the raw segmentations (to then be used in the seg_finetune dataset if it exists. this is like for admin mode. I am the admin)

  On the process page:
  - display what is happening: like show the postprocessing steps happening

other to do

the _measurement and _steps images should go into an "image_segs" folder (as child folder of predicted_ folder)

 Zhang vs lee skeletonization comparison?
  audit the git situation. I want a user to be able to clone the repo and be able to run inference. 




 DONE:
--reprocess for single image. It overwrites its current postprocessing files (its current _measurement and _steps files, as well as the postprocessed_segmentations .txt file)
 add more logging for postprocessing. 
- the editing tools should hover so that they are always visible. clicking to next image should be along side the editing tools 
  - show editing pane first, and turn on "edit segmentation" by default
  - the user should edit the postprocessed segmentations rather than the raw segmentations
  - the user's segmentations should overwrite the postprocessed_segmentations .txt file and the _measurement and _steps files. 
  - the user's corrected segmentations should be added as another pane to the _steps file.
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



### Pre-train a skin segmentation model using this data: https://pmc.ncbi.nlm.nih.gov/articles/PMC11803237/
How to train:
Download the data, convert to yolo format, and run UNET model to segment the "Keratin", "Epidermis", and "gland" classes. 
OG dataset was 0.28um/px (20x), but published at 0.7 um/px (8x) So the pretrained model was trained at 0.7um/px
pretrained 3 class model: C:\Users\plas.ka\skin_layer_segmentation\pretrained_model\3cls_hist_seg 

### Add or adjust P&G segmentations (pre-finetuning)

```
python add_segmentations.py --dataset "dataset_for_finetuning"
```

### then finetune on our segmented images in dataset_for_finetuning

```
.\.venv\Scripts\python.exe -u pretrained_model\3cls_hist_seg\train_semseg.py train `
>>   --data dataset_for_finetuning `
>>   --init-weights pretrained_model\3cls_hist_seg\best.pt `
>>   --pretrained-classes "SC,Epidermis,glands" `
>>   --encoder resnet34 --epochs 200 --batch 2 --accumulate 1 --crop 512 --lr 2e-4 `
>>   --class-weights "SC=3" --oversample "gland=1.5" `
>>   --blur-p 0.3 --downup-p 0.3 --jpeg-p 0.3 `
>>   --val-every 5 --patience 15 --workers 0 --device auto `
>>   --project runs\semseg --name dataset_3cls_gland_v2
```


### then pseudolabeling on nicely segmented images
  use .txt segmentation cooredinates in the labels folder (corresponding to an image with the same prefix) 

### then for interence on data in yolo_skin_seg\inference_images

python predict.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S_EX 8_24 TIFF_H_E"

python predict.py --source "runs\predict_sample_src" --limit 5

python predict.py --weights "runs\semseg\dataset_3cls_gland_v2\best.pt" --source "difficult_test_images" --limit 2

### to know:

BX61 10X → 0.645 µm/px
Motic 40X scan → 0.260 µm/px


## Other Models

### Rotation model: 
Rotates the image so that the SC and Epidermis layers are at the top. 
------------- train:--------------
python rotate.py train --dataset dataset_rotate --output runs/rotation/rotate_model.pt --pretrained --device cuda --epochs 50 --samples-per-image 4 --batch-size 4
----------- inference:---------------
python rotate.py predict --weights runs/rotation/rotate_model.pt --source difficult_test_images --out runs/rotation/difficult_test_predictions --device cuda


### Follicle detection model:
YOLO11n-OBB, copied from pretrained_model\runs\follicle_det\rot1-2\weights\best.pt to models\follicle_det.pt.
In postprocess.py it runs on the rotated image and adds a "Follicles (N)" panel
to the steps image. Configure via the `follicle_*` keys in base_config.yaml (`follicle_enabled: false` to turn off).

### ROI detection model:
YOLO11n-OBB (class `roi`), copied from pretrained_model\runs\roi_det\train\weights\best.pt to models\roi_det.pt.
Runs on the rotated image and adds a "ROI Boxes (N)" panel to the steps image. Configure via the `roi_*` keys
in base_config.yaml; if no ROI is detected (or `roi_enabled: false`) the full frame is used.

### Postprocessing order (postprocess.py)
1. Rotate
2. Detect follicles
3. Detect ROI boxes
4. Remove SC/Epidermis outside the union of ROI boxes and inside the bottom `follicle_mask_frac` of each follicle box
5. Per-class elliptical opening then closing (`sc_erod_dil_kernel_size_um`, `epidermal_erod_dil_kernel_size_um`)
6. Trim `lateral_clip_frac` per side, then measure

# to do
 make better at different image sizes and resolutions

folders to do: 
  S_EX_NOV_24_D2_H_E (log entry) and S_Exp 6.24 D5


improve follicle detection 

Validate epidermal thickness on "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\Skin Biology Technology Development - Fundamental_MDE (Multi Decade & Ethnicity study)\Results-Learnings\Makio Analysis\AntiAging\MDE credentialling\Histology\Epi_Thickness_CA_face.csv"

 # Skin layer segmentation and quantification
 Measures the Epidermis and Stratum Corneum layers on H&E skin images.

Train (on PMC data), finetune on (P&G data), pseudolabeling (on generated segmented P&G data) and run inference on H&E stained skin 

## Download the repo from github (https://github.com/plaska-pg/skin_layer_segmentation)

```powershell
git clone https://github.com/plaska-pg/skin_layer_segmentation.git "$HOME\skin_layer_segmentation"
cd "$HOME\skin_layer_segmentation"
```
## Environment setup

Install python3.14. You can do this by typing python in the powershell terminal (if on windows). This'll pull up an install window. The commands above clone the repo into your home folder (for example, `C:\Users\plas.ka\skin_layer_segmentation` on this computer), keeping the virtual environment off OneDrive.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins the exact packages. The background `run_predict_background*.ps1` scripts resolve the interpreter from `.venv\Scripts\python.exe` under `$ProjectDir`, so keep the venv there (or update `$ProjectDir`).



## Train a skin segmentation model using this data: https://pmc.ncbi.nlm.nih.gov/articles/PMC11803237/
How to train:
Download the data, convert to yolo format, and run UNET model to segment just the "Keratin" and "Epidermis" classes. Keratin is equivalent to "SC layer" and Epidermis is equivalnt to "Granular Layer" in finetuning but then back to Epidermis. I want to make this consistent throughout the pipeline later.

## Pretrained a skin segmentation model using this data: https://pmc.ncbi.nlm.nih.gov/articles/PMC11803237/
data and model saved here: pretrained_model and also I copied the one being used to models/best.pt
training resolution was 0.645 µm/px

## then finetuned on our segmented images in dataset_for_finetuning

## then pseudolabeling on nicely segmented images
  use .txt segmentation cooredinates in the labels folder (corresponding to an image with the same prefix) 

## then for interence on data in yolo_skin_seg\inference_images

python predict.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S_EX 8_24 TIFF_H_E"

python predict.py --source "runs\predict_sample_src" --limit 5

python predict.py --source "difficult_test_images" --limit 3

## to do:>

python predict.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S_Exp_SEP_2024_D3_H_E"

python predict.py --source "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S_Exp 6.24 D5"


## to check on process status:

Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'predict\.py' } | Select-Object ProcessId, CommandLine

## to know:

BX61 10X → 0.645 µm/px
Motic 40X scan → 0.260 µm/px

## commands

to kill:
Stop-Process -Id 8092

to check on processes:
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'predict\.py' } | Select-Object ProcessId, CommandLine

to run:
.\run_predict_background.ps1


## rotation model: 
------------- train:--------------
python rotate.py train --dataset dataset_rotate --output runs/rotation/rotate_model.pt --pretrained --device cuda --epochs 50 --samples-per-image 4 --batch-size 4
----------- inference:---------------
python rotate.py predict --weights runs/rotation/rotate_model.pt --source difficult_test_images --out runs/rotation/difficult_test_predictions --device cuda

## examples where segmentation/postprocessing is not good
"C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S-EX_SEP_2024_D2_H_E_predicted\S_EX_9_24_D2_L1C4_steps.jpg"

"C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S-EX_SEP_2024_D2_H_E_predicted\S_EX_9_24_D2_L1C4_steps.jpg"

"C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S-EX_SEP_2024_D2_H_E_predicted\S_EX_9_24_D2_L4C4_steps.jpg"


"C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S-EX_SEP_2024_D2_H_E_predicted\S_EX_9_24_D2_L3C1_steps.jpg"


try 25.8 morphological closing


$one = "$env:TEMP\yolo_one_image"
New-Item -ItemType Directory -Force $one

$input = Get-ChildItem `
  "C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images\S-EX_SEP_2024_D2_H_E" `
  -File |
  Where-Object { $_.BaseName -eq "S_EX_9_24_D2_L1C4" } |
  Select-Object -First 1

Copy-Item $input.FullName $one

python predict.py `
  --source $one `
  --out "C:\Users\plas.ka\Desktop\one_prediction"


  ## next try follicle detection model 
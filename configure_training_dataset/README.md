# TIF (Image-Pro) → `annotations.json`

to use:

```
python configure_training_dataset/build_yolo_dataset.py "training_dataset/images/GMDE 217 1 N-1 10X_seg.tif" --split train --classes "SC,Granular Layer"
python configure_training_dataset/build_yolo_dataset.py "training_dataset/images/OTHER_FILE.tif" --split val --classes "SC,Granular Layer"
```

Converts hand-traced boundary curves/regions embedded by MediaCy Image-Pro's
manual-measurement tool directly into the tif's own metadata tags into the
yolo-format.
Image-Pro embeds the traced annotations as XML in a few TIFF tags - no
separate annotation file is exported:

| Tag     | Contents                                                              |
|---------|------------------------------------------------------------------------|
| `38124` | Open boundary curves (polylines) - e.g. the SC/Granular Layer bands    |
| `38129` | Class registry (`McSettings`): class names + `ClassValue` ordering     |
| `38123` | Raster `CmGraphOverlay` - standalone closed blobs (e.g. Follicle)      |



## YOLO dataset (`dataset/`) - one pass, tif → PNG + label in one command

`yolo_train.py` reads from a separate YOLO-segmentation dataset at
[`dataset/`](../dataset) (`images/<split>/*.png` + `labels/<split>/*.txt` +
`dataset.yaml`).
[`build_yolo_dataset.py`](build_yolo_dataset.py) builds this directly from a
tif in one pass - no separate preview step, no intermediate COCO file:

```
python configure_training_dataset/build_yolo_dataset.py "training_dataset/images/GMDE 217 1 N-1 10X_seg.tif" --split train --classes "SC,Granular Layer"
python configure_training_dataset/build_yolo_dataset.py "training_dataset/images/OTHER_FILE.tif" --split val --classes "SC,Granular Layer"
```

- You must pass `--split train` or `--split val` explicitly - there's no
  automatic splitting, so you decide per image.
- Use `--classes "SC,Granular Layer"` to discard Follicle/Gland annotations and
  train those regions as implicit background.
- Converts the tif straight to `dataset/images/<split>/<name>.png` and
  writes one normalized YOLO-segmentation polygon line per traced instance
  to `dataset/labels/<split>/<name>.txt`.
- Class ids are tracked in `dataset/dataset.yaml`'s `names` map and reused
  by name (case/whitespace-insensitive) across runs - a brand-new class
  name gets the next free id appended automatically.
- Skips an image already present in that split unless you pass
  `--overwrite`; warns (but still proceeds) if the same tif is also present
  in the *other* split, since a segmentation model shouldn't be evaluated on
  an image it was trained on.




# inference: 
python for_transfer/train_semseg.py predict --weights runs/semseg/dataset_2cls_finetune_v2/best.pt --source inference_images --out runs/semseg_preds --um-per-px 0.26 --min-blob-px 200 --require-adjacent "SC:Granular Layer" --adjacency-margin-px 15 --touch-classes "SC,Granular Layer" --min-touch-frac 0.4 --canonicalize "SC,Granular Layer"



# Zoom/magnification situation#


# input data

"C:\Users\plas.ka\OneDrive - Procter and Gamble\Shortcuts\W Cheng Section (BDT-Skin) - Histology\Raw images"

# Output data
CSV file

folder name, file name, SC mean height, SC mean width, SC min height, SC max height, SC min width, SC max width, SC area, Granular Layer mean height, Granular Layer mean width, Granular Layer min height, Granular Layer max height, Granular Layer min width, Granular Layer max width, Granular Layer area


Segmentation folder for each folder 


# to postprocess
find out the dims of what a good segmentation is (like height to width ratio)
remove those that are not good. 



for each image during inference, show png file with side by side: original image, predicted, the predicted + postprocessing, and the metrics on the image. 



# to run prediction:

cd "C:\Users\plas.ka\OneDrive - Procter and Gamble\Desktop\yolo_skin_seg"

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

.\run_predict_background.ps1

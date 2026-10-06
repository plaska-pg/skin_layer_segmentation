original dataset is Histo-Set. It was converted to yolo format: train_dataset_yolo_format_Histo-Set_4cls_2x.  contains these classes:
   0: SC
  1: Epidermis
  2: glands
  3: follicles

Original dataset. Whole-slide H&E images captured at 20x on a HEIDSTAR HDS-MS-200A brightfield microscope, about 15,000 by 4,000 to 8,000 px per slide. AT20× magnification, the HEIDSTAR HDS‑MS‑200A brightfield microscope has an image resolution of approximately 0.25 µm per pixel

## onverted Histo-Seg dataset to Histo-Seg_yolo_train_dataset (shrunk and subsettted) 
  Converted datasets. Every dataset is a fixed shrink of the source, then cut into 1024 px tiles. The shrink sets the effective magnification.

  ┌────────────────────────────────────┬────────┬───────────────┬────────────────┬────────────┬─────────────┐
  │              Dataset               │ Shrink │ Effective mag │  Tile covers   │ Keratin px │ Train tiles │
  ├────────────────────────────────────┼────────┼───────────────┼────────────────┼────────────┼─────────────┤ │
  ├────────────────────────────────────┼────────┼───────────────┼────────────────┼────────────┼─────────────┤
  │ 4-class 2x                         │ 2x     │ ~10x          │ 2048 source px │ 70 to 150  │ 601         │
  └────────────────────────────────────┴────────┴───────────────┴────────────────┴────────────┴─────────────┘

  Zoom during training. Augmentation adds a second scale change on top of the dataset scale.

  - U-Net: random 512 px crops from the tile after a 0.7x to 1.3x rescale, validated on full tiles at exactly 1x. The narrowest range of the three.

  At inference each model expects the scale it trained at, so a new 20x slide must be shrunk by the same factor before tiling.

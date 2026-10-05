
yolo_dataset_4cls_2x was created from another original dataset. 

Original dataset. Whole-slide H&E images captured at 20x on a HEIDSTAR HDS-MS-200A brightfield microscope, about 15,000 by 4,000 to 8,000 px per slide, with no micrometre-per-pixel metadata in the files. At this scale the keratin layer is 140 to 300 px thick and the epidermis 120 to 600 px.

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

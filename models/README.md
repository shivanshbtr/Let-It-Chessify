# Models

This folder contains trained model weights for the Chess-OCR pipeline. Only the
final `best.pt` from the valid completed run of each stage is kept in this repo:

- `corner_detection/train-2/weights/best.pt`
- `piece_detection/train/weights/best.pt`

`corner_detection/train/` was a dead run interrupted at epoch 10 and is not the
one to use.

## Folder structure

```
models
├── corner_detection
│   └── train-2
│       └── weights
│           └── best.pt
└── piece_detection
    └── train
        └── weights
            └── best.pt
```

## Full training artifacts

Full training outputs (all epoch checkpoints, logs, curves, batch previews,
confusion matrices, etc.) were zipped and uploaded to Google Drive before this
repo was pushed to GitHub, since they're too large/numerous for version control.

**Google Drive link:** [drive.google.com/file/d/1ToSGFEDoMVAqu-FxFT13QL_iTxMkQ_sY/view?usp=sharing](https://drive.google.com/file/d/1ToSGFEDoMVAqu-FxFT13QL_iTxMkQ_sY/view?usp=sharing)

### Google Drive zip structure

```
models
├── corner_detection
│   ├── train
│   │   ├── weights
│   │   │   ├── best.pt
│   │   │   ├── epoch0.pt
│   │   │   ├── epoch10.pt
│   │   │   └── last.pt
│   │   ├── args.yaml
│   │   ├── labels.jpg
│   │   ├── results.csv
│   │   ├── train_batch0.jpg
│   │   ├── train_batch1.jpg
│   │   └── train_batch2.jpg
│   └── train-2
│       ├── weights
│       │   ├── best.pt
│       │   ├── epoch0.pt
│       │   ├── epoch10.pt
│       │   ├── epoch20.pt
│       │   ├── epoch30.pt
│       │   ├── epoch40.pt
│       │   ├── epoch50.pt
│       │   ├── epoch60.pt
│       │   ├── epoch70.pt
│       │   ├── epoch80.pt
│       │   ├── epoch90.pt
│       │   ├── epoch100.pt
│       │   ├── epoch110.pt
│       │   ├── epoch120.pt
│       │   └── last.pt
│       ├── args.yaml
│       ├── labels.jpg
│       ├── results.csv
│       ├── train_batch0.jpg
│       ├── train_batch1.jpg
│       └── train_batch2.jpg
└── piece_detection
    └── train
        ├── weights
        │   ├── best.pt
        │   └── last.pt
        ├── args.yaml
        ├── BoxF1_curve.png
        ├── BoxPR_curve.png
        ├── BoxP_curve.png
        ├── BoxR_curve.png
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── labels.jpg
        ├── results.csv
        ├── results.png
        ├── train_batch0.jpg
        ├── train_batch1.jpg
        ├── train_batch2.jpg
        ├── train_batch3230.jpg
        ├── train_batch3231.jpg
        ├── train_batch3232.jpg
        ├── val_batch0_labels.jpg
        ├── val_batch0_pred.jpg
        ├── val_batch1_labels.jpg
        └── val_batch1_pred.jpg
```

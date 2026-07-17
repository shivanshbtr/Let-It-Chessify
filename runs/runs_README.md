# Runs

This folder contains trained CNN square classifiers, generated during training
of the piece classifier on physical and synthetic data. Everything here is
small enough to keep in the repo as-is (no pruning, no Drive backup).

## Folder structure

```
runs
├── physical
│   ├── best.pt
│   ├── history.csv
│   ├── label_map.json
│   ├── square_classifier.onnx
│   └── square_classifier.onnx.data
└── synthetic
    ├── best.pt
    ├── history.csv
    ├── label_map.json
    ├── square_classifier.onnx
    └── square_classifier.onnx.data
```

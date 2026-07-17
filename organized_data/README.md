# Organized Data

This folder contains the **processed/organized** version of the raw datasets, structured and split for model training. Due to their size, they are **not tracked in this GitHub repository**. Instead, the full `organized_data` folder has been zipped and uploaded to Google Drive.

**Download link: [drive.google.com/file/d/1CF8VzC0JuQotPhF-P576zeE_jMEViK-V/view?usp=sharing](https://drive.google.com/file/d/1CF8VzC0JuQotPhF-P576zeE_jMEViK-V/view?usp=sharing)**

## Folder Structure

```
organized_data/
├── classification/
│   ├── physical/            # Real (physical board) square crops, organized by class
│   │   ├── bB/ bK/ bN/ bP/ bQ/ bR/   # Black pieces
│   │   ├── wB/ wK/ wN/ wP/ wQ/ wR/   # White pieces
│   │   └── Empty/                   # Empty squares
│   │
│   ├── synthetic/           # Digitally rendered square crops, organized by class
│   │   ├── bB/ bK/ bN/ bP/ bQ/ bR/   # Black pieces
│   │   ├── wB/ wK/ wN/ wP/ wQ/ wR/   # White pieces
│   │   └── empty/                   # Empty squares
│   │
│   └── splits/              # Train/val/test split definitions
│       ├── physical/        # train.csv, val.csv, test.csv (physical set)
│       ├── synthetic/       # train.csv, val.csv, test.csv (synthetic set)
│       ├── label_map.json           # Class-to-index mapping (synthetic)
│       └── label_map_physical.json  # Class-to-index mapping (physical)
│
├── corner_detection/
│   ├── images/
│   │   ├── train/           # Training images, grouped by game ID (e.g. G000, G019, ...)
│   │   └── val/             # Validation images, grouped by game ID (e.g. G083, ...)
│   ├── labels/
│   │   ├── train/           # YOLO-format corner labels for training images
│   │   └── val/             # YOLO-format corner labels for validation images
│   └── data.yaml            # Dataset config for corner detection training
│
└── piece_detection/
    ├── images/
    │   ├── train/            # Training images, named by FEN string
    │   └── val/               # Validation images, named by FEN string
    ├── labels/
    │   ├── train/            # YOLO-format bbox labels, single class (piece)
    │   └── val/               # YOLO-format bbox labels, single class (piece)
    └── data.yaml              # Dataset config for piece detection training
```

## Notes

- `physical` refers to data derived from real photographs of physical chess boards/pieces; `synthetic` refers to digital/2D chessboard squares.
- After downloading, extract the zip and place the `organized_data` folder in the project root to match the expected paths used in the code.

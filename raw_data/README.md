# Raw Data

This folder contains the raw datasets used in this project. Due to their size, they are **not tracked in this GitHub repository**. Instead, the full `raw_data` folder has been zipped and uploaded to Google Drive.

**Download link:** [drive.google.com/file/d/1CDZ7xPwHqXZSiGEIdCSVykBbjqqR981P/view?usp=sharing](https://drive.google.com/file/d/1CDZ7xPwHqXZSiGEIdCSVykBbjqqR981P/view?usp=sharing)

## Folder Structure

```
raw_data/
├── Chess Piece Detection/
│   ├── annotations/      # XML annotation files (Pascal VOC format)
│   └── images/           # Raw physical chessboard images (.JPG)
│
├── chessboard-corner-detect/
│   ├── images/           #Physical(real) Board Images grouped by game/session ID (e.g. 0, 19, 22, ...)
│   └── labels/           # Corresponding YOLO-format label files, bbox coordinates for board
│
├── FENiT-FEN/
│   ├── images/           # Physical(real) Board Images named using their FEN string
│   └── labels/           # YOLO-format bbox labels, 13 classes (12 piece types + board)
│
└── self_synthetic_made/
    ├── images/           # Screenshots of various digital/online chess board interfaces
    └── labels/           # Corresponding FEN strings for each image
```

## Notes

- After downloading, extract the zip and place the `raw_data` folder in the project root to match the expected paths used in the code.
- File/folder naming conventions are preserved as-is from the original data collection process.

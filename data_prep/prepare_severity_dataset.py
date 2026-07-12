"""
Converts the raw Kaggle SeverityScore folders (1 / 2 / 3) into the folder
layout Ultralytics YOLO classification training expects:

    dataset_severity/
        train/
            Minor/*.jpg
            Substantial/*.jpg
            Critical/*.jpg
        val/
            Minor/*.jpg
            Substantial/*.jpg
            Critical/*.jpg

Raw Kaggle layout is expected to be:

    <raw_root>/SeverityScore/Severity Score Dataset/1/*.jpg   (Minor Impact)
    <raw_root>/SeverityScore/Severity Score Dataset/2/*.jpg   (Substantial Impact)
    <raw_root>/SeverityScore/Severity Score Dataset/3/*.jpg   (Critical Impact)

Usage:
    python data_prep/prepare_severity_dataset.py \
        --raw_root /path/to/kaggle/download \
        --out_dir  /path/to/dataset_severity \
        --val_split 0.15
"""
import argparse
import random
import shutil
from pathlib import Path

LABEL_MAP = {"1": "Minor", "2": "Substantial", "3": "Critical"}


def find_severity_root(raw_root: Path) -> Path:
    candidates = [
        raw_root / "SeverityScore" / "Severity Score Dataset",
        raw_root / "SeverityScore",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Could not find SeverityScore folder under {raw_root}")


def split_and_copy(src_dir: Path, out_dir: Path, class_name: str, val_split: float, seed: int):
    images = [p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    random.Random(seed).shuffle(images)
    n_val = int(len(images) * val_split)
    val_files = images[:n_val]
    train_files = images[n_val:]

    for split_name, files in (("train", train_files), ("val", val_files)):
        dest = out_dir / split_name / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)

    print(f"[{class_name}] total={len(images)} train={len(train_files)} val={len(val_files)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="Path to unzipped Kaggle dataset root")
    ap.add_argument("--out_dir", required=True, help="Where to write the YOLO-format dataset")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    severity_root = find_severity_root(raw_root)
    for folder_num, class_name in LABEL_MAP.items():
        src = severity_root / folder_num
        if not src.is_dir():
            raise FileNotFoundError(f"Expected folder {src} not found")
        split_and_copy(src, out_dir, class_name, args.val_split, args.seed)

    print(f"\nDone. Dataset ready at: {out_dir}")
    print("Train with:  yolo classify train data=<out_dir> model=yolov8n-cls.pt")


if __name__ == "__main__":
    main()

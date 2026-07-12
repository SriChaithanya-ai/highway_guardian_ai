"""
Converts the raw Kaggle "Road Accidents from CCTV Footages Dataset" into the
folder layout Ultralytics YOLO classification training expects:

    dataset_accident/
        train/
            Accident/*.jpg
            NonAccident/*.jpg
        val/
            Accident/*.jpg
            NonAccident/*.jpg

Raw Kaggle layout (after download & unzip) is expected to be:

    <raw_root>/Accident/Accident/*.jpg
    <raw_root>/NonAccident/NonAccident/*.jpg

Usage:
    python data_prep/prepare_accident_dataset.py \
        --raw_root /path/to/kaggle/download \
        --out_dir  /path/to/dataset_accident \
        --val_split 0.15
"""
import argparse
import random
import shutil
from pathlib import Path


def find_class_dir(raw_root: Path, class_name: str) -> Path:
    """Handle the dataset's doubled folder naming (Accident/Accident/...)."""
    candidates = [
        raw_root / class_name / class_name,
        raw_root / class_name,
    ]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        f"Could not locate images for class '{class_name}' under {raw_root}. "
        f"Checked: {[str(c) for c in candidates]}"
    )


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

    for class_name in ("Accident", "NonAccident"):
        src = find_class_dir(raw_root, class_name)
        split_and_copy(src, out_dir, class_name, args.val_split, args.seed)

    print(f"\nDone. Dataset ready at: {out_dir}")
    print("Train with:  yolo classify train data=<out_dir> model=yolov8n-cls.pt")


if __name__ == "__main__":
    main()

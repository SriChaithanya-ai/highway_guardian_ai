"""
Trains the Accident / NonAccident classifier.

Run data_prep/prepare_accident_dataset.py first to build the dataset folder.

Usage:
    python training/train_accident_classifier.py \
        --data /path/to/dataset_accident \
        --epochs 30 \
        --imgsz 224 \
        --model yolov8n-cls.pt
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset_accident (train/ val/ folders)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", default="yolov8n-cls.pt", help="Base model to fine-tune from")
    ap.add_argument("--project", default="models", help="Where ultralytics writes run outputs")
    ap.add_argument("--name", default="accident_classifier")
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=8,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights saved at: {best_weights}")
    print("This path is what config/settings.py -> ACCIDENT_CLASSIFIER_WEIGHTS expects.")


if __name__ == "__main__":
    main()

"""
Trains the accident-severity classifier (Minor / Substantial / Critical).

Run data_prep/prepare_severity_dataset.py first to build the dataset folder.

Usage:
    python training/train_severity_classifier.py \
        --data /path/to/dataset_severity \
        --epochs 40 \
        --imgsz 224 \
        --model yolov8n-cls.pt
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset_severity (train/ val/ folders)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", default="yolov8n-cls.pt")
    ap.add_argument("--project", default="models")
    ap.add_argument("--name", default="severity_classifier")
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
        patience=10,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights saved at: {best_weights}")
    print("This path is what config/settings.py -> SEVERITY_CLASSIFIER_WEIGHTS expects.")
    print("Class order should be: Critical, Minor, Substantial (alphabetical, verify in results.csv / model.names)")


if __name__ == "__main__":
    main()

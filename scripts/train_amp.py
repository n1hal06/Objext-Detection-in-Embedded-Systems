import argparse
import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from training_common import build_train_args, load_config


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n with AMP.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "yolov8n_baseline.yaml")
    parser.add_argument("--baseline-ckpt", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "baseline" / "weights" / "best.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Override AMP epochs from config")
    parser.add_argument("--name", type=str, default="amp")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    project_root = args.project_root.resolve()

    if not args.baseline_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_ckpt}")

    cfg = load_config(args.config)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    logging.info("Initialized GradScaler (enabled=%s)", scaler.is_enabled())

    model = YOLO(str(args.baseline_ckpt))
    train_args = build_train_args(
        project_root,
        cfg,
        name=args.name,
        stage="amp",
        epochs_override=args.epochs,
        amp=True,
        pretrained=False,
    )
    train_args["fraction"] = 1.0
    train_args["imgsz"] = 640

    logging.info("Starting AMP fine-tuning with args: %s", train_args)
    model.train(**train_args)

    amp_ckpt = project_root / "runs" / args.name / "weights" / "best.pt"
    amp_model = YOLO(str(amp_ckpt))
    baseline_model = YOLO(str(args.baseline_ckpt))

    baseline_map50 = float(baseline_model.val(data=str(project_root / "data" / "dataset.yaml")).box.map50)
    amp_map50 = float(amp_model.val(data=str(project_root / "data" / "dataset.yaml")).box.map50)

    logging.info("Saved AMP checkpoint: %s", amp_ckpt)
    logging.info("mAP50 comparison | baseline: %.4f | amp: %.4f | delta: %.4f", baseline_map50, amp_map50, amp_map50 - baseline_map50)


if __name__ == "__main__":
    main()

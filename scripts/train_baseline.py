import argparse
import logging
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from training_common import build_train_args, load_config


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n baseline in FP32.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "yolov8n_baseline.yaml")
    parser.add_argument("--name", type=str, default="baseline")
    return parser.parse_args()


def log_epoch_metrics(results_csv: Path) -> None:
    if not results_csv.exists():
        logging.warning("results.csv not found at %s", results_csv)
        return

    df = pd.read_csv(results_csv)
    metric_cols = [
        "epoch",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "train/box_loss",
        "train/cls_loss",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    for _, row in df[metric_cols].iterrows():
        logging.info("Epoch metrics: %s", {c: float(row[c]) for c in metric_cols})


def main() -> None:
    setup_logging()
    args = parse_args()

    project_root = args.project_root.resolve()
    cfg = load_config(args.config)

    model = YOLO(cfg.get("model", "yolov8n.pt"))
    train_args = build_train_args(
        project_root,
        cfg,
        name=args.name,
        stage="baseline",
        amp=False,
        pretrained=True,
    )

    logging.info("Starting baseline training with args: %s", train_args)
    model.train(**train_args)

    run_dir = project_root / "runs" / args.name
    results_csv = run_dir / "results.csv"
    best_ckpt = run_dir / "weights" / "best.pt"

    log_epoch_metrics(results_csv)

    eval_model = YOLO(str(best_ckpt)) if best_ckpt.exists() else model
    val_metrics = eval_model.val(data=str(project_root / "data" / "dataset.yaml"))
    map50 = float(val_metrics.box.map50)
    map5095 = float(val_metrics.box.map)

    logging.info("Final baseline checkpoint: %s", best_ckpt)
    logging.info("Final baseline mAP50: %.4f", map50)
    logging.info("Final baseline mAP50-95: %.4f", map5095)

    if map50 >= 0.45:
        logging.info("Target met: baseline mAP50 >= 45%%")
    else:
        logging.warning("Target not met: baseline mAP50 %.2f%% < 45%%", map50 * 100)


if __name__ == "__main__":
    main()

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.ao.quantization as tq
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.qat_config import get_qat_backend
from training_common import build_train_args, load_config


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QAT fine-tuning for YOLOv8n.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "yolov8n_baseline.yaml")
    parser.add_argument("--amp-ckpt", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "amp" / "weights" / "best.pt")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "fbgemm", "qnnpack"])
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None, help="Override QAT epochs from config")
    return parser.parse_args()


def calibrate_model(qat_model: torch.nn.Module, train_image_dir: Path, imgsz: int, calibration_images: int) -> None:
    qat_model.eval()
    image_paths = sorted(list(train_image_dir.glob("*.jpg")) + list(train_image_dir.glob("*.png")))
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {train_image_dir}")

    used = 0
    with torch.no_grad():
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (imgsz, imgsz))
            arr = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            x = torch.from_numpy(arr).unsqueeze(0)
            _ = qat_model(x)
            used += 1
            if used >= calibration_images:
                break

    logging.info("Calibration pass complete with %d training images.", used)


def main() -> None:
    setup_logging()
    args = parse_args()

    project_root = args.project_root.resolve()
    if not args.amp_ckpt.exists():
        raise FileNotFoundError(f"AMP checkpoint not found: {args.amp_ckpt}")

    cfg = load_config(args.config)

    amp_model = YOLO(str(args.amp_ckpt))
    pt_model = amp_model.model.float()
    pt_model.train()   # ✅ required for QAT

    backend, qconfig_mapping = get_qat_backend(args.backend)
    logging.info("Using QAT backend: %s", backend)

    # ✅ Assign qconfig (Eager mode)
    if hasattr(qconfig_mapping, "global_qconfig"):
        pt_model.qconfig = qconfig_mapping.global_qconfig
    else:
        pt_model.qconfig = qconfig_mapping

    # ✅ Prepare QAT (Eager instead of FX)
    tq.prepare_qat(pt_model, inplace=True)
    qat_graph = pt_model

    logging.info("Inserted fake-quant nodes with prepare_qat (Eager mode)")

    calibrate_model(
        qat_graph,
        train_image_dir=project_root / "data" / "images" / "train",
        imgsz=int(cfg.get("imgsz", 512)),
        calibration_images=args.calibration_images,
    )

    amp_model.model = qat_graph
    logging.info("Starting QAT fine-tuning for %d epochs", args.epochs)

    qat_train_args = build_train_args(
        project_root,
        cfg,
        name="qat",
        stage="qat",
        epochs_override=args.epochs,
        amp=False,
        pretrained=False,
        lr0_override=float(cfg.get("qat_lr0", cfg.get("lr0", 0.0001))),
    )

    amp_model.train(**qat_train_args)

    # ✅ Convert to quantized model (Eager)
    quantized_model = tq.convert(qat_graph.eval(), inplace=False)
    amp_model.model = quantized_model

    out_dir = project_root / "runs" / "qat" / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    qat_pt = out_dir / "best_qat.pt"
    torch.save({"model": quantized_model.state_dict(), "backend": backend}, qat_pt)

    logging.info("Saved QAT INT8 state dict to %s", qat_pt)

    baseline_ckpt = project_root / "runs" / "baseline" / "weights" / "best.pt"

    baseline_map50 = float(
        YOLO(str(baseline_ckpt)).val(data=str(project_root / "data" / "dataset.yaml")).box.map50
    ) if baseline_ckpt.exists() else float("nan")

    amp_map50 = float(
        YOLO(str(args.amp_ckpt)).val(data=str(project_root / "data" / "dataset.yaml")).box.map50
    )

    qat_best = project_root / "runs" / "qat" / "weights" / "best.pt"
    qat_model_for_eval = YOLO(str(qat_best)) if qat_best.exists() else amp_model

    qat_map50 = float(
        qat_model_for_eval.val(data=str(project_root / "data" / "dataset.yaml")).box.map50
    )

    logging.info("mAP50 baseline: %.4f", baseline_map50)
    logging.info("mAP50 AMP: %.4f", amp_map50)
    logging.info("mAP50 QAT: %.4f", qat_map50)

    drop = amp_map50 - qat_map50
    if drop <= 0.02:
        logging.info("QAT target met: mAP50 drop from AMP to QAT <= 2 points")
    else:
        logging.warning("QAT drop exceeded: %.4f (> 0.02)", drop)


if __name__ == "__main__":
    main()
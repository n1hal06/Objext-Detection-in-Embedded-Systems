import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import psutil
from ultralytics import YOLO


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 models across backends.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", type=str, default="all", choices=["cpu", "cuda", "trt", "onnx", "all"])
    parser.add_argument("--num-images", type=int, default=200)
    return parser.parse_args()


def percentile_ms(values: List[float], p: float) -> float:
    return float(np.percentile(np.array(values), p)) * 1000.0


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else float("nan")


def load_images(val_dir: Path, limit: int) -> List[Path]:
    images = sorted([p for p in val_dir.glob("*.jpg")]) + sorted([p for p in val_dir.glob("*.png")])
    if not images:
        raise RuntimeError(f"No validation images found in {val_dir}")
    return images[: min(limit, len(images))]


def benchmark_yolo_predict(model: YOLO, images: List[Path], device: str) -> Dict[str, float]:
    latencies: List[float] = []
    process = psutil.Process()
    peak_mem_mb = process.memory_info().rss / (1024 * 1024)

    for img_path in images:
        start = time.perf_counter()
        _ = model.predict(source=str(img_path), imgsz=640, device=device, verbose=False)
        end = time.perf_counter()
        latencies.append(end - start)
        peak_mem_mb = max(peak_mem_mb, process.memory_info().rss / (1024 * 1024))

    total_time = sum(latencies)
    fps = len(images) / total_time if total_time > 0 else 0.0
    return {
        "fps": fps,
        "p50_ms": percentile_ms(latencies, 50),
        "p95_ms": percentile_ms(latencies, 95),
        "peak_ram_mb": peak_mem_mb,
    }


def benchmark_onnx(onnx_path: Path, images: List[Path]) -> Dict[str, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    latencies: List[float] = []
    process = psutil.Process()
    peak_mem_mb = process.memory_info().rss / (1024 * 1024)

    for img_path in images:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (640, 640))
        x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        start = time.perf_counter()
        _ = sess.run(None, {input_name: x})
        end = time.perf_counter()

        latencies.append(end - start)
        peak_mem_mb = max(peak_mem_mb, process.memory_info().rss / (1024 * 1024))

    total_time = sum(latencies)
    fps = len(images) / total_time if total_time > 0 else 0.0
    return {
        "fps": fps,
        "p50_ms": percentile_ms(latencies, 50),
        "p95_ms": percentile_ms(latencies, 95),
        "peak_ram_mb": peak_mem_mb,
    }


def safe_trt_benchmark(engine_path: Path, images: List[Path]) -> Dict[str, float]:
    try:
        model = YOLO(str(engine_path))
        return benchmark_yolo_predict(model, images, device="0")
    except Exception as exc:
        logging.warning("TensorRT benchmark unavailable: %s", exc)
        return {"fps": float("nan"), "p50_ms": float("nan"), "p95_ms": float("nan"), "peak_ram_mb": float("nan")}


def get_map50(model_path: Path, data_yaml: Path, device: str = "cpu") -> float:
    try:
        model = YOLO(str(model_path))
        metrics = model.val(data=str(data_yaml), device=device)
        return float(metrics.box.map50)
    except Exception as exc:
        logging.warning("mAP50 evaluation failed for %s: %s", model_path, exc)
        return float("nan")


def main() -> None:
    setup_logging()
    args = parse_args()

    root = args.project_root.resolve()
    val_dir = root / "data" / "images" / "val"
    data_yaml = root / "data" / "dataset.yaml"
    images = load_images(val_dir, args.num_images)

    baseline = root / "runs" / "baseline" / "weights" / "best.pt"
    amp = root / "runs" / "amp" / "weights" / "best.pt"
    qat_pt = root / "runs" / "qat" / "weights" / "best.pt"
    qat_onnx = root / "runs" / "qat" / "weights" / "model_qat.onnx"
    trt_engine = root / "runs" / "qat" / "weights" / "model.trt"

    rows = []

    if args.device in ["cpu", "all"] and baseline.exists():
        b_model = YOLO(str(baseline))
        stats = benchmark_yolo_predict(b_model, images, device="cpu")
        rows.append({
            "Model": "YOLOv8n FP32",
            "mAP50": get_map50(baseline, data_yaml, device="cpu") * 100,
            "FPS": stats["fps"],
            "P50 lat (ms)": stats["p50_ms"],
            "P95 lat (ms)": stats["p95_ms"],
            "Peak RAM (MB)": stats["peak_ram_mb"],
            "Size MB": size_mb(baseline),
        })

    if args.device in ["cuda", "all"] and amp.exists():
        a_model = YOLO(str(amp))
        stats = benchmark_yolo_predict(a_model, images, device="0")
        rows.append({
            "Model": "YOLOv8n AMP",
            "mAP50": get_map50(amp, data_yaml, device="0") * 100,
            "FPS": stats["fps"],
            "P50 lat (ms)": stats["p50_ms"],
            "P95 lat (ms)": stats["p95_ms"],
            "Peak RAM (MB)": stats["peak_ram_mb"],
            "Size MB": size_mb(amp),
        })

    if args.device in ["onnx", "all"] and qat_onnx.exists():
        stats = benchmark_onnx(qat_onnx, images)
        map_source = qat_pt if qat_pt.exists() else amp
        rows.append({
            "Model": "YOLOv8n QAT INT8",
            "mAP50": get_map50(map_source, data_yaml, device="cpu") * 100,
            "FPS": stats["fps"],
            "P50 lat (ms)": stats["p50_ms"],
            "P95 lat (ms)": stats["p95_ms"],
            "Peak RAM (MB)": stats["peak_ram_mb"],
            "Size MB": size_mb(qat_onnx),
        })

    if args.device in ["trt", "all"] and trt_engine.exists():
        stats = safe_trt_benchmark(trt_engine, images)
        map_source = qat_pt if qat_pt.exists() else amp
        rows.append({
            "Model": "TRT INT8",
            "mAP50": get_map50(map_source, data_yaml, device="0") * 100,
            "FPS": stats["fps"],
            "P50 lat (ms)": stats["p50_ms"],
            "P95 lat (ms)": stats["p95_ms"],
            "Peak RAM (MB)": stats["peak_ram_mb"],
            "Size MB": size_mb(trt_engine),
        })

    if not rows:
        raise RuntimeError("No benchmark rows produced. Ensure checkpoints/exports exist and --device is valid.")

    df = pd.DataFrame(rows)
    out_csv = root / "runs" / "benchmark_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    logging.info("Benchmark Results Table:\n%s", df.to_markdown(index=False))
    logging.info("Saved benchmark CSV to %s", out_csv)


if __name__ == "__main__":
    main()

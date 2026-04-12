import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
from ultralytics import YOLO


class DummyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name: str, num_samples: int = 32):
        self.input_name = input_name
        self.samples = [np.random.rand(1, 3, 640, 640).astype(np.float32) for _ in range(num_samples)]
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        batch = {self.input_name: self.samples[self.idx]}
        self.idx += 1
        return batch


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export QAT model to ONNX/TRT and ORT-INT8.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-ckpt", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "baseline" / "weights" / "best.pt")
    parser.add_argument("--amp-ckpt", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "amp" / "weights" / "best.pt")
    parser.add_argument("--qat-ckpt", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "qat" / "weights" / "best.pt")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else float("nan")


def export_onnx(model: YOLO, out_dir: Path, opset: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = model.export(format="onnx", imgsz=640, dynamic=True, opset=opset)
    onnx_path = Path(exported)

    target = out_dir / "model_qat.onnx"
    if onnx_path.resolve() != target.resolve():
        target.write_bytes(onnx_path.read_bytes())

    onnx_model = onnx.load(str(target))
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX checker passed for %s", target)
    return target


def _extract_first_shape(output) -> Tuple[int, ...]:
    if isinstance(output, torch.Tensor):
        return tuple(output.shape)
    if isinstance(output, (list, tuple)) and output:
        return _extract_first_shape(output[0])
    raise RuntimeError("Unsupported PyTorch output type for shape extraction")


def validate_onnx_output_shape(onnx_path: Path, model: YOLO) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    out = sess.run(None, {input_name: x})
    onnx_shape = tuple(out[0].shape)

    model.model.eval()
    with torch.no_grad():
        pt_out = model.model(torch.from_numpy(x))
    pt_shape = _extract_first_shape(pt_out)

    if pt_shape != onnx_shape:
        raise RuntimeError(f"Output shape mismatch: PyTorch={pt_shape}, ONNX={onnx_shape}")

    logging.info("ONNX runtime inference successful | output shape=%s", onnx_shape)
    return pt_shape, onnx_shape


def export_pi_int8_onnx(onnx_path: Path, out_path: Path) -> Optional[Path]:
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        reader = DummyCalibrationDataReader(input_name=input_name, num_samples=32)
        quantize_static(
            model_input=str(onnx_path),
            model_output=str(out_path),
            calibration_data_reader=reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
        logging.info("Saved Raspberry Pi ORT INT8 model: %s", out_path)
        return out_path
    except Exception as exc:
        logging.warning("ONNX Runtime static quantization failed: %s", exc)
        return None


def build_tensorrt_engine(onnx_path: Path, engine_path: Path) -> Optional[Path]:
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        if not parser.parse(onnx_path.read_bytes()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"TensorRT parse failed: {errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build returned None")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(serialized_engine)
        logging.info("Saved TensorRT engine: %s", engine_path)

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        logging.info("TensorRT dummy load successful")

        return engine_path
    except Exception as exc:
        logging.warning("TensorRT export skipped or unavailable: %s", exc)
        return None


def main() -> None:
    setup_logging()
    args = parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "runs" / "qat" / "weights"

    source_ckpt = args.qat_ckpt if args.qat_ckpt.exists() else args.amp_ckpt
    if not source_ckpt.exists():
        raise FileNotFoundError(f"No source checkpoint found for export: {source_ckpt}")

    model = YOLO(str(source_ckpt))
    onnx_path = export_onnx(model, out_dir, args.opset)
    _, onnx_shape = validate_onnx_output_shape(onnx_path, model)

    pi_onnx_path = out_dir / "model_pi.onnx"
    quantized_pi = export_pi_int8_onnx(onnx_path, pi_onnx_path)

    trt_path = out_dir / "model.trt"
    engine = build_tensorrt_engine(onnx_path, trt_path)

    logging.info("Export summary:")
    logging.info("FP32 baseline size (MB): %.2f", mb(args.baseline_ckpt))
    logging.info("AMP FP16 checkpoint size (MB): %.2f", mb(args.amp_ckpt))
    logging.info("QAT ONNX size (MB): %.2f", mb(onnx_path))
    logging.info("TensorRT engine size (MB): %.2f", mb(engine) if engine else float("nan"))
    logging.info("Pi ORT INT8 ONNX size (MB): %.2f", mb(quantized_pi) if quantized_pi else float("nan"))
    logging.info("ONNX output shape verified: %s", onnx_shape)


if __name__ == "__main__":
    main()

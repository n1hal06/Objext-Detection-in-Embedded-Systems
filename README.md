# Real-Time Object Detection on Edge Devices (YOLOv8n)

This project implements a training-to-deployment pipeline for YOLOv8n optimized for:
- Raspberry Pi 4 (CPU inference)
- NVIDIA Jetson Nano (GPU + TensorRT INT8 inference)

## Targets

- Baseline mAP50 >= 45%
- Post-QAT mAP50 drop <= 2 points vs AMP
- Raspberry Pi 4 CPU inference >= 15 FPS
- Jetson Nano TensorRT INT8 inference >= 30 FPS

## Project Structure

project-root/
- data/
  - images/train
  - images/val
  - labels/train
  - labels/val
  - dataset.yaml
- configs/
  - yolov8n_baseline.yaml
  - qat_config.py
- scripts/
  - prepare_dataset.py
  - train_baseline.py
  - train_amp.py
  - train_qat.py
  - export_model.py
  - benchmark.py
- notebooks/
  - results_analysis.ipynb
- requirements.txt
- README.md

## Setup

1. Create and activate Python 3.10+ environment.
2. Install dependencies:

   pip install -r requirements.txt

## Run Pipeline (in order)

1) Phase 1: Dataset preparation

   python scripts/prepare_dataset.py --project-root . --dataset coco2017
   # or simply: python scripts/prepare_dataset.py --dataset coco2017

Expected output:
- Full COCO2017 downloaded and extracted
- Training and validation images plus YOLO labels generated from official annotations
- data/dataset.yaml generated/updated

2) Phase 2: Baseline FP32 training

   python scripts/train_baseline.py --project-root .

Expected output:
- Checkpoint: runs/baseline/weights/best.pt
- Per-epoch logs including mAP50, mAP50-95, precision, recall, box_loss, cls_loss
- Default baseline schedule: 150 epochs, lower learning rate, lighter augmentation

3) Phase 3: AMP fine-tuning

   python scripts/train_amp.py --project-root . --epochs 100

Expected output:
- Checkpoint: runs/amp/weights/best.pt
- Baseline vs AMP mAP50 comparison

4) Phase 4: QAT fine-tuning

   python scripts/train_qat.py --project-root . --backend auto

Expected output:
- QAT state dict: runs/qat/weights/best_qat.pt
- Validation comparison: baseline vs AMP vs QAT
- Warning if AMP->QAT drop exceeds 2 points

5) Phase 5: Export

   python scripts/export_model.py --project-root .

Expected output:
- ONNX: runs/qat/weights/model_qat.onnx
- TensorRT engine (if TRT available): runs/qat/weights/model.trt
- Pi ORT INT8 model: runs/qat/weights/model_pi.onnx
- Size report for baseline/AMP/QAT/TRT artifacts

6) Phase 6: Benchmark

   python scripts/benchmark.py --project-root . --device all --num-images 200

Expected output:
- Markdown table with mAP50, FPS, P50/P95 latency, peak RAM, size MB
- CSV: runs/benchmark_results.csv

## Hardware-Specific Notes

### Raspberry Pi 4
- Use backend qnnpack for QAT target behavior.
- Use ONNX Runtime INT8 model generated as model_pi.onnx.
- For best CPU speed, reduce background services and pin governor to performance.

### Jetson Nano
- Install TensorRT Python bindings from JetPack.
- export_model.py uses TensorRT build with explicit batch and FP16 fallback.
- If TensorRT is unavailable, scripts continue with warning and no crash.

## Expected Results (typical)

- FP32 baseline: about 45%+ mAP50 on COCO128, around 5 FPS on Pi CPU
- AMP: comparable mAP50 with faster training time
- QAT INT8: mAP50 within 2 points of baseline, improved Pi CPU FPS
- TRT INT8 on Jetson: large FPS boost vs FP32 GPU path

## Troubleshooting

- If TensorRT export fails, verify JetPack/TensorRT installation and CUDA compatibility.
- If mAP drops after QAT, lower learning rate or increase QAT epochs gradually.
- If benchmark rows are missing, ensure expected checkpoints/exports exist under runs/.

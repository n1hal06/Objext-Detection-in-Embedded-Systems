from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def dataset_yaml_path(project_root: Path) -> Path:
    return project_root / "data" / "dataset.yaml"


def build_train_args(
    project_root: Path,
    cfg: Dict[str, Any],
    *,
    name: str,
    stage: str,
    epochs_override: int | None = None,
    amp: bool,
    pretrained: bool,
    lr0_override: float | None = None,
) -> Dict[str, Any]:
    stage_prefix = f"{stage}_"

    train_args = {
        "data": str(dataset_yaml_path(project_root)),
        "epochs": int(epochs_override if epochs_override is not None else cfg.get(f"{stage_prefix}epochs", cfg.get("epochs", 150))),
        "imgsz": int(cfg.get("imgsz", 640)),
        "batch": int(cfg.get("batch", 16)),
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": float(lr0_override if lr0_override is not None else cfg.get(f"{stage_prefix}lr0", cfg.get("lr0", 0.0005))),
        "lrf": float(cfg.get("lrf", 0.005)),
        "warmup_epochs": float(cfg.get("warmup_epochs", 5.0)),
        "weight_decay": float(cfg.get("weight_decay", 0.0005)),
        "amp": amp,
        "mosaic": float(cfg.get("mosaic", 0.7)),
        "mixup": float(cfg.get("mixup", 0.05)),
        "hsv_h": float(cfg.get("hsv_h", 0.01)),
        "hsv_s": float(cfg.get("hsv_s", 0.5)),
        "hsv_v": float(cfg.get("hsv_v", 0.3)),
        "degrees": float(cfg.get("degrees", 3.0)),
        "translate": float(cfg.get("translate", 0.08)),
        "scale": float(cfg.get("scale", 0.35)),
        "close_mosaic": int(cfg.get("close_mosaic", 20)),
        "patience": int(cfg.get("patience", 40)),
        "fraction": float(cfg.get("fraction", 1.0)),
        "workers": int(cfg.get("workers", 8)),
        "project": str(project_root / "runs"),
        "name": name,
        "exist_ok": True,
        "pretrained": pretrained,
        "cos_lr": True,
    }

    return train_args

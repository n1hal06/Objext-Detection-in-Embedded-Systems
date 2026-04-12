import platform
from typing import Tuple

import torch
from torch.ao.quantization import get_default_qat_qconfig_mapping


def get_qat_backend(preferred: str = "auto") -> Tuple[str, object]:
    """Return backend name and qconfig mapping for FX QAT."""
    if preferred != "auto":
        backend = preferred
    else:
        machine = platform.machine().lower()
        backend = "qnnpack" if any(k in machine for k in ["arm", "aarch64"]) else "fbgemm"

    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError(f"Unsupported backend: {backend}. Use 'fbgemm' or 'qnnpack'.")

    torch.backends.quantized.engine = backend
    qconfig_mapping = get_default_qat_qconfig_mapping(backend)
    return backend, qconfig_mapping

"""
core/device.py - Compute Device Detection & Model Warmup
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

import logging
from typing import Any, Tuple
import numpy as np
import torch

logger = logging.getLogger("eco_engine.device")


def detect_best_device(requested_device: str = "auto") -> Tuple[str, str]:
    """
    Safely resolves compute device with validation and graceful CPU fallback.

    Args:
        requested_device: "auto", "0", "cuda", "cuda:0", or "cpu".

    Returns:
        Tuple[str, str]: (device_identifier, human_readable_display_name)
    """
    req = str(requested_device).lower().strip()
    if req in ("cpu", "host"):
        return "cpu", "CPU (Host Processing)"

    has_cuda = False
    try:
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if req in ("auto", "0", "cuda", "cuda:0", "gpu"):
        if has_cuda:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return "0", f"CUDA:0 ({gpu_name}, {vram_gb:.1f} GB)"
            except Exception as e:
                return "cpu", f"CPU (Fallback: CUDA query failed: {e})"
        else:
            return "cpu", "CPU (Host Processing)"

    return "cpu", "CPU (Host Processing)"


def get_inference_device() -> Tuple[str, str]:
    """Backwards-compatibility alias for detect_best_device('auto')."""
    return detect_best_device("auto")


def warmup_engine(model: Any, device: str = "cpu") -> None:
    """
    Warms up the YOLO model graph to absorb the initial inference latency (~2.4s)
    before real-time processing begins.
    """
    dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
    try:
        model(dummy_input, device=device, verbose=False, imgsz=320)
    except Exception:
        if device != "cpu":
            try:
                model(dummy_input, device="cpu", verbose=False, imgsz=320)
            except Exception:
                pass

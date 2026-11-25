"""Flatten critic component weights for inference."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def flatten_component_weights(components: list[dict[str, Any]]) -> np.ndarray:
    if not components:
        raise ValueError("Probe v4 requires components.")
    ensemble = np.asarray([float(component.get("ensemble_weight", 1.0)) for component in components])
    if np.any(ensemble < 0) or float(ensemble.sum()) <= 0:
        raise ValueError("Invalid component ensemble weights.")
    ensemble = ensemble / ensemble.sum()
    parts = []
    for component, coefficient in zip(components, ensemble, strict=True):
        weight = component["candidate_weight"]
        if isinstance(weight, torch.Tensor):
            weight = weight.detach().float().cpu().numpy()
        scale = max(float(component.get("score_scale", 1.0)), 1e-12)
        parts.append(np.asarray(weight, dtype=np.float32) * float(coefficient) / scale)
    return np.concatenate(parts).astype(np.float32)

from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from src.common import resolve
from src.models import render_chat_with_assistant_prefix

def state_action_probe_prompt(tokenizer: Any, prompt: str, prefix: str, candidate: str) -> str:
    """Render the actual generation context ending at the proposed action."""
    response = "\n\n".join(part.strip() for part in (prefix, candidate) if part and part.strip())
    return render_chat_with_assistant_prefix(tokenizer, "", prompt, response)


def find_transformer_layers(model: Any) -> tuple[Any, str]:
    for path in ("model.layers", "base_model.model.model.layers", "base_model.model.model.model.layers"):
        try:
            current = model
            for part in path.split("."):
                current = getattr(current, part)
            if len(current) > 0:
                return current, path
        except Exception:
            continue
    raise RuntimeError("Could not locate transformer layers.")


def feature_name(layer: int, pooling: str) -> str:
    return f"layer_{layer}_{pooling}"


def normalize_feature_specs(
    specs: Sequence[dict[str, Any] | tuple[int, str]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in specs:
        if isinstance(raw, tuple):
            layer, pooling = raw
            name = feature_name(int(layer), str(pooling))
        else:
            layer = int(raw["layer"])
            pooling = str(raw["pooling"])
            name = str(raw.get("name") or feature_name(layer, pooling))
        if layer < 0:
            raise ValueError(f"Feature layer must be non-negative, got {layer}.")
        if pooling != "last" and re.fullmatch(r"mean_last_(\d+)", pooling) is None:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        if name in names:
            raise ValueError(f"Duplicate feature spec name: {name}")
        names.add(name)
        normalized.append({"name": name, "layer": layer, "pooling": pooling})
    if not normalized:
        raise ValueError("At least one feature spec is required.")
    return normalized


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "last":
        mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
        positions = torch.arange(mask.shape[1], device=hidden.device).unsqueeze(0).expand_as(mask)
        last_positions = positions.masked_fill(~mask, -1).max(dim=1).values.clamp_min(0)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_positions]
    match = re.fullmatch(r"mean_last_(\d+)", pooling)
    if not match:
        raise ValueError(f"Unknown pooling mode: {pooling}")
    n_tokens = int(match.group(1))
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    reverse_rank = torch.flip(
        torch.cumsum(torch.flip(mask.to(dtype=torch.int64), dims=[1]), dim=1),
        dims=[1],
    )
    pool_mask = mask & (reverse_rank <= n_tokens)
    denom = pool_mask.sum(dim=1).clamp_min(1).to(dtype=hidden.dtype).unsqueeze(1)
    return (hidden * pool_mask.unsqueeze(2).to(dtype=hidden.dtype)).sum(dim=1) / denom


def extract_candidate_feature_bundle(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    texts: list[str],
    specs: Sequence[dict[str, Any] | tuple[int, str]],
    batch_size: int,
    runtime_stats: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    normalized = normalize_feature_specs(specs)
    layers, _ = find_transformer_layers(model)
    invalid = sorted({spec["layer"] for spec in normalized if spec["layer"] >= len(layers)})
    if invalid:
        raise ValueError(f"Layers {invalid} out of range for model with {len(layers)} layers.")
    chunks: dict[str, list[torch.Tensor]] = {spec["name"]: [] for spec in normalized}
    current: dict[str, torch.Tensor] = {}
    current_mask: torch.Tensor | None = None
    specs_by_layer: dict[int, list[dict[str, Any]]] = {}
    for spec in normalized:
        specs_by_layer.setdefault(spec["layer"], []).append(spec)

    def make_hook(layer_specs: list[dict[str, Any]]) -> Any:
        def hook(module: Any, inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if current_mask is None:
                raise RuntimeError("Missing current attention mask.")
            for spec in layer_specs:
                current[spec["name"]] = (
                    pool_hidden(value, current_mask, spec["pooling"]).detach().float().cpu()
                )

        return hook

    handles = [
        layers[layer].register_forward_hook(make_hook(layer_specs))
        for layer, layer_specs in specs_by_layer.items()
    ]
    try:
        with torch.inference_mode():
            start = 0
            active_batch_size = max(1, int(batch_size))
            while start < len(texts):
                batch = texts[start : start + active_batch_size]
                try:
                    encoded = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=int(cfg["generation"]["max_input_tokens"]),
                        add_special_tokens=False,
                    )
                    inputs = {key: value.to(device) for key, value in encoded.items()}
                    current.clear()
                    current_mask = inputs.get("attention_mask")
                    out = model(**inputs, use_cache=False)
                    missing = [spec["name"] for spec in normalized if spec["name"] not in current]
                    if missing:
                        raise RuntimeError(f"Forward hooks did not capture hidden states: {missing}")
                    for spec in normalized:
                        chunks[spec["name"]].append(current[spec["name"]])
                    del out, inputs, encoded
                    start += len(batch)
                    if device.type == "cuda" and start % max(active_batch_size * 50, 1) == 0:
                        torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    current.clear()
                    current_mask = None
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if active_batch_size <= 1:
                        raise
                    active_batch_size = max(1, active_batch_size // 2)
                    print(f"feature extraction CUDA OOM; reducing batch size to {active_batch_size}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if runtime_stats is not None:
        runtime_stats["requested_batch_size"] = int(batch_size)
        runtime_stats["final_batch_size"] = int(active_batch_size)
    if not texts:
        return {spec["name"]: np.empty((0, 0), dtype=np.float32) for spec in normalized}
    return {
        spec["name"]: torch.cat(chunks[spec["name"]], dim=0).numpy()
        for spec in normalized
    }


def load_probe_checkpoint(path: str | Path) -> dict[str, Any]:
    path = resolve(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

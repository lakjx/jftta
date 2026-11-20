from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    from accelerate import Accelerator


ROOT = Path(__file__).resolve().parents[1]
MODEL_PROFILE_ENV = "TTA_MODEL_PROFILE"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(dataset_config: Path, method_config: Path) -> dict[str, Any]:
    cfg = deep_merge(read_yaml(dataset_config), read_yaml(method_config))
    cfg["_dataset_config_path"] = str(dataset_config)
    cfg["_method_config_path"] = str(method_config)
    profile_path = os.environ.get(MODEL_PROFILE_ENV, "").strip()
    if profile_path:
        apply_model_profile(cfg, Path(profile_path))
    configure_output_paths(cfg)
    return cfg


def load_model_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate a portable model/artifact profile."""
    resolved = resolve(path)
    profile = read_yaml(resolved)
    metadata = profile.get("model_profile", {})
    slug = str(metadata.get("slug", "")).strip()
    if not slug or not all(character.isalnum() or character in {"_", "-"} for character in slug):
        raise ValueError(f"Invalid model_profile.slug in {resolved}: {slug!r}")
    layer = metadata.get("value_head_layer")
    if not isinstance(layer, int) or layer < 0:
        raise ValueError(f"model_profile.value_head_layer must be a non-negative integer in {resolved}")
    policy = profile.get("models", {}).get("policy")
    if not policy:
        raise ValueError(f"models.policy is required in model profile {resolved}")
    return profile


def apply_model_profile(cfg: dict[str, Any], path: str | Path) -> None:
    """Apply a new policy model without mutating legacy experiment configs.

    Checkpoints, tokenizer-rendered SFT data, outputs, queues, logs, and reports
    are namespaced by the profile slug. Raw/frozen prompt pools and GenARM pair
    labels remain shared because they do not depend on the policy tokenizer.
    """
    profile_path = resolve(path)
    profile = load_model_profile(profile_path)
    metadata = dict(profile["model_profile"])
    slug = str(metadata["slug"])
    layer = int(metadata["value_head_layer"])

    overrides = {key: value for key, value in profile.items() if key != "model_profile"}
    merged = deep_merge(cfg, overrides)
    cfg.clear()
    cfg.update(merged)
    cfg["model_profile"] = metadata
    cfg["_model_profile_path"] = str(profile_path)

    paths = cfg.setdefault("paths", {})
    for key, raw_value in list(paths.items()):
        if not isinstance(raw_value, str) or not raw_value:
            continue
        candidate = Path(raw_value)
        if not candidate.is_absolute() and candidate.parts and candidate.parts[0] == "checkpoints":
            paths[key] = str(Path("checkpoints") / slug / Path(*candidate.parts[1:]))
        elif key == "sft_train_dataset":
            paths[key] = str(Path("data/processed/model_profiles") / slug / candidate.name)

    method_name = str(cfg.get("method", {}).get("name", ""))
    if method_name.startswith("tpo") or method_name == "ultrarm_bon":
        adapter = paths.get("sft_adapter")
        if isinstance(adapter, str):
            paths["sft_adapter"] = adapter.replace(
                "/sft_helpsteer2/adapter", "/sft_helpsteer2_weak/adapter"
            ).replace(
                "/sft_ultrafeedback/adapter", "/sft_ultrafeedback_weak/adapter"
            )

    output_root = Path(str(cfg.get("output", {}).get("root", "output")))
    if not output_root.is_absolute() and (not output_root.parts or output_root.parts[-1] != slug):
        cfg.setdefault("output", {})["root"] = str(output_root / slug)

    judge = cfg.get("judge")
    if isinstance(judge, dict):
        judge["layer"] = layer
        probe = judge.get("probe_path")
        if isinstance(probe, str):
            probe_path = Path(probe)
            if not probe_path.is_absolute() and probe_path.parts and probe_path.parts[0] == "checkpoints":
                probe = str(Path("checkpoints") / slug / Path(*probe_path.parts[1:]))
            judge["probe_path"] = rebase_value_head_layer(probe, layer)


def rebase_value_head_layer(path: str, layer: int) -> str:
    """Replace the layer component in a conventional probe artifact name."""
    return re.sub(r"_layer\d+_", f"_layer{int(layer)}_", path)


def configure_output_paths(cfg: dict[str, Any]) -> None:
    if "method" not in cfg or "evaluation" not in cfg:
        return
    dataset_name = str(cfg["evaluation"]["name"])
    output_name = str(cfg["method"].get("output_name", cfg["method"]["name"]))
    output_root = Path(str(cfg.get("output", {}).get("root", "output")))
    output_dir = output_root / dataset_name / output_name
    cfg.setdefault("paths", {}).update(
        {
            "output_dir": str(output_dir),
            "benchmark_generations": str(output_dir / "generations.jsonl"),
            "benchmark_scores": str(output_dir / "armorm_scores.jsonl"),
            "benchmark_report_json": str(output_dir / "metrics.json"),
        }
    )


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # U+2028/U+2029 are legal inside JSON strings; str.splitlines() would split them.
    lines = path.read_text(encoding="utf-8").split("\n")
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty[-1] if nonempty else -1
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != last_nonempty:
                    raise
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def apply_overrides(
    cfg: dict[str, Any],
    sample: int | None = None,
    max_new_tokens: int | None = None,
    max_steps: int | None = None,
) -> None:
    if sample is not None:
        cfg.setdefault("runtime_overrides", {})["sample"] = int(sample)
    if max_new_tokens is not None:
        cfg["generation"]["max_new_tokens"] = int(max_new_tokens)
        if "grpo" in cfg:
            cfg["grpo"]["max_completion_length"] = int(max_new_tokens)
    if max_steps is not None:
        if "grpo" in cfg:
            cfg["grpo"]["max_steps"] = int(max_steps)
        if "sft" in cfg:
            cfg["sft"]["max_steps"] = int(max_steps)


def enforce_cache_env(cfg: dict[str, Any]) -> None:
    cache = cfg["cache"]
    os.environ["HF_HOME"] = str(resolve(cache["hf_home"]))
    os.environ["HF_HUB_CACHE"] = str(resolve(cache["hf_hub_cache"]))
    os.environ["HF_DATASETS_CACHE"] = str(resolve(cache["hf_datasets_cache"]))
    os.environ["TRANSFORMERS_CACHE"] = str(resolve(cache["transformers_cache"]))
    os.environ["XDG_CACHE_HOME"] = str(resolve(cache["xdg_cache_home"]))
    if "modelscope_cache" in cache:
        os.environ["MODELSCOPE_CACHE"] = str(resolve(cache["modelscope_cache"]))
        os.environ["MODELSCOPE_HOME"] = str(resolve(cache["modelscope_cache"]))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def bind_cuda_to_local_rank() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)


def log_rank_device(accelerator: Accelerator) -> None:
    if not torch.cuda.is_available():
        print(f"rank={accelerator.process_index} device={accelerator.device}", flush=True)
        return
    current = torch.cuda.current_device()
    print(
        f"rank={accelerator.process_index} local_rank={os.environ.get('LOCAL_RANK', '0')} "
        f"device={accelerator.device} current_cuda={current} gpu={torch.cuda.get_device_name(current)}",
        flush=True,
    )


def setup_run(
    config_path: Path,
    dataset_config_path: Path | None = None,
    sample: int | None = None,
    max_new_tokens: int | None = None,
    max_steps: int | None = None,
) -> tuple[dict[str, Any], Accelerator]:
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    if dataset_config_path is None:
        cfg = read_yaml(config_path)
        cfg["_config_path"] = str(config_path)
        configure_output_paths(cfg)
    else:
        cfg = load_experiment_config(dataset_config_path, config_path)
    apply_overrides(cfg, sample=sample, max_new_tokens=max_new_tokens, max_steps=max_steps)
    enforce_cache_env(cfg)
    ensure_dirs(cfg)
    bind_cuda_to_local_rank()
    timeout_seconds = int(cfg.get("runtime", {}).get("distributed_timeout_seconds", 7200))
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_seconds))])
    bind_cuda_to_local_rank()
    set_seed(int(cfg.get("seed", 42)) + int(accelerator.process_index))
    log_rank_device(accelerator)
    return cfg, accelerator


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for value in cfg.get("cache", {}).values():
        resolve(value).mkdir(parents=True, exist_ok=True)
    for value in cfg.get("paths", {}).values():
        path = resolve(value)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_prompt_seed(base_seed: int, prompt_id: Any, method: str) -> int:
    """Return a process- and sharding-independent RNG seed for one generation."""
    payload = f"{int(base_seed)}\0{prompt_id}\0{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def shard_items(items: list[Any], accelerator: Accelerator) -> list[Any]:
    return [item for index, item in enumerate(items) if index % accelerator.num_processes == accelerator.process_index]


def shard_path(path: Path, rank: int) -> Path:
    return path.with_suffix(path.suffix + f".rank{rank}")


def sync_point(cfg: dict[str, Any], accelerator: Accelerator, label: str) -> None:
    accelerator.wait_for_everyone()


def env_file_sync(
    cfg: dict[str, Any],
    accelerator: Accelerator,
    label: str,
    timeout_seconds: int | None = None,
) -> None:
    accelerator.wait_for_everyone()


def merge_shards(path: Path, accelerator: Accelerator, key_fields: list[str]) -> list[dict[str, Any]]:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        seen = set()
        rows: list[dict[str, Any]] = []
        for rank in range(accelerator.num_processes):
            for row in read_jsonl(shard_path(path, rank)):
                key = tuple(row.get(field) for field in key_fields)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        rows.sort(key=lambda row: tuple(str(row.get(field, "")) for field in key_fields))
        write_jsonl(path, rows)
        for rank in range(accelerator.num_processes):
            spath = shard_path(path, rank)
            if spath.exists():
                spath.unlink()
    accelerator.wait_for_everyone()
    return read_jsonl(path)


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = set(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def adapter_ready(path: str | Path) -> bool:
    return (resolve(path) / "adapter_config.json").exists()


def override_sample(cfg: dict[str, Any], default: int) -> int:
    return int(cfg.get("runtime_overrides", {}).get("sample", default))


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def repeat_4gram_rate(text: str) -> float:
    toks = normalize_space(text).split()
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    counts = Counter(grams)
    return sum(count - 1 for count in counts.values() if count > 1) / max(1, len(grams))


def mean(values: list[Any]) -> float | None:
    vals = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else None


def none_sub(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson(rank_values(left), rank_values(right))


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}"


def patch_armorm_transformers_compat() -> None:
    import transformers.models.llama.modeling_llama as llama_modeling

    if not hasattr(llama_modeling, "LLAMA_INPUTS_DOCSTRING"):
        llama_modeling.LLAMA_INPUTS_DOCSTRING = ""


def unload(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

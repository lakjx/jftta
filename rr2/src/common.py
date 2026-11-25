from __future__ import annotations
from collections import Counter
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def adapter_ready(path: str | Path) -> bool:
    return (resolve(path) / "adapter_config.json").exists()


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def repeat_4gram_rate(text: str) -> float:
    toks = normalize_space(text).split()
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    counts = Counter(grams)
    return sum(count - 1 for count in counts.values() if count > 1) / max(1, len(grams))


def patch_armorm_transformers_compat() -> None:
    import transformers.models.llama.modeling_llama as llama_modeling

    if not hasattr(llama_modeling, "LLAMA_INPUTS_DOCSTRING"):
        llama_modeling.LLAMA_INPUTS_DOCSTRING = ""



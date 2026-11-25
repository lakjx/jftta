from __future__ import annotations
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_prompt_seed(base_seed: int, prompt_id: Any, method: str) -> int:
    payload = f"{int(base_seed)}\x00{prompt_id}\x00{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def repeat_4gram_rate(text: str) -> float:
    toks = normalize_space(text).split()
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    counts = Counter(grams)
    return sum((count - 1 for count in counts.values() if count > 1)) / max(
        1, len(grams)
    )


def patch_armorm_transformers_compat() -> None:
    import transformers.models.llama.modeling_llama as llama_modeling

    if not hasattr(llama_modeling, "LLAMA_INPUTS_DOCSTRING"):
        llama_modeling.LLAMA_INPUTS_DOCSTRING = ""

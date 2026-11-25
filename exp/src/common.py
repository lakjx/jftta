from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]

def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path

def model_reference(path: str | Path) -> str:
    value = str(path)
    local = Path(value).expanduser()
    if local.is_absolute():
        return str(local)
    packaged = ROOT / local
    if packaged.exists():
        return str(packaged)
    if local.exists():
        return str(local.resolve())
    if value.startswith(('./', '../', '~')):
        return str(packaged)
    return value

def adapter_ready(path: str | Path) -> bool:
    return (resolve(path) / 'adapter_config.json').exists()

def normalize_space(text: str) -> str:
    return ' '.join(str(text).split())

def repeat_4gram_rate(text: str) -> float:
    toks = normalize_space(text).split()
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    counts = Counter(grams)
    return sum((count - 1 for count in counts.values() if count > 1)) / max(1, len(grams))

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def select_bon(candidates, n_values):
    selections = []
    for n in n_values:
        if not 0 < n <= len(candidates):
            raise ValueError(f"Insufficient candidates for N={n}.")
        bank = candidates[:n]
        if any(not math.isfinite(c["armorm"]["armorm_score"]) for c in bank):
            raise ValueError("Non-finite ArmoRM overall score.")
        selected = max(
            range(n), key=lambda index: bank[index]["armorm"]["armorm_score"]
        )
        selections.append(
            {
                "n": n,
                "selector": "armorm_score",
                "selected_candidate_idx": selected,
                "total_generated_tokens": sum(
                    c["total_generated_tokens"] for c in bank
                ),
                "response": bank[selected]["text"],
                "armorm": bank[selected]["armorm"],
            }
        )
    return selections

from __future__ import annotations

import math
import time
from typing import Any

from src.models import generate_texts, render_chat
from src.search_common import generation_row


def generate_bon(cfg, tokenizer, model, judge, device, prompt) -> dict[str, Any]:
    started = time.monotonic()
    count = int(cfg.get("search", {}).get("bon_candidates", 4))
    if count < 1:
        raise ValueError("BoN requires at least one candidate")
    if judge is None:
        raise ValueError("BoN requires a reward scorer")
    batch_size = max(
        1, min(count, int(cfg.get("decoding", {}).get("candidate_batch_size", count)))
    )
    text = render_chat(
        tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True
    )
    candidates = []
    calls = 0
    while len(candidates) < count:
        current = min(batch_size, count - len(candidates))
        outputs = generate_texts(
            cfg, tokenizer, model, device, [text], current, temperature=0.3
        )
        if len(outputs) != current:
            raise RuntimeError(
                f"Requested {current} candidates, received {len(outputs)}"
            )
        candidates.extend(outputs)
        calls += 1
    scores = [
        float(value)
        for value in judge.score(
            prompt["prompt"], "", [item["text"] for item in candidates]
        )
    ]
    if len(scores) != count or not all(math.isfinite(value) for value in scores):
        raise ValueError("Reward scorer must return one finite scalar per candidate")
    selected_index = max(range(count), key=scores.__getitem__)
    selected = candidates[selected_index]
    row = generation_row(
        prompt, "bon", selected["text"], selected, candidate_idx=selected_index
    )
    row.update(
        {
            "selection_score": scores[selected_index],
            "candidate_scores": scores,
            "bon_candidates": count,
            "lm_forward_calls": calls,
            "tta_generated_tokens": sum(
                int(item["raw_token_count"]) for item in candidates
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return row

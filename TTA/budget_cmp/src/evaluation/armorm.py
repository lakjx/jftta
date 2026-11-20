from __future__ import annotations
from typing import Any
import torch
from transformers import AutoModelForSequenceClassification
from oval.common import patch_armorm_transformers_compat, resolve
from oval.models import render_chat

ARMORM_ATTRIBUTES = [
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is_safe",
    "prometheus-score",
    "argilla-overall_quality",
    "argilla-judge_lm",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",
]


def load_armorm_model(cfg: dict[str, Any], device: torch.device) -> Any:
    patch_armorm_transformers_compat()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        str(resolve(cfg["models"]["armorm"])),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = int(cfg["armorm"]["pad_token_id"])
    model.eval().to(device)
    return model


def score_armorm_batch(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    texts = [
        render_chat(
            tokenizer,
            "",
            row["prompt"],
            response=row["response"],
            add_generation_prompt=False,
        )
        for row in batch
    ]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(cfg["armorm"]["max_length"]),
        add_special_tokens=False,
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        )
    rewards = output.rewards.detach().float().cpu().tolist()
    scores = output.score.detach().float().cpu().reshape(-1).tolist()
    out = []
    for row, reward_values, score in zip(batch, rewards, scores, strict=True):
        item = {
            "selection_id": row["generation_id"],
            "prompt_id": row["prompt_id"],
            "method": row.get("method"),
            "candidate_idx": row.get("candidate_idx"),
            "response_token_count": row.get("response_token_count"),
            "armorm_score": float(score),
        }
        for attr, value in zip(ARMORM_ATTRIBUTES, reward_values, strict=True):
            item[f"armorm_{attr}"] = float(value)
        out.append(item)
    return out


class ArmoRMScorer:
    def __init__(self, cfg, device, batch_size=16):
        from oval.models import load_tokenizer

        self.cfg = cfg
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = load_tokenizer(cfg["models"]["armorm"])
        self.model = load_armorm_model(cfg, device)

    def evaluate(self, jobs):
        import gc

        scores, offset = [], 0
        while offset < len(jobs):
            batch = jobs[offset : offset + self.batch_size]
            try:
                scores.extend(
                    score_armorm_batch(
                        self.cfg, self.tokenizer, self.model, self.device, batch
                    )
                )
            except torch.cuda.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            offset += len(batch)
        return scores

    def score(self, prompt, responses):
        jobs = [
            dict(
                generation_id=f"tpo::{i}",
                prompt_id="tpo",
                prompt=prompt,
                response=response,
            )
            for i, response in enumerate(responses)
        ]
        target = "armorm_" + self.cfg["experiment"]["reward_attribute"]
        return [row[target] for row in self.evaluate(jobs)]

    def close(self):
        self.model = None
        self.tokenizer = None

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from src.common import (
    merge_shards,
    patch_armorm_transformers_compat,
    read_jsonl,
    resolve,
    shard_items,
    shard_path,
    sync_point,
    unload,
    write_jsonl,
)
from src.models import load_tokenizer, render_chat

if TYPE_CHECKING:
    from accelerate import Accelerator


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


def score_jobs_distributed(
    cfg: dict[str, Any],
    accelerator: Accelerator,
    jobs: list[dict[str, Any]],
    output_path: Any,
    force: bool,
) -> list[dict[str, Any]]:
    output_path = resolve(output_path)
    if force and accelerator.is_main_process and output_path.exists():
        output_path.unlink()
    sync_point(cfg, accelerator, f"reset_scores_{output_path.name}")
    shard = shard_path(output_path, accelerator.process_index)
    if force and shard.exists():
        shard.unlink()
    local_jobs = shard_items(jobs, accelerator)
    rows = read_jsonl(shard)
    if not force and not rows and output_path.exists():
        local_ids = {job["generation_id"] for job in local_jobs}
        rows = [row for row in read_jsonl(output_path) if row.get("selection_id") in local_ids]
    done = {row.get("selection_id") for row in rows}
    pending_jobs = [job for job in local_jobs if job["generation_id"] not in done]
    if pending_jobs:
        tokenizer = load_tokenizer(cfg["models"]["armorm"])
        model = load_armorm_model(cfg, accelerator.device)
        batch_size = max(1, int(cfg["runtime"]["eval_batch_size"]))
        index = 0
        progress = tqdm(total=len(pending_jobs), desc=f"armorm-rank{accelerator.process_index}", disable=False)
        while index < len(pending_jobs):
            batch = pending_jobs[index : index + batch_size]
            try:
                rows.extend(score_armorm_batch(cfg, tokenizer, model, accelerator.device, batch))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"ArmoRM CUDA OOM; reducing eval batch size to {batch_size}", flush=True)
                continue
            index += len(batch)
            progress.update(len(batch))
            write_jsonl(shard, rows)
        progress.close()
        unload(model)
        del tokenizer
    write_jsonl(shard, rows)
    return merge_shards(output_path, accelerator, ["selection_id"])


def score_armorm_batch(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    texts = [
        render_chat(tokenizer, "", row["prompt"], response=row["response"], add_generation_prompt=False)
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
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
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

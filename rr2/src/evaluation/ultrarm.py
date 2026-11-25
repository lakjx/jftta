from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, PreTrainedModel
from src.common import resolve

class LlamaRewardModel(PreTrainedModel):
    """Corrected UltraRM architecture; the model repository's Python file is broken."""

    config_class = LlamaConfig
    base_model_prefix = "model"

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True, **kwargs)
        token_rewards = self.regression_head(output.last_hidden_state).squeeze(-1)
        if attention_mask is None:
            ends = torch.full(
                (token_rewards.shape[0], 1),
                token_rewards.shape[1] - 1,
                device=token_rewards.device,
                dtype=torch.long,
            )
        else:
            ends = attention_mask.long().cumsum(dim=1).argmax(dim=1, keepdim=True)
        return token_rewards.gather(1, ends).squeeze(1)


def render_ultrarm_text(prompt: str, response: str) -> str:
    return f"Human: {prompt.strip()}\nAssistant: {response.strip()}"


def load_ultrarm(cfg: dict[str, Any], device: torch.device) -> tuple[Any, LlamaRewardModel]:
    path = resolve(cfg["models"]["ultrarm"])
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = LlamaRewardModel.from_pretrained(
        str(path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    return tokenizer, model


def score_ultrarm_batch(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: LlamaRewardModel,
    device: torch.device,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    texts = [render_ultrarm_text(row["prompt"], row["response"]) for row in batch]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(cfg.get("ultrarm", {}).get("max_length", 4096)),
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        scores = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
    values = scores.detach().float().cpu().tolist()
    return [
        {
            "selection_id": row["generation_id"],
            "prompt_id": row["prompt_id"],
            "method": row.get("method"),
            "response_token_count": row.get("response_token_count"),
            "ultrarm_score": float(score),
        }
        for row, score in zip(batch, values, strict=True)
    ]

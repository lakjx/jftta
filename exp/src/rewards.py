from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import model_reference
from src.models import render_chat

ARMORM_ATTRIBUTES = (
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
)
DEFAULT_MODELS = {
    'armorm': 'RLHFlow/ArmoRM-Llama3-8B-v0.1',
    'ultrarm': 'openbmb/UltraRM-13b',
}


class RewardScorer:
    def __init__(self, kind: str, tokenizer: Any, model: Any, device: torch.device,
                 batch_size: int = 4, attribute: str | None = None) -> None:
        if kind not in DEFAULT_MODELS:
            raise ValueError(f'Unknown reward model: {kind}')
        if attribute and (kind != 'armorm' or attribute not in ARMORM_ATTRIBUTES):
            raise ValueError('Dimension scoring requires a supported ArmoRM attribute.')
        self.kind = kind
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.batch_size = max(1, batch_size)
        self.attribute = attribute

    def score(self, prompt: str, prefix: str, candidates: list[str]) -> list[float]:
        responses = [' '.join(part for part in (prefix.strip(), c.strip()) if part) for c in candidates]
        if self.kind == 'armorm':
            texts = [render_chat(self.tokenizer, '', prompt, response=r, add_generation_prompt=False) for r in responses]
        else:
            texts = [f'Human: {prompt.strip()}\nAssistant: {r}' for r in responses]
        scores = []
        start = 0
        batch_size = self.batch_size
        while start < len(texts):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(batch, return_tensors='pt', padding=True,
                                     truncation=False, add_special_tokens=self.kind == 'ultrarm')
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            try:
                with torch.inference_mode():
                    output = self.model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'))
                if self.kind == 'armorm':
                    values = output.rewards[:, ARMORM_ATTRIBUTES.index(self.attribute)] if self.attribute else output.score
                else:
                    values = output
                scores.extend(values.detach().float().cpu().reshape(-1).tolist())
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue
            start += len(batch)
        return scores


def load_reward(kind: str, device: torch.device, model_path: str | None = None,
                batch_size: int = 4, attribute: str | None = None) -> RewardScorer:
    if kind not in DEFAULT_MODELS:
        raise ValueError(f'Unknown reward model: {kind}')
    path = model_reference(model_path or DEFAULT_MODELS[kind])
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if kind == 'armorm':
        import transformers.models.llama.modeling_llama as llama_modeling
        if not hasattr(llama_modeling, 'LLAMA_INPUTS_DOCSTRING'):
            llama_modeling.LLAMA_INPUTS_DOCSTRING = ''
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        from src.ultrarm import LlamaRewardModel
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        model = LlamaRewardModel.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval().to(device)
    return RewardScorer(kind, tokenizer, model, device, batch_size, attribute)

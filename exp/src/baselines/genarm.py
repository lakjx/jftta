from __future__ import annotations
import time
from typing import Any
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM
from src.common import adapter_ready, model_reference, resolve
from src.models import (
    decode_generated_tokens,
    generation_stop_token_ids,
    load_policy_tokenizer,
    policy_model_path,
    render_chat,
)
from src.search_common import generation_row


def genarm_log_probs(
    policy_logits: torch.Tensor,
    arm_logits: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    if alpha < 0:
        raise ValueError("GenARM alpha must be non-negative")
    if temperature <= 0:
        raise ValueError("GenARM temperature must be positive")
    policy_lp = F.log_softmax(policy_logits.float(), dim=-1)
    arm_lp = F.log_softmax(arm_logits.float(), dim=-1)
    normalized_formula = (policy_lp + float(alpha) * arm_lp) / (1.0 + float(alpha))
    effective_temperature = float(temperature) / (1.0 + float(alpha))
    return F.log_softmax(normalized_formula / effective_temperature, dim=-1)


def sample_genarm(log_probs: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
    filtered = log_probs.clone()
    if top_k > 0 and top_k < filtered.shape[-1]:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    probabilities = F.softmax(filtered, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(
            probabilities, descending=True, dim=-1
        )
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= float(top_p)
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        sampled = torch.multinomial(sorted_probs, 1)
        return sorted_indices.gather(-1, sampled)
    return torch.multinomial(probabilities, 1)


def load_genarm_model(cfg: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    sft_adapter = resolve(cfg["paths"]["sft_adapter"])
    arm_adapter = resolve(cfg["paths"]["genarm_adapter"])
    if not adapter_ready(sft_adapter):
        raise FileNotFoundError(f"SFT adapter missing at {sft_adapter}")
    if not adapter_ready(arm_adapter):
        raise FileNotFoundError(
            f"GenARM adapter missing at {arm_adapter}; train it with python -m src.baselines.train genarm"
        )
    tokenizer = load_policy_tokenizer(cfg)
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_reference(policy_model_path(cfg)),
        revision=cfg.get("models", {}).get("revision"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    sft_model = PeftModel.from_pretrained(
        base, str(sft_adapter), is_trainable=False
    ).merge_and_unload()
    model = PeftModel.from_pretrained(sft_model, str(arm_adapter), is_trainable=False)
    model.eval().to(device)
    return (tokenizer, model)


def generate_genarm(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    method_cfg = cfg.get("decoding", {}).get("genarm", {})
    alpha = float(method_cfg.get("alpha", 1.0))
    temperature = float(method_cfg.get("temperature", 1.0))
    top_p = float(method_cfg.get("top_p", 1.0))
    top_k = int(method_cfg.get("top_k", 0))
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    prompt_text = render_chat(
        tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True
    )
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg["generation"]["max_input_tokens"]),
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    stop_ids = set(generation_stop_token_ids(tokenizer))
    generated: list[int] = []
    with torch.inference_mode(), model.disable_adapter():
        policy_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    with torch.inference_mode():
        arm_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    policy_cache = policy_output.past_key_values
    arm_cache = arm_output.past_key_values
    policy_logits = policy_output.logits[:, -1]
    arm_logits = arm_output.logits[:, -1]
    del policy_output, arm_output
    lm_calls = 2
    for _ in range(max_new_tokens):
        log_probs = genarm_log_probs(policy_logits, arm_logits, alpha, temperature)
        selected = sample_genarm(log_probs, top_p=top_p, top_k=top_k)
        selected_id = int(selected.item())
        generated.append(selected_id)
        if selected_id in stop_ids:
            break
        with torch.inference_mode(), model.disable_adapter():
            policy_output = model(
                input_ids=selected,
                past_key_values=policy_cache,
                use_cache=True,
                return_dict=True,
            )
        with torch.inference_mode():
            arm_output = model(
                input_ids=selected,
                past_key_values=arm_cache,
                use_cache=True,
                return_dict=True,
            )
        lm_calls += 2
        policy_cache = policy_output.past_key_values
        arm_cache = arm_output.past_key_values
        policy_logits = policy_output.logits[:, -1]
        arm_logits = arm_output.logits[:, -1]
        del policy_output, arm_output
    decoded = decode_generated_tokens(
        cfg,
        tokenizer,
        torch.tensor(generated, dtype=torch.long),
        list(stop_ids),
        max_new_tokens,
    )
    row = generation_row(prompt, "genarm", decoded["text"], decoded, candidate_idx=None)
    row.update(
        {
            "genarm_alpha": alpha,
            "genarm_base_temperature": temperature,
            "genarm_effective_temperature": temperature / (1.0 + alpha),
            "genarm_top_p": top_p,
            "genarm_top_k": top_k,
            "genarm_formula": "log_p_sft + alpha * log_p_arm",
            "genarm_shared_sft_backbone": True,
            "lm_forward_calls": lm_calls,
            "rm_forward_calls": 0,
            "rm_scored_sequences": 0,
            "tta_generated_tokens": decoded["raw_token_count"],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return row

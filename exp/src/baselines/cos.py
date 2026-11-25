from __future__ import annotations
import time
from typing import Any
import torch
import torch.nn.functional as F
from src.models import decode_generated_tokens, generation_stop_token_ids, render_chat
from src.search_common import generation_row


def context_steered_log_probs(
    context_logits: torch.Tensor,
    no_context_logits: torch.Tensor,
    steering_lambda: float,
    temperature: float,
) -> torch.Tensor:
    context_lp = F.log_softmax(context_logits / float(temperature), dim=-1)
    no_context_lp = F.log_softmax(no_context_logits / float(temperature), dim=-1)
    steered = (1.0 + float(steering_lambda)) * context_lp - float(
        steering_lambda
    ) * no_context_lp
    return F.log_softmax(steered, dim=-1)


def sample_top_p(log_probs: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = log_probs.exp()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative - sorted_probs >= float(top_p)
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled)


def generate_cos(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    cos_cfg = cfg.get("decoding", {}).get("cos", {})
    steering_lambda = float(cos_cfg.get("lambda", 1.0))
    context = str(cos_cfg.get("context", ""))
    temperature = float(cfg["generation"]["temperature"])
    top_p = float(cfg["generation"]["top_p"])
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    context_text = render_chat(
        tokenizer, context, prompt["prompt"], response=None, add_generation_prompt=True
    )
    no_context_text = render_chat(
        tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True
    )
    context_ids = tokenizer(
        context_text,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg["generation"]["max_input_tokens"]),
        add_special_tokens=False,
    )["input_ids"].to(device)
    no_context_ids = tokenizer(
        no_context_text,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg["generation"]["max_input_tokens"]),
        add_special_tokens=False,
    )["input_ids"].to(device)
    stop_ids = set(generation_stop_token_ids(tokenizer))
    generated: list[int] = []
    with torch.inference_mode():
        context_output = model(input_ids=context_ids, use_cache=True, return_dict=True)
        no_context_output = model(
            input_ids=no_context_ids, use_cache=True, return_dict=True
        )
    lm_calls = 2
    context_cache = context_output.past_key_values
    no_context_cache = no_context_output.past_key_values
    context_logits = context_output.logits[:, -1]
    no_context_logits = no_context_output.logits[:, -1]
    del context_output, no_context_output
    for _ in range(max_new_tokens):
        log_probs = context_steered_log_probs(
            context_logits, no_context_logits, steering_lambda, temperature
        )
        selected = sample_top_p(log_probs, top_p)
        selected_id = int(selected.item())
        generated.append(selected_id)
        if selected_id in stop_ids:
            break
        with torch.inference_mode():
            context_output = model(
                input_ids=selected,
                past_key_values=context_cache,
                use_cache=True,
                return_dict=True,
            )
            no_context_output = model(
                input_ids=selected,
                past_key_values=no_context_cache,
                use_cache=True,
                return_dict=True,
            )
        lm_calls += 2
        context_cache = context_output.past_key_values
        no_context_cache = no_context_output.past_key_values
        context_logits = context_output.logits[:, -1]
        no_context_logits = no_context_output.logits[:, -1]
        del context_output, no_context_output
    decoded = decode_generated_tokens(
        cfg,
        tokenizer,
        torch.tensor(generated, dtype=torch.long),
        list(stop_ids),
        max_new_tokens,
    )
    row = generation_row(prompt, "cos", decoded["text"], decoded, candidate_idx=None)
    row.update(
        {
            "cos_lambda": steering_lambda,
            "cos_context": context,
            "lm_forward_calls": lm_calls,
            "rm_forward_calls": 0,
            "rm_scored_sequences": 0,
            "tta_generated_tokens": decoded["raw_token_count"],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return row

from __future__ import annotations

import time
from typing import Any

import torch

from src.common import unload
from src.evaluation.armorm import (
    ARMORM_ATTRIBUTES,
    load_armorm_model,
    score_armorm_batch,
)
from src.models import decode_generated_tokens, generation_stop_token_ids, load_tokenizer, render_chat
from src.search_common import generation_row

def args_combined_scores(lm_logits: torch.Tensor, rewards: torch.Tensor, weight: float) -> torch.Tensor:
    """ARGS uses raw LM logits plus a weighted partial-sequence reward."""
    return lm_logits + float(weight) * rewards.to(device=lm_logits.device, dtype=lm_logits.dtype)


class ArmoRMScorer:
    def __init__(
        self,
        cfg: dict[str, Any],
        device: torch.device,
        policy_tokenizer: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.tokenizer = load_tokenizer(cfg["models"]["armorm"])
        self.model = load_armorm_model(cfg, device)
        attribute = str(cfg["experiment"]["reward_attribute"])
        self.target_key = f"armorm_{attribute}"
        self.target_index = ARMORM_ATTRIBUTES.index(attribute)
        self.policy_tokenizer = policy_tokenizer
        self.token_ids_compatible = bool(
            policy_tokenizer is not None
            and token_id_spaces_compatible(policy_tokenizer, self.tokenizer)
        )
        self.forward_calls = 0
        self.scored_sequences = 0

    def score(self, prompt: str, responses: list[str]) -> list[float]:
        jobs = [
            {
                "generation_id": f"tta_reward_{index}",
                "prompt_id": f"tta_reward_{index}",
                "prompt": prompt,
                "response": response,
                "method": "tta_reward",
                "candidate_idx": index,
            }
            for index, response in enumerate(responses)
        ]
        scores: list[float] = []
        batch_size = max(1, int(self.cfg["runtime"]["eval_batch_size"]))
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            rows = score_armorm_batch(self.cfg, self.tokenizer, self.model, self.device, batch)
            scores.extend(float(row[self.target_key]) for row in rows)
            self.forward_calls += 1
            self.scored_sequences += len(batch)
        return scores

    def initial_partial_cache(self, prompt: str) -> Any:
        if not self.token_ids_compatible:
            raise RuntimeError("Incremental ArmoRM cache requires compatible policy/reward token IDs.")
        text = render_chat(self.tokenizer, "", prompt, response=None, add_generation_prompt=True)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.cfg["generation"]["max_input_tokens"]),
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        with torch.inference_mode():
            output = self.model.model(input_ids=input_ids, use_cache=True, return_dict=True)
        self.forward_calls += 1
        return output.past_key_values

    def partial_candidate_rewards(self, cache: Any, candidate_ids: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Score top-k next tokens at a virtual EOT and retain all candidate cache rows."""
        if not self.token_ids_compatible:
            raise RuntimeError("Raw candidate token IDs cannot cross incompatible tokenizers.")
        candidate_ids = candidate_ids.to(self.device).reshape(-1, 1)
        candidate_count = int(candidate_ids.shape[0])
        cache.batch_repeat_interleave(candidate_count)
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if not isinstance(eot_id, int) or eot_id < 0:
            eot_id = int(self.tokenizer.eos_token_id)
        eot = torch.full_like(candidate_ids, int(eot_id))
        with torch.inference_mode():
            candidate_output = self.model.model(
                input_ids=candidate_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            eot_output = self.model.model(
                input_ids=eot,
                past_key_values=candidate_output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            rewards = self.model.regression_layer(eot_output.last_hidden_state[:, -1])[:, self.target_index]
        self.forward_calls += 2
        self.scored_sequences += candidate_count
        return rewards.detach().float(), eot_output.past_key_values

    def score_policy_token_candidates(
        self,
        policy_tokenizer: Any,
        generated_ids: list[int],
        candidate_ids: torch.Tensor,
        cache_state: dict[str, Any],
    ) -> tuple[torch.Tensor, Any, list[list[int]], int]:
        """Score cross-tokenizer candidates with an exact shared-prefix cache.

        The full partial response is decoded in policy space and retokenized in
        reward space, so Qwen IDs are never reinterpreted as Llama IDs. Only
        the reward-token suffix after the candidates' stable common prefix is
        forwarded through ArmoRM.
        """
        candidate_reward_ids = [
            self.tokenizer(
                cache_state["prompt_text"]
                + policy_tokenizer.decode(
                    [*generated_ids, int(candidate_id)],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                add_special_tokens=False,
            )["input_ids"]
            for candidate_id in candidate_ids.detach().cpu().tolist()
        ]
        common_length = common_token_prefix_length(
            [cache_state["input_ids"], *candidate_reward_ids]
        )
        cache = cache_state["cache"]
        cache.crop(common_length)
        candidate_count = len(candidate_reward_ids)
        cache.batch_repeat_interleave(candidate_count)
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if not isinstance(eot_id, int) or eot_id < 0:
            eot_id = int(self.tokenizer.eos_token_id)
        suffixes = [ids[common_length:] + [int(eot_id)] for ids in candidate_reward_ids]
        suffix_lengths = [len(ids) for ids in suffixes]
        max_suffix = max(suffix_lengths)
        raw_pad_id = self.tokenizer.pad_token_id
        pad_id = int(raw_pad_id if raw_pad_id is not None else self.tokenizer.eos_token_id)
        input_ids = torch.tensor(
            [ids + [pad_id] * (max_suffix - len(ids)) for ids in suffixes],
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [1] * common_length + [1] * length + [0] * (max_suffix - length)
                for length in suffix_lengths
            ],
            device=self.device,
            dtype=torch.long,
        )
        with torch.inference_mode():
            output = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            eot_positions = torch.tensor(
                [length - 1 for length in suffix_lengths],
                device=self.device,
                dtype=torch.long,
            )
            hidden = output.last_hidden_state[
                torch.arange(candidate_count, device=self.device),
                eot_positions,
            ]
            rewards = self.model.regression_layer(hidden)[:, self.target_index]
        self.forward_calls += 1
        self.scored_sequences += candidate_count
        return rewards.detach().float(), output.past_key_values, candidate_reward_ids, common_length

    def initial_policy_bridge_cache(self, prompt: str) -> dict[str, Any]:
        prompt_text = render_chat(self.tokenizer, "", prompt, response=None, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            output = self.model.model(input_ids=tensor, use_cache=True, return_dict=True)
        self.forward_calls += 1
        return {"cache": output.past_key_values, "input_ids": input_ids, "prompt_text": prompt_text}

    def select_policy_bridge_cache(
        self,
        expanded_cache: Any,
        candidate_reward_ids: list[list[int]],
        selected_position: int,
        common_length: int,
        prompt_text: str,
    ) -> dict[str, Any]:
        expanded_cache.batch_select_indices(
            torch.tensor([selected_position], device=self.device, dtype=torch.long)
        )
        selected_ids = candidate_reward_ids[selected_position]
        expanded_cache.crop(len(selected_ids))
        if len(selected_ids) < common_length:
            raise RuntimeError("Selected reward sequence is shorter than its cached common prefix.")
        return {"cache": expanded_cache, "input_ids": selected_ids, "prompt_text": prompt_text}

    def close(self) -> None:
        unload(self.model)
        del self.tokenizer


def generate_args(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    reward: ArmoRMScorer,
    device: torch.device,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    args_cfg = cfg.get("decoding", {}).get("args", {})
    weight = float(args_cfg.get("weight", 1.0))
    width = int(args_cfg.get("top_k", 10))
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    prompt_text = render_chat(tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg["generation"]["max_input_tokens"]),
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(device)
    stop_ids = set(generation_stop_token_ids(tokenizer))
    generated: list[int] = []
    lm_calls = 0
    with torch.inference_mode():
        lm_output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    lm_calls += 1
    lm_cache = lm_output.past_key_values
    logits = lm_output.logits[:, -1]
    rm_cache = (
        reward.initial_partial_cache(prompt["prompt"])
        if reward.token_ids_compatible
        else reward.initial_policy_bridge_cache(prompt["prompt"])
    )
    del lm_output
    for _ in range(max_new_tokens):
        candidate_logits, candidate_ids = torch.topk(logits, k=min(width, logits.shape[-1]), dim=-1)
        candidate_ids_flat = candidate_ids.reshape(-1)
        if reward.token_ids_compatible:
            candidate_rewards, expanded_cache = reward.partial_candidate_rewards(rm_cache, candidate_ids_flat)
        else:
            candidate_rewards, expanded_cache, candidate_reward_ids, common_length = reward.score_policy_token_candidates(
                tokenizer,
                generated,
                candidate_ids_flat,
                rm_cache,
            )
        combined = args_combined_scores(candidate_logits.reshape(-1), candidate_rewards, weight)
        selected_position = int(torch.argmax(combined).item())
        selected_id = int(candidate_ids_flat[selected_position].item())
        if reward.token_ids_compatible:
            expanded_cache.batch_select_indices(torch.tensor([selected_position], device=device, dtype=torch.long))
            expanded_cache.crop(expanded_cache.get_seq_length() - 1)
            rm_cache = expanded_cache
        else:
            rm_cache = reward.select_policy_bridge_cache(
                expanded_cache,
                candidate_reward_ids,
                selected_position,
                common_length,
                rm_cache["prompt_text"],
            )
        generated.append(selected_id)
        if selected_id in stop_ids:
            break
        selected = torch.tensor([[selected_id]], device=device, dtype=input_ids.dtype)
        with torch.inference_mode():
            lm_output = model(input_ids=selected, past_key_values=lm_cache, use_cache=True, return_dict=True)
        lm_calls += 1
        lm_cache = lm_output.past_key_values
        logits = lm_output.logits[:, -1]
        del lm_output
    decoded = decode_generated_tokens(
        cfg,
        tokenizer,
        torch.tensor(generated, dtype=torch.long),
        list(stop_ids),
        max_new_tokens,
    )
    row = generation_row(prompt, "args", decoded["text"], decoded, candidate_idx=None)
    row.update(
        {
            "args_weight": weight,
            "args_top_k": width,
            "args_reward_token_bridge": "shared_token_ids" if reward.token_ids_compatible else "text_retokenize",
            "lm_forward_calls": lm_calls,
            "rm_forward_calls": reward.forward_calls,
            "rm_scored_sequences": reward.scored_sequences,
            "tta_generated_tokens": decoded["raw_token_count"],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return row


def token_id_spaces_compatible(policy_tokenizer: Any, reward_tokenizer: Any) -> bool:
    """Whether every policy token has the same integer ID in reward space."""
    policy_vocab = policy_tokenizer.get_vocab()
    reward_vocab = reward_tokenizer.get_vocab()
    return all(reward_vocab.get(token) == token_id for token, token_id in policy_vocab.items())


def common_token_prefix_length(sequences: list[list[int]]) -> int:
    if not sequences:
        raise ValueError("At least one token sequence is required.")
    limit = min(len(sequence) for sequence in sequences)
    for index in range(limit):
        value = sequences[0][index]
        if any(sequence[index] != value for sequence in sequences[1:]):
            return index
    return limit

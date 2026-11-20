from __future__ import annotations
from typing import Any
import numpy as np
import torch
from src.models import generate_texts, render_chat_with_assistant_prefix
from src.search_common import adjust_candidate_score, append_segment, normalize_mpc_segment, token_count

def adjusted_candidate_scores(
    cfg: dict[str, Any],
    tokenizer: Any,
    candidates: list[dict[str, Any]],
    scores: np.ndarray,
) -> list[float]:
    return [
        adjust_candidate_score(
            cfg,
            tokenizer,
            {"text": row["segment"], "raw": row["raw"]},
            float(score),
        )
        for row, score in zip(candidates, scores, strict=True)
    ]


def prepare_ftrl_targets(
    targets: list[float],
    normalization: str,
) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in targets if np.isfinite(float(value))],
        dtype=np.float32,
    )
    if finite.size == 0:
        return {
            "values": [float("-inf") for _ in targets],
            "mean": None,
            "std": None,
            "gap": 0.0,
            "gap_z": 0.0,
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    ordered = sorted((float(value) for value in finite), reverse=True)
    gap = ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
    gap_z = gap / max(std, 1e-8)
    if normalization == "zscore":
        values = [
            (float(value) - mean) / max(std, 1e-8)
            if np.isfinite(float(value))
            else float("-inf")
            for value in targets
        ]
    elif normalization == "none":
        values = [float(value) for value in targets]
    else:
        raise ValueError(f"Unknown FTRL target normalization: {normalization}")
    return {
        "values": values,
        "mean": mean,
        "std": std,
        "gap": float(gap),
        "gap_z": float(gap_z),
    }


def ftrl_update_allowed(alpha: float, gate_passed: bool, pair_count: int) -> bool:
    return bool(float(alpha) > 0.0 and gate_passed and int(pair_count) > 0)


def generate_ftrl_lookahead_records(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: str,
    prefix: str,
    candidates: list[dict[str, Any]],
    lookahead_per_action: int,
    lookahead_max_tokens: int,
    final_max_tokens: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jobs: list[tuple[int, str, tuple[int, int, str, str]]] = []
    for action_index, candidate in enumerate(candidates):
        segment = candidate["segment"]
        prefix_after = append_segment(prefix, segment)
        remaining = final_max_tokens - token_count(tokenizer, prefix_after)
        if lookahead_per_action <= 0 or remaining <= 0 or candidate["raw"]["stop_reason"] == "stop_token":
            records.append(
                {
                    "action_index": action_index,
                    "rollout_index": 0,
                    "target_candidate": segment,
                    "text": prefix_after,
                    "raw": candidate["raw"],
                    "lookahead_continuation": "",
                }
            )
            continue
        rollout_tokens = min(lookahead_max_tokens, remaining)
        prompt_text = render_chat_with_assistant_prefix(tokenizer, "", prompt, prefix_after)
        for rollout_index in range(lookahead_per_action):
            jobs.append((rollout_tokens, prompt_text, (action_index, rollout_index, segment, prefix_after)))
    for rollout_tokens in sorted({tokens for tokens, _, _ in jobs}, reverse=True):
        group = [(prompt_text, metadata) for tokens, prompt_text, metadata in jobs if tokens == rollout_tokens]
        outputs = generate_texts(
            cfg,
            tokenizer,
            model,
            device,
            [prompt_text for prompt_text, _ in group],
            1,
            max_new_tokens=rollout_tokens,
        )
        for (_, (action_index, rollout_index, segment, prefix_after)), item in zip(group, outputs, strict=True):
            continuation = item["text"].strip()
            target_candidate = append_segment(segment, continuation) if continuation else segment
            records.append(
                {
                    "action_index": action_index,
                    "rollout_index": rollout_index,
                    "target_candidate": target_candidate,
                    "text": append_segment(prefix_after, continuation) if continuation else prefix_after,
                    "raw": item,
                    "lookahead_continuation": continuation,
                }
            )
    return records


def top_mean(values: list[float], top_k: int) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return float("-inf")
    ordered = sorted(finite, reverse=True)
    top = ordered[: max(1, min(int(top_k), len(ordered)))]
    return float(sum(top) / len(top))


def ftrl_pairwise_gradient(
    features: np.ndarray,
    targets: list[float],
    weight: np.ndarray,
    margin: float,
    max_pairs: int,
) -> tuple[np.ndarray, int]:
    pairs: list[tuple[float, int, int]] = []
    for left_idx, left_target in enumerate(targets):
        if not np.isfinite(float(left_target)):
            continue
        for right_idx, right_target in enumerate(targets):
            if left_idx == right_idx or not np.isfinite(float(right_target)):
                continue
            gap = float(left_target) - float(right_target)
            if gap > margin:
                pairs.append((gap, left_idx, right_idx))
    if not pairs:
        return np.zeros(features.shape[1], dtype=np.float32), 0
    pairs.sort(reverse=True)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    grad = np.zeros(features.shape[1], dtype=np.float32)
    for _, winner_idx, loser_idx in pairs:
        diff = features[winner_idx] - features[loser_idx]
        logit = float(np.clip(weight @ diff, -40.0, 40.0))
        coeff = -1.0 / (1.0 + float(np.exp(logit)))
        grad += coeff * diff
    grad /= float(len(pairs))
    return grad.astype(np.float32), len(pairs)


def ftrl_proximal_delta(
    grad_sum: np.ndarray,
    grad_sq_sum: np.ndarray,
    alpha: float,
    beta: float,
    l1: float,
    l2: float,
    max_delta_norm: float,
) -> np.ndarray:
    denom = (float(beta) + np.sqrt(np.maximum(grad_sq_sum, 0.0))) / max(float(alpha), 1e-8) + float(l2)
    if l1 > 0.0:
        adjusted = np.where(np.abs(grad_sum) > l1, grad_sum - np.sign(grad_sum) * l1, 0.0)
        delta = -adjusted / denom
    else:
        delta = -grad_sum / denom
    return clip_vector_norm(delta.astype(np.float32), max_delta_norm)


def clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0:
        return vector
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= max_norm:
        return vector
    return (vector * (float(max_norm) / max(norm, 1e-8))).astype(np.float32)

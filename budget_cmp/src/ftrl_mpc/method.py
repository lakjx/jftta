from __future__ import annotations
from typing import Any
import numpy as np
import torch
from oval.generation_cost import generation_stage, record_prompt_cost
from oval.models import generate_texts, render_chat_with_assistant_prefix
from oval.search_common import (
    adjust_candidate_score,
    append_segment,
    normalize_mpc_segment,
    token_count,
)

PairwiseJudge = Any


@record_prompt_cost
def generate_ftrl_mpc(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    judge: PairwiseJudge,
    device: torch.device,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    feature_matrix = getattr(judge, "feature_matrix", None)
    weight = getattr(judge, "weight", None)
    if not callable(feature_matrix) or weight is None:
        raise RuntimeError(
            "ftrl_mpc requires a candidate-hidden judge with direct feature access."
        )
    search_cfg = cfg["search"]
    steps = int(search_cfg["mpc_steps"])
    candidates_per_step = int(search_cfg["mpc_candidates_per_step"])
    segment_max = int(search_cfg["mpc_segment_max_tokens"])
    min_segment = int(search_cfg["mpc_segment_min_tokens"])
    final_max = int(cfg["generation"]["max_new_tokens"])
    lookahead_per_action = max(0, int(search_cfg.get("ftrl_lookahead_per_action", 2)))
    lookahead_max = int(search_cfg.get("ftrl_lookahead_max_tokens", segment_max))
    target_top_k = max(1, int(search_cfg.get("ftrl_target_top_k", 1)))
    margin = float(search_cfg.get("ftrl_margin", 0.03))
    max_pairs = int(search_cfg.get("ftrl_max_pairs", 128))
    alpha = float(search_cfg.get("ftrl_alpha", 0.1))
    beta = float(search_cfg.get("ftrl_beta", 1.0))
    l1 = float(search_cfg.get("ftrl_l1", 0.0))
    l2 = float(search_cfg.get("ftrl_l2", 0.5))
    max_delta_norm = float(search_cfg.get("ftrl_max_delta_norm", 1.0))
    target_score_blend = float(search_cfg.get("ftrl_target_score_blend", 0.0))
    target_normalization = str(search_cfg.get("ftrl_target_normalization", "none"))
    confidence_gate_z = float(search_cfg.get("ftrl_confidence_gate_z", 0.0))
    if target_normalization not in {"none", "zscore"}:
        raise ValueError(
            "search.ftrl_target_normalization must be one of: none, zscore"
        )
    w0 = np.asarray(weight, dtype=np.float32)
    grad_sum = np.zeros_like(w0, dtype=np.float32)
    grad_sq_sum = np.zeros_like(w0, dtype=np.float32)
    delta = np.zeros_like(w0, dtype=np.float32)
    prefix = ""
    selected_raw_tokens = 0
    selected_stop = False
    boundary_fallback_count = 0
    total_pair_count = 0
    update_count = 0
    lookahead_record_count = 0
    selected_scores: list[float] = []
    selected_target_scores: list[float] = []
    step_diagnostics: list[dict[str, Any]] = []
    gate_pass_count = 0
    frozen_selection_count = 0
    adapted_selection_count = 0
    for iteration in range(steps):
        prefix_tokens = token_count(tokenizer, prefix)
        remaining = final_max - prefix_tokens
        if remaining <= 0:
            break
        current_max = min(segment_max, remaining)
        prompt_text = render_chat_with_assistant_prefix(
            tokenizer, "", prompt["prompt"], prefix
        )
        with generation_stage(model, "ftrl_mpc.candidates", iteration=iteration):
            outputs = generate_texts(
                cfg,
                tokenizer,
                model,
                device,
                [prompt_text],
                candidates_per_step,
                max_new_tokens=current_max,
            )
        candidates: list[dict[str, Any]] = []
        for idx, item in enumerate(outputs):
            segment, fallback = normalize_mpc_segment(
                item["text"],
                tokenizer,
                min_segment,
                allow_short=item["stop_reason"] == "stop_token",
            )
            if not segment:
                continue
            candidates.append(
                {
                    "idx": idx,
                    "segment": segment,
                    "raw": item,
                    "boundary_fallback": fallback,
                    "iteration": iteration,
                    "candidate_index": idx,
                }
            )
        if not candidates:
            break
        action_segments = [row["segment"] for row in candidates]
        action_features = feature_matrix(
            prompt["prompt"], prefix, action_segments
        ).astype(np.float32)
        lookahead_records = generate_ftrl_lookahead_records(
            cfg,
            tokenizer,
            model,
            device,
            prompt["prompt"],
            prefix,
            candidates,
            lookahead_per_action=lookahead_per_action,
            lookahead_max_tokens=lookahead_max,
            final_max_tokens=final_max,
        )
        lookahead_record_count += len(lookahead_records)
        raw_target_values = ftrl_backed_up_targets(
            feature_matrix,
            prompt["prompt"],
            prefix,
            len(candidates),
            lookahead_records,
            w0,
            target_top_k=target_top_k,
        )
        target_info = prepare_ftrl_targets(raw_target_values, target_normalization)
        target_values = target_info["values"]
        gate_passed = (
            confidence_gate_z <= 0.0 or float(target_info["gap_z"]) >= confidence_gate_z
        )
        if gate_passed:
            gate_pass_count += 1
        grad, pair_count = ftrl_pairwise_gradient(
            action_features, target_values, w0 + delta, margin, max_pairs
        )
        total_pair_count += pair_count
        update_allowed = ftrl_update_allowed(alpha, gate_passed, pair_count)
        if update_allowed:
            grad_sum += grad
            grad_sq_sum += grad * grad
            delta = ftrl_proximal_delta(
                grad_sum,
                grad_sq_sum,
                alpha=alpha,
                beta=beta,
                l1=l1,
                l2=l2,
                max_delta_norm=max_delta_norm,
            )
            update_count += 1
        frozen_scores = action_features @ w0
        adapted_scores = action_features @ (w0 + delta)
        ranking_scores = (
            adapted_scores if gate_passed and alpha > 0.0 else frozen_scores
        )
        if target_score_blend != 0.0 and gate_passed and (alpha > 0.0):
            blend_values = np.asarray(
                [
                    0.0 if value == float("-inf") else float(value)
                    for value in target_values
                ],
                dtype=np.float32,
            )
            ranking_scores = ranking_scores + target_score_blend * blend_values
        frozen_adjusted = adjusted_candidate_scores(
            cfg, tokenizer, candidates, frozen_scores
        )
        adapted_adjusted = adjusted_candidate_scores(
            cfg, tokenizer, candidates, adapted_scores
        )
        adjusted_scores = [
            adjust_candidate_score(
                cfg,
                tokenizer,
                {"text": row["segment"], "raw": row["raw"]},
                float(score),
            )
            for row, score in zip(candidates, ranking_scores, strict=True)
        ]
        selected_pos = int(
            max(range(len(adjusted_scores)), key=lambda idx: adjusted_scores[idx])
        )
        frozen_pos = int(
            max(range(len(frozen_adjusted)), key=lambda idx: frozen_adjusted[idx])
        )
        adapted_pos = int(
            max(range(len(adapted_adjusted)), key=lambda idx: adapted_adjusted[idx])
        )
        frozen_selection_count += int(selected_pos == frozen_pos)
        adapted_selection_count += int(selected_pos == adapted_pos)
        selected = candidates[selected_pos]
        prefix = append_segment(prefix, selected["segment"])
        selected_raw_tokens += int(selected["raw"]["raw_token_count"])
        selected_scores.append(float(adjusted_scores[selected_pos]))
        selected_target_scores.append(float(raw_target_values[selected_pos]))
        step_diagnostics.append(
            {
                "iteration": iteration,
                "target_normalization": target_normalization,
                "target_mean": target_info["mean"],
                "target_std": target_info["std"],
                "target_gap": target_info["gap"],
                "target_gap_z": target_info["gap_z"],
                "confidence_gate_z": confidence_gate_z,
                "gate_passed": gate_passed,
                "pair_count": pair_count,
                "updated": update_allowed,
                "delta_norm": float(np.linalg.norm(delta)),
                "frozen_candidate_position": frozen_pos,
                "adapted_candidate_position": adapted_pos,
                "selected_candidate_position": selected_pos,
                "frozen_candidate_index": int(candidates[frozen_pos]["idx"]),
                "adapted_candidate_index": int(candidates[adapted_pos]["idx"]),
                "selected_candidate_index": int(selected["idx"]),
            }
        )
        boundary_fallback_count += 1 if selected["boundary_fallback"] else 0
        if selected["raw"]["stop_reason"] == "stop_token":
            selected_stop = True
            break
    final_tokens = token_count(tokenizer, prefix)
    hit_max = final_tokens >= final_max
    stop_reason = (
        "stop_token" if selected_stop else "max_new_tokens" if hit_max else "max_steps"
    )
    finite_target_scores = [
        score for score in selected_target_scores if np.isfinite(float(score))
    ]
    return {
        "generation_id": f"{prompt['prompt_id']}::ftrl_mpc",
        "prompt_id": prompt["prompt_id"],
        "source_index": prompt.get("source_index"),
        "prompt": prompt["prompt"],
        "system_prompt": "",
        "method": "ftrl_mpc",
        "candidate_idx": None,
        "response": prefix.strip(),
        "response_token_count": final_tokens,
        "raw_generated_token_count": selected_raw_tokens,
        "hit_max_new_tokens": hit_max,
        "sentence_trimmed": False,
        "stop_reason": stop_reason,
        "mpc_strategy": "ftrl_segments",
        "selection_failures": 0,
        "boundary_fallback_count": boundary_fallback_count,
        "ftrl_update_count": update_count,
        "ftrl_pair_count": total_pair_count,
        "ftrl_lookahead_count": lookahead_record_count,
        "ftrl_delta_norm": float(np.linalg.norm(delta)),
        "ftrl_target_normalization": target_normalization,
        "ftrl_confidence_gate_z": confidence_gate_z,
        "ftrl_gate_pass_count": gate_pass_count,
        "ftrl_frozen_selection_count": frozen_selection_count,
        "ftrl_adapted_selection_count": adapted_selection_count,
        "ftrl_steps": step_diagnostics,
        "ftrl_selected_score_mean": float(np.mean(selected_scores))
        if selected_scores
        else None,
        "ftrl_selected_target_mean": float(np.mean(finite_target_scores))
        if finite_target_scores
        else None,
    }


def adjusted_candidate_scores(
    cfg: dict[str, Any],
    tokenizer: Any,
    candidates: list[dict[str, Any]],
    scores: np.ndarray,
) -> list[float]:
    return [
        adjust_candidate_score(
            cfg, tokenizer, {"text": row["segment"], "raw": row["raw"]}, float(score)
        )
        for row, score in zip(candidates, scores, strict=True)
    ]


def prepare_ftrl_targets(targets: list[float], normalization: str) -> dict[str, Any]:
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
    gap_z = gap / max(std, 1e-08)
    if normalization == "zscore":
        values = [
            (float(value) - mean) / max(std, 1e-08)
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
    return bool(float(alpha) > 0.0 and gate_passed and (int(pair_count) > 0))


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
        if (
            lookahead_per_action <= 0
            or remaining <= 0
            or candidate["raw"]["stop_reason"] == "stop_token"
        ):
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
        prompt_text = render_chat_with_assistant_prefix(
            tokenizer, "", prompt, prefix_after
        )
        for rollout_index in range(lookahead_per_action):
            jobs.append(
                (
                    rollout_tokens,
                    prompt_text,
                    (action_index, rollout_index, segment, prefix_after),
                )
            )
    for rollout_tokens in sorted({tokens for tokens, _, _ in jobs}, reverse=True):
        group = [
            (prompt_text, metadata)
            for tokens, prompt_text, metadata in jobs
            if tokens == rollout_tokens
        ]
        with generation_stage(model, "ftrl_mpc.lookahead", horizon=rollout_tokens):
            outputs = generate_texts(
                cfg,
                tokenizer,
                model,
                device,
                [prompt_text for prompt_text, _ in group],
                1,
                max_new_tokens=rollout_tokens,
            )
        for (_, (action_index, rollout_index, segment, prefix_after)), item in zip(
            group, outputs, strict=True
        ):
            continuation = item["text"].strip()
            target_candidate = (
                append_segment(segment, continuation) if continuation else segment
            )
            records.append(
                {
                    "action_index": action_index,
                    "rollout_index": rollout_index,
                    "target_candidate": target_candidate,
                    "text": append_segment(prefix_after, continuation)
                    if continuation
                    else prefix_after,
                    "raw": item,
                    "lookahead_continuation": continuation,
                }
            )
    return records


def ftrl_backed_up_targets(
    feature_matrix: Any,
    prompt: str,
    prefix: str,
    action_count: int,
    lookahead_records: list[dict[str, Any]],
    target_weight: np.ndarray,
    target_top_k: int,
) -> list[float]:
    grouped: list[list[float]] = [[] for _ in range(action_count)]
    if not lookahead_records:
        return [float("-inf") for _ in range(action_count)]
    features = feature_matrix(
        prompt, prefix, [record["target_candidate"] for record in lookahead_records]
    ).astype(np.float32)
    scores = features @ target_weight
    for record, score in zip(lookahead_records, scores, strict=True):
        action_index = int(record["action_index"])
        if 0 <= action_index < len(grouped):
            grouped[action_index].append(float(score))
    return [top_mean(values, target_top_k) for values in grouped]


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
        return (np.zeros(features.shape[1], dtype=np.float32), 0)
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
    return (grad.astype(np.float32), len(pairs))


def ftrl_proximal_delta(
    grad_sum: np.ndarray,
    grad_sq_sum: np.ndarray,
    alpha: float,
    beta: float,
    l1: float,
    l2: float,
    max_delta_norm: float,
) -> np.ndarray:
    denom = (float(beta) + np.sqrt(np.maximum(grad_sq_sum, 0.0))) / max(
        float(alpha), 1e-08
    ) + float(l2)
    if l1 > 0.0:
        adjusted = np.where(
            np.abs(grad_sum) > l1, grad_sum - np.sign(grad_sum) * l1, 0.0
        )
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
    return (vector * (float(max_norm) / max(norm, 1e-08))).astype(np.float32)

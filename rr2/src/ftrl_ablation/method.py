from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.ftrl_mpc.method import (
    adjusted_candidate_scores,
    ftrl_pairwise_gradient,
    ftrl_proximal_delta,
    ftrl_update_allowed,
    generate_ftrl_lookahead_records,
    prepare_ftrl_targets,
    top_mean,
)
from src.models import generate_texts, render_chat_with_assistant_prefix
from src.search_common import adjust_candidate_score, append_segment, normalize_mpc_segment, token_count


PairwiseJudge = Any
ABLATION_MODES = ("fixed", "direct_backup", "step_local_reset", "accumulated_ftrl")


@dataclass
class FTRLState:
    grad_sum: np.ndarray
    grad_sq_sum: np.ndarray
    delta: np.ndarray

    @classmethod
    def zeros_like(cls, weight: np.ndarray) -> "FTRLState":
        zeros = np.zeros_like(weight, dtype=np.float32)
        return cls(zeros.copy(), zeros.copy(), zeros.copy())


def update_ftrl_state(
    mode: str,
    state: FTRLState,
    features: np.ndarray,
    targets: list[float],
    base_weight: np.ndarray,
    *,
    margin: float,
    max_pairs: int,
    alpha: float,
    beta: float,
    l1: float,
    l2: float,
    max_delta_norm: float,
    gate_passed: bool,
) -> tuple[FTRLState, np.ndarray, int, bool]:
    """Apply the controlled local or cumulative update and return its gradient."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown FTRL ablation mode: {mode!r}")
    if mode in {"fixed", "direct_backup"}:
        gradient, pair_count = ftrl_pairwise_gradient(
            features,
            targets,
            base_weight,
            margin,
            max_pairs,
        )
        return state, gradient, pair_count, False

    working = FTRLState.zeros_like(base_weight) if mode == "step_local_reset" else state
    gradient, pair_count = ftrl_pairwise_gradient(
        features,
        targets,
        base_weight + working.delta,
        margin,
        max_pairs,
    )
    allowed = ftrl_update_allowed(alpha, gate_passed, pair_count)
    if not allowed:
        return working, gradient, pair_count, False
    grad_sum = working.grad_sum + gradient
    grad_sq_sum = working.grad_sq_sum + gradient * gradient
    delta = ftrl_proximal_delta(
        grad_sum,
        grad_sq_sum,
        alpha=alpha,
        beta=beta,
        l1=l1,
        l2=l2,
        max_delta_norm=max_delta_norm,
    )
    return FTRLState(grad_sum, grad_sq_sum, delta), gradient, pair_count, True


def ranking_scores_for_mode(
    mode: str,
    frozen_scores: np.ndarray,
    adapted_scores: np.ndarray,
    raw_targets: list[float],
    normalized_targets: list[float],
    *,
    target_score_blend: float,
    gate_passed: bool,
    alpha: float,
) -> np.ndarray:
    if mode == "fixed":
        return frozen_scores.copy()
    if mode == "direct_backup":
        return finite_score_array(raw_targets)
    if mode not in {"step_local_reset", "accumulated_ftrl"}:
        raise ValueError(f"Unknown FTRL ablation mode: {mode!r}")
    scores = adapted_scores.copy() if gate_passed and alpha > 0.0 else frozen_scores.copy()
    if target_score_blend != 0.0 and gate_passed and alpha > 0.0:
        scores = scores + float(target_score_blend) * finite_score_array(normalized_targets)
    return scores


def generate_ftrl_ablation(
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
        raise RuntimeError("FTRL ablations require a candidate-hidden judge with direct feature access.")

    search_cfg = cfg["search"]
    mode = str(search_cfg["ftrl_ablation_mode"])
    if mode not in ABLATION_MODES:
        raise ValueError(f"search.ftrl_ablation_mode must be one of {ABLATION_MODES}, got {mode!r}")
    method_name = str(cfg["method"]["name"])
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
        raise ValueError("search.ftrl_target_normalization must be one of: none, zscore")

    base_weight = np.asarray(weight, dtype=np.float32)
    state = FTRLState.zeros_like(base_weight)
    diagnostic_gradient_sum = np.zeros_like(base_weight, dtype=np.float32)
    previous_gradient: np.ndarray | None = None
    prefix = ""
    selected_raw_tokens = 0
    total_generated_tokens = 0
    selected_stop = False
    boundary_fallback_count = 0
    total_pair_count = 0
    update_count = 0
    lookahead_record_count = 0
    selected_scores: list[float] = []
    selected_target_scores: list[float] = []
    step_diagnostics: list[dict[str, Any]] = []

    for iteration in range(steps):
        prefix_tokens = token_count(tokenizer, prefix)
        remaining = final_max - prefix_tokens
        if remaining <= 0:
            break
        current_max = min(segment_max, remaining)
        prompt_text = render_chat_with_assistant_prefix(tokenizer, "", prompt["prompt"], prefix)
        outputs = generate_texts(
            cfg,
            tokenizer,
            model,
            device,
            [prompt_text],
            candidates_per_step,
            max_new_tokens=current_max,
        )
        total_generated_tokens += sum(int(item.get("raw_token_count", 0)) for item in outputs)
        candidates: list[dict[str, Any]] = []
        for idx, item in enumerate(outputs):
            segment, fallback = normalize_mpc_segment(
                item["text"],
                tokenizer,
                min_segment,
                allow_short=item["stop_reason"] == "stop_token",
            )
            if segment:
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
        action_features = feature_matrix(prompt["prompt"], prefix, action_segments).astype(np.float32)
        frozen_scores = action_features @ base_weight
        lookahead_records: list[dict[str, Any]] = []
        raw_targets = [float("-inf") for _ in candidates]
        rollout_scores: list[list[float]] = [[] for _ in candidates]
        if mode != "fixed":
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
            total_generated_tokens += sum(
                int((record.get("raw") or {}).get("raw_token_count", 0))
                for record in lookahead_records
            )
            raw_targets, rollout_scores = backed_up_target_details(
                feature_matrix,
                prompt["prompt"],
                prefix,
                len(candidates),
                lookahead_records,
                base_weight,
                target_top_k,
            )

        target_info = prepare_ftrl_targets(raw_targets, target_normalization)
        normalized_targets = target_info["values"]
        gate_passed = (
            mode == "fixed"
            or confidence_gate_z <= 0.0
            or float(target_info["gap_z"]) >= confidence_gate_z
        )
        state_before = FTRLState(
            state.grad_sum.copy(),
            state.grad_sq_sum.copy(),
            state.delta.copy(),
        )
        pre_update_weight = base_weight + (
            np.zeros_like(base_weight) if mode == "step_local_reset" else state_before.delta
        )
        state, gradient, pair_count, updated = update_ftrl_state(
            mode,
            state,
            action_features,
            normalized_targets,
            base_weight,
            margin=margin,
            max_pairs=max_pairs,
            alpha=alpha,
            beta=beta,
            l1=l1,
            l2=l2,
            max_delta_norm=max_delta_norm,
            gate_passed=gate_passed,
        )
        total_pair_count += pair_count
        update_count += int(updated)
        adapted_scores = action_features @ (base_weight + state.delta)
        ranking_scores = ranking_scores_for_mode(
            mode,
            frozen_scores,
            adapted_scores,
            raw_targets,
            normalized_targets,
            target_score_blend=target_score_blend,
            gate_passed=gate_passed,
            alpha=alpha,
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
        frozen_adjusted = adjusted_candidate_scores(cfg, tokenizer, candidates, frozen_scores)
        adapted_adjusted = adjusted_candidate_scores(cfg, tokenizer, candidates, adapted_scores)
        direct_adjusted = adjusted_candidate_scores(
            cfg,
            tokenizer,
            candidates,
            finite_score_array(raw_targets),
        ) if mode != "fixed" else [float("-inf") for _ in candidates]
        selected_pos = argmax_finite(adjusted_scores)
        frozen_pos = argmax_finite(frozen_adjusted)
        adapted_pos = argmax_finite(adapted_adjusted)
        direct_pos = argmax_finite(direct_adjusted) if mode != "fixed" else None

        pre_metrics = pairwise_ranking_metrics(
            action_features,
            normalized_targets,
            pre_update_weight,
            margin,
            max_pairs,
        )
        post_metrics = pairwise_ranking_metrics(
            action_features,
            normalized_targets,
            base_weight + state.delta,
            margin,
            max_pairs,
        )
        previous_cosine = vector_cosine(previous_gradient, gradient)
        history_cosine = vector_cosine(diagnostic_gradient_sum, gradient)
        if pair_count > 0:
            diagnostic_gradient_sum += gradient
            previous_gradient = gradient.copy()

        selected = candidates[selected_pos]
        prefix = append_segment(prefix, selected["segment"])
        selected_raw_tokens += int(selected["raw"]["raw_token_count"])
        selected_scores.append(float(adjusted_scores[selected_pos]))
        selected_target_scores.append(float(raw_targets[selected_pos]))
        step_diagnostics.append(
            {
                "iteration": iteration,
                "mode": mode,
                "candidate_count": len(candidates),
                "target_normalization": target_normalization,
                "target_mean": target_info["mean"],
                "target_std": target_info["std"],
                "target_gap": target_info["gap"],
                "target_gap_z": target_info["gap_z"],
                "raw_targets": json_floats(raw_targets),
                "rollout_scores_by_action": [json_floats(values) for values in rollout_scores],
                "rollout_std_mean": finite_mean(
                    [float(np.std(values)) for values in rollout_scores if len(values) > 1]
                ),
                "confidence_gate_z": confidence_gate_z,
                "gate_passed": gate_passed,
                "pair_count": pair_count,
                "updated": updated,
                "delta_norm_before": float(np.linalg.norm(state_before.delta)),
                "delta_norm": float(np.linalg.norm(state.delta)),
                "gradient_norm": float(np.linalg.norm(gradient)),
                "gradient_cosine_previous": previous_cosine,
                "gradient_cosine_history": history_cosine,
                "pre_update_pair_loss": pre_metrics["loss"],
                "pre_update_pair_accuracy": pre_metrics["accuracy"],
                "post_update_pair_loss": post_metrics["loss"],
                "post_update_pair_accuracy": post_metrics["accuracy"],
                "frozen_candidate_position": frozen_pos,
                "direct_candidate_position": direct_pos,
                "adapted_candidate_position": adapted_pos,
                "selected_candidate_position": selected_pos,
                "frozen_candidate_index": int(candidates[frozen_pos]["idx"]),
                "direct_candidate_index": (
                    int(candidates[direct_pos]["idx"]) if direct_pos is not None else None
                ),
                "adapted_candidate_index": int(candidates[adapted_pos]["idx"]),
                "selected_candidate_index": int(selected["idx"]),
            }
        )
        boundary_fallback_count += int(bool(selected["boundary_fallback"]))
        if selected["raw"]["stop_reason"] == "stop_token":
            selected_stop = True
            break

    final_tokens = token_count(tokenizer, prefix)
    hit_max = final_tokens >= final_max
    stop_reason = "stop_token" if selected_stop else ("max_new_tokens" if hit_max else "max_steps")
    finite_selected_targets = [value for value in selected_target_scores if np.isfinite(value)]
    return {
        "generation_id": f"{prompt['prompt_id']}::{method_name}",
        "prompt_id": prompt["prompt_id"],
        "source_index": prompt.get("source_index"),
        "prompt": prompt["prompt"],
        "system_prompt": "",
        "method": method_name,
        "ablation_mode": mode,
        "candidate_idx": None,
        "response": prefix.strip(),
        "response_token_count": final_tokens,
        "raw_generated_token_count": selected_raw_tokens,
        "tta_generated_tokens": total_generated_tokens,
        "hit_max_new_tokens": hit_max,
        "sentence_trimmed": False,
        "stop_reason": stop_reason,
        "mpc_strategy": "ftrl_ablation_segments",
        "selection_failures": 0,
        "boundary_fallback_count": boundary_fallback_count,
        "ftrl_update_count": update_count,
        "ftrl_pair_count": total_pair_count,
        "ftrl_lookahead_count": lookahead_record_count,
        "ftrl_delta_norm": float(np.linalg.norm(state.delta)),
        "ftrl_target_normalization": target_normalization,
        "ftrl_confidence_gate_z": confidence_gate_z,
        "ftrl_steps": step_diagnostics,
        "ftrl_selected_score_mean": finite_mean(selected_scores),
        "ftrl_selected_target_mean": finite_mean(finite_selected_targets),
    }


def backed_up_target_details(
    feature_matrix: Any,
    prompt: str,
    prefix: str,
    action_count: int,
    lookahead_records: list[dict[str, Any]],
    target_weight: np.ndarray,
    target_top_k: int,
) -> tuple[list[float], list[list[float]]]:
    grouped: list[list[float]] = [[] for _ in range(action_count)]
    if not lookahead_records:
        return [float("-inf") for _ in range(action_count)], grouped
    features = feature_matrix(
        prompt,
        prefix,
        [record["target_candidate"] for record in lookahead_records],
    ).astype(np.float32)
    scores = features @ target_weight
    for record, score in zip(lookahead_records, scores, strict=True):
        action_index = int(record["action_index"])
        if 0 <= action_index < action_count:
            grouped[action_index].append(float(score))
    return [top_mean(values, target_top_k) for values in grouped], grouped


def pairwise_ranking_metrics(
    features: np.ndarray,
    targets: list[float],
    weight: np.ndarray,
    margin: float,
    max_pairs: int,
) -> dict[str, float | int | None]:
    pairs: list[tuple[float, int, int]] = []
    for winner, winner_target in enumerate(targets):
        if not np.isfinite(float(winner_target)):
            continue
        for loser, loser_target in enumerate(targets):
            if winner == loser or not np.isfinite(float(loser_target)):
                continue
            gap = float(winner_target) - float(loser_target)
            if gap > margin:
                pairs.append((gap, winner, loser))
    pairs.sort(reverse=True)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    if not pairs:
        return {"pairs": 0, "loss": None, "accuracy": None}
    logits = np.asarray(
        [float(weight @ (features[winner] - features[loser])) for _, winner, loser in pairs],
        dtype=np.float64,
    )
    return {
        "pairs": len(pairs),
        "loss": float(np.mean(np.logaddexp(0.0, -logits))),
        "accuracy": float(np.mean(logits > 0.0)),
    }


def finite_score_array(values: list[float]) -> np.ndarray:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    floor = min(finite) - max(1.0, abs(min(finite))) if finite else -1e9
    return np.asarray(
        [float(value) if np.isfinite(float(value)) else floor for value in values],
        dtype=np.float32,
    )


def argmax_finite(values: list[float] | np.ndarray) -> int:
    if len(values) == 0:
        raise ValueError("Cannot select from an empty score vector.")
    return int(max(range(len(values)), key=lambda index: float(values[index])))


def vector_cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def finite_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def json_floats(values: list[float]) -> list[float | None]:
    return [float(value) if np.isfinite(float(value)) else None for value in values]

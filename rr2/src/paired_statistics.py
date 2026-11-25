from collections import defaultdict

import numpy as np


def paired_interval(values, texts, draws=20000):
    grouped = defaultdict(list)
    for value, text in zip(values, texts, strict=True):
        grouped[text.strip()].append(float(value))
    if not grouped:
        raise ValueError("No paired observations")
    sums = np.array([sum(group) for group in grouped.values()])
    counts = np.array([len(group) for group in grouped.values()])
    rng = np.random.default_rng()
    means = []
    for start in range(0, draws, 500):
        indices = rng.integers(0, len(grouped), (min(500, draws - start), len(grouped)))
        means.extend(sums[indices].sum(axis=1) / counts[indices].sum(axis=1))
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": len(values),
        "delta": float(np.mean(values)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def label_agreement(trajectories, scores, draws=10000):
    blocks = []
    gaps = []
    for row in trajectories:
        prompt_id = row["prompt_id"]
        for state in row["trace"]:
            iteration = state["iteration"]
            targets = np.array(state["diagnostics"]["raw_targets"], dtype=float)
            rewards = [[] for _ in targets]
            for index, rollout in enumerate(state["rollouts"]):
                key = f"{prompt_id}::{iteration}::{index}"
                rewards[rollout["action_index"]].append(scores[key])
            independent = np.array([max(group) if group else np.nan for group in rewards])
            pairs = [
                (left, right)
                for left in range(len(targets))
                for right in range(left + 1, len(targets))
                if np.isfinite(targets[left])
                and np.isfinite(targets[right])
                and abs(targets[left] - targets[right]) > 0.03
            ]
            gaps.extend(abs(targets[left] - targets[right]) for left, right in pairs)
            blocks.append((prompt_id, iteration + 1, targets, independent, pairs))
    if not gaps:
        raise ValueError("No valid lookahead comparisons")
    rows = []
    for quantile in (0, 0.75):
        threshold = float(np.quantile(gaps, quantile)) if quantile else 0.03
        for scope in ("all_steps", "later_steps_2plus"):
            per_prompt = defaultdict(list)
            pair_count = 0
            for prompt_id, step, targets, independent, pairs in blocks:
                if scope == "later_steps_2plus" and step < 2:
                    continue
                selected = [
                    (left, right)
                    for left, right in pairs
                    if abs(targets[left] - targets[right]) >= threshold
                    and np.isfinite(independent[left])
                    and np.isfinite(independent[right])
                    and independent[left] != independent[right]
                ]
                if not selected:
                    continue
                agreement = np.mean([
                    (targets[left] - targets[right])
                    * (independent[left] - independent[right]) > 0
                    for left, right in selected
                ])
                per_prompt[prompt_id].append(float(agreement))
                pair_count += len(selected)
            if not per_prompt:
                continue
            values = [np.mean(group) for group in per_prompt.values()]
            interval = paired_interval(values, list(per_prompt), draws)
            rows.append({
                "scope": scope,
                "gap_quantile": quantile,
                "prompts": len(per_prompt),
                "pairs": pair_count,
                "accuracy": interval["delta"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
            })
    return rows

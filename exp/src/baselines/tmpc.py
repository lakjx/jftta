from __future__ import annotations
from typing import Any
import torch
from src.common import repeat_4gram_rate
from src.models import generate_texts, render_chat
from src.search_common import (
    boundary_cuts,
    contains_low_value_phrase,
    normalize_for_dedup,
    normalize_mpc_segment,
    score_with_external_signal,
    token_count,
)


def generate_tmpc_subgoal_regeneration(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    judge: Any,
    device: torch.device,
    prompt: dict[str, Any],
) -> dict[str, Any]:
    search_cfg = cfg["search"]
    iterations = int(search_cfg.get("mpc_steps", 3))
    candidates_per_step = int(search_cfg.get("mpc_candidates_per_step", 3))
    final_max = int(cfg["generation"]["max_new_tokens"])
    segment_sentences = int(
        search_cfg.get(
            "tmpc_segment_sentences", search_cfg.get("mpc_subgoal_segment_sentences", 3)
        )
    )
    buffer_size = int(
        search_cfg.get(
            "tmpc_buffer_size", search_cfg.get("mpc_subgoal_max_segments", 3)
        )
    )
    min_subgoal_tokens = int(
        search_cfg.get(
            "tmpc_min_subgoal_tokens", search_cfg.get("mpc_subgoal_min_tokens", 24)
        )
    )
    prompt_char_budget = int(
        search_cfg.get(
            "tmpc_prompt_char_budget",
            search_cfg.get("mpc_subgoal_prompt_max_chars", 2400),
        )
    )
    candidate_records: list[dict[str, Any]] = []
    candidate_scores: list[float] = []
    subgoal_buffer: list[str] = []
    buffer_updates: list[dict[str, Any]] = []
    selection_failures = 0
    rejected_subgoal_count = 0
    total_extracted_subgoals = 0
    initial_prompt = render_chat(
        tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True
    )
    initial_output = generate_texts(
        cfg, tokenizer, model, device, [initial_prompt], 1, max_new_tokens=final_max
    )[0]
    initial_record = tmpc_candidate_record(
        initial_output, iteration=0, rollout_index=0, system_prompt="", initial=True
    )
    if initial_record["text"]:
        candidate_records.append(initial_record)
        scores, failures = score_with_external_signal(
            cfg, judge, prompt["prompt"], "", [initial_record["text"]]
        )
        selection_failures += failures
        candidate_scores.extend(scores)
        subgoals, rejected = select_tmpc_subgoals(
            cfg,
            tokenizer,
            [initial_record],
            scores,
            segment_sentences=segment_sentences,
            min_tokens=min_subgoal_tokens,
            max_segments=buffer_size,
        )
        rejected_subgoal_count += rejected
        total_extracted_subgoals += len(subgoals)
        before = len(subgoal_buffer)
        subgoal_buffer = update_subgoal_buffer(subgoal_buffer, subgoals, buffer_size)
        buffer_updates.append(
            {
                "iteration": 0,
                "rollout_count": 1,
                "selected_subgoals": len(subgoals),
                "buffer_size": len(subgoal_buffer),
                "new_buffer_items": max(0, len(subgoal_buffer) - before),
            }
        )
    for iteration in range(1, iterations + 1):
        rollout_records = generate_tmpc_conditioned_rollouts(
            cfg,
            tokenizer,
            model,
            device,
            prompt["prompt"],
            subgoal_buffer,
            iteration=iteration,
            candidates_per_step=candidates_per_step,
            max_new_tokens=final_max,
            prompt_char_budget=prompt_char_budget,
        )
        if not rollout_records:
            continue
        scores, failures = score_with_external_signal(
            cfg,
            judge,
            prompt["prompt"],
            "",
            [record["text"] for record in rollout_records],
        )
        selection_failures += failures
        candidate_records.extend(rollout_records)
        candidate_scores.extend(scores)
        subgoals, rejected = select_tmpc_subgoals(
            cfg,
            tokenizer,
            rollout_records,
            scores,
            segment_sentences=segment_sentences,
            min_tokens=min_subgoal_tokens,
            max_segments=buffer_size,
        )
        rejected_subgoal_count += rejected
        total_extracted_subgoals += len(subgoals)
        before_items = set((normalize_for_dedup(item) for item in subgoal_buffer))
        subgoal_buffer = update_subgoal_buffer(subgoal_buffer, subgoals, buffer_size)
        added = sum(
            (
                1
                for item in subgoal_buffer
                if normalize_for_dedup(item) not in before_items
            )
        )
        buffer_updates.append(
            {
                "iteration": iteration,
                "rollout_count": len(rollout_records),
                "selected_subgoals": len(subgoals),
                "buffer_size": len(subgoal_buffer),
                "new_buffer_items": added,
            }
        )
    if not candidate_records:
        fallback_prompt = render_chat(
            tokenizer, "", prompt["prompt"], response=None, add_generation_prompt=True
        )
        fallback = generate_texts(
            cfg,
            tokenizer,
            model,
            device,
            [fallback_prompt],
            1,
            max_new_tokens=final_max,
        )[0]
        fallback_record = tmpc_candidate_record(
            fallback, iteration=0, rollout_index=0, system_prompt="", initial=True
        )
        candidate_records.append(fallback_record)
        scores, failures = score_with_external_signal(
            cfg, judge, prompt["prompt"], "", [fallback_record["text"]]
        )
        candidate_scores.extend(scores)
        selection_failures += failures
    selected_idx = max(
        range(len(candidate_scores)), key=lambda idx: candidate_scores[idx]
    )
    selected = candidate_records[selected_idx]
    selected_text = selected["text"].strip()
    selected_raw = selected["raw"]
    return {
        "generation_id": f"{prompt['prompt_id']}::tmpc",
        "prompt_id": prompt["prompt_id"],
        "source_index": prompt.get("source_index"),
        "prompt": prompt["prompt"],
        "system_prompt": selected.get("system_prompt", ""),
        "method": "tmpc",
        "candidate_idx": selected_idx,
        "response": selected_text,
        "response_token_count": token_count(tokenizer, selected_text),
        "raw_generated_token_count": int(selected_raw["raw_token_count"]),
        "hit_max_new_tokens": bool(selected_raw["hit_max_new_tokens"]),
        "sentence_trimmed": bool(selected_raw["sentence_trimmed"]),
        "stop_reason": selected_raw["stop_reason"],
        "mpc_strategy": "tmpc_subgoal_regeneration",
        "selection_failures": selection_failures,
        "mpc_candidate_count": len(candidate_records),
        "mpc_final_candidate_count": len(candidate_records),
        "mpc_iteration_winner_count": sum(
            (1 for update in buffer_updates if int(update["selected_subgoals"]) > 0)
        ),
        "mpc_subgoal_count": len(subgoal_buffer),
        "mpc_accepted_subgoal_count": total_extracted_subgoals,
        "mpc_rejected_subgoal_count": rejected_subgoal_count,
        "tmpc_buffer_size": len(subgoal_buffer),
        "tmpc_buffer_update_count": sum(
            (1 for update in buffer_updates if int(update["new_buffer_items"]) > 0)
        ),
        "tmpc_buffer_updates": buffer_updates,
        "tmpc_selected_iteration": selected["iteration"],
        "tmpc_selected_rollout_index": selected["rollout_index"],
        "tmpc_selected_initial": bool(selected.get("initial", False)),
        "mpc_selected_external_score": float(candidate_scores[selected_idx]),
        "mpc_total_generated_token_count": sum(
            (int(record["raw"]["raw_token_count"]) for record in candidate_records)
        ),
    }


def tmpc_candidate_record(
    item: dict[str, Any],
    iteration: int,
    rollout_index: int,
    system_prompt: str,
    initial: bool,
) -> dict[str, Any]:
    return {
        "text": item["text"].strip(),
        "raw": item,
        "iteration": iteration,
        "rollout_index": rollout_index,
        "system_prompt": system_prompt,
        "initial": initial,
        "final_pass": True,
        "simulation_pass": False,
    }


def generate_tmpc_conditioned_rollouts(
    cfg: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: str,
    subgoals: list[str],
    iteration: int,
    candidates_per_step: int,
    max_new_tokens: int,
    prompt_char_budget: int,
) -> list[dict[str, Any]]:
    search_cfg = cfg["search"]
    prompt_texts: list[str] = []
    system_prompts: list[str] = []
    for rollout_index in range(candidates_per_step):
        system_prompt = tmpc_system_prompt(search_cfg, rollout_index)
        prompt_texts.append(
            render_chat(
                tokenizer,
                system_prompt,
                build_tmpc_user_prompt(prompt, subgoals, prompt_char_budget),
                response=None,
                add_generation_prompt=True,
            )
        )
        system_prompts.append(system_prompt)
    outputs = generate_texts(
        cfg, tokenizer, model, device, prompt_texts, 1, max_new_tokens=max_new_tokens
    )
    records: list[dict[str, Any]] = []
    for rollout_index, (system_prompt, item) in enumerate(
        zip(system_prompts, outputs, strict=True)
    ):
        if not item["text"].strip():
            continue
        records.append(
            tmpc_candidate_record(
                item,
                iteration=iteration,
                rollout_index=rollout_index,
                system_prompt=system_prompt,
                initial=False,
            )
        )
    return records


def tmpc_system_prompt(search_cfg: dict[str, Any], rollout_index: int) -> str:
    prompts = search_cfg.get("tmpc_system_prompts")
    if isinstance(prompts, list) and prompts:
        return str(prompts[rollout_index % len(prompts)])
    default_prompts = [
        "You are a concise and clear assistant. Prioritize direct usefulness, safety, and plain wording.",
        "You are a detailed and informative assistant. Prioritize completeness, accurate caveats, and concrete examples.",
        "You are a balanced and well-structured assistant. Prioritize coherence, relevance, and actionable organization.",
    ]
    return default_prompts[rollout_index % len(default_prompts)]


def build_tmpc_user_prompt(
    prompt: str, subgoals: list[str], prompt_char_budget: int
) -> str:
    parts = ["User request:", prompt.strip(), ""]
    if subgoals:
        parts.extend(
            [
                "High-scoring ideas and phrases discovered in previous attempts:",
                format_subgoals_for_prompt(subgoals, prompt_char_budget),
                "",
                "Write one complete assistant response that synthesizes the useful ideas above if they are relevant.",
                "Do not copy the snippets verbatim, do not stitch them together, and do not mention rewards, candidates, buffers, or previous attempts.",
            ]
        )
    else:
        parts.append("Write one complete assistant response.")
    parts.extend(
        [
            "Answer the user directly.",
            "Be helpful and harmless. Refuse or redirect unsafe requests when appropriate.",
        ]
    )
    return "\n".join(parts)


def select_tmpc_subgoals(
    cfg: dict[str, Any],
    tokenizer: Any,
    records: list[dict[str, Any]],
    scores: list[float],
    segment_sentences: int,
    min_tokens: int,
    max_segments: int,
) -> tuple[list[str], int]:
    if not records or max_segments <= 0:
        return ([], 0)
    search_cfg = cfg.get("search", {})
    margin = float(
        search_cfg.get(
            "tmpc_reward_margin", search_cfg.get("mpc_subgoal_score_margin", 0.0)
        )
    )
    scored_records = sorted(
        zip(records, scores, strict=True), key=lambda item: float(item[1]), reverse=True
    )
    best_score = float(scored_records[0][1])
    selected: list[str] = []
    seen: set[str] = set()
    rejected = 0
    for record, score in scored_records:
        if float(score) < best_score - margin:
            rejected += 1
            continue
        segments = extract_subgoal_segments(
            record["text"],
            tokenizer,
            segment_sentences=segment_sentences,
            min_tokens=min_tokens,
            max_segments=max_segments,
        )
        if not segments:
            rejected += 1
            continue
        for segment in segments:
            if not usable_subgoal_segment(cfg, tokenizer, segment):
                rejected += 1
                continue
            key = normalize_for_dedup(segment)
            if not key or key in seen:
                rejected += 1
                continue
            selected.append(segment)
            seen.add(key)
            if len(selected) >= max_segments:
                return (selected, rejected)
    return (selected, rejected)


def usable_subgoal_segment(cfg: dict[str, Any], tokenizer: Any, segment: str) -> bool:
    search_cfg = cfg.get("search", {})
    text = segment.strip()
    if not text:
        return False
    if token_count(tokenizer, text) < int(search_cfg.get("mpc_subgoal_min_tokens", 32)):
        return False
    if repeat_4gram_rate(text) > float(
        search_cfg.get("mpc_subgoal_repeat_threshold", 0.12)
    ):
        return False
    return not contains_low_value_phrase(text)


def format_subgoals_for_prompt(subgoals: list[str], char_budget: int) -> str:
    if char_budget <= 0:
        return ""
    selected: list[str] = []
    used = 0
    for idx, subgoal in enumerate(reversed(subgoals), start=1):
        text = subgoal.strip()
        if not text:
            continue
        line = f"{idx}. {text}"
        extra = len(line) + (1 if selected else 0)
        if selected and used + extra > char_budget:
            break
        if not selected and extra > char_budget:
            line = line[: max(0, char_budget - 3)].rstrip() + "..."
            extra = len(line)
        selected.append(line)
        used += extra
    return "\n".join(selected)


def extract_subgoal_segments(
    text: str,
    tokenizer: Any,
    segment_sentences: int,
    min_tokens: int,
    max_segments: int,
) -> list[str]:
    if max_segments <= 0:
        return []
    units = split_text_units(text)
    if not units:
        stripped = text.strip()
        return (
            [stripped]
            if stripped and token_count(tokenizer, stripped) >= min_tokens
            else []
        )
    group_size = max(1, segment_sentences)
    segments: list[str] = []
    for index in range(0, len(units), group_size):
        segment = " ".join(units[index : index + group_size]).strip()
        if token_count(tokenizer, segment) >= min_tokens:
            segments.append(segment)
        if len(segments) >= max_segments:
            break
    if not segments:
        normalized, _ = normalize_mpc_segment(
            text, tokenizer, min_tokens=min_tokens, allow_short=False
        )
        if normalized:
            segments.append(normalized)
    return segments[:max_segments]


def split_text_units(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    cuts = boundary_cuts(stripped)
    if not cuts:
        return [stripped]
    units: list[str] = []
    start = 0
    for cut in cuts:
        piece = stripped[start:cut].strip()
        if piece:
            units.append(piece)
        start = cut
    tail = stripped[start:].strip()
    if tail:
        units.append(tail)
    return units


def update_subgoal_buffer(
    existing: list[str], new_segments: list[str], max_segments: int
) -> list[str]:
    if max_segments <= 0:
        return []
    merged = list(existing)
    seen = {normalize_for_dedup(segment) for segment in merged}
    for segment in new_segments:
        key = normalize_for_dedup(segment)
        if not key or key in seen:
            continue
        merged.append(segment.strip())
        seen.add(key)
    return merged[-max_segments:]

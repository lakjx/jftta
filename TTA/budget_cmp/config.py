from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS = {
    "seed": 11112,
    "models": {
        "policy": "weights/Llama-3-8B-HelpSteer2-SFT",
        "armorm": "weights/ArmoRM-Llama3-8B-v0.1",
    },
    "experiment": {"reward_attribute": "helpsteer-correctness"},
    "runtime": {"generation_batch_size": 32, "record_generation_cost": False},
    "generation": {
        "max_new_tokens": 512,
        "max_input_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "trim_incomplete_on_max": True,
        "trim_min_token_fraction": 0.35,
    },
    "armorm": {"max_length": 4096, "pad_token_id": 128256},
    "judge": {
        "type": "candidate_hidden",
        "probe_path": "weights/critic/correctness.pt",
        "input_format": "state_action",
        "template": "state_action",
        "batch_size": 32,
    },
    "search": {
        "seed_namespace": "correctness_ftrl_v4",
        "mpc_steps": 4,
        "mpc_candidates_per_step": 8,
        "mpc_segment_max_tokens": 128,
        "mpc_segment_min_tokens": 32,
        "ftrl_lookahead_per_action": 4,
        "ftrl_lookahead_max_tokens": 128,
        "ftrl_target_top_k": 1,
        "ftrl_margin": 0.03,
        "ftrl_max_pairs": 64,
        "ftrl_alpha": 0.1,
        "ftrl_beta": 1.0,
        "ftrl_l1": 0.0,
        "ftrl_l2": 0.5,
        "ftrl_max_delta_norm": 1.0,
        "ftrl_target_score_blend": 0.5,
        "ftrl_target_normalization": "zscore",
        "ftrl_confidence_gate_z": 0.5,
        "mpc_quality_repeat_threshold": 0.12,
        "mpc_quality_repeat_penalty": 0.8,
        "mpc_quality_hit_max_penalty": 0.08,
        "mpc_quality_trimmed_penalty": 0.04,
        "mpc_quality_long_token_threshold": 512,
        "mpc_quality_long_token_penalty": 0.04,
        "mpc_quality_meta_penalty": 0.15,
    },
    "bon": {"n_values": [8, 16, 32], "selector": "armorm_score"},
    "methods": ["oval", "bon", "tmpc", "tpo"],
    "tmpc": {
        "judge": {
            "type": "candidate_hidden",
            "probe_path": "weights/critic/tmpc_correctness.pt",
            "input_format": "state_action",
            "template": "state_action",
            "layer": 20,
            "pooling": "mean_last_128",
            "batch_size": 32,
        },
        "search": {
            "mpc_steps": 4,
            "mpc_candidates_per_step": 8,
            "mpc_selection_batch_size": 4,
            "tmpc_buffer_size": 3,
            "tmpc_segment_sentences": 3,
            "tmpc_reward_margin": 0.04,
            "tmpc_min_subgoal_tokens": 24,
            "tmpc_prompt_char_budget": 2400,
            "mpc_subgoal_min_tokens": 24,
            "mpc_subgoal_repeat_threshold": 0.12,
            "mpc_quality_repeat_threshold": 0.12,
            "mpc_quality_repeat_penalty": 0.8,
            "mpc_quality_hit_max_penalty": 0.08,
            "mpc_quality_trimmed_penalty": 0.04,
            "mpc_quality_long_token_threshold": 512,
            "mpc_quality_long_token_penalty": 0.04,
            "mpc_quality_meta_penalty": 0.15,
            "tmpc_system_prompts": [
                "You are a precise assistant. Prioritize factual "
                "accuracy and clear, verifiable reasoning.",
                "You are a rigorous assistant. Make assumptions "
                "explicit, check logical steps, and avoid unsupported "
                "claims.",
                "You are a cautious assistant. Prioritize "
                "consistency, calibrated uncertainty, and correction "
                "of possible errors.",
            ],
        },
    },
    "decoding": {
        "tpo": {
            "iterations": 2,
            "samples": 3,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 0,
            "response_max_tokens": 512,
            "feedback_max_tokens": 512,
            "max_input_tokens": 7168,
        }
    },
}


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        path = Path(path).resolve()
        overrides = json.loads(path.read_text())

        def merge(target, values):
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = value

        for group, key in [
            ("models", "policy"),
            ("models", "armorm"),
            ("judge", "probe_path"),
        ]:
            if key in overrides.get(group, {}):
                overrides[group][key] = str(
                    (path.parent / overrides[group][key]).resolve()
                )
        if "probe_path" in overrides.get("tmpc", {}).get("judge", {}):
            overrides["tmpc"]["judge"]["probe_path"] = str(
                (path.parent / overrides["tmpc"]["judge"]["probe_path"]).resolve()
            )
        merge(cfg, overrides)
    return cfg

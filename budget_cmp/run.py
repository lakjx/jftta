from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

from config import load_config
from oval.io import read_jsonl, save_json, select_bon

HERE = Path(__file__).resolve().parent
METHODS = ("oval", "bon", "tmpc", "tpo")


def run(args, cfg, prompts):
    import torch
    from oval.common import set_seed, stable_prompt_seed
    from oval.models import load_lm, generate_texts, render_chat
    from oval.value_head.judge import make_judge
    from oval.ftrl_mpc.method import generate_ftrl_mpc
    from oval.tmpc.method import generate_tmpc_subgoal_regeneration
    from oval.tpo.method import generate_tpo
    from oval.generation_cost import GenerationCostRecorder
    from oval.evaluation.armorm import ArmoRMScorer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for evaluation.")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0")
    reward_device = torch.device(args.reward_device or "cuda:0")
    signature = hashlib.sha256(
        json.dumps({"config": cfg, "seed": args.seed}, sort_keys=True).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = cfg["methods"]

    def path_for(prompt):
        return args.output_dir / f"{prompt['prompt_id']}.json"

    def load_record(prompt):
        path = path_for(prompt)
        if path.exists():
            record = json.loads(path.read_text())
            if (
                record.get("run_signature") != signature
                or record.get("prompt") != prompt["prompt"]
            ):
                raise ValueError(f"Existing result belongs to a different run: {path}")
            return record
        return {
            "prompt_id": prompt["prompt_id"],
            "prompt": prompt["prompt"],
            "seed": args.seed,
            "run_signature": signature,
            "methods": methods,
            "primary_selector": "armorm_score",
        }

    pending = [p for p in prompts if not load_record(p).get("generation_complete")]
    save_json(
        args.output_dir / f"run-shard{args.shard}.json",
        {
            "config": cfg,
            "seed": args.seed,
            "signature": signature,
            "prompt_ids": [p["prompt_id"] for p in prompts],
            "shard": args.shard,
            "shards": args.shards,
        },
    )
    reward = None
    if pending:
        tokenizer, model = load_lm(cfg, device)
        judges = {}
        tmpc_cfg = {
            **cfg,
            "judge": cfg["tmpc"]["judge"],
            "search": cfg["tmpc"]["search"],
        }
        for prompt in pending:
            record = load_record(prompt)
            pid = prompt["prompt_id"]
            if "bon" in methods and "candidates" not in record:
                set_seed(stable_prompt_seed(args.seed, pid, "full_bon"))
                text = render_chat(
                    tokenizer,
                    "",
                    prompt["prompt"],
                    response=None,
                    add_generation_prompt=True,
                )
                with GenerationCostRecorder(model, device, "bon_armorm") as meter:
                    candidates = generate_texts(
                        cfg,
                        tokenizer,
                        model,
                        device,
                        [text],
                        max(cfg["bon"]["n_values"]),
                    )
                costs = meter.summary()
                del meter
                counts = [
                    count
                    for call in costs["calls"]
                    for count in call["generated_token_counts"]
                ]
                if len(candidates) != len(counts):
                    raise RuntimeError("Candidate and generated-token counts disagree.")
                for index, (candidate, count) in enumerate(
                    zip(candidates, counts, strict=True)
                ):
                    candidate.update(candidate_idx=index, total_generated_tokens=count)
                record.update(candidates=candidates, bon_generation_cost=costs)
                save_json(path_for(prompt), record)
            for method in ("oval", "tmpc", "tpo"):
                if method not in methods or method in record:
                    continue
                namespace = (
                    cfg["search"]["seed_namespace"] if method == "oval" else method
                )
                set_seed(stable_prompt_seed(args.seed, pid, namespace))
                if method in ("oval", "tmpc") and method not in judges:
                    judges[method] = make_judge(
                        cfg if method == "oval" else tmpc_cfg,
                        device,
                        tokenizer=tokenizer,
                        model=model,
                    )
                if method == "tpo" and reward is None:
                    reward = ArmoRMScorer(cfg, reward_device, args.scoring_batch_size)
                with GenerationCostRecorder(model, device, method) as meter:
                    if method == "oval":
                        row = generate_ftrl_mpc(
                            cfg, tokenizer, model, judges[method], device, prompt
                        )
                    elif method == "tmpc":
                        row = generate_tmpc_subgoal_regeneration(
                            tmpc_cfg, tokenizer, model, judges[method], device, prompt
                        )
                    else:
                        row = generate_tpo(
                            cfg, tokenizer, model, reward, device, prompt
                        )
                meter.attach(row)
                del meter
                row.update(method=method, generation_id=f"{pid}::{method}")
                record[method] = row
                save_json(path_for(prompt), record)
            record["generation_complete"] = True
            save_json(path_for(prompt), record)
            print(json.dumps({"event": "generated", "prompt_id": pid}), flush=True)
        for name in judges:
            judges[name].close()
        judges.clear()
        del tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()

    pending = [p for p in prompts if not load_record(p).get("scoring_complete")]
    if pending:
        if reward is None:
            reward = ArmoRMScorer(cfg, reward_device, args.scoring_batch_size)
        for prompt in pending:
            record = load_record(prompt)
            pid = prompt["prompt_id"]
            targets = [
                (f"candidate{i}", item, item["text"])
                for i, item in enumerate(record.get("candidates", []))
            ]
            targets.extend(
                (method, record[method], record[method]["response"])
                for method in ("oval", "tmpc", "tpo")
                if method in record
            )
            jobs = [
                dict(
                    generation_id=f"{pid}::{name}",
                    prompt_id=pid,
                    prompt=prompt["prompt"],
                    response=response,
                )
                for name, _, response in targets
            ]
            scores = reward.evaluate(jobs)
            for (_, item, _), score in zip(targets, scores, strict=True):
                item["armorm"] = score
            if "candidates" in record:
                record["bon_selections"] = select_bon(
                    record["candidates"], cfg["bon"]["n_values"]
                )
            record["scoring_complete"] = True
            save_json(path_for(prompt), record)
            print(json.dumps({"event": "scored", "prompt_id": pid}), flush=True)
    if reward is not None:
        reward.close()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Compare OVAL, BoN, TMPC and TPO.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prompts", type=Path, default=HERE / "data/prompts.jsonl")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--critic", type=Path)
    parser.add_argument("--tmpc-critic", type=Path)
    parser.add_argument("--reward-model", type=Path)
    parser.add_argument("--reward-device")
    parser.add_argument("--tpo-iterations", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, help="Override seed in config.py")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("Require 0 <= shard < shards.")
    if (
        args.threads < 1
        or args.scoring_batch_size < 1
        or (args.max_prompts is not None and args.max_prompts < 1)
    ):
        parser.error("Thread, batch and prompt counts must be positive.")
    cfg = load_config(args.config)
    args.seed = cfg["seed"] if args.seed is None else args.seed
    cfg["seed"] = args.seed
    if args.methods is not None:
        cfg["methods"] = args.methods
    if (
        not cfg["methods"]
        or any(m not in METHODS for m in cfg["methods"])
        or len(set(cfg["methods"])) != len(cfg["methods"])
    ):
        parser.error("methods must contain distinct entries from oval, bon, tmpc, tpo.")
    if args.tpo_iterations is not None:
        cfg["decoding"]["tpo"]["iterations"] = args.tpo_iterations
    if (
        cfg["decoding"]["tpo"]["iterations"] < 0
        or cfg["decoding"]["tpo"]["samples"] < 1
    ):
        parser.error("TPO requires non-negative iterations and positive samples.")
    assets = [
        (cfg["models"], "policy", args.model_dir),
        (cfg["models"], "armorm", args.reward_model),
    ]
    if "oval" in cfg["methods"]:
        assets.append((cfg["judge"], "probe_path", args.critic))
    if "tmpc" in cfg["methods"]:
        assets.append((cfg["tmpc"]["judge"], "probe_path", args.tmpc_critic))
    for group, key, override in assets:
        value = (
            override.resolve()
            if override is not None
            else (HERE / group[key]).resolve()
        )
        group[key] = str(value)
        if not value.exists():
            parser.error(f"Missing {key}: {value}")
    if (Path(cfg["models"]["policy"]) / "adapter_config.json").exists():
        parser.error("--model-dir must contain a merged model, not an adapter.")
    cfg["runtime"]["record_generation_cost"] = False
    n_values = cfg["bon"]["n_values"]
    if (
        not n_values
        or any(type(n) is not int or n < 1 for n in n_values)
        or len(set(n_values)) != len(n_values)
    ):
        parser.error("bon.n_values must contain distinct positive integers.")
    if cfg["bon"]["selector"] != "armorm_score":
        parser.error("BoN selection must use armorm_score (overall).")
    prompts = read_jsonl(args.prompts)
    ids = [p["prompt_id"] for p in prompts]
    if len(set(ids)) != len(ids) or not prompts:
        parser.error("Prompts must be non-empty with unique prompt_id values.")
    for prompt in prompts:
        pid = prompt["prompt_id"]
        if (
            not isinstance(pid, str)
            or not pid
            or any(
                char
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for char in pid
            )
        ):
            parser.error(
                "prompt_id must contain only letters, digits, underscores or hyphens."
            )
        if not isinstance(prompt.get("prompt"), str) or not prompt["prompt"].strip():
            parser.error(f"Missing prompt text: {pid}")
    prompts = prompts[: args.max_prompts][args.shard :: args.shards]
    args.output_dir = (args.output_dir or HERE / "outputs").resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "assigned_prompts": len(prompts),
                    "methods": cfg["methods"],
                    "models": cfg["models"],
                    "bon_selector": cfg["bon"]["selector"],
                    "tpo_iterations": cfg["decoding"]["tpo"]["iterations"],
                    "output_dir": str(args.output_dir),
                },
                indent=2,
            )
        )
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(args, cfg, prompts)


if __name__ == "__main__":
    main()

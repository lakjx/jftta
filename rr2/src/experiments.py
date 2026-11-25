import argparse
import copy
import csv
import gc
import functools
import json
from pathlib import Path
import sys

import numpy as np

from src.paired_statistics import label_agreement, paired_interval


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_rows(rows):
    if rows:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments(lookahead):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "score", "compare", "all"), default="all")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=48 if lookahead else None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluators", nargs="+", choices=("armorm", "ultrarm"), default=["armorm", "ultrarm"])
    parser.add_argument("--armorm-key")
    parser.add_argument("--draws", type=int, default=10000 if lookahead else 20000)
    if not lookahead:
        parser.add_argument("--cohorts", type=Path, nargs="+")
        parser.add_argument("--ultrarm-scores", type=Path)
        parser.add_argument("--short-prompts", action="store_true")
    args = parser.parse_args()
    if args.stage in ("generate", "score", "all") and args.config is None:
        parser.error("--config is required for generation and scoring")
    if args.stage in ("generate", "all") and args.prompts is None:
        parser.error("--prompts is required for generation")
    if args.batch_size < 1 or args.draws < 1:
        parser.error("--batch-size and --draws must be positive")
    return args


def load_config(args):
    from src import common

    common.ROOT = args.project_root.resolve()
    value = json.loads(args.config.read_text())
    cfg = copy.deepcopy(value.get("config", value))
    return cfg


def release_model(model, tokenizer):
    import torch

    model.cpu()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(args, cfg, modes, capture=False):
    import torch
    import src.ftrl_ablation.method as ftrl
    from src.models import load_lm
    from src.value_head.judge import CandidateHiddenJudge

    prompts = read_rows(args.prompts)
    if args.limit is not None:
        prompts = prompts[:args.limit]
    device = torch.device(args.device)
    tokenizer, model = load_lm(cfg, device, adapter_path=cfg["paths"].get("sft_adapter"))
    original_forward = model.forward

    @functools.wraps(original_forward)
    def feature_forward(*positional, **keywords):
        if keywords.get("use_cache") is False:
            keywords.setdefault("logits_to_keep", 1)
        return original_forward(*positional, **keywords)

    model.forward = feature_forward
    judge = CandidateHiddenJudge(cfg, device, tokenizer, model)
    paths = {
        mode: args.output / mode / "generations.jsonl"
        for mode in modes
    }
    if capture:
        paths = {modes[0]: args.output / "trajectories.jsonl"}
    existing = {
        mode: {row["prompt_id"]: row for row in read_rows(path)}
        for mode, path in paths.items()
    }
    original_lookahead = ftrl.generate_ftrl_lookahead_records
    lookaheads = []

    def record_lookahead(*positional, **keywords):
        records = original_lookahead(*positional, **keywords)
        lookaheads.append(records)
        return records

    if capture:
        ftrl.generate_ftrl_lookahead_records = record_lookahead
    try:
        for prompt in prompts:
            for mode in modes:
                if prompt["prompt_id"] in existing[mode]:
                    if existing[mode][prompt["prompt_id"]]["prompt"] != prompt["prompt"]:
                        raise ValueError("Existing prompt ID has different text")
                    continue
                current = copy.deepcopy(cfg)
                current.setdefault("method", {})["name"] = mode
                current["search"]["ftrl_ablation_mode"] = mode
                lookaheads.clear()
                row = ftrl.generate_ftrl_ablation(current, tokenizer, model, judge, device, prompt)
                if capture:
                    row["trace"] = [
                        {"iteration": index, "diagnostics": {"raw_targets": step["raw_targets"]}, "rollouts": records}
                        for index, (step, records) in enumerate(zip(row["ftrl_steps"], lookaheads, strict=True))
                    ]
                append_row(paths[mode], row)
                existing[mode][prompt["prompt_id"]] = row
    finally:
        ftrl.generate_ftrl_lookahead_records = original_lookahead
        judge.close()
        del judge
        release_model(model, tokenizer)


def load_evaluator(cfg, name, device):
    if name == "armorm":
        from src.evaluation.armorm import load_armorm_model, score_armorm_batch
        from src.models import load_tokenizer

        tokenizer = load_tokenizer(cfg["models"]["armorm"])
        model = load_armorm_model(cfg, device)
        return tokenizer, model, score_armorm_batch
    from src.evaluation.ultrarm import load_ultrarm, score_ultrarm_batch

    tokenizer, model = load_ultrarm(cfg, device)
    return tokenizer, model, score_ultrarm_batch


def score(args, cfg, modes, capture=False):
    import torch

    jobs = []
    if capture:
        for row in read_rows(args.output / "trajectories.jsonl"):
            for state in row["trace"]:
                for index, rollout in enumerate(state["rollouts"]):
                    jobs.append({
                        "generation_id": f"{row['prompt_id']}::{state['iteration']}::{index}",
                        "prompt_id": row["prompt_id"],
                        "prompt": row["prompt"],
                        "response": rollout["text"],
                        "method": "lookahead",
                    })
    else:
        for mode in modes:
            for row in read_rows(args.output / mode / "generations.jsonl"):
                jobs.append({**row, "method": mode})
    device = torch.device(args.device)
    for evaluator in args.evaluators:
        paths = {
            mode: args.output / mode / f"{evaluator}_scores.jsonl"
            for mode in modes
        }
        if capture:
            paths = {"lookahead": args.output / f"{evaluator}_scores.jsonl"}
        done = {
            row.get("generation_id", row.get("selection_id"))
            for path in paths.values()
            for row in read_rows(path)
        }
        pending = [job for job in jobs if job["generation_id"] not in done]
        if not pending:
            continue
        tokenizer, model, score_batch = load_evaluator(cfg, evaluator, device)
        attribute = args.armorm_key or cfg["experiment"]["reward_attribute"]
        if not attribute.startswith("armorm_"):
            attribute = "armorm_" + attribute
        key = attribute if evaluator == "armorm" else "ultrarm_score"
        try:
            start = 0
            batch_size = args.batch_size
            while start < len(pending):
                batch = pending[start:start + batch_size]
                try:
                    values = score_batch(cfg, tokenizer, model, device, batch)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    continue
                for job, value in zip(batch, values, strict=True):
                    reward = float(value[key])
                    if not np.isfinite(reward):
                        raise ValueError("Nonfinite reward")
                    append_row(paths[job["method"]], {
                        "generation_id": job["generation_id"],
                        "prompt_id": job["prompt_id"],
                        "method": job["method"],
                        "score": reward,
                    })
                start += len(batch)
        finally:
            release_model(model, tokenizer)


def reward_values(directory, mode, evaluator, args, external):
    if evaluator == "ultrarm" and args.ultrarm_scores is not None:
        rows = [row for row in external if row.get("corpus") == directory.name and row.get("method") == mode]
    else:
        rows = read_rows(directory / mode / f"{evaluator}_scores.jsonl")
    fields = {"HS2": "armorm_helpsteer-helpfulness", "UF": "armorm_ultrafeedback-helpfulness"}
    key = args.armorm_key or fields.get(directory.name)
    if key and not key.startswith("armorm_"):
        key = "armorm_" + key
    values = {}
    for row in rows:
        if "score" in row:
            value = row["score"]
        elif evaluator == "ultrarm":
            value = row["ultrarm_score"]
        else:
            value = row[key]
        values[row["prompt_id"]] = float(value) * (100 if evaluator == "armorm" else 1)
    return values


def compare(args, control):
    modes = ("accumulated_ftrl", control)
    directories = args.cohorts or [args.output]
    external = read_rows(args.ultrarm_scores) if args.ultrarm_scores is not None else []
    rows = []
    for evaluator in args.evaluators:
        pooled_values, pooled_texts = [], []
        short_values, short_texts = [], []
        for directory in directories:
            generations = {
                mode: {row["prompt_id"]: row for row in read_rows(directory / mode / "generations.jsonl")}
                for mode in modes
            }
            rewards = {
                mode: reward_values(directory, mode, evaluator, args, external)
                for mode in modes
            }
            ids = sorted(generations[modes[0]])
            if not ids or any(set(generations[mode]) != set(ids) or set(rewards[mode]) != set(ids) for mode in modes):
                raise ValueError(f"Incomplete paired records in {directory}")
            texts = [generations[modes[0]][key]["prompt"] for key in ids]
            if any(generations[modes[1]][key]["prompt"] != text for key, text in zip(ids, texts, strict=True)):
                raise ValueError("Paired prompt texts differ")
            differences = [rewards[modes[0]][key] - rewards[modes[1]][key] for key in ids]
            rows.append({"scope": directory.name, "evaluator": evaluator, **paired_interval(differences, texts, args.draws)})
            pooled_values.extend(differences)
            pooled_texts.extend(texts)
            if args.short_prompts:
                threshold = float(np.quantile([len(text) for text in texts], 0.25))
                for value, text in zip(differences, texts, strict=True):
                    if len(text) <= threshold:
                        short_values.append(value)
                        short_texts.append(text)
        rows.append({"scope": "pooled", "evaluator": evaluator, **paired_interval(pooled_values, pooled_texts, args.draws)})
        if args.short_prompts:
            rows.append({"scope": "short_prompts_exploratory", "evaluator": evaluator, **paired_interval(short_values, short_texts, args.draws)})
    print_rows(rows)


def paired_main(control):
    args = parse_arguments(False)
    modes = ("accumulated_ftrl", control)
    if args.stage in ("generate", "score", "all"):
        cfg = load_config(args)
        if args.stage in ("generate", "all"):
            generate(args, cfg, modes)
        if args.stage in ("score", "all"):
            score(args, cfg, modes)
    if args.stage in ("compare", "all"):
        compare(args, control)


def lookahead_main():
    args = parse_arguments(True)
    if args.stage in ("generate", "score", "all"):
        cfg = load_config(args)
        if args.stage in ("generate", "all"):
            generate(args, cfg, ("accumulated_ftrl",), capture=True)
        if args.stage in ("score", "all"):
            score(args, cfg, ("accumulated_ftrl",), capture=True)
    if args.stage in ("compare", "all"):
        trajectories = read_rows(args.output / "trajectories.jsonl")
        rows = []
        for evaluator in args.evaluators:
            scores = {row["generation_id"]: row["score"] for row in read_rows(args.output / f"{evaluator}_scores.jsonl")}
            rows.extend({"evaluator": evaluator, **row} for row in label_agreement(trajectories, scores, args.draws))
        print_rows(rows)

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from oval.io import read_jsonl, save_json, select_bon
from config import load_config

HERE = Path(__file__).resolve().parent


def summarize(records, n_values):
    rows = []
    seeds = {record["seed"] for record in records}
    if len(seeds) != 1:
        raise ValueError("Collect one seed at a time.")
    signatures = {record.get("run_signature") for record in records}
    if len(signatures) != 1:
        raise ValueError("Results have different run configurations.")
    for record in records:
        if not record.get("generation_complete") or not record.get("scoring_complete"):
            raise ValueError(f"Incomplete result: {record['prompt_id']}")
        methods = []
        for key in ("oval", "tmpc", "tpo"):
            if key in record:
                item = record[key]
                label = key.upper()
                if key == "tpo":
                    label += f" (iter={item['tpo_iterations']})"
                methods.append(
                    (label, "", "", item["armorm"], item["tta_generated_tokens"])
                )
        if "oval_reference" in record:
            methods.extend(
                [
                    (
                        "OVAL_frozen_reference",
                        "",
                        "",
                        record["oval_reference"]["armorm"],
                        record["oval_reference_total_generated_tokens"],
                    ),
                    (
                        "OVAL_metered_replay",
                        "",
                        "",
                        record["oval_replay"]["armorm"],
                        record["oval_replay"]["tta_generated_tokens"],
                    ),
                ]
            )
        if "candidates" in record:
            methods.extend(
                (
                    "BoN-ArmoRM",
                    item["n"],
                    item["selector"],
                    item["armorm"],
                    item["total_generated_tokens"],
                )
                for item in select_bon(record["candidates"], n_values)
            )
        for method, n, selector, scores, tokens in methods:
            correctness, overall = (
                scores["armorm_helpsteer-correctness"] * 100,
                scores["armorm_score"] * 100,
            )
            if not math.isfinite(correctness) or not math.isfinite(overall):
                raise ValueError("Non-finite evaluation score.")
            rows.append(
                dict(
                    prompt_id=record["prompt_id"],
                    seed=record["seed"],
                    method=method,
                    n=n,
                    selector=selector,
                    total_generated_tokens=tokens,
                    armorm_correctness=correctness,
                    armorm_overall=overall,
                )
            )
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["n"], row["selector"]), []).append(row)
    summary = []
    for (method, n, selector), items in groups.items():
        tokens = [
            r["total_generated_tokens"]
            for r in items
            if r["total_generated_tokens"] is not None
        ]
        summary.append(
            dict(
                method=method,
                n=n,
                selector=selector,
                prompt_count=len(items),
                average_total_generated_tokens=statistics.mean(tokens)
                if len(tokens) == len(items)
                else None,
                recorded_token_prompt_count=len(tokens),
                mean_armorm_correctness=statistics.mean(
                    r["armorm_correctness"] for r in items
                ),
                mean_armorm_overall=statistics.mean(r["armorm_overall"] for r in items),
            )
        )
    report = {
        "prompt_count": len(records),
        "seed": next(iter(seeds)),
        "bon_selector": "armorm_score",
        "summary": summary,
    }
    if all("oval_replay_matches_reference" in r for r in records):
        report["oval_exact_reproduction_count"] = sum(
            r["oval_replay_matches_reference"] for r in records
        )
    return rows, report


def main():
    parser = argparse.ArgumentParser(description="Summarize OVAL, BoN, TMPC and TPO.")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompts", type=Path, default=HERE / "data/prompts.jsonl")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, help="Override seed in config.py")
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.seed = cfg["seed"] if args.seed is None else args.seed
    cfg["seed"] = args.seed
    args.results_dir = args.results_dir or HERE / "outputs"
    prompts = read_jsonl(args.prompts)[: args.max_prompts]
    ids = [p["prompt_id"] for p in prompts]
    if not ids or len(set(ids)) != len(ids):
        parser.error("Prompts must be non-empty with unique IDs.")
    records = []
    for pid in ids:
        if Path(pid).name != pid or "/" in pid or "\\" in pid:
            parser.error("Invalid prompt_id.")
        path = args.results_dir / f"{pid}.json"
        if not path.is_file():
            parser.error(f"Missing result: {path}")
        record = json.loads(path.read_text())
        if record["prompt_id"] != pid or record["seed"] != args.seed:
            parser.error(f"Mismatched prompt or seed: {path}")
        records.append(record)
    rows, report = summarize(records, cfg["bon"]["n_values"])
    output = args.output_dir or args.results_dir / "report"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

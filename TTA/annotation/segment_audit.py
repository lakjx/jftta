import argparse
import collections
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from _common import print_json, read_records


DATASETS = [
    ('HelpSteer2', 'Helpfulness', 'helpsteer2_validation518'),
    ('HelpSteer2', 'Correctness', 'helpsteer2_validation518_correctness'),
    ('HelpSteer2', 'Coherence', 'helpsteer2_validation518_coherence'),
    ('HelpSteer2', 'Complexity', 'helpsteer2_validation518_complexity'),
    ('HelpSteer2', 'Verbosity', 'helpsteer2_validation518_verbosity'),
    ('UltraFeedback', 'Helpfulness', 'ultrafeedback_helpfulness_eval500'),
    ('UltraFeedback', 'Instruction following', 'ultrafeedback_instruction_following_eval500'),
    ('UltraFeedback', 'Truthfulness', 'ultrafeedback_truthfulness_eval500'),
    ('UltraFeedback', 'Honesty', 'ultrafeedback_honesty_eval500'),
]


def aggregate(rows, backbone, dataset):
    result = dict(backbone=backbone, dataset=dataset, settings=len(rows))
    for key in ('responses', 'committed_segments', 'boundary_fallbacks', 'eos_segments'):
        result[key] = sum(row[key] for row in rows)
    result['normalized_nonterminal_segments'] = (
        result['committed_segments'] - result['boundary_fallbacks'] - result['eos_segments'])
    total = result['committed_segments']
    result['boundary_or_eos_pct'] = 100 * (1 - result['boundary_fallbacks'] / total) if total else None
    return result


def audit(path, backbone, dataset, preference, expected=None):
    data = read_records(path)
    if not data or len({row['generation_id'] for row in data}) != len(data):
        raise ValueError('Empty input or repeated generation IDs: ' + str(path))
    if expected is not None and len(data) != expected:
        raise ValueError('Unexpected response count: ' + str(path))
    if not all('boundary_fallback_count' in row for row in data):
        raise ValueError('Missing boundary fallback counts: ' + str(path))
    if not all('ftrl_steps' in row for row in data):
        return None, dict(path=str(path), responses=len(data),
                          boundary_fallbacks=sum(row['boundary_fallback_count'] for row in data),
                          reason='No per-step records; the segment denominator is unavailable')
    config = json.loads((path.parent / 'run_manifest.json').read_text())['config']
    raw = dict(responses=len(data), committed_segments=sum(len(row['ftrl_steps']) for row in data),
               boundary_fallbacks=sum(row['boundary_fallback_count'] for row in data),
               eos_segments=sum(row['stop_reason'] == 'stop_token' for row in data))
    result = aggregate([raw], backbone, dataset)
    result.update(preference=preference, path=str(path),
                  segment_max_tokens=config['search']['mpc_segment_max_tokens'])
    return result, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--inputs', nargs='+', type=Path)
    parser.add_argument('--segment-max-tokens', type=int, default=128)
    args = parser.parse_args()
    sources = []
    if args.inputs:
        for path in args.inputs:
            manifest = json.loads((path.parent / 'run_manifest.json').read_text())
            cfg = manifest.get('config', manifest)
            corpus = cfg.get('data', {}).get('dataset', path.parents[1].name)
            sources.append((path, str(cfg.get('models', {}).get('policy', 'unspecified')),
                            str(corpus), cfg.get('experiment', {}).get('reward_attribute', 'unspecified'), None))
    else:
        for backbone, subdir in [('Llama-3-8B', ''), ('Qwen3-8B', 'qwen3_8b')]:
            for dataset, preference, dirname in DATASETS:
                sources.append((args.project_root / 'output' / subdir / dirname / 'ftrl_mpc' / 'generations.jsonl',
                                backbone, dataset, preference, 518 if dataset == 'HelpSteer2' else 500))
    rows, legacy = [], []
    for source in sources:
        row, skipped = audit(*source)
        if row is not None:
            rows.append(row)
        if skipped is not None:
            legacy.append(skipped)
    matching = [row for row in rows if row['segment_max_tokens'] == args.segment_max_tokens]
    grouped = collections.defaultdict(list)
    for row in matching:
        grouped[row['backbone'], row['dataset']].append(row)
    summary = [aggregate(values, *key) for key, values in sorted(grouped.items())]
    summary.append(aggregate(matching, 'Both', 'Both'))
    print_json(dict(segment_max_tokens=args.segment_max_tokens, per_setting=rows,
                    matching_summary=summary, legacy=legacy))


if __name__ == '__main__':
    main()

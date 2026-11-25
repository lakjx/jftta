import argparse
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from _common import agreement_report, load_data, print_json, read_records


ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm', 'code-complexity',
    'code-style', 'code-explanation', 'code-instruction-following', 'code-readability',
]


def render(row):
    return ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
            + row['prompt'].strip() + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            + row['response'].strip() + '<|eot_id|>')


def validate_scores(rows, originals, complete=False):
    lookup = {row['id']: row for row in originals}
    scored = {}
    for row in rows:
        rid = row['id']
        if rid not in lookup or rid in scored:
            raise ValueError('Unexpected or duplicate scored response: ' + str(rid))
        if any(row[key] != lookup[rid][key] for key in ('dataset', 'group_id')):
            raise ValueError('Score metadata disagrees with the input response')
        values = [row['aggregate']] + [row['scores'][name] for name in ATTRIBUTES]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Nonfinite ArmoRM score')
        scored[rid] = row
    if complete and set(scored) != set(lookup):
        raise ValueError('Complete response coverage is required for agreement verification')
    return scored


def compare(rows, originals, pairs, direct):
    scored = validate_scores(rows, originals, complete=True)

    def score(rid, dataset, dimension):
        prefix = 'helpsteer' if dataset == 'helpsteer2' else 'ultrafeedback'
        return scored[rid]['scores'][prefix + '-' + dimension]

    print_json(dict(evaluator='ArmoRM', responses=len(scored),
                    truncated=sum(bool(row.get('truncated')) for row in rows),
                    **agreement_report(pairs, direct, score, lambda rid: scored[rid]['aggregate'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--batch-tokens', type=int, default=32768)
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--scores', type=Path, nargs='+')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('Require 0 <= shard < shards')
    if min(args.max_length, args.batch_size, args.batch_tokens) < 1:
        parser.error('Length and batch limits must be positive')
    originals, pairs, direct = load_data(args.data_dir)
    if args.check:
        print_json(dict(responses=len(originals), attribute_pairs=len(pairs), direct_pairs=len(direct)))
        return
    if args.compare_only:
        if not args.scores:
            parser.error('--compare-only requires --scores')
        compare([row for path in args.scores for row in read_records(path)], originals, pairs, direct)
        return
    if args.model is None or args.output is None:
        parser.error('Scoring requires --model and --output')
    selected = originals[args.shard::args.shards]
    old = read_records(args.output) if args.output.exists() else []
    done = validate_scores(old, selected)
    pending = [row.copy() for row in selected if row['id'] not in done]
    if not pending:
        if args.shards == 1:
            compare(old, originals, pairs, direct)
        return
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import transformers.models.llama.modeling_llama as llama_modeling

    if not hasattr(llama_modeling, 'LLAMA_INPUTS_DOCSTRING'):
        llama_modeling.LLAMA_INPUTS_DOCSTRING = ''
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id != 128256:
        raise ValueError('Expected the original ArmoRM pad token ID 128256')
    for row in pending:
        row['_ids'] = tokenizer(render(row), add_special_tokens=False)['input_ids']
    pending.sort(key=lambda row: (len(row['_ids']), row['id']))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='sdpa').eval().to(args.device)
    model.config.pad_token_id = 128256
    model.config.use_cache = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cursor, batch_tokens, batch_size = 0, args.batch_tokens, args.batch_size
    import json

    with args.output.open('a', encoding='utf-8') as handle:
        while cursor < len(pending):
            size = 1
            while size < batch_size and cursor + size < len(pending):
                maxlen = min(args.max_length, len(pending[cursor + size]['_ids']))
                if (size + 1) * maxlen > batch_tokens:
                    break
                size += 1
            batch = pending[cursor:cursor + size]
            encoded = tokenizer.pad({'input_ids': [row['_ids'][:args.max_length] for row in batch]},
                                    padding=True, return_tensors='pt')
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            try:
                with torch.inference_mode():
                    output = model(**encoded, use_cache=False)
                rewards = output.rewards.float().cpu().tolist()
                aggregate = output.score.float().cpu().tolist()
            except torch.cuda.OutOfMemoryError:
                del encoded
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                batch_size = min(batch_size, max(1, size // 2))
                batch_tokens = max(1, batch_tokens // 2)
                continue
            if len(rewards) != len(batch) or len(aggregate) != len(batch):
                raise ValueError('ArmoRM output count differs from its input batch')
            for row, values, value in zip(batch, rewards, aggregate):
                if len(values) != len(ATTRIBUTES) or not all(math.isfinite(v) for v in values + [value]):
                    raise ValueError('Invalid ArmoRM output')
                result = dict(id=row['id'], dataset=row['dataset'], group_id=row['group_id'],
                              input_tokens=len(row['_ids']), truncated=len(row['_ids']) > args.max_length,
                              aggregate=value, scores=dict(zip(ATTRIBUTES, values)))
                handle.write(json.dumps(result) + '\n')
            handle.flush()
            cursor += size
            print('ArmoRM %d/%d' % (len(old) + cursor, len(selected)), file=sys.stderr, flush=True)
            del output, encoded
    completed = read_records(args.output)
    validate_scores(completed, selected, complete=True)
    if args.shards == 1:
        compare(completed, originals, pairs, direct)


if __name__ == '__main__':
    main()

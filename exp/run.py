from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIMENSIONS = ('helpfulness', 'honesty', 'instruction_following', 'truthfulness')


def checkpoint_directory(reference: str, subdirectory: str = '') -> Path:
    subdir = Path(subdirectory)
    if subdir.is_absolute() or '..' in subdir.parts:
        raise ValueError('--checkpoint-subdir must be a relative directory within the checkpoint repository.')
    path = Path(reference).expanduser()
    local = path if path.is_dir() else ROOT / path
    if local.is_dir():
        selected = local / subdir
        if not selected.is_dir():
            raise FileNotFoundError(f'Checkpoint directory not found: {selected}')
        return selected.resolve()
    if path.is_absolute() or reference.startswith(('./', '../', '~')):
        raise FileNotFoundError(f'Checkpoint directory not found: {path}')
    from huggingface_hub import HfApi, snapshot_download
    prefix = f'{subdir.as_posix()}/' if subdir.parts else ''
    required = [f'{prefix}model/config.json', *(f'{prefix}critics/{dimension}.pt' for dimension in DIMENSIONS)]
    available = set(HfApi().list_repo_files(repo_id=reference, repo_type='model'))
    missing = [name for name in required if name not in available]
    if missing:
        raise FileNotFoundError(
            f'Hugging Face checkpoint {reference!r} is missing required files: '
            + ', '.join(missing)
            + '. Check the repository ID and --checkpoint-subdir; the release must contain model/ and critics/.'
        )
    snapshot = snapshot_download(repo_id=reference, allow_patterns=[f'{prefix}model/*', f'{prefix}critics/*.pt'])
    return Path(snapshot) / subdir


def load_prompts(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError('Each input row must contain a nonempty prompt.')
            if 'prompt_id' not in row:
                raise ValueError('Each input row must contain prompt_id.')
            key = str(row['prompt_id'])
            if key in seen:
                raise ValueError(f'Duplicate prompt_id: {key}')
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError('The prompt file is empty.')
    return rows


def completed_ids(path: Path, prompts: list[dict], dimension: str, resume: bool) -> set[str]:
    if not path.exists():
        return set()
    if not resume:
        raise FileExistsError(f'{path} already exists. Use --resume to continue it.')
    expected = {str(row['prompt_id']): row['prompt'] for row in prompts}
    done = set()
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row['prompt_id'])
            if key in done or expected.get(key) != row.get('prompt') or row.get('dimension') != dimension:
                raise ValueError(f'Existing output does not match this run: {path}')
            if not isinstance(row.get('response'), str):
                raise ValueError(f'Existing output contains an incomplete response: {path}')
            done.add(key)
    return done


def main():
    parser = argparse.ArgumentParser(description='Run OVAL on the included 500 prompts with released merged checkpoints.')
    parser.add_argument('--checkpoint', default=str(ROOT / 'checkpoints'), help='Local checkpoint directory or Hugging Face repository ID containing model/ and critics/.')
    parser.add_argument('--checkpoint-subdir', default='', help='Optional subdirectory inside the checkpoint repository, such as ckpt0.')
    parser.add_argument('--dimension', choices=(*DIMENSIONS, 'all'), default='all')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'oval.json')
    parser.add_argument('--prompts', type=Path, default=ROOT / 'data' / 'eval500.jsonl')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.expanduser().read_text(encoding='utf-8'))
    prompts = load_prompts(args.prompts.expanduser())
    dimensions = DIMENSIONS if args.dimension == 'all' else (args.dimension,)
    outputs = {d: args.output_dir.expanduser() / f'{d}.jsonl' for d in dimensions}
    done = {d: completed_ids(outputs[d], prompts, d, args.resume) for d in dimensions}
    if all(len(done[d]) == len(prompts) for d in dimensions):
        print('All requested prompts are already complete.', flush=True)
        return
    root = checkpoint_directory(args.checkpoint, args.checkpoint_subdir)
    model_dir = root / 'model'
    if not (model_dir / 'config.json').is_file() or (model_dir / 'adapter_config.json').exists():
        raise ValueError(f'Expected a complete merged model at {model_dir}.')
    from src.value_head.model import load_probe_checkpoint
    for dimension in dimensions:
        probe = load_probe_checkpoint(root / 'critics' / f'{dimension}.pt')
        if probe.get('reward_attribute') != f'ultrafeedback-{dimension}':
            raise ValueError(f'Critic dimension mismatch: {dimension}')
    import torch
    from src.models import load_lm
    from src.oval import generate_ftrl_mpc
    from src.value_head.judge import CandidateHiddenJudge
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.')
    cfg['models']['policy'] = str(model_dir)
    tokenizer, model = load_lm(cfg, device)
    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    for dimension in dimensions:
        cfg['judge']['probe_path'] = str(root / 'critics' / f'{dimension}.pt')
        cfg['experiment'] = {'reward_attribute': f'ultrafeedback-{dimension}'}
        judge = CandidateHiddenJudge(cfg, device, tokenizer, model)
        with outputs[dimension].open('a' if args.resume else 'x', encoding='utf-8') as handle:
            for prompt in prompts:
                if str(prompt['prompt_id']) in done[dimension]:
                    continue
                result = generate_ftrl_mpc(cfg, tokenizer, model, judge, device, prompt)
                row = {'prompt_id': prompt['prompt_id'], 'prompt': prompt['prompt'], 'dimension': dimension, 'response': result['response']}
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                done[dimension].add(str(prompt['prompt_id']))
                print(f'{dimension}: {len(done[dimension])}/{len(prompts)}', flush=True)


if __name__ == '__main__':
    main()

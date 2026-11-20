import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import signal
import ssl
from string import Template
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
from _common import DIMENSIONS, agreement_report, load_data, print_json, read_records
from _glm_prompts import RUBRICS, SYSTEM_PROMPT, USER_TEMPLATE


STOP = threading.Event()


def messages(task):
    rubric = RUBRICS[task['rubric_key']]
    candidates = [{'candidate_id': row['candidate_id'], 'response': row['response']}
                  for row in task['candidates']]
    content = Template(USER_TEMPLATE).substitute(
        dataset_display=rubric['dataset_display'], dimension_display=rubric['dimension_display'],
        definition=rubric['definition'], criteria=rubric['criteria'],
        do_not_reward=rubric['do_not_reward'], anchors=rubric['anchors'],
        candidate_count=len(candidates), instruction=task['prompt'],
        candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
        rubric_key=task['rubric_key'])
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]


def parse_scores(content, task):
    decoder, value = json.JSONDecoder(), None
    for index, char in enumerate(content):
        if char != '{':
            continue
        try:
            candidate, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get('scores'), list):
            value = candidate
            break
    if value is None or value.get('dimension') != task['rubric_key']:
        raise ValueError('Missing score JSON or incorrect dimension')
    mapping = {row['candidate_id']: row['id'] for row in task['candidates']}
    scores = {}
    for row in value['scores']:
        cid, score = row.get('candidate_id'), row.get('score')
        if cid in scores or cid not in mapping:
            raise ValueError('Duplicate or unexpected candidate ID')
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not float(score).is_integer() or not 1 <= score <= 10):
            raise ValueError('Every score must be an integer in [1, 10]')
        scores[cid] = dict(candidate_id=cid, id=mapping[cid], score=int(score),
                           rationale=str(row.get('rationale', '')))
    if set(scores) != set(mapping):
        raise ValueError('Missing candidate scores')
    return [scores[row['candidate_id']] for row in task['candidates']]


def validate_tasks(tasks, responses):
    originals = {row['id']: row for row in responses}
    expected = {(row['id'], dimension) for row in responses for dimension in DIMENSIONS[row['dataset']]}
    covered, task_ids = set(), set()
    for task in tasks:
        if task['task_id'] in task_ids:
            raise ValueError('Duplicate task ID')
        task_ids.add(task['task_id'])
        if task['rubric_key'] != task['dataset'] + '_' + task['dimension']:
            raise ValueError('Task rubric disagrees with its dataset and dimension')
        candidate_ids = set()
        for candidate in task['candidates']:
            if set(candidate) != {'candidate_id', 'id', 'response'}:
                raise ValueError('Task candidates must contain only IDs and response text')
            cid, rid = candidate['candidate_id'], candidate['id']
            if cid in candidate_ids:
                raise ValueError('Duplicate candidate ID')
            candidate_ids.add(cid)
            original = originals[rid]
            if any(original[key] != task[key] for key in ('dataset', 'group_id', 'prompt')):
                raise ValueError('Candidate is bound to a different prompt group')
            if original['response'] != candidate['response']:
                raise ValueError('Task response differs from the original response')
            identity = (rid, task['dimension'])
            if identity in covered:
                raise ValueError('Response dimension occurs in multiple tasks')
            covered.add(identity)
        messages(task)
    if covered != expected:
        raise ValueError('Tasks must cover all original responses and dimensions')


def compare(rows, tasks, pairs, direct):
    expected = {row['task_id']: row for row in tasks}
    scores, done, providers = {}, set(), set()
    for row in rows:
        tid = row['task_id']
        if tid not in expected or tid in done:
            raise ValueError('Unexpected or duplicate GLM result')
        done.add(tid)
        task = expected[tid]
        if any(row[key] != task[key] for key in ('dataset', 'group_id', 'dimension')):
            raise ValueError('GLM result metadata disagrees with its task')
        judge = row['judge']
        url = urllib.parse.urlsplit(judge['base_url'])
        if not url.hostname or 'siliconflow' in url.hostname.casefold():
            raise ValueError('GLM result is not from the original internal evaluator')
        for model in (judge['requested_model'], judge['returned_model']):
            if 'glm52' not in ''.join(char for char in model.casefold() if char.isalnum()):
                raise ValueError('GLM result uses a different evaluator model')
        providers.add((judge['base_url'].rstrip('/'), judge['requested_model'].strip().casefold(),
                       judge['returned_model'].strip().casefold(), judge.get('disable_thinking', False)))
        if row['scores'] != parse_scores(row['raw_judge_response'], task):
            raise ValueError('Parsed scores disagree with the judge response')
        for value in row['scores']:
            scores[value['id'], task['dimension']] = value['score']
    if done != set(expected):
        raise ValueError('Complete GLM task coverage is required for agreement verification')
    if len(providers) != 1:
        raise ValueError('GLM results mix different providers, models or thinking settings')
    print_json(dict(evaluator='GLM', tasks=len(done),
                    **agreement_report(pairs, direct, lambda rid, dataset, dimension: scores[rid, dimension])))


def evaluate(task, args, api_key):
    payload = dict(model=args.model, messages=messages(task), temperature=0,
                   max_tokens=4096, stream=False)
    if args.disable_thinking:
        payload['enable_thinking'] = False
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    error = ''
    for attempt in range(args.retries):
        if STOP.is_set():
            break
        try:
            request = urllib.request.Request(
                args.base_url + '/chat/completions', data=body,
                headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
                method='POST')
            with urllib.request.urlopen(request, timeout=args.timeout,
                                        context=ssl.create_default_context()) as response:
                envelope = json.load(response)
            returned = envelope.get('model')
            if str(returned).strip().casefold() != args.expected_returned_model.strip().casefold():
                STOP.set()
                raise ValueError('Returned model does not match the requested evaluator: ' + str(returned))
            content = envelope['choices'][0]['message'].get('content')
            if not isinstance(content, str):
                raise ValueError('Missing textual judge output')
            return dict(task_id=task['task_id'], dataset=task['dataset'], group_id=task['group_id'],
                        paper_prompt_id=task['paper_prompt_id'], dimension=task['dimension'],
                        scores=parse_scores(content, task), raw_judge_response=content,
                        judge=dict(base_url=args.base_url, requested_model=args.model, returned_model=returned,
                                   disable_thinking=args.disable_thinking))
        except urllib.error.HTTPError as exc:
            error = 'HTTP %s: %s' % (exc.code, exc.read().decode('utf-8', errors='replace')[:500])
            if exc.code in (401, 402, 403, 404):
                STOP.set()
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)[:500]
        if STOP.is_set():
            break
        if attempt + 1 < args.retries:
            STOP.wait(min(30, 2 ** (attempt + 1)))
    raise RuntimeError(error.replace(api_key, '[REDACTED]'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-url', default='https://antchat.alipay.com/v1')
    parser.add_argument('--model', default='GLM-5.2')
    parser.add_argument('--expected-returned-model')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--retries', type=int, default=6)
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--scores', type=Path, nargs='+')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip('/')
    args.expected_returned_model = args.expected_returned_model or args.model
    url = urllib.parse.urlsplit(args.base_url)
    if (url.scheme not in ('http', 'https') or not url.hostname
            or url.username or url.password or url.query or url.fragment):
        parser.error('--base-url must be an HTTP(S) URL without embedded credentials')
    if 'siliconflow' in url.hostname.casefold():
        parser.error('Use the original internal GLM evaluator endpoint')
    for model in (args.model, args.expected_returned_model):
        if 'glm52' not in ''.join(char for char in model.casefold() if char.isalnum()):
            parser.error('Specify the actual GLM-5.2 model alias')
    if min(args.concurrency, args.retries, args.timeout) <= 0:
        parser.error('Concurrency, retries and timeout must be positive')
    originals, pairs, direct = load_data(args.data_dir)
    tasks = read_records(args.tasks)
    validate_tasks(tasks, originals)
    if args.check:
        print_json(dict(tasks=len(tasks), responses=len(originals), network_requests=0))
        return
    if args.compare_only:
        if not args.scores:
            parser.error('--compare-only requires --scores')
        compare([row for path in args.scores for row in read_records(path)], tasks, pairs, direct)
        return
    if args.output is None:
        parser.error('Scoring requires --output')
    old = read_records(args.output) if args.output.exists() else []
    lookup, done = {task['task_id']: task for task in tasks}, set()
    for row in old:
        tid, judge = row['task_id'], row['judge']
        if tid not in lookup or tid in done:
            raise ValueError('Unexpected or duplicate completed task')
        if (judge['base_url'] != args.base_url or judge['requested_model'] != args.model
                or judge['returned_model'].strip().casefold() != args.expected_returned_model.strip().casefold()
                or judge.get('disable_thinking', False) != args.disable_thinking):
            raise ValueError('Existing scores use a different provider, model or thinking setting')
        if row['scores'] != parse_scores(row['raw_judge_response'], lookup[tid]):
            raise ValueError('Existing parsed scores disagree with the judge response')
        done.add(tid)
    if len(done) == len(tasks):
        compare(old, tasks, pairs, direct)
        return
    api_key = os.environ.get('ANTCHAT_API_KEY', '').strip()
    if not api_key:
        raise ValueError('Set ANTCHAT_API_KEY before running GLM scoring')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    STOP.clear()
    signal.signal(signal.SIGINT, lambda *unused: STOP.set())
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, lambda *unused: STOP.set())
    pending = iter(task for task in tasks if task['task_id'] not in done)
    failures = 0
    with args.output.open('a', encoding='utf-8') as handle, concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency) as pool:
        active = {}

        def submit():
            if not STOP.is_set():
                task = next(pending, None)
                if task is not None:
                    active[pool.submit(evaluate, task, args, api_key)] = task

        for _ in range(args.concurrency):
            submit()
        while active:
            ready, _ = concurrent.futures.wait(active, timeout=30,
                                               return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                task = active.pop(future)
                try:
                    row = future.result()
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    handle.flush()
                    done.add(task['task_id'])
                except Exception as exc:
                    failures += 1
                    print(task['task_id'] + ': ' + str(exc).replace(api_key, '[REDACTED]'),
                          file=sys.stderr, flush=True)
                submit()
            if ready:
                print('GLM %d/%d, failures=%d' % (len(done), len(tasks), failures),
                      file=sys.stderr, flush=True)
    if len(done) != len(tasks):
        raise SystemExit(2)
    compare(read_records(args.output), tasks, pairs, direct)


if __name__ == '__main__':
    main()

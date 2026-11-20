import collections
import gzip
import json
import math
from pathlib import Path


DIMENSIONS = {
    'helpsteer2': ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'],
    'ultrafeedback': ['instruction_following', 'truthfulness', 'honesty', 'helpfulness'],
}


def read_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        text = handle.read()
    decoder, position, rows = json.JSONDecoder(), 0, []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        value, position = decoder.raw_decode(text, position)
        if not isinstance(value, dict):
            raise ValueError('Expected JSON objects in ' + str(path))
        rows.append(value)
    return rows


def load_data(directory):
    directory = Path(directory)
    responses = read_records(directory / 'responses.jsonl')
    lookup = {row['id']: row for row in responses}
    if len(lookup) != len(responses) or not responses:
        raise ValueError('Responses must have unique IDs and be nonempty')
    for row in responses:
        if row['dataset'] not in DIMENSIONS:
            raise ValueError('Unknown response dataset: ' + str(row['dataset']))
        if not isinstance(row['prompt'], str) or not isinstance(row['response'], str):
            raise ValueError('Prompt and response must be strings')
    pairs = read_records(directory / 'pairs.jsonl')
    direct_path = directory / 'direct_preference_pairs.jsonl'
    direct = read_records(direct_path) if direct_path.exists() else []
    for row in pairs + direct:
        a, b = lookup[row['response_a_id']], lookup[row['response_b_id']]
        if a['group_id'] != b['group_id'] or a['group_id'] != row['group_id']:
            raise ValueError('A preference pair crosses prompt groups')
        if 'preference_strength' in row:
            delta = -row['preference_strength']
        else:
            delta = a['annotations'][row['dimension']] - b['annotations'][row['dimension']]
        if row['label_a_minus_b'] != ((delta > 0) - (delta < 0)):
            raise ValueError('Preference direction disagrees with the original annotation')
    return responses, pairs, direct


def agreement(pairs, prediction):
    counts = collections.Counter()
    for row in pairs:
        a, b = prediction(row['response_a_id']), prediction(row['response_b_id'])
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Evaluator scores must be finite')
        direction = row['label_a_minus_b'] * ((a > b) - (a < b))
        counts['wins' if direction > 0 else 'losses' if direction < 0 else 'ties'] += 1
    n = sum(counts.values())
    return {
        'n_pairs': n,
        'wins': counts['wins'],
        'ties': counts['ties'],
        'losses': counts['losses'],
        'accuracy': (counts['wins'] + 0.5 * counts['ties']) / n if n else None,
    }


def agreement_report(pairs, direct, score, aggregate=None):
    dimensions, macros, human = [], [], []
    for dataset, names in DIMENSIONS.items():
        for gap in (1, 2):
            rates = []
            for dimension in names:
                selected = [row for row in pairs if row['dataset'] == dataset
                            and row['dimension'] == dimension and row['label_a_minus_b']
                            and row['annotation_gap'] >= gap]
                result = agreement(selected, lambda rid: score(rid, dataset, dimension))
                dimensions.append(dict(dataset=dataset, dimension=dimension,
                                       minimum_annotation_gap=gap, **result))
                rates.append(result['accuracy'])
            macros.append(dict(dataset=dataset, minimum_annotation_gap=gap,
                               accuracy=sum(rates) / len(rates) if all(v is not None for v in rates) else None))
    for strength in (1, 2, 3):
        selected = [row for row in direct if row['label_a_minus_b']
                    and abs(row['preference_strength']) >= strength]
        if selected:
            human.append(dict(evaluator='helpfulness', minimum_preference_strength=strength,
                              **agreement(selected, lambda rid: score(rid, 'helpsteer2', 'helpfulness'))))
            if aggregate is not None:
                human.append(dict(evaluator='aggregate', minimum_preference_strength=strength,
                                  **agreement(selected, aggregate)))
    return dict(dimensions=dimensions, macro=macros, direct_human_preference=human)


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

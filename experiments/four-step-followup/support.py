"""Reuse the frozen four-step protocol without changing its source fingerprint."""
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_HERE = ROOT / 'experiments/four-step-lr'
OLD_ROOT = ROOT / 'results/four-step-lr'
OUT = ROOT / 'results/four-step-followup'
sys.path.append(str(OLD_HERE))
import common as old  # noqa: E402

SPEC = yaml.safe_load((HERE / 'protocol.yml').read_text())['followup']
SEEDS = tuple(SPEC['seeds'])
SNAPSHOTS = tuple(SPEC['snapshots'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def old_config():
    cfg = old.config()
    _, current = old.identity(cfg)
    training = read(OLD_ROOT / 'experiment.json')
    evaluation = read(OLD_ROOT / 'evaluation.json')
    # Later experiments added unrelated files under drills/. The original
    # fingerprint cannot be recomputed, but every source it recorded must match.
    original_sources = training['sources']
    if (training['status'] != 'complete' or evaluation['status'] != 'complete' or
            evaluation['training_fingerprint'] != training['fingerprint'] or
            any(current['sources'].get(name) != digest for name, digest in original_sources.items()) or
            json.dumps(current['config'], sort_keys=True) !=
            json.dumps(training['config'], sort_keys=True) or
            current['tools'] != training['tools'] or
            current['benchmarks'] != training['benchmarks'] or
            current['python'] != training['python'] or
            current['torch'] != training['torch'] or
            current['numpy'] != training['numpy']):
        raise ValueError('The original four-step evidence is incomplete or its source fingerprint changed.')
    return cfg, training['fingerprint']


def training_config(fingerprint):
    cfg, _ = old_config()
    cfg = copy.deepcopy(cfg)
    cfg['method']['optimizer']['learning_rate'] = SPEC['learning_rate']
    cfg['runtime'].update(output_dir=str(OUT / 'training'), experiment_group='rolling8',
                          experiment_fingerprint=fingerprint)
    return cfg


def identity():
    cfg, original_fingerprint = old_config()
    paths = sorted(HERE.glob('*.py')) + [HERE / 'protocol.yml']
    payload = dict(original_fingerprint=original_fingerprint,
                   candidates_sha256=sha(ROOT / 'experiments/ten-step-optimality/candidates.csv'),
                   sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
                   return_spec=SPEC, old_config=cfg)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def candidates():
    cfg, _ = old_config()
    action_names = tuple(cfg['protocol']['actions'])
    mapping = {name: i for i, name in enumerate(action_names)}
    result = {}
    path = ROOT / 'experiments/ten-step-optimality/candidates.csv'
    with path.open(newline='') as stream:
        for raw in csv.DictReader(stream):
            sequence = tuple(mapping[name] for name in json.loads(raw['actions']))
            key = (raw['circuit'], sequence)
            if key in result or len(sequence) != int(raw['depth']):
                raise ValueError('Duplicate or malformed four-step enumeration row.')
            result[key] = dict(luts=int(raw['luts']), levels=int(raw['levels']),
                               feasible=raw['feasible'] == 'True',
                               structural_sha256=raw['structural_sha256'],
                               mapped_structural_sha256=raw['mapped_structural_sha256'])
    expected = sum(7 ** depth for depth in range(5))
    for circuit in cfg['protocol']['circuits']:
        subset = {seq for name, seq in result if name == circuit}
        required = {seq for depth in range(5) for seq in itertools.product(range(7), repeat=depth)}
        if len(subset) != expected or subset != required:
            raise ValueError(f'Incomplete four-step enumeration for {circuit}.')
    return result


def rank(row):
    return (0, row['luts'], row['levels']) if row['feasible'] else (1, row['levels'], row['luts'])


def best_for_sequence(table, circuit, sequence):
    return min((table[circuit, sequence[:depth]] for depth in range(5)), key=rank)

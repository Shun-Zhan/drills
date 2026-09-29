"""Shared fixed protocol, provenance, and atomic result helpers."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from drills.experiment import load_config  # noqa: E402

OUT = ROOT / 'results/lr-reward-two-stage'
SPEC = yaml.safe_load((HERE / 'protocol.yml').read_text())
TEN = SPEC['ten_step']
FOUR = SPEC['four_step']
GROUPS = tuple(TEN['groups'])
CIRCUITS = tuple(TEN['circuits'])
SEEDS = tuple(TEN['seeds'])
OLD_LEARNING = ROOT / 'results/learning-effectiveness'
OLD_REWARD = ROOT / 'results/four-step-reward'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def base_config():
    cfg = load_config(ROOT / 'experiments/learning-effectiveness/protocol.yml')
    original = yaml.safe_load((ROOT / 'params.yml').read_text())
    expected_protocol = copy.deepcopy(original['protocol'])
    for circuit in expected_protocol['circuits'].values():
        circuit['file'] = str((ROOT / circuit['file']).resolve())
    for key, expected in (('protocol', expected_protocol),
                          ('method', original['method']),
                          ('environment', original['environment'])):
        if cfg[key] != expected:
            raise ValueError(f'Verified legacy protocol differs from params.yml: {key}')
    return cfg


def source_payload(stage):
    cfg = base_config()
    own = [HERE / 'study.py', HERE / f'stage{stage}.py', HERE / 'protocol.yml']
    inherited = [ROOT / 'params.yml', ROOT / 'requirements.txt',
                 ROOT / 'experiments/learning-effectiveness/protocol.yml',
                 *sorted((ROOT / 'drills').glob('*.py')),
                 ROOT / 'experiments/four-step-followup/run.py',
                 ROOT / 'experiments/four-step-followup/support.py',
                 ROOT / 'experiments/four-step-reward/context.py',
                 ROOT / 'experiments/four-step-reward/run.py',
                 ROOT / 'experiments/four-step-reward/reward.py',
                 ROOT / 'experiments/four-step-reward/table.py',
                 ROOT / 'experiments/four-step-reward/protocol.yml',
                 ROOT / 'experiments/ten-step-reward-validation/run.py',
                 ROOT / 'experiments/ten-step-reward-validation/context.py',
                 ROOT / 'experiments/ten-step-reward-validation/protocol.yml',
                 ROOT / 'experiments/ten-step-optimality/candidates.csv']
    paths = sorted(set(own + inherited))
    return dict(stage=stage, protocol=SPEC['ten_step' if stage == 1 else 'four_step'],
                sources={str(path.relative_to(ROOT)): sha(path) for path in paths},
                tools={key: sha(cfg['runtime'][key])
                       for key in ('abc_binary', 'yosys_binary')},
                circuits={name: sha(circuit['file'])
                          for name, circuit in cfg['protocol']['circuits'].items()},
                python=sys.version, torch=torch.__version__, numpy=np.__version__)


def identity(stage):
    payload = source_payload(stage)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def training_config(group, stage=1):
    if stage != 1 or group not in GROUPS:
        raise ValueError('Unknown ten-step group.')
    cfg = copy.deepcopy(base_config())
    cfg['protocol'].update(episodes=TEN['episodes'], iterations=TEN['iterations'],
                           seeds=list(SEEDS))
    if group != 'legacy':
        cfg['method']['optimizer']['learning_rate'] = TEN['learning_rates'][group[:4]]
        if group.endswith('_new'):
            cfg['method']['reward'] = dict(kind='best_feasible_v1', **TEN['reward'])
    cfg['runtime'].update(output_dir=str(OUT / 'stage1/training' / group),
                          experiment_group=group,
                          experiment_fingerprint=identity(1)[0])
    return cfg


def train_folder(group, circuit, seed):
    return OUT / 'stage1/training' / group / circuit / f'seed-{seed}'


def eval_folder(group, circuit, seed):
    return OUT / 'stage1/evaluation' / group / circuit / f'seed-{seed}'


def eval_seeds(seed):
    first = TEN['evaluation']['first_seed'] + SEEDS.index(seed) * TEN['evaluation']['rollouts_per_training_seed']
    return tuple(range(first, first + TEN['evaluation']['rollouts_per_training_seed']))


def reuse_candidate(group, circuit, seed):
    if group == 'legacy':
        return OLD_LEARNING / 'training/trained' / circuit / f'seed-{seed}'
    if group == 'w010_new' and circuit in ('i2c', 'max'):
        return OLD_REWARD / 'ten-step/reward' / circuit / f'seed-{seed}'
    return None


def source_for(group, circuit, seed):
    decision = read(OUT / 'stage1/preflight.json')['reuse'][f'{group}/{circuit}/{seed}']
    return Path(decision['path']) if decision['reused'] else train_folder(group, circuit, seed)

"""Immutable protocol and provenance for the reward-only follow-up."""
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
FOLLOWUP = ROOT / 'experiments/four-step-followup'
sys.path.append(str(FOLLOWUP))
import support as prior  # noqa: E402

OUT = ROOT / 'results/four-step-reward'
SPEC = yaml.safe_load((HERE / 'protocol.yml').read_text())['reward_experiment']
SEEDS4 = tuple(SPEC['four_step_seeds'])
SEEDS10 = tuple(SPEC['ten_step']['seeds'])
SNAPSHOTS4 = tuple(SPEC['snapshots'])
SNAPSHOTS10 = (0, 50, 100)


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


def identity():
    cfg, old_fingerprint = prior.old_config()
    followup_fingerprint, _ = prior.identity()
    paths = sorted(HERE.glob('*.py')) + [HERE / 'protocol.yml']
    payload = dict(
        old_fingerprint=old_fingerprint,
        followup_fingerprint=followup_fingerprint,
        candidate_sha256=sha(ROOT / 'experiments/ten-step-optimality/candidates.csv'),
        prefix_sha256={name: sha(prior.OUT / 'prefix-states' / f'{name}.json')
                       for name in ('i2c', 'max', 'int2float')},
        sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
        tools={name: sha(cfg['runtime'][name]) for name in ('abc_binary', 'yosys_binary')},
        circuits={name: sha(row['file']) for name, row in cfg['protocol']['circuits'].items()},
        protocol=SPEC, python=sys.version, torch=torch.__version__, numpy=np.__version__,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def training_config(circuit, group, ten_step=False):
    fingerprint, _ = identity()
    cfg, _ = prior.old_config()
    cfg = copy.deepcopy(cfg)
    cfg['method']['optimizer']['learning_rate'] = SPEC['learning_rate']
    cfg['method']['gamma'] = SPEC['gamma']
    cfg['runtime'].update(experiment_group=group, experiment_fingerprint=fingerprint,
                          output_dir=str(OUT / ('ten-step' if ten_step else 'four-step') / group))
    if ten_step:
        cfg['protocol'].update(episodes=SPEC['ten_step']['episodes'],
                               iterations=SPEC['ten_step']['iterations'], seeds=list(SEEDS10))
    else:
        cfg['protocol'].update(episodes=SPEC['four_step_episodes'],
                               iterations=SPEC['four_step_iterations'], seeds=list(SEEDS4))
    if group == 'reward':
        cfg['method']['reward'] = dict(kind='best_feasible_v1', **SPEC['reward'])
    return cfg

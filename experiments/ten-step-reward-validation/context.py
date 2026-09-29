"""Fixed protocol, paths, and provenance for the ten-step reward comparison."""
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
REWARD = ROOT / 'experiments/four-step-reward'
sys.path.append(str(FOLLOWUP))
sys.path.append(str(REWARD))
import support as prior  # noqa: E402

OUT = ROOT / 'results/ten-step-reward-validation'
ARCHIVE = ROOT.parent / 'experiment-archives/four-step-reward-682ee92.tar.zst'
SPEC = yaml.safe_load((HERE / 'protocol.yml').read_text())['validation']
SEEDS = tuple(SPEC['training_seeds'])
CIRCUITS = tuple(SPEC['circuits'])
GROUPS = tuple(SPEC['groups'])
EVAL_GROUPS = tuple(SPEC['evaluation']['groups'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def identity():
    cfg, old_fingerprint = prior.old_config()
    inherited = [
        FOLLOWUP / 'run.py', FOLLOWUP / 'support.py',
        REWARD / 'reward.py', REWARD / 'protocol.yml',
        ROOT / 'drills/model.py', ROOT / 'drills/fpga_session.py',
        ROOT / 'drills/features.py', ROOT / 'drills/experiment.py',
        ROOT / 'params.yml',
    ]
    sources = sorted(HERE.glob('*.py')) + [HERE / 'protocol.yml'] + inherited
    payload = dict(
        prior_fingerprint=old_fingerprint,
        sources={str(path.relative_to(ROOT)): sha(path) for path in sources},
        tools={name: sha(cfg['runtime'][name])
               for name in ('abc_binary', 'yosys_binary')},
        circuits={name: sha(cfg['protocol']['circuits'][name]['file'])
                  for name in CIRCUITS},
        protocol=SPEC,
        python=sys.version, torch=torch.__version__, numpy=np.__version__,
    )
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def training_config(circuit, group):
    if circuit not in CIRCUITS or group not in GROUPS:
        raise ValueError('Training task is outside the fixed protocol.')
    cfg, _ = prior.old_config()
    cfg = copy.deepcopy(cfg)
    digest, _ = identity()
    cfg['protocol'].update(episodes=SPEC['episodes'],
                           iterations=SPEC['iterations'], seeds=list(SEEDS))
    cfg['method']['optimizer']['learning_rate'] = SPEC['learning_rate']
    cfg['method']['gamma'] = SPEC['gamma']
    cfg['runtime'].update(output_dir=str(OUT / 'training' / group),
                          experiment_group=group,
                          experiment_fingerprint=digest)
    if group == 'candidate':
        cfg['method']['reward'] = dict(kind='best_feasible_v1', **SPEC['reward'])
    return cfg


def evaluation_config(circuit):
    cfg = training_config(circuit, 'original')
    cfg['runtime']['output_dir'] = str(OUT / 'evaluation')
    return cfg


def evaluation_seeds(training_seed):
    if training_seed not in SEEDS:
        raise ValueError('Evaluation seed bank is outside the fixed protocol.')
    count = SPEC['evaluation']['rollouts_per_training_seed']
    start = SPEC['evaluation']['first_seed'] + SEEDS.index(training_seed) * count
    return tuple(range(start, start + count))


def training_folder(group, circuit, seed):
    return OUT / 'training' / group / circuit / f'seed-{seed}'


def evaluation_folder(group, circuit, seed):
    return OUT / 'evaluation' / group / circuit / f'seed-{seed}'

"""Shared, fixed protocol and provenance helpers for the four-step study."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from drills.experiment import load_config
from drills.fpga_session import FPGASession
from drills.model import A2C, ActorCritic, Normalizer

ORIGINAL = 'adf7bef53cdf6c4161586c03e6e2caecffe26bc4'
SPEC = yaml.safe_load((HERE / 'protocol.yml').read_text())
ASSESSMENT = SPEC['assessment']
GROUPS = tuple(ASSESSMENT['groups'])
POLICIES = ('lr001-final', 'lr010-final', 'initial', 'uniform')
SNAPSHOTS = tuple(ASSESSMENT['snapshots'])


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def state_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def flattened(module):
    return torch.cat([p.detach().flatten().clone() for p in module.parameters()])


def rms(value):
    return float(value.double().square().mean().sqrt())


def config():
    cfg = load_config(HERE / 'protocol.yml')
    original = yaml.safe_load(subprocess.check_output(['git', 'show', f'{ORIGINAL}:params.yml'], cwd=ROOT))
    expected = copy.deepcopy(original['protocol'])
    expected.update(episodes=250, iterations=4, seeds=list(range(10)))
    for circuit in expected['circuits'].values():
        circuit['file'] = str((ROOT / circuit['file']).resolve())
    if cfg['protocol'] != expected or cfg['method'] != original['method'] or cfg['environment'] != original['environment']:
        raise ValueError('Four-step protocol unexpectedly changes the original method or circuit settings.')
    required = dict(learning_rates={'lr001': 0.001, 'lr010': 0.01}, groups=list(GROUPS),
                    optimum_luts={'int2float': 44, 'i2c': 312, 'max': 781}, evaluation_attempts=10,
                    evaluation_seed_start=30000, probe_pattern='all_ordered_action_pairs_repeated',
                    snapshots=[0, 50, 100, 150, 200, 250], task_timeout_seconds=1800)
    if ASSESSMENT != required or GROUPS != ('lr001', 'lr010', 'uniform') or cfg['runtime']['workers'] != 3:
        raise ValueError('Assessment deviates from its predeclared 250x4 comparison.')
    return cfg


def group_config(cfg, group, fingerprint):
    if group not in GROUPS:
        raise ValueError(group)
    result = copy.deepcopy(cfg)
    if group != 'uniform':
        result['method']['optimizer']['learning_rate'] = ASSESSMENT['learning_rates'][group]
    result['runtime'].update(
        output_dir=str(Path(cfg['runtime']['output_dir']) / 'training' / group),
        experiment_group=group, experiment_fingerprint=fingerprint)
    return result


def source_hashes():
    paths = [ROOT / 'drills.py', ROOT / 'requirements.txt',
             *sorted((ROOT / 'drills').glob('*.py')), *sorted(HERE.glob('*.py')), HERE / 'protocol.yml']
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def identity(cfg):
    payload = dict(sources=source_hashes(), config=cfg,
                   tools={key: sha(cfg['runtime'][key]) for key in ('abc_binary', 'yosys_binary')},
                   benchmarks={name: sha(row['file']) for name, row in cfg['protocol']['circuits'].items()},
                   python=sys.version, torch=torch.__version__, numpy=np.__version__)
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return fingerprint, payload


def probe_path(cfg, name):
    return Path(cfg['runtime']['output_dir']) / 'probes' / f'{name}.npy'


def create_probes(cfg, name):
    """One deterministic state bank per circuit, independent of any training RNG."""
    path = probe_path(cfg, name)
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if data.shape != (196, len(cfg['method']['features'])) or not np.isfinite(data).all():
            raise ValueError(f'Invalid fixed probe bank: {path}')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    game = FPGASession(cfg, cfg['protocol']['circuits'][name], path.parent / f'{name}-generation')
    rows = []
    for first in range(7):
        for second in range(7):
            state = game.reset()
            normalizer = Normalizer(len(state), cfg['method']['normalization'], cfg['method']['features'], state)
            for action in (first, second, first, second):
                rows.append(normalizer.normalize(state))
                state, _, _ = game.step(action)
    data = np.asarray(rows, dtype=np.float32)
    if data.shape != (196, len(cfg['method']['features'])) or not np.isfinite(data).all():
        raise ValueError('Probe generation produced invalid states.')
    np.save(path, data, allow_pickle=False)
    return path


def evaluation_seeds(training_seed):
    start = ASSESSMENT['evaluation_seed_start'] + 10 * training_seed
    return tuple(range(start, start + ASSESSMENT['evaluation_attempts']))


def supervise(tasks, log_dir, workers=3, timeout=1800):
    """Monitor subprocesses, retain failures, and enforce a per-task hard timeout."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending, active, completed = list(tasks), [], []
    while pending or active:
        while pending and len(active) < workers:
            label, argv = pending.pop(0)
            log = log_dir / f'{label}.log'
            stream = log.open('a')
            process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            active.append(dict(label=label, argv=argv, log=str(log), stream=stream,
                               process=process, started_at=now(), started=time.monotonic()))
            print(f'Started {label}', flush=True)
        for job in list(active):
            process = job['process']
            timed_out = False
            if process.poll() is None and time.monotonic() - job['started'] > timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            exit_code = process.poll()
            if exit_code is None:
                continue
            job['stream'].close()
            record = dict(label=job['label'], argv=job['argv'], log=job['log'],
                          started_at=job['started_at'], finished_at=now(),
                          elapsed_seconds=time.monotonic() - job['started'],
                          exit_code=exit_code, timed_out=timed_out)
            completed.append(record)
            active.remove(job)
            print(f'Finished {job["label"]}: exit={exit_code}', flush=True)
        if active:
            time.sleep(2)
    return completed

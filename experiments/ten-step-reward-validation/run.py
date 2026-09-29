"""Train original and candidate rewards with one ten-step window learner."""
import argparse
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys
import traceback

from filelock import FileLock
import numpy as np
import torch
import yaml

from context import (ARCHIVE, CIRCUITS, GROUPS, HERE, OUT, REWARD, ROOT, SEEDS, SPEC, dump,
                     identity, prior, read, sha, training_config, training_folder)
from drills.experiment import verify_netlist
from drills.fpga_session import FPGASession
from reward import RewardSession

_window_spec = importlib.util.spec_from_file_location(
    'certified_window_run', prior.HERE / 'run.py')
window = importlib.util.module_from_spec(_window_spec)
_window_spec.loader.exec_module(window)
WindowA2C = window.WindowA2C


class TenStepWindowA2C(WindowA2C):
    """Keep sampling, normalization, and updates identical between rewards."""

    def __init__(self, cfg, circuit, seed, folder, group, resume=False):
        super().__init__(cfg, cfg['protocol']['circuits'][circuit], seed, folder, resume)
        if group == 'candidate':
            previous = self.game
            self.game = RewardSession(cfg, cfg['protocol']['circuits'][circuit], folder)
            self.game.episode = self.episodes_completed
            self.game.best = previous.best
            self.game.best_netlists = previous.best_netlists

    def _update(self, states, actions, rewards):
        raw = np.empty(len(rewards), dtype=np.float32)
        cumulative = 0.0
        for index in reversed(range(len(rewards))):
            cumulative = rewards[index] + self.method['gamma'] * cumulative
            raw[index] = cumulative
        if len(raw) != self.protocol['iterations']:
            raise ValueError('Return vector does not match ten-step episode length.')
        if self.episodes_completed < SPEC['warmup_episodes']:
            norm = self.method['normalization']
            returns = (raw - raw.mean()) / max(float(raw.std()), norm['returns_epsilon'])
            mode = 'within_episode_warmup'
        else:
            if len(self.return_window) != SPEC['return_window']:
                raise ValueError('Return window is not full after warmup.')
            history = np.asarray(self.return_window, dtype=np.float32)
            mean = history.mean(axis=0)
            scale = np.maximum(history.std(axis=0), SPEC['return_std_floor'])
            returns = (raw - mean) / scale
            mode = 'lagged_window'
        actor_before = prior.old.flattened(self.network.actor)
        logits, values = self.network(torch.tensor(np.asarray(states), device='cpu'))
        advantage = torch.tensor(returns, device='cpu') - values
        log_probs = logits.log_softmax(-1).gather(
            1, torch.tensor(actions, device='cpu')[:, None]).squeeze(1)
        settings = self.method['loss']
        actor = getattr(-log_probs * advantage.detach(), settings['reduction'])()
        critic = getattr(advantage.square(), settings['reduction'])()
        loss = settings['actor_weight'] * actor + settings['critic_weight'] * critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.return_window.append([float(value) for value in raw])
        self.return_window = self.return_window[-SPEC['return_window']:]
        metric = dict(episode=self.episodes_completed + 1, mode=mode,
                      mean_abs_advantage=float(advantage.detach().abs().mean()),
                      mean_abs_target=float(np.abs(returns).mean()),
                      actor_update_rms=prior.old.rms(
                          prior.old.flattened(self.network.actor) - actor_before),
                      critic_loss=float(critic.detach()))
        if not all(math.isfinite(value) for value in metric.values()
                   if isinstance(value, float)):
            raise ValueError('Nonfinite return diagnostic.')
        with self.metrics_path.open('a') as stream:
            stream.write(json.dumps(metric, allow_nan=False) + '\n')


def check_config_isolation():
    original = training_config('i2c', 'original')
    candidate = training_config('i2c', 'candidate')
    if candidate['method']['reward'] != dict(kind='best_feasible_v1', **SPEC['reward']):
        raise ValueError('Candidate reward differs from the fixed protocol.')
    if original['method']['reward'] == candidate['method']['reward']:
        raise ValueError('The two reward definitions are identical.')
    original = copy.deepcopy(original)
    candidate = copy.deepcopy(candidate)
    for cfg in (original, candidate):
        del cfg['method']['reward']
        del cfg['runtime']['experiment_group']
        del cfg['runtime']['output_dir']
    if original != candidate:
        raise ValueError('Training configuration differs beyond the reward.')
    old_window = prior.SPEC
    for key in ('return_window', 'warmup_episodes', 'return_std_floor'):
        if SPEC[key] != old_window[key]:
            raise ValueError(f'Window setting changed: {key}')
    prior_reward = read(ROOT / 'experiments/four-step-reward/audit.json')
    if prior_reward['status'] != 'pass':
        raise ValueError('The four-step candidate reward audit did not pass.')
    reward_spec = yaml.safe_load((REWARD / 'protocol.yml').read_text())['reward_experiment']
    if SPEC['reward'] != reward_spec['reward']:
        raise ValueError('The candidate reward differs from the audited version.')


def preflight():
    check_config_isolation()
    digest, payload = identity()
    if sha(ARCHIVE) != SPEC['legacy_snapshot_sha256'] or \
            stat.S_IMODE(ARCHIVE.stat().st_mode) & 0o222:
        raise ValueError('The read-only legacy results snapshot failed verification.')
    if SEEDS != tuple(range(20, 30)) or CIRCUITS != ('i2c', 'max') or \
            GROUPS != ('original', 'candidate'):
        raise ValueError('Unexpected confirmatory circuit or seed protocol.')
    if SPEC['episodes'] != 100 or SPEC['iterations'] != 10 or \
            SPEC['learning_rate'] != 0.01 or SPEC['gamma'] != 0.99:
        raise ValueError('Unexpected training budget.')
    if SPEC['evaluation'] != dict(rollouts_per_training_seed=30, first_seed=30000,
                                  groups=['original', 'candidate', 'initial']):
        raise ValueError('Unexpected frozen evaluation protocol.')
    if SPEC['statistics']['alpha'] != 0.05 or \
            SPEC['statistics']['ties_count_as_nonwins'] is not True:
        raise ValueError('Unexpected confirmatory decision protocol.')
    if SPEC['workers'] != 3 or SPEC['task_timeout_seconds'] != 1800:
        raise ValueError('Unexpected supervision settings.')
    existing = OUT / 'preflight.json'
    if existing.exists() and read(existing)['fingerprint'] != digest:
        raise ValueError('An earlier experiment exists with a different fingerprint.')
    dump(existing, dict(status='pass', fingerprint=digest, source_payload=payload,
                        legacy_snapshot=dict(path=str(ARCHIVE),
                                             sha256=SPEC['legacy_snapshot_sha256']),
                        comparisons=['reward-only', 'initial state', 'first episode']))
    print(f'Preflight passed: {digest}', flush=True)


def train_task(group, circuit, seed):
    if group not in GROUPS or circuit not in CIRCUITS or seed not in SEEDS:
        raise ValueError('Training task is outside the fixed protocol.')
    digest, _ = identity()
    cfg = training_config(circuit, group)
    folder = training_folder(group, circuit, seed)
    result_path = folder / 'result.json'
    if result_path.exists():
        existing = read(result_path)
        if existing['fingerprint'] == digest and existing['status'] == 'complete':
            print(f'Already complete: {group}/{circuit}/{seed}', flush=True)
            return
        raise ValueError(f'Conflicting result: {result_path}')
    folder.mkdir(parents=True, exist_ok=True)
    completed = 0
    try:
        agent = TenStepWindowA2C(
            cfg, circuit, seed, folder, group, resume=(folder / 'checkpoint.pt').exists())
        completed = agent.episodes_completed
        for _ in range(completed, SPEC['episodes']):
            agent.run_episode()
            completed = agent.episodes_completed
            if completed % 10 == 0:
                print(f'{group}/{circuit}/seed-{seed}: {completed}/100', flush=True)
            dump(folder / 'status.json', dict(status='running', episodes=completed))
        verify_netlist(cfg, cfg['protocol']['circuits'][circuit],
                       folder / 'best-mapped.v', agent.game.best)
        first = prior.old.load(folder / 'snapshots/0.pt')
        final = prior.old.load(folder / 'checkpoint.pt')
        if len(final['return_window']) != SPEC['return_window']:
            raise ValueError('Final checkpoint lacks the complete return window.')
        hashes = {name: prior.old.state_hash(first[name])
                  for name in ('network', 'optimizer', 'rng_state')}
        dump(result_path, dict(
            status='complete', fingerprint=digest, group=group, circuit=circuit,
            seed=seed, episodes_completed=completed,
            mapping_calls=completed * (SPEC['iterations'] + 1),
            best=agent.game.best, initial_hashes=hashes,
            final_network_sha256=prior.old.state_hash(final['network']),
            checkpoint_sha256=sha(agent.checkpoint),
            best_mapped_sha256=sha(folder / 'best-mapped.v'),
            equivalence_log_sha256=sha(folder / 'equivalence.log'),
            training_seconds=agent.training_seconds))
        dump(folder / 'status.json', dict(status='complete', episodes=completed))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', episodes=completed,
                                         error=repr(error),
                                         traceback=traceback.format_exc()))
        raise


def first_episode(folder):
    with (folder / 'episodes/1/log.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != SPEC['iterations'] + 1:
        raise ValueError(f'First episode has incomplete mapping log: {folder}')
    return [(int(row['iteration']), row['optimization'],
             int(row['luts']), int(row['levels'])) for row in rows]


def validate_pairs():
    digest, _ = identity()
    rows = []
    for circuit in CIRCUITS:
        for seed in SEEDS:
            a = training_folder('original', circuit, seed)
            b = training_folder('candidate', circuit, seed)
            old, new = read(a / 'result.json'), read(b / 'result.json')
            if (old['status'], new['status']) != ('complete', 'complete') or \
                    old['fingerprint'] != digest or new['fingerprint'] != digest:
                raise ValueError(f'Incomplete or stale training pair: {circuit}/{seed}')
            if old['initial_hashes'] != new['initial_hashes']:
                raise ValueError(f'Initial network/optimizer/RNG differs: {circuit}/{seed}')
            if first_episode(a) != first_episode(b):
                raise ValueError(f'First ABC episode differs: {circuit}/{seed}')
            rows.append(dict(circuit=circuit, seed=seed,
                             initial_hashes=old['initial_hashes'],
                             first_episode_equal=True))
    dump(OUT / 'pair-validation.json',
         dict(status='pass', fingerprint=digest, rows=rows))


def train():
    digest, payload = identity()
    if not (OUT / 'preflight.json').exists() or \
            read(OUT / 'preflight.json')['fingerprint'] != digest:
        raise ValueError('Run preflight with the current sources first.')
    manifest_path = OUT / 'train-manifest.json'
    if manifest_path.exists() and read(manifest_path)['fingerprint'] != digest:
        raise ValueError('Cannot resume training with changed sources.')
    jobs = [(group, circuit, seed) for circuit in CIRCUITS
            for seed in SEEDS for group in GROUPS]
    pending = []
    for group, circuit, seed in jobs:
        path = training_folder(group, circuit, seed) / 'result.json'
        if path.exists():
            row = read(path)
            if row['status'] != 'complete' or row['fingerprint'] != digest:
                raise ValueError(f'Conflicting completed result: {path}')
            continue
        pending.append((f'{group}-{circuit}-{seed}',
                        [sys.executable, '-B', '-u', str(HERE / 'run.py'),
                         '--task', group, circuit, str(seed)]))
    with FileLock(str(OUT / '.train.lock'), timeout=0):
        manifest = dict(status='running', fingerprint=digest,
                        source_payload=payload, jobs=[list(job) for job in jobs])
        dump(manifest_path, manifest)
        processes = prior.old.supervise(
            pending, OUT / 'logs/training', workers=SPEC['workers'],
            timeout=SPEC['task_timeout_seconds'])
        missing = [list(job) for job in jobs
                   if not (training_folder(*job) / 'result.json').exists()]
        manifest.update(
            status='complete' if not missing and all(p['exit_code'] == 0 for p in processes)
            else 'incomplete', processes=processes, missing=missing)
        dump(manifest_path, manifest)
        if manifest['status'] != 'complete':
            raise RuntimeError('Training is incomplete; inspect manifest and task logs.')
        validate_pairs()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'train'))
    parser.add_argument('--task', nargs=3, metavar=('GROUP', 'CIRCUIT', 'SEED'))
    args = parser.parse_args()
    if args.task:
        group, circuit, raw_seed = args.task
        train_task(group, circuit, int(raw_seed))
    elif args.phase == 'preflight':
        preflight()
    elif args.phase == 'train':
        train()
    else:
        parser.error('--phase or --task is required')


if __name__ == '__main__':
    main()

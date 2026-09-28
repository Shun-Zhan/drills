"""Run parity, pilot, paired four-step training, and conditional ten-step search."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from context import (HERE, OUT, ROOT, SEEDS4, SEEDS10, SPEC, dump, identity,
                     prior, read, sha, training_config)
from reward import RewardSession
from table import TableSession
from drills.experiment import verify_netlist
from drills.fpga_session import FPGASession
from drills.model import A2C, ActorCritic, Normalizer

_window_spec = importlib.util.spec_from_file_location('certified_window_run', prior.HERE / 'run.py')
window = importlib.util.module_from_spec(_window_spec)
_window_spec.loader.exec_module(window)
WindowA2C = window.WindowA2C


class TableWindowA2C(WindowA2C):
    """Use the untouched window update with exact cached states and mappings."""
    def __init__(self, cfg, circuit_name, seed, folder, group, resume=False):
        super().__init__(cfg, cfg['protocol']['circuits'][circuit_name], seed, folder, resume)
        previous = self.game
        self.game = TableSession(cfg, circuit_name, folder, group,
                                 completed=self.episodes_completed, best=previous.best)


class ABCRewardA2C(WindowA2C):
    """Use the untouched window update with the reward-only ABC environment."""
    def __init__(self, cfg, circuit_name, seed, folder, resume=False):
        super().__init__(cfg, cfg['protocol']['circuits'][circuit_name], seed, folder, resume)
        previous = self.game
        self.game = RewardSession(cfg, cfg['protocol']['circuits'][circuit_name], folder)
        self.game.episode = self.episodes_completed
        self.game.best = previous.best
        self.game.best_netlists = previous.best_netlists


class TenStepRewardA2C(ABCRewardA2C):
    """Identical window update, generalized from four to ten actions."""
    def _update(self, states, actions, rewards):
        raw = np.empty(len(rewards), dtype=np.float32)
        cumulative = 0.0
        for index in reversed(range(len(rewards))):
            cumulative = rewards[index] + self.method['gamma'] * cumulative
            raw[index] = cumulative
        if len(raw) != self.protocol['iterations']:
            raise ValueError('Return vector does not match the episode length.')
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
                      actor_update_rms=prior.old.rms(prior.old.flattened(self.network.actor) - actor_before),
                      critic_loss=float(critic.detach()))
        if not all(math.isfinite(value) for value in metric.values() if isinstance(value, float)):
            raise ValueError('Nonfinite ten-step return diagnostic.')
        with self.metrics_path.open('a') as stream:
            stream.write(json.dumps(metric, allow_nan=False) + '\n')


class UniformA2C(A2C):
    """Original no-update agent, sampling each action from uniform probabilities."""
    def run_episode(self):
        if self.episodes_completed >= self.protocol['episodes']:
            raise RuntimeError('The prescribed budget is already complete.')
        started = time.perf_counter()
        state = self.game.reset()
        normalizer = Normalizer(len(state), self.method['normalization'],
                                self.method['features'], state)
        rewards = []
        done = False
        while not done:
            state = normalizer.normalize(state)
            with torch.no_grad():
                logits, _ = self.network(torch.as_tensor(state, device='cpu'))
                action = torch.multinomial(torch.full_like(logits, 1 / len(logits)),
                                           1, generator=self.rng).item()
            state, reward, done = self.game.step(action)
            rewards.append(reward)
        self.training_seconds += time.perf_counter() - started
        self.episodes_completed += 1
        self.rewards.append(sum(rewards))
        self.save_model()
        return self.rewards[-1]


def check_audit():
    result = read(HERE / 'audit.json')
    fingerprint, _ = identity()
    if result['status'] != 'pass' or result['fingerprint'] != fingerprint:
        raise ValueError('Reward audit is missing or belongs to different sources.')


def table_or_abc():
    equivalent = read(OUT / 'equivalence.json')
    fingerprint, _ = identity()
    if equivalent['fingerprint'] != fingerprint or equivalent['status'] != 'complete':
        raise ValueError('Missing or stale lookup equivalence check.')
    return 'table' if equivalent['table_equivalent'] else 'abc'


def validate_pilot():
    pilot = read(OUT / 'pilot.json')
    fingerprint, _ = identity()
    if pilot['fingerprint'] != fingerprint or pilot['status'] != 'pass':
        raise ValueError('The 20-episode scale pilot has not passed.')


def replay_table_best(cfg, circuit_name, folder, best, group):
    """Recreate only the best prefix in ABC, then verify the mapped netlist."""
    circuit = cfg['protocol']['circuits'][circuit_name]
    cls = RewardSession if group == 'reward' else FPGASession
    replay = cls(cfg, circuit, folder / 'best-replay')
    replay.reset()
    actions = cfg['protocol']['actions']
    for name in best['sequence'][1:]:
        replay.step(actions.index(name))
    if (replay.luts, replay.levels) != (best['luts'], best['levels']):
        raise ValueError('ABC replay differs from the table best result.')
    for source, target in (('current.v', 'best.v'), ('mapped.v', 'best-mapped.v')):
        shutil.copyfile(replay.episode_dir / source, folder / target)
    verify_netlist(cfg, circuit, folder / 'best-mapped.v', best)


def _agent_for(phase, group, circuit, seed, folder, cfg, resume, environment):
    if phase in ('parity', 'pilot', 'four'):
        if environment == 'table':
            return TableWindowA2C(cfg, circuit, seed, folder, group, resume)
        if group == 'window':
            return WindowA2C(cfg, cfg['protocol']['circuits'][circuit], seed, folder, resume)
        return ABCRewardA2C(cfg, circuit, seed, folder, resume)
    if group == 'reward':
        return TenStepRewardA2C(cfg, circuit, seed, folder, resume)
    cfg['method']['learning_enabled'] = False
    cls = UniformA2C if group == 'uniform' else A2C
    return cls(cfg, cfg['protocol']['circuits'][circuit], seed, folder, resume)


def _task_folder(phase, group, circuit, seed):
    if phase == 'parity':
        return OUT / 'parity' / circuit / f'seed-{seed}'
    if phase == 'pilot':
        return OUT / 'pilot' / circuit / f'seed-{seed}'
    return OUT / ('ten-step' if phase == 'ten' else 'four-step') / group / circuit / f'seed-{seed}'


def train_task(phase, group, circuit, seed):
    fingerprint, _ = identity()
    ten_step = phase == 'ten'
    cfg = training_config(circuit, group, ten_step=ten_step)
    if phase == 'pilot':
        cfg['protocol']['episodes'] = SPEC['pilot']['episodes']
    folder = _task_folder(phase, group, circuit, seed)
    result_path = folder / 'result.json'
    if result_path.exists():
        existing = read(result_path)
        if existing['status'] == 'complete' and existing['fingerprint'] == fingerprint:
            print(f'Already complete: {phase}/{group}/{circuit}/{seed}', flush=True)
            return
        raise ValueError(f'Conflicting result: {result_path}')
    folder.mkdir(parents=True, exist_ok=True)
    completed = 0
    environment = ('table' if phase == 'parity' else 'abc' if ten_step else table_or_abc())
    try:
        agent = _agent_for(phase, group, circuit, seed, folder, cfg,
                           (folder / 'checkpoint.pt').exists(), environment)
        completed = agent.episodes_completed
        for _ in range(completed, cfg['protocol']['episodes']):
            agent.run_episode()
            completed = agent.episodes_completed
            if completed % (25 if not ten_step else 10) == 0:
                print(f'{phase}/{group}/{circuit}/seed-{seed}: '
                      f'{completed}/{cfg["protocol"]["episodes"]}', flush=True)
            dump(folder / 'status.json', dict(status='running', episodes=completed))
        if phase not in ('parity', 'pilot'):
            if environment == 'table':
                replay_table_best(cfg, circuit, folder, agent.game.best, group)
            else:
                verify_netlist(cfg, cfg['protocol']['circuits'][circuit],
                               folder / 'best-mapped.v', agent.game.best)
        initial = prior.old.load(folder / 'snapshots/0.pt') if group in ('window', 'reward') else None
        final = prior.old.load(agent.checkpoint)
        result = dict(status='complete', fingerprint=fingerprint, phase=phase, group=group,
                      circuit=circuit, seed=seed, environment=environment,
                      episodes_completed=completed,
                      optimization_actions=completed * cfg['protocol']['iterations'],
                      mapping_calls=completed * (cfg['protocol']['iterations'] + 1),
                      best=agent.game.best, training_seconds=agent.training_seconds,
                      initial_network_sha256=prior.old.state_hash(initial['network']) if initial else None,
                      final_network_sha256=prior.old.state_hash(final['network']),
                      checkpoint_sha256=sha(agent.checkpoint))
        dump(result_path, result)
        dump(folder / 'status.json', dict(status='complete', episodes=completed))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', episodes=completed,
                                        error=repr(error), traceback=traceback.format_exc()))
        raise


def _supervise(phase, jobs):
    fingerprint, payload = identity()
    manifest_path = OUT / f'{phase}-manifest.json'
    if manifest_path.exists() and read(manifest_path)['fingerprint'] != fingerprint:
        raise ValueError(f'Cannot resume {phase} with changed source fingerprint.')
    pending = []
    for group, circuit, seed in jobs:
        path = _task_folder(phase, group, circuit, seed) / 'result.json'
        if path.exists() and read(path).get('status') == 'complete':
            if read(path).get('fingerprint') != fingerprint:
                raise ValueError(f'Stale completed result: {path}')
            continue
        argv = [sys.executable, '-B', '-u', str(HERE / 'run.py'), '--task',
                phase, group, circuit, str(seed)]
        pending.append((f'{group}-{circuit}-{seed}', argv))
    manifest = dict(status='running', phase=phase, fingerprint=fingerprint,
                    source_payload=payload, jobs=[list(job) for job in jobs])
    dump(manifest_path, manifest)
    processes = prior.old.supervise(pending, OUT / 'logs' / phase,
                                    workers=SPEC['workers'], timeout=SPEC['task_timeout_seconds'])
    missing = [list(job) for job in jobs if not (_task_folder(phase, *job) / 'result.json').exists()]
    manifest.update(status='complete' if not missing and
                    all(row['exit_code'] == 0 for row in processes) else 'incomplete',
                    processes=processes, missing=missing)
    dump(manifest_path, manifest)
    if manifest['status'] != 'complete':
        raise RuntimeError(f'{phase} incomplete; inspect manifest and per-task logs.')


def run_parity():
    check_audit()
    _supervise('parity', [('window', 'i2c', seed) for seed in SEEDS4])
    rows = []
    for seed in SEEDS4:
        current = read(_task_folder('parity', 'window', 'i2c', seed) / 'result.json')
        reference = read(prior.OUT / 'training/i2c' / f'seed-{seed}/result.json')
        rows.append(dict(seed=seed, reproduced=current['final_network_sha256'],
                         original=reference['final_network_sha256'],
                         initial_match=current['initial_network_sha256'] ==
                         reference['initial_network_sha256'],
                         final_match=current['final_network_sha256'] ==
                         reference['final_network_sha256'],
                         best_match={k: current['best'][k] == reference['best'][k]
                                     for k in ('luts', 'levels', 'feasible', 'sequence')}))
    equivalent = all(row['initial_match'] and row['final_match'] for row in rows)
    digest, _ = identity()
    dump(OUT / 'equivalence.json', dict(status='complete', fingerprint=digest,
                                        table_equivalent=equivalent, rows=rows,
                                        selected_environment='table' if equivalent else 'abc'))
    print('Table environment:', 'equivalent' if equivalent else 'ABC fallback', flush=True)


def run_pilot():
    check_audit()
    environment = table_or_abc()
    _supervise('pilot', [('reward', circuit, SPEC['pilot']['seed'])
                         for circuit in SPEC['circuits']])
    rows = []
    for circuit in SPEC['circuits']:
        folder = _task_folder('pilot', 'reward', circuit, SPEC['pilot']['seed'])
        diagnostics = [json.loads(line) for line in
                       (folder / 'return-diagnostics.jsonl').read_text().splitlines()]
        recent = diagnostics[SPEC['warmup_episodes']:]
        mean_advantage = float(np.mean([row['mean_abs_advantage'] for row in recent]))
        finite = len(diagnostics) == SPEC['pilot']['episodes'] and all(
            math.isfinite(row['mean_abs_advantage']) and math.isfinite(row['actor_update_rms'])
            for row in diagnostics)
        changed = any(row['actor_update_rms'] > 0 for row in diagnostics)
        rows.append(dict(circuit=circuit, mean_abs_advantage=mean_advantage,
                         finite=finite, actor_changed=changed))
    i2c = next(row for row in rows if row['circuit'] == 'i2c')
    pass_scale = (SPEC['pilot']['same_order_lower'] <= i2c['mean_abs_advantage'] <=
                  SPEC['pilot']['same_order_upper'])
    passed = pass_scale and all(row['finite'] and row['actor_changed'] for row in rows)
    digest, _ = identity()
    dump(OUT / 'pilot.json', dict(status='pass' if passed else 'failed', fingerprint=digest,
                                  environment=environment, pass_scale=pass_scale, rows=rows))
    if not passed:
        raise RuntimeError('Pilot failed; no formal training will start.')


def run_four():
    check_audit()
    validate_pilot()
    jobs = [('window', circuit, seed) for circuit in ('max', 'int2float') for seed in SEEDS4]
    jobs += [('reward', circuit, seed) for circuit in SPEC['circuits'] for seed in SEEDS4]
    _supervise('four', jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('parity', 'pilot', 'four', 'ten'))
    parser.add_argument('--task', nargs=4, metavar=('PHASE', 'GROUP', 'CIRCUIT', 'SEED'))
    args = parser.parse_args()
    if args.task:
        phase, group, circuit, raw_seed = args.task
        seed = int(raw_seed)
        if phase not in ('parity', 'pilot', 'four', 'ten') or circuit not in SPEC['circuits']:
            raise ValueError('Task outside fixed protocol.')
        if phase == 'parity' and (group, circuit, seed) not in [('window', 'i2c', s) for s in SEEDS4]:
            raise ValueError('Parity task outside fixed protocol.')
        if phase == 'pilot' and (group != 'reward' or seed != SPEC['pilot']['seed']):
            raise ValueError('Pilot task outside fixed protocol.')
        if phase == 'four' and (group not in ('window', 'reward') or seed not in SEEDS4):
            raise ValueError('Four-step task outside fixed protocol.')
        if phase == 'ten' and (group not in ('reward', 'frozen', 'uniform') or
                               seed not in SEEDS10 or circuit not in SPEC['ten_step']['circuits_in_priority_order']):
            raise ValueError('Ten-step task outside fixed protocol.')
        train_task(phase, group, circuit, seed)
        return
    if args.phase == 'parity':
        run_parity()
    elif args.phase == 'pilot':
        run_pilot()
    elif args.phase == 'four':
        run_four()
    elif args.phase == 'ten':
        from ten import run_ten
        run_ten(_supervise)
    else:
        parser.error('--phase or --task is required')


if __name__ == '__main__':
    main()

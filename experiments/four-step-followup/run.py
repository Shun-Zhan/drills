"""Train the 250x4 A2C variant with a lagged eight-episode return window."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

from filelock import FileLock
import numpy as np
import torch

from support import (HERE, OLD_ROOT, OUT, SEEDS, SNAPSHOTS, SPEC, dump, identity,
                     old, read, sha, training_config)
from drills.experiment import verify_netlist


class WindowA2C(old.A2C):
    """Change only return normalization; preserve one optimizer step per episode."""

    def __init__(self, config, circuit, seed, directory, resume=False):
        self.folder = Path(directory)
        self.metrics_path = self.folder / 'return-diagnostics.jsonl'
        self.return_window = []
        super().__init__(config, circuit, seed, directory, resume)
        if resume:
            saved = old.load(self.checkpoint)
            if 'return_window' not in saved:
                raise ValueError('A rolling-window checkpoint lacks its return history.')
            self.return_window = saved['return_window']
            if len(self.return_window) != min(self.episodes_completed, SPEC['return_window']):
                raise ValueError('Checkpoint return history disagrees with completed episodes.')
            records = ([json.loads(line) for line in self.metrics_path.read_text().splitlines()]
                       if self.metrics_path.exists() else [])
            records = [row for row in records if row['episode'] <= self.episodes_completed]
            if [row['episode'] for row in records] != list(range(1, self.episodes_completed + 1)):
                raise ValueError('Committed return diagnostics are incomplete.')
            self.metrics_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n'
                                                for row in records))
        elif self.metrics_path.exists():
            raise FileExistsError(self.metrics_path)
        if self.episodes_completed in SNAPSHOTS:
            target = self.folder / 'snapshots' / f'{self.episodes_completed}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(self.checkpoint, target)

    def _update(self, states, actions, rewards):
        raw = np.empty(len(rewards), dtype=np.float32)
        cumulative = 0.0
        for index in reversed(range(len(rewards))):
            cumulative = rewards[index] + self.method['gamma'] * cumulative
            raw[index] = cumulative
        if len(raw) != 4:
            raise ValueError('The rolling return protocol requires four steps.')
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
        actor_before = old.flattened(self.network.actor)
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
                      actor_update_rms=old.rms(old.flattened(self.network.actor) - actor_before),
                      critic_loss=float(critic.detach()))
        if not all(math.isfinite(value) for value in metric.values() if isinstance(value, float)):
            raise ValueError('Nonfinite rolling-return diagnostic.')
        with self.metrics_path.open('a') as stream:
            stream.write(json.dumps(metric, allow_nan=False) + '\n')

    def save_model(self):
        # Keep the rolling statistics in the same atomic checkpoint as the weights.
        temporary = self.checkpoint.with_suffix('.tmp')
        torch.save(dict(network=self.network.state_dict(), optimizer=self.optimizer.state_dict(),
                        rng_state=self.rng.get_state(), episodes_completed=self.episodes_completed,
                        rewards=self.rewards, training_seconds=self.training_seconds,
                        learning_enabled=self.method.get('learning_enabled', True),
                        experiment_group=self.config['runtime'].get('experiment_group'),
                        experiment_fingerprint=self.config['runtime'].get('experiment_fingerprint'),
                        state_normalization=dict(mode=self.method['normalization']['state'],
                                                 features=self.method['features']),
                        best=self.game.best, best_netlists=self.game.best_netlists,
                        return_window=self.return_window), temporary)
        temporary.replace(self.checkpoint)
        self.game.export_best()
        if self.episodes_completed in SNAPSHOTS:
            target = self.folder / 'snapshots' / f'{self.episodes_completed}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.checkpoint, target)


def train_task(cfg, circuit, seed, fingerprint, resume):
    folder = OUT / 'training' / circuit / f'seed-{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    if resume and result_path.exists() and read(result_path).get('status') == 'complete':
        print(f'Already complete: {circuit}/seed-{seed}', flush=True)
        return
    completed = 0
    try:
        agent = WindowA2C(cfg, cfg['protocol']['circuits'][circuit], seed, folder,
                          resume=resume and (folder / 'checkpoint.pt').exists())
        completed = agent.episodes_completed
        for _ in range(completed, cfg['protocol']['episodes']):
            agent.run_episode()
            completed = agent.episodes_completed
            if completed % 25 == 0:
                print(f'rolling8/{circuit}/seed-{seed}: {completed}/250', flush=True)
            dump(folder / 'status.json', dict(status='running', episodes=completed, time=old.now()))
        verify_netlist(cfg, cfg['protocol']['circuits'][circuit], folder / 'best-mapped.v',
                       agent.game.best)
        initial = old.load(folder / 'snapshots/0.pt')
        final = old.load(folder / 'snapshots/250.pt')
        control = old.load(OLD_ROOT / 'training/lr010' / circuit /
                           f'seed-{seed}/snapshots/0.pt')
        if old.state_hash(initial['network']) != old.state_hash(control['network']):
            raise ValueError('Rolling-window and original LR=0.01 initial weights differ.')
        result = dict(status='complete', fingerprint=fingerprint, circuit=circuit, seed=seed,
                      episodes_completed=completed, optimization_actions=completed * 4,
                      best=agent.game.best, training_seconds=agent.training_seconds,
                      initial_network_sha256=old.state_hash(initial['network']),
                      final_network_sha256=old.state_hash(final['network']),
                      checkpoint_sha256=sha(agent.checkpoint))
        dump(result_path, result)
        dump(folder / 'status.json', dict(status='complete', episodes=completed, time=old.now()))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', episodes=completed,
                                        error=repr(error), traceback=traceback.format_exc(),
                                        time=old.now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuit', choices=('i2c', 'int2float', 'max'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--task', nargs=2, metavar=('CIRCUIT', 'SEED'))
    args = parser.parse_args()
    fingerprint, payload = identity()
    cfg = training_config(fingerprint)
    if args.task:
        circuit, raw_seed = args.task
        seed = int(raw_seed)
        if circuit not in cfg['protocol']['circuits'] or seed not in SEEDS:
            raise ValueError('Training task outside the predeclared follow-up protocol.')
        train_task(cfg, circuit, seed, fingerprint, args.resume)
        return
    if args.circuit is None:
        parser.error('--circuit is required outside --task mode')
    manifest_path = OUT / f'train-{args.circuit}.json'
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f'Existing training manifest requires --resume: {manifest_path}')
    if args.resume and (not manifest_path.exists() or
                        read(manifest_path)['fingerprint'] != fingerprint):
        raise ValueError('Resume requires the identical follow-up source fingerprint.')
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT / '.run.lock'), timeout=0):
        manifest = dict(status='running', fingerprint=fingerprint, sources=payload['sources'],
                        original_fingerprint=payload['original_fingerprint'],
                        circuit=args.circuit, seeds=list(SEEDS), started_at=old.now())
        if args.resume:
            manifest['previous_invocation'] = read(manifest_path)
        dump(manifest_path, manifest)
        tasks = []
        for seed in SEEDS:
            result = OUT / 'training' / args.circuit / f'seed-{seed}/result.json'
            if args.resume and result.exists() and read(result).get('status') == 'complete':
                continue
            command = [sys.executable, '-B', '-u', str(HERE / 'run.py'),
                       '--task', args.circuit, str(seed)]
            if args.resume:
                command.append('--resume')
            tasks.append((f'{args.circuit}-{seed}', command))
        processes = old.supervise(tasks, OUT / 'logs' / args.circuit,
                                  workers=SPEC['workers'], timeout=SPEC['task_timeout_seconds'])
        missing = [seed for seed in SEEDS if not (OUT / 'training' / args.circuit /
                   f'seed-{seed}/result.json').exists()]
        complete = not missing and all(row['exit_code'] == 0 for row in processes)
        manifest.update(status='complete' if complete else 'incomplete', processes=processes,
                        missing=missing, finished_at=old.now())
        dump(manifest_path, manifest)
        if not complete:
            raise SystemExit('Follow-up training incomplete; inspect per-seed status and logs.')


if __name__ == '__main__':
    main()

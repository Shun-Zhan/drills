"""Run 250x4 paired A2C learning-rate searches and uniform searches."""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

from filelock import FileLock
import numpy as np
import torch
from torch import nn

from common import (ASSESSMENT, GROUPS, HERE, ROOT, SNAPSHOTS, ActorCritic, A2C,
                    FPGASession, config, create_probes, dump, flattened, group_config,
                    identity, load, now, probe_path, read, rms, sha, state_hash, supervise)
from drills.experiment import verify_netlist


class DiagnosticA2C(A2C):
    """Run the original A2C path; observe parameters after each original update."""

    def __init__(self, cfg, circuit, seed, folder, probes, resume=False):
        self.folder = Path(folder)
        self.probes = torch.as_tensor(np.load(probes, allow_pickle=False), dtype=torch.float32)
        self.probe_sha256 = sha(probes)
        self.metrics_path = self.folder / 'metrics.jsonl'
        self.last_update = dict(actor=0.0, critic=0.0)
        super().__init__(cfg, circuit, seed, folder, resume)
        initial_file = self.folder / 'snapshots/0.pt'
        if not initial_file.exists():
            if self.episodes_completed != 0:
                raise ValueError('Missing initial model snapshot during a resumed run.')
            initial_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.checkpoint, initial_file)
        self.reference = copy.deepcopy(self.network)
        self.reference.load_state_dict(load(initial_file)['network'])
        self.initial_actor = flattened(self.reference.actor)
        self.initial_critic = flattened(self.reference.critic)
        if resume:
            rows = [json.loads(line) for line in self.metrics_path.read_text().splitlines()] if self.metrics_path.exists() else []
            rows = [row for row in rows if row['episode'] <= self.episodes_completed]
            if rows and [row['episode'] for row in rows] != list(range(rows[-1]['episode'] + 1)):
                raise ValueError('Metric episodes are incomplete or duplicated.')
            self.metrics_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
            if not rows and self.episodes_completed == 0:
                self._append_metric()
            elif not rows or rows[-1]['episode'] != self.episodes_completed:
                raise ValueError('Committed checkpoint lacks its diagnostic row.')
            if self.episodes_completed in SNAPSHOTS:
                target = self.folder / 'snapshots' / f'{self.episodes_completed}.pt'
                if not target.exists():
                    shutil.copyfile(self.checkpoint, target)
        else:
            self._append_metric()

    def _update(self, states, actions, rewards):
        before = {name: flattened(getattr(self.network, name)) for name in ('actor', 'critic')}
        super()._update(states, actions, rewards)
        self.last_update = {name: rms(flattened(getattr(self.network, name)) - before[name])
                            for name in ('actor', 'critic')}

    def _append_metric(self):
        with torch.no_grad():
            logits, _ = self.network(self.probes)
            logp = logits.log_softmax(-1)
            probs = logp.exp()
            entropy = float((-probs * logp).sum(-1).mean())
        row = dict(episode=self.episodes_completed, entropy=entropy,
                   probe_sha256=self.probe_sha256)
        for name in ('actor', 'critic'):
            module = getattr(self.network, name)
            weights = []
            layers = {}
            for layer_name, layer in module.named_modules():
                if isinstance(layer, nn.Linear):
                    values = layer.weight.detach().double().flatten()
                    weights.append(values)
                    layers[layer_name] = float(values.var(unbiased=False))
            row[name + '_weight_variance'] = float(torch.cat(weights).var(unbiased=False))
            row[name + '_layer_variance'] = layers
            row[name + '_drift_rms'] = rms(flattened(module) - getattr(self, 'initial_' + name))
            row[name + '_update_rms'] = self.last_update[name]
        scalars = [value for value in row.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError('Nonfinite weight or entropy diagnostic.')
        with self.metrics_path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def save_model(self):
        # The metric is committed before the checkpoint. Resume discards an uncommitted tail.
        if hasattr(self, 'initial_actor'):
            self._append_metric()
        super().save_model()
        if hasattr(self, 'initial_actor') and self.episodes_completed in SNAPSHOTS:
            target = self.folder / 'snapshots' / f'{self.episodes_completed}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.checkpoint, target)


def save_uniform(folder, game, generator, episodes_completed, fingerprint):
    target = Path(folder) / 'checkpoint.pt'
    temporary = target.with_suffix('.tmp')
    torch.save(dict(episodes_completed=episodes_completed, rng_state=generator.get_state(),
                    best=game.best, best_netlists=game.best_netlists, fingerprint=fingerprint), temporary)
    temporary.replace(target)
    game.export_best()


def uniform_task(cfg, name, seed, folder, fingerprint, resume):
    game = FPGASession(cfg, cfg['protocol']['circuits'][name], folder)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    checkpoint = folder / 'checkpoint.pt'
    if resume and checkpoint.exists():
        saved = load(checkpoint)
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Uniform checkpoint belongs to a different experiment.')
        completed = saved['episodes_completed']
        generator.set_state(saved['rng_state'])
        game.best, game.best_netlists, game.episode = saved['best'], saved['best_netlists'], completed
        game.export_best()
    else:
        completed = 0
        save_uniform(folder, game, generator, completed, fingerprint)
    for episode in range(completed, cfg['protocol']['episodes']):
        game.reset()
        for _ in range(cfg['protocol']['iterations']):
            action = torch.multinomial(torch.full((7,), 1 / 7), 1, generator=generator).item()
            game.step(action)
        save_uniform(folder, game, generator, episode + 1, fingerprint)
        if (episode + 1) % 25 == 0:
            print(f'uniform/{name}/seed-{seed}: {episode + 1}/250', flush=True)
    verify_netlist(cfg, cfg['protocol']['circuits'][name], folder / 'best-mapped.v', game.best)
    return dict(group='uniform', circuit=name, seed=seed, status='complete',
                episodes_completed=cfg['protocol']['episodes'], best=game.best,
                checkpoint_sha256=sha(checkpoint))


def train_task(cfg, group, name, seed, fingerprint, resume):
    local = group_config(cfg, group, fingerprint)
    folder = Path(local['runtime']['output_dir']) / name / f'seed-{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if resume and (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'complete':
        print(f'Already complete: {group}/{name}/{seed}', flush=True)
        return
    completed = 0
    try:
        if group == 'uniform':
            result = uniform_task(local, name, seed, folder, fingerprint, resume)
        else:
            agent = DiagnosticA2C(local, local['protocol']['circuits'][name], seed, folder,
                                  probe_path(cfg, name), resume and (folder / 'checkpoint.pt').exists())
            completed = agent.episodes_completed
            for _ in range(completed, local['protocol']['episodes']):
                agent.run_episode()
                completed = agent.episodes_completed
                if completed % 25 == 0:
                    print(f'{group}/{name}/seed-{seed}: {completed}/250', flush=True)
                dump(folder / 'status.json', dict(status='running', episodes=completed, time=now()))
            verify_netlist(local, local['protocol']['circuits'][name], folder / 'best-mapped.v', agent.game.best)
            initial, final = load(folder / 'snapshots/0.pt'), load(agent.checkpoint)
            result = dict(group=group, circuit=name, seed=seed, status='complete',
                          episodes_completed=completed, best=agent.game.best,
                          training_seconds=agent.training_seconds,
                          initial_network_sha256=state_hash(initial['network']),
                          final_network_sha256=state_hash(final['network']),
                          checkpoint_sha256=sha(agent.checkpoint))
        dump(folder / 'result.json', result)
        dump(folder / 'status.json', dict(status='complete', episodes=250, time=now()))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', episodes=completed,
                                        error=repr(error), traceback=traceback.format_exc(), time=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--task', nargs=3, metavar=('GROUP', 'CIRCUIT', 'SEED'))
    args = parser.parse_args()
    cfg = config()
    fingerprint, evidence = identity(cfg)
    if args.task:
        group, name, raw_seed = args.task
        seed = int(raw_seed)
        if group not in GROUPS or name not in cfg['protocol']['circuits'] or seed not in cfg['protocol']['seeds']:
            raise ValueError('Training task outside the predeclared protocol.')
        train_task(cfg, group, name, seed, fingerprint, args.resume)
        return
    root = Path(cfg['runtime']['output_dir'])
    manifest_file = root / 'experiment.json'
    if root.exists() and not args.resume:
        raise FileExistsError(f'Existing experiment output requires --resume: {root}')
    if args.resume and (not manifest_file.exists() or read(manifest_file)['fingerprint'] != fingerprint):
        raise ValueError('Resume requires identical source, tools, circuits, and protocol.')
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.suite.lock'), timeout=0):
        probes = {name: sha(create_probes(cfg, name)) for name in cfg['protocol']['circuits']}
        if args.resume and read(manifest_file)['probes'] != probes:
            raise ValueError('A fixed state probe bank changed.')
        manifest = dict(fingerprint=fingerprint, **evidence, probes=probes,
                        original_algorithm_commit='adf7bef53cdf6c4161586c03e6e2caecffe26bc4',
                        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        platform=platform.platform(), started_at=now(),
                        tool_versions={key: subprocess.check_output([cfg['runtime'][key], *args], text=True).strip()
                                       for key, args in [('abc_binary', ['-c', 'version']), ('yosys_binary', ['-V'])]})
        if args.resume:
            manifest['previous_invocation'] = read(manifest_file)
        dump(manifest_file, manifest)
        tasks = []
        for group in GROUPS:
            for name in cfg['protocol']['circuits']:
                for seed in cfg['protocol']['seeds']:
                    result_file = root / 'training' / group / name / f'seed-{seed}/result.json'
                    if args.resume and result_file.exists() and read(result_file)['status'] == 'complete':
                        continue
                    argv = [sys.executable, '-B', '-u', str(HERE / 'run.py'), '--task', group, name, str(seed)]
                    if args.resume:
                        argv.append('--resume')
                    tasks.append((f'{group}-{name}-{seed}', argv))
        manifest['processes'] = supervise(tasks, root / 'logs/training', workers=cfg['runtime']['workers'],
                                          timeout=ASSESSMENT['task_timeout_seconds'])
        missing = [f'{g}/{n}/{s}' for g in GROUPS for n in cfg['protocol']['circuits'] for s in cfg['protocol']['seeds']
                   if not (root / 'training' / g / n / f'seed-{s}/result.json').exists()]
        manifest.update(finished_at=now(), missing=missing,
                        status='complete' if not missing and all(r['exit_code'] == 0 for r in manifest['processes'])
                        else 'incomplete')
        dump(manifest_file, manifest)
        if manifest['status'] != 'complete':
            raise SystemExit('Training incomplete; failures retained in the manifest and logs.')


if __name__ == '__main__':
    main()

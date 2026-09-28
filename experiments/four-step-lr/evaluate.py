"""Evaluate each final policy, its initial weights, and uniform actions on ten new rollouts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from filelock import FileLock
import torch

from common import (ASSESSMENT, HERE, ROOT, POLICIES, ActorCritic, FPGASession,
                    Normalizer, config, dump, evaluation_seeds, identity, load,
                    now, read, sha, state_hash, supervise)
from drills.experiment import verify_netlist


def checkpoint_path(root, policy, name, seed):
    if policy == 'uniform':
        return root / 'training/uniform' / name / f'seed-{seed}/checkpoint.pt'
    group = 'lr010' if policy == 'lr010-final' else 'lr001'
    episode = 0 if policy == 'initial' else 250
    return root / 'training' / group / name / f'seed-{seed}/snapshots/{episode}.pt'


def inference_rollout(cfg, circuit, network, evaluation_seed, folder, uniform=False):
    game = FPGASession(cfg, circuit, folder)
    generator = torch.Generator(device='cpu').manual_seed(evaluation_seed)
    state = game.reset()
    normalizer = Normalizer(len(state), cfg['method']['normalization'], cfg['method']['features'], state)
    actions, rewards = [], []
    done = False
    while not done:
        normalized = normalizer.normalize(state)
        with torch.no_grad():
            if uniform:
                probabilities = torch.full((len(cfg['protocol']['actions']),),
                                           1 / len(cfg['protocol']['actions']))
            else:
                logits, _ = network(torch.as_tensor(normalized, device='cpu'))
                probabilities = logits.softmax(-1)
            action = torch.multinomial(probabilities, 1, generator=generator).item()
        state, reward, done = game.step(action)
        actions.append(action)
        rewards.append(reward)
    game.export_best()
    verify_netlist(cfg, circuit, Path(folder) / 'best-mapped.v', game.best)
    return dict(evaluation_seed=evaluation_seed, actions=actions, rewards=rewards,
                best=game.best, terminal=dict(luts=game.luts, levels=game.levels,
                                              feasible=game.levels <= circuit['max_levels']))


def evaluate_task(cfg, policy, name, seed, fingerprint, resume):
    root = Path(cfg['runtime']['output_dir'])
    folder = root / 'evaluation' / policy / name / f'seed-{seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(root, policy, name, seed)
    digest = sha(checkpoint)
    saved = load(checkpoint)
    if policy == 'uniform':
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Uniform checkpoint fingerprint mismatch.')
        network, network_hash = None, None
    else:
        if saved['experiment_fingerprint'] != fingerprint:
            raise ValueError('Model checkpoint fingerprint mismatch.')
        torch.set_num_threads(cfg['environment']['torch_threads'])
        with torch.random.fork_rng(devices=[]):
            network = ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']),
                                  cfg['method']['network'])
        network.load_state_dict(saved['network'])
        network.eval().requires_grad_(False)
        network_hash = state_hash(network.state_dict())
    rows = []
    try:
        for evaluation_seed in evaluation_seeds(seed):
            destination = folder / f'rollout-{evaluation_seed}'
            record = destination / 'rollout.json'
            if resume and record.exists():
                row = read(record)
                if row['checkpoint_sha256'] != digest or row['status'] != 'complete':
                    raise ValueError('Cannot reuse a rollout from another checkpoint.')
            else:
                row = inference_rollout(cfg, cfg['protocol']['circuits'][name], network,
                                        evaluation_seed, destination, uniform=policy == 'uniform')
                row.update(policy=policy, circuit=name, training_seed=seed,
                           checkpoint_sha256=digest, status='complete')
                dump(record, row)
            rows.append(row)
            dump(folder / 'status.json', dict(status='running', completed=len(rows), time=now()))
        if sha(checkpoint) != digest or (network is not None and state_hash(network.state_dict()) != network_hash):
            raise RuntimeError('Independent evaluation changed a checkpoint or model.')
        dump(folder / 'result.json', dict(status='complete', policy=policy, circuit=name, training_seed=seed,
                                         checkpoint_sha256=digest, network_state_sha256=network_hash,
                                         network_unchanged=True, checkpoint_unchanged=True, rollouts=rows))
        dump(folder / 'status.json', dict(status='complete', completed=len(rows), time=now()))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', completed=len(rows), error=repr(error),
                                         traceback=traceback.format_exc(), time=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--task', nargs=3, metavar=('POLICY', 'CIRCUIT', 'SEED'))
    args = parser.parse_args()
    cfg = config()
    fingerprint, _ = identity(cfg)
    root = Path(cfg['runtime']['output_dir'])
    training_manifest = read(root / 'experiment.json')
    if training_manifest['fingerprint'] != fingerprint or training_manifest['status'] != 'complete':
        raise ValueError('Complete training under identical sources is required.')
    if args.task:
        policy, name, raw_seed = args.task
        seed = int(raw_seed)
        if policy not in POLICIES or name not in cfg['protocol']['circuits'] or seed not in cfg['protocol']['seeds']:
            raise ValueError('Evaluation task outside the predeclared protocol.')
        evaluate_task(cfg, policy, name, seed, fingerprint, args.resume)
        return
    target = root / 'evaluation.json'
    if target.exists() and not args.resume:
        raise FileExistsError('Existing independent evaluation requires --resume.')
    if args.resume and (not target.exists() or read(target)['training_fingerprint'] != fingerprint):
        raise ValueError('Resume requires the identical completed training experiment.')
    for name in cfg['protocol']['circuits']:
        for seed in cfg['protocol']['seeds']:
            left = load(root / 'training/lr001' / name / f'seed-{seed}/snapshots/0.pt')
            right = load(root / 'training/lr010' / name / f'seed-{seed}/snapshots/0.pt')
            if state_hash(left['network']) != state_hash(right['network']):
                raise ValueError(f'Learning-rate arms have different initial weights: {name}/{seed}')
    with FileLock(str(root / '.suite.lock'), timeout=0):
        manifest = dict(training_fingerprint=fingerprint, policies=list(POLICIES),
                        attempts_per_model=ASSESSMENT['evaluation_attempts'],
                        evaluation_seed_start=ASSESSMENT['evaluation_seed_start'],
                        uniform_banks_shared_across_training_seeds=False,
                        started_at=now(), command=[sys.executable, '-B', '-u', str(HERE / 'evaluate.py'), *sys.argv[1:]])
        if args.resume:
            manifest['previous_invocation'] = read(target)
        dump(target, manifest)
        tasks = []
        for policy in POLICIES:
            for name in cfg['protocol']['circuits']:
                for seed in cfg['protocol']['seeds']:
                    result_file = root / 'evaluation' / policy / name / f'seed-{seed}/result.json'
                    if args.resume and result_file.exists() and read(result_file)['status'] == 'complete':
                        continue
                    argv = [sys.executable, '-B', '-u', str(HERE / 'evaluate.py'), '--task', policy, name, str(seed)]
                    if args.resume:
                        argv.append('--resume')
                    tasks.append((f'{policy}-{name}-{seed}', argv))
        manifest['processes'] = supervise(tasks, root / 'logs/evaluation', workers=cfg['runtime']['workers'],
                                          timeout=ASSESSMENT['task_timeout_seconds'])
        missing = [f'{p}/{n}/{s}' for p in POLICIES for n in cfg['protocol']['circuits'] for s in cfg['protocol']['seeds']
                   if not (root / 'evaluation' / p / n / f'seed-{s}/result.json').exists()]
        manifest.update(finished_at=now(), missing=missing,
                        status='complete' if not missing and all(r['exit_code'] == 0 for r in manifest['processes'])
                        else 'incomplete')
        dump(target, manifest)
        if manifest['status'] != 'complete':
            raise SystemExit('Independent evaluation incomplete; failures retained.')


if __name__ == '__main__':
    main()

"""Train the fixed four-step candidate to 1000 episodes and evaluate it exactly."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

from filelock import FileLock
import torch

from study import (FOUR, HERE, OLD_REWARD, OUT, ROOT, dump, identity, read, sha)

REWARD = ROOT / 'experiments/four-step-reward'
sys.path.insert(0, str(REWARD))
_run_spec = importlib.util.spec_from_file_location('prior_four_step_runner', REWARD / 'run.py')
prior = importlib.util.module_from_spec(_run_spec)
_run_spec.loader.exec_module(prior)
context = prior
supervise = prior.prior.old.supervise
state_hash = prior.prior.old.state_hash

_exact_spec = importlib.util.spec_from_file_location(
    'prior_four_step_exact', ROOT / 'experiments/four-step-followup/exact.py')
exact = importlib.util.module_from_spec(_exact_spec)
_exact_spec.loader.exec_module(exact)


def folder_for(circuit, seed):
    return OUT / 'stage2/training' / circuit / f'seed-{seed}'


def old_folder(circuit, seed):
    return OLD_REWARD / 'four-step/reward' / circuit / f'seed-{seed}'


def extension_config(circuit):
    cfg = prior.training_config(circuit, 'reward', ten_step=False)
    cfg['protocol']['episodes'] = FOUR['episodes']
    cfg['runtime'].update(output_dir=str(OUT / 'stage2/training'),
                          experiment_group='extension_reward',
                          experiment_fingerprint=identity(2)[0])
    return cfg


def preflight():
    if (tuple(FOUR['circuits']) != ('int2float', 'i2c', 'max') or
            tuple(FOUR['seeds']) != tuple(range(10)) or
            FOUR['episodes'] != 1000 or FOUR['iterations'] != 4 or
            FOUR['snapshots'] != [0, 250, 500, 750, 1000] or
            FOUR['learning_rate'] != 0.01):
        raise ValueError('The 1000x4 extension protocol changed.')
    prior.check_audit()
    if prior.table_or_abc() != 'table':
        raise ValueError('Certified four-step lookup environment is unavailable.')
    old_digest, old_payload = prior.identity()
    manifest = read(OLD_REWARD / 'four-manifest.json')
    if manifest['status'] != 'complete' or manifest['fingerprint'] != old_digest:
        raise ValueError('Historical 250-episode training is incomplete or stale.')
    old_sources = {}
    for circuit in FOUR['circuits']:
        for seed in FOUR['seeds']:
            path = old_folder(circuit, seed)
            result = read(path / 'result.json')
            checkpoint = path / 'snapshots/250.pt'
            if (result['status'] != 'complete' or result['fingerprint'] != old_digest or
                    result['episodes_completed'] != 250 or not checkpoint.is_file()):
                raise ValueError(f'Historical 250-episode result invalid: {path}')
            old_sources[f'{circuit}/{seed}'] = sha(checkpoint)
    digest, payload = identity(2)
    target = OUT / 'stage2/preflight.json'
    if target.exists() and read(target)['fingerprint'] != digest:
        raise ValueError('A stage-two result exists with another fingerprint.')
    dump(target, dict(status='pass', fingerprint=digest, source_payload=payload,
                      historical_fingerprint=old_digest,
                      historical_source_payload=old_payload,
                      checkpoint_250_sha256=old_sources))
    print('Stage 2 preflight:', digest)


def verify_250(circuit, seed, checkpoint):
    prior_path = old_folder(circuit, seed) / 'snapshots/250.pt'
    expected_sha = read(OUT / 'stage2/preflight.json')['checkpoint_250_sha256'][f'{circuit}/{seed}']
    if sha(prior_path) != expected_sha:
        raise ValueError(f'Historical checkpoint changed: {circuit}/{seed}')
    new = torch.load(checkpoint, map_location='cpu', weights_only=True)
    old = torch.load(prior_path, map_location='cpu', weights_only=True)
    for key in ('network', 'optimizer', 'rng_state', 'return_window', 'rewards', 'best'):
        if state_hash(new[key]) != state_hash(old[key]):
            raise ValueError(f'250-episode state differs: {circuit}/{seed}/{key}')
    if new['episodes_completed'] != 250 or old['episodes_completed'] != 250:
        raise ValueError('250-episode checkpoint count differs.')


def train_task(circuit, seed):
    if circuit not in FOUR['circuits'] or seed not in FOUR['seeds']:
        raise ValueError('Training task outside the four-step protocol.')
    digest, _ = identity(2)
    if read(OUT / 'stage2/preflight.json')['fingerprint'] != digest:
        raise ValueError('Run stage-two preflight first.')
    folder = folder_for(circuit, seed)
    result_path = folder / 'result.json'
    if result_path.exists():
        result = read(result_path)
        if result['status'] == 'complete' and result['fingerprint'] == digest:
            return
        raise ValueError(f'Conflicting training result: {result_path}')
    folder.mkdir(parents=True, exist_ok=True)
    cfg = extension_config(circuit)
    completed = 0
    try:
        agent = prior.TableWindowA2C(cfg, circuit, seed, folder, 'reward',
                                     resume=(folder / 'checkpoint.pt').exists())
        completed = agent.episodes_completed
        if completed >= 250:
            verify_250(circuit, seed, folder / 'snapshots/250.pt')
        for _ in range(completed, FOUR['episodes']):
            agent.run_episode()
            completed = agent.episodes_completed
            if completed in FOUR['snapshots']:
                target = folder / 'snapshots' / f'{completed}.pt'
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copyfile(agent.checkpoint, target)
                if completed == 250:
                    verify_250(circuit, seed, target)
            if completed % 100 == 0:
                print(f'{circuit}/{seed}: {completed}/1000', flush=True)
            dump(folder / 'status.json', dict(status='running', episodes=completed))
        prior.replay_table_best(cfg, circuit, folder, agent.game.best, 'reward')
        dump(result_path, dict(status='complete', fingerprint=digest, circuit=circuit,
                               seed=seed, episodes_completed=completed,
                               lookup_mappings=completed * 5, best=agent.game.best,
                               training_seconds=agent.training_seconds,
                               checkpoint_sha256=sha(agent.checkpoint),
                               best_mapped_sha256=sha(folder / 'best-mapped.v'),
                               equivalence_log_sha256=sha(folder / 'equivalence.log'),
                               snapshot_sha256={str(ep): sha(folder / 'snapshots' / f'{ep}.pt')
                                                for ep in FOUR['snapshots']}))
        dump(folder / 'status.json', dict(status='complete', episodes=completed))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', episodes=completed,
                                         error=repr(error), traceback=traceback.format_exc()))
        raise


def train_all():
    digest, payload = identity(2)
    if read(OUT / 'stage2/preflight.json')['fingerprint'] != digest:
        raise ValueError('Stage-two preflight fingerprint changed.')
    jobs = [(c, s) for c in FOUR['circuits'] for s in FOUR['seeds']]
    pending = [(f'{c}-{s}', [sys.executable, '-B', '-u', str(HERE / 'stage2.py'),
                              '--task', c, str(s)]) for c, s in jobs
               if not (folder_for(c, s) / 'result.json').exists()]
    with FileLock(str(OUT / 'stage2/.train.lock'), timeout=0):
        manifest = dict(status='running', fingerprint=digest, source_payload=payload,
                        jobs=[list(job) for job in jobs])
        dump(OUT / 'stage2/train-manifest.json', manifest)
        processes = supervise(pending, OUT / 'stage2/logs/training',
                              workers=FOUR['workers'], timeout=FOUR['task_timeout_seconds'])
        missing = [list(job) for job in jobs if not (folder_for(*job) / 'result.json').exists()]
        manifest.update(status='complete' if not missing and
                        all(p['exit_code'] == 0 for p in processes) else 'incomplete',
                        processes=processes, missing=missing)
        dump(OUT / 'stage2/train-manifest.json', manifest)
        if manifest['status'] != 'complete':
            raise RuntimeError('Stage-two training is incomplete.')


def evaluate_exact():
    digest, _ = identity(2)
    manifest = read(OUT / 'stage2/train-manifest.json')
    if manifest['status'] != 'complete' or manifest['fingerprint'] != digest:
        raise ValueError('Stage-two training has not completed.')
    cfg, old_fingerprint = prior.prior.old_config()
    torch.set_num_threads(cfg['environment']['torch_threads'])
    table = prior.prior.candidates()
    rows = []
    for circuit in FOUR['circuits']:
        raw = exact.build_feature_cache(cfg, old_fingerprint, table, circuit)
        states = exact.normalized_features(cfg, raw, circuit)
        outcome = exact.outcomes(table, circuit)
        for seed in FOUR['seeds']:
            folder = folder_for(circuit, seed)
            result = read(folder / 'result.json')
            if (result['status'] != 'complete' or result['fingerprint'] != digest or
                    result['episodes_completed'] != 1000 or
                    result['checkpoint_sha256'] != sha(folder / 'checkpoint.pt')):
                raise ValueError(f'Incomplete extension result: {folder}')
            verify_250(circuit, seed, folder / 'snapshots/250.pt')
            for episode in FOUR['snapshots']:
                checkpoint = folder / 'snapshots' / f'{episode}.pt'
                if result['snapshot_sha256'][str(episode)] != sha(checkpoint):
                    raise ValueError(f'Snapshot changed: {checkpoint}')
                network = exact.load_network(cfg, checkpoint, digest)
                probs = exact.action_probabilities(network, states)
                row = exact.evaluate(probs, outcome, circuit, seed,
                                     'w010_new_extended', episode, checkpoint)
                row.pop('curves')
                rows.append(row)
        print('Exact extension evaluation:', circuit, flush=True)
    dump(OUT / 'stage2/exact.json', dict(status='complete', fingerprint=digest,
                                         rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'train', 'exact'))
    parser.add_argument('--task', nargs=2, metavar=('CIRCUIT', 'SEED'))
    args = parser.parse_args()
    if args.task:
        circuit, seed = args.task
        train_task(circuit, int(seed))
    elif args.phase == 'preflight':
        preflight()
    elif args.phase == 'train':
        train_all()
    elif args.phase == 'exact':
        evaluate_exact()
    else:
        parser.error('Specify --phase or --task.')


if __name__ == '__main__':
    main()

"""Run the 100x10 original-versus-learning-rate/reward pilot."""
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

from filelock import FileLock
import torch

from study import (CIRCUITS, GROUPS, HERE, OLD_LEARNING, OLD_REWARD, OUT, ROOT,
                   SEEDS, TEN, base_config, dump, eval_folder, eval_seeds,
                   identity, read, reuse_candidate, sha, source_for,
                   train_folder, training_config)
from drills.experiment import verify_netlist
from drills.fpga_session import FPGASession
from drills.model import A2C, ActorCritic, Normalizer

VALIDATION = ROOT / 'experiments/ten-step-reward-validation'
sys.path.insert(0, str(VALIDATION))
_spec = importlib.util.spec_from_file_location('prior_ten_step_runner', VALIDATION / 'run.py')
_prior_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prior_run)
TenStepWindowA2C = _prior_run.TenStepWindowA2C
supervise = _prior_run.prior.old.supervise
state_hash = _prior_run.prior.old.state_hash


def _same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _manifest_matches(manifest, circuit):
    cfg = base_config()
    payload = manifest.get('source_payload', manifest)
    return (manifest.get('status') == 'complete' and
            all(payload['tools'][key] == sha(cfg['runtime'][key])
                for key in ('abc_binary', 'yosys_binary')) and
            payload.get('benchmarks', payload.get('circuits', {})).get(circuit)
            == sha(cfg['protocol']['circuits'][circuit]['file']))


def validated_reuse(group, circuit, seed):
    folder = reuse_candidate(group, circuit, seed)
    if folder is None:
        return dict(reused=False, reason='no historical run for this cell')
    result_path, checkpoint = folder / 'result.json', folder / 'checkpoint.pt'
    if not result_path.is_file() or not checkpoint.is_file() or not (folder / 'snapshots/0.pt').is_file():
        return dict(reused=False, reason='historical files missing')
    result = read(result_path)
    if (result.get('status') != 'complete' or result.get('circuit') != circuit or
            result.get('seed') != seed or result.get('episodes_completed') != 100):
        return dict(reused=False, reason='historical run incomplete')
    if group == 'legacy':
        manifest = read(OLD_LEARNING / 'experiment.json')
        if not _manifest_matches(manifest, circuit):
            return dict(reused=False, reason='legacy tool/circuit provenance mismatch')
        expected = copy.deepcopy(training_config('legacy'))
        observed = result.get('config', {})
        expected['method']['learning_enabled'] = True
        for key in ('protocol', 'method', 'environment'):
            if not _same(expected[key], observed.get(key)):
                return dict(reused=False, reason=f'legacy {key} configuration mismatch')
        if observed.get('runtime', {}).get('experiment_fingerprint') != manifest.get('fingerprint'):
            return dict(reused=False, reason='legacy source fingerprint mismatch')
    else:
        manifest = read(OLD_REWARD / 'ten-manifest.json')
        if (manifest.get('status') != 'complete' or
                result.get('fingerprint') != manifest.get('fingerprint') or
                not _manifest_matches(manifest, circuit) or
                result.get('mapping_calls') != 1100 or
                result.get('checkpoint_sha256') != sha(checkpoint)):
            return dict(reused=False, reason='candidate provenance mismatch')
        prior = manifest['source_payload']
        if any(sha(ROOT / path) != digest for path, digest in prior['sources'].items()):
            return dict(reused=False, reason='candidate source changed')
        protocol = prior['protocol']
        if (protocol['learning_rate'] != 0.01 or protocol['ten_step']['episodes'] != 100 or
                protocol['ten_step']['iterations'] != 10 or protocol['reward'] != TEN['reward'] or
                protocol['return_window'] != TEN['return_window'] or
                protocol['warmup_episodes'] != TEN['warmup_episodes'] or
                protocol['return_std_floor'] != TEN['return_std_floor']):
            return dict(reused=False, reason='candidate method mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved.get('episodes_completed') != 100 or saved.get('best') != result.get('best') or
            len(saved.get('rewards', [])) != 100):
        return dict(reused=False, reason='historical checkpoint mismatch')
    return dict(reused=True, path=str(folder), checkpoint_sha256=sha(checkpoint),
                result_sha256=sha(result_path), source_fingerprint=(
                    manifest['fingerprint']))


def preflight():
    if (GROUPS != ('legacy', 'w001_old', 'w001_new', 'w010_old', 'w010_new') or
            CIRCUITS != ('int2float', 'i2c', 'max') or SEEDS != (0, 1, 2) or
            TEN['episodes'] != 100 or TEN['iterations'] != 10 or
            TEN['learning_rates'] != {'w001': 0.001, 'w010': 0.01} or
            TEN['evaluation'] != {'rollouts_per_training_seed': 30, 'first_seed': 40000}):
        raise ValueError('The 100x10 pilot protocol changed.')
    old = _prior_run.SPEC
    for key in ('return_window', 'warmup_episodes', 'return_std_floor'):
        if TEN[key] != old[key]:
            raise ValueError(f'Window protocol mismatch: {key}')
    if TEN['reward'] != old['reward']:
        raise ValueError('The reward no longer matches the audited ten-step study.')
    canonical = training_config('legacy')
    for group in GROUPS:
        cfg = training_config(group)
        if group == 'legacy':
            continue
        for key in ('protocol', 'environment'):
            if cfg[key] != canonical[key]:
                raise ValueError(f'Unplanned {key} change: {group}')
        control = copy.deepcopy(canonical['method'])
        control['optimizer']['learning_rate'] = TEN['learning_rates'][group[:4]]
        if group.endswith('_new'):
            control['reward'] = dict(kind='best_feasible_v1', **TEN['reward'])
        if cfg['method'] != control:
            raise ValueError(f'Unplanned method change: {group}')
    digest, payload = identity(1)
    target = OUT / 'stage1/preflight.json'
    if target.exists() and read(target)['fingerprint'] != digest:
        raise ValueError('A stage-one result already exists with another fingerprint.')
    reuse = {f'{g}/{c}/{s}': validated_reuse(g, c, s)
             for c in CIRCUITS for s in SEEDS for g in GROUPS}
    dump(target, dict(status='pass', fingerprint=digest, source_payload=payload,
                      reuse=reuse))
    print('Stage 1 preflight:', digest, 'reused:', sum(r['reused'] for r in reuse.values()))


def _checkpoint(group, circuit, seed):
    return source_for(group, circuit, seed) / 'checkpoint.pt'


def train_task(group, circuit, seed):
    if group not in GROUPS or circuit not in CIRCUITS or seed not in SEEDS:
        raise ValueError('Training task outside the protocol.')
    digest, _ = identity(1)
    if read(OUT / 'stage1/preflight.json')['fingerprint'] != digest:
        raise ValueError('Run preflight before training.')
    if read(OUT / 'stage1/preflight.json')['reuse'][f'{group}/{circuit}/{seed}']['reused']:
        return
    folder = train_folder(group, circuit, seed)
    result_path = folder / 'result.json'
    if result_path.exists():
        result = read(result_path)
        if result.get('status') == 'complete' and result.get('fingerprint') == digest:
            return
        raise ValueError(f'Conflicting training result: {result_path}')
    folder.mkdir(parents=True, exist_ok=True)
    cfg = training_config(group)
    completed = 0
    try:
        resume = (folder / 'checkpoint.pt').exists()
        if group == 'legacy':
            agent = A2C(cfg, cfg['protocol']['circuits'][circuit], seed, folder, resume=resume)
            if not resume:
                (folder / 'snapshots').mkdir(exist_ok=True)
                shutil.copyfile(folder / 'checkpoint.pt', folder / 'snapshots/0.pt')
        else:
            kind = 'candidate' if group.endswith('_new') else 'original'
            agent = TenStepWindowA2C(cfg, circuit, seed, folder, kind, resume=resume)
        completed = agent.episodes_completed
        for _ in range(completed, TEN['episodes']):
            agent.run_episode()
            completed = agent.episodes_completed
            if completed % 10 == 0:
                print(f'{group}/{circuit}/{seed}: {completed}/100', flush=True)
            dump(folder / 'status.json', dict(status='running', episodes=completed))
        verify_netlist(cfg, cfg['protocol']['circuits'][circuit],
                       folder / 'best-mapped.v', agent.game.best)
        dump(result_path, dict(status='complete', fingerprint=digest, group=group,
                               circuit=circuit, seed=seed, episodes_completed=completed,
                               mapping_calls=completed * 11, best=agent.game.best,
                               training_seconds=agent.training_seconds,
                               checkpoint_sha256=sha(folder / 'checkpoint.pt'),
                               best_mapped_sha256=sha(folder / 'best-mapped.v'),
                               equivalence_log_sha256=sha(folder / 'equivalence.log')))
        dump(folder / 'status.json', dict(status='complete', episodes=completed))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', episodes=completed,
                                         error=repr(error), traceback=traceback.format_exc()))
        raise


def _first_episode(folder):
    with (folder / 'episodes/1/log.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 11:
        raise ValueError(f'Incomplete first episode: {folder}')
    return [(row['iteration'], row['optimization'], row['luts'], row['levels'])
            for row in rows]


def validate_training():
    digest, _ = identity(1)
    pairs = []
    for circuit in CIRCUITS:
        for seed in SEEDS:
            reference = source_for('legacy', circuit, seed)
            first = _first_episode(reference)
            baseline = torch.load(reference / 'snapshots/0.pt', map_location='cpu', weights_only=True)
            for group in GROUPS:
                folder = source_for(group, circuit, seed)
                record = read(folder / 'result.json')
                saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
                initial = torch.load(folder / 'snapshots/0.pt', map_location='cpu', weights_only=True)
                if (record['status'] != 'complete' or saved['episodes_completed'] != 100 or
                        saved['best'] != record['best'] or _first_episode(folder) != first or
                        state_hash(initial['network']) != state_hash(baseline['network']) or
                        state_hash(initial['rng_state']) != state_hash(baseline['rng_state'])):
                    raise ValueError(f'Training pair mismatch: {circuit}/{seed}/{group}')
                if not read(OUT / 'stage1/preflight.json')['reuse'][f'{group}/{circuit}/{seed}']['reused']:
                    if record['fingerprint'] != digest or record['mapping_calls'] != 1100 or \
                            record['checkpoint_sha256'] != sha(folder / 'checkpoint.pt'):
                        raise ValueError(f'New training evidence mismatch: {folder}')
                pairs.append(dict(circuit=circuit, seed=seed, group=group,
                                  checkpoint_sha256=sha(folder / 'checkpoint.pt')))
    dump(OUT / 'stage1/pair-validation.json', dict(status='pass', fingerprint=digest,
                                                    rows=pairs))


def train_all():
    digest, payload = identity(1)
    if read(OUT / 'stage1/preflight.json')['fingerprint'] != digest:
        raise ValueError('Preflight fingerprint changed.')
    jobs = [(g, c, s) for c in CIRCUITS for s in SEEDS for g in GROUPS
            if not read(OUT / 'stage1/preflight.json')['reuse'][f'{g}/{c}/{s}']['reused']]
    pending = [(f'{g}-{c}-{s}', [sys.executable, '-B', '-u', str(HERE / 'stage1.py'),
                                  '--train-task', g, c, str(s)]) for g, c, s in jobs
               if not (train_folder(g, c, s) / 'result.json').exists()]
    with FileLock(str(OUT / 'stage1/.train.lock'), timeout=0):
        manifest = dict(status='running', fingerprint=digest, source_payload=payload,
                        jobs=[list(job) for job in jobs])
        dump(OUT / 'stage1/train-manifest.json', manifest)
        processes = supervise(pending, OUT / 'stage1/logs/training',
                              workers=TEN['workers'], timeout=TEN['task_timeout_seconds'])
        missing = [list(job) for job in jobs if not (train_folder(*job) / 'result.json').exists()]
        manifest.update(status='complete' if not missing and
                        all(p['exit_code'] == 0 for p in processes) else 'incomplete',
                        processes=processes, missing=missing)
        dump(OUT / 'stage1/train-manifest.json', manifest)
        if manifest['status'] != 'complete':
            raise RuntimeError('Stage-one training is incomplete.')
    validate_training()


def evaluate_task(group, circuit, seed):
    if group not in GROUPS or circuit not in CIRCUITS or seed not in SEEDS:
        raise ValueError('Evaluation task outside the protocol.')
    digest, _ = identity(1)
    source = _checkpoint(group, circuit, seed)
    checkpoint_hash = sha(source)
    saved = torch.load(source, map_location='cpu', weights_only=True)
    cfg = training_config('legacy')
    torch.set_num_threads(cfg['environment']['torch_threads'])
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']),
                              cfg['method']['network'])
    network.load_state_dict(saved['network'])
    network.eval().requires_grad_(False)
    before = state_hash(network.state_dict())
    folder = eval_folder(group, circuit, seed)
    output = folder / 'result.json'
    if output.exists():
        old = read(output)
        if old['status'] == 'complete' and old['fingerprint'] == digest and \
                old['checkpoint_sha256'] == checkpoint_hash:
            return
        raise ValueError(f'Conflicting evaluation result: {output}')
    rows = []
    try:
        for eval_seed in eval_seeds(seed):
            destination = folder / f'rollout-{eval_seed}'
            record = destination / 'rollout.json'
            if record.exists():
                row = read(record)
                if (row['status'] != 'complete' or row['fingerprint'] != digest or
                        row['checkpoint_sha256'] != checkpoint_hash or
                        row['evaluation_seed'] != eval_seed or
                        row['log_sha256'] != sha(destination / 'episodes/1/log.csv') or
                        row['best_mapped_sha256'] != sha(destination / 'best-mapped.v')):
                    raise ValueError(f'Conflicting rollout: {record}')
            else:
                game = FPGASession(cfg, cfg['protocol']['circuits'][circuit], destination)
                generator = torch.Generator(device='cpu').manual_seed(eval_seed)
                state = game.reset()
                normalizer = Normalizer(len(state), cfg['method']['normalization'],
                                        cfg['method']['features'], state)
                actions = []
                done = False
                while not done:
                    normalized = normalizer.normalize(state)
                    with torch.no_grad():
                        logits, _ = network(torch.as_tensor(normalized, device='cpu'))
                        action = int(torch.multinomial(logits.softmax(-1), 1,
                                                       generator=generator))
                    actions.append(action)
                    state, _, done = game.step(action)
                game.export_best()
                verify_netlist(cfg, cfg['protocol']['circuits'][circuit],
                               destination / 'best-mapped.v', game.best)
                row = dict(status='complete', fingerprint=digest, group=group,
                           circuit=circuit, training_seed=seed, evaluation_seed=eval_seed,
                           checkpoint_sha256=checkpoint_hash, mapping_calls=11,
                           actions=actions, best=game.best,
                           log_sha256=sha(destination / 'episodes/1/log.csv'),
                           best_mapped_sha256=sha(destination / 'best-mapped.v'),
                           equivalence_log_sha256=sha(destination / 'equivalence.log'))
                dump(record, row)
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f'{group}/{circuit}/{seed}: {len(rows)}/30 eval', flush=True)
        if before != state_hash(network.state_dict()) or checkpoint_hash != sha(source):
            raise ValueError('Frozen evaluation changed the network or source checkpoint.')
        dump(output, dict(status='complete', fingerprint=digest, group=group,
                          circuit=circuit, training_seed=seed,
                          checkpoint_sha256=checkpoint_hash, network_sha256=before,
                          rollouts=rows))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', completed=len(rows),
                                         error=repr(error), traceback=traceback.format_exc()))
        raise


def evaluate_all():
    digest, payload = identity(1)
    pair = read(OUT / 'stage1/pair-validation.json')
    if pair['status'] != 'pass' or pair['fingerprint'] != digest:
        raise ValueError('Training pairs have not passed validation.')
    jobs = [(g, c, s) for c in CIRCUITS for s in SEEDS for g in GROUPS]
    pending = [(f'{g}-{c}-{s}', [sys.executable, '-B', '-u', str(HERE / 'stage1.py'),
                                  '--eval-task', g, c, str(s)]) for g, c, s in jobs
               if not (eval_folder(g, c, s) / 'result.json').exists()]
    with FileLock(str(OUT / 'stage1/.evaluation.lock'), timeout=0):
        manifest = dict(status='running', fingerprint=digest, source_payload=payload,
                        jobs=[list(job) for job in jobs])
        dump(OUT / 'stage1/evaluation-manifest.json', manifest)
        processes = supervise(pending, OUT / 'stage1/logs/evaluation',
                              workers=TEN['workers'], timeout=TEN['task_timeout_seconds'])
        missing = [list(job) for job in jobs if not (eval_folder(*job) / 'result.json').exists()]
        manifest.update(status='complete' if not missing and
                        all(p['exit_code'] == 0 for p in processes) else 'incomplete',
                        processes=processes, missing=missing)
        dump(OUT / 'stage1/evaluation-manifest.json', manifest)
        if manifest['status'] != 'complete':
            raise RuntimeError('Stage-one evaluation is incomplete.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'train', 'evaluate'))
    parser.add_argument('--train-task', nargs=3, metavar=('GROUP', 'CIRCUIT', 'SEED'))
    parser.add_argument('--eval-task', nargs=3, metavar=('GROUP', 'CIRCUIT', 'SEED'))
    args = parser.parse_args()
    if args.train_task:
        g, c, s = args.train_task
        train_task(g, c, int(s))
    elif args.eval_task:
        g, c, s = args.eval_task
        evaluate_task(g, c, int(s))
    elif args.phase == 'preflight':
        preflight()
    elif args.phase == 'train':
        train_all()
    elif args.phase == 'evaluate':
        evaluate_all()
    else:
        parser.error('Specify --phase, --train-task, or --eval-task.')


if __name__ == '__main__':
    main()

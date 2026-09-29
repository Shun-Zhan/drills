"""Independent, no-update ABC rollouts for trained and initial policies."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from filelock import FileLock
import torch

from context import (CIRCUITS, EVAL_GROUPS, HERE, OUT, ROOT, SEEDS, SPEC,
                     dump, evaluation_config, evaluation_folder,
                     evaluation_seeds, identity, prior, read, sha,
                     training_folder)
from drills.experiment import verify_netlist
from drills.fpga_session import FPGASession
from drills.model import ActorCritic, Normalizer


def checkpoint_for(group, circuit, seed):
    if group == 'initial':
        return training_folder('candidate', circuit, seed) / 'snapshots/0.pt'
    return training_folder(group, circuit, seed) / 'checkpoint.pt'


def rollout(cfg, circuit, network, eval_seed, folder):
    game = FPGASession(cfg, cfg['protocol']['circuits'][circuit], folder)
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
            action = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        actions.append(action)
        state, _, done = game.step(action)
    game.export_best()
    verify_netlist(cfg, cfg['protocol']['circuits'][circuit],
                   folder / 'best-mapped.v', game.best)
    return dict(evaluation_seed=eval_seed, actions=actions,
                best=game.best, mapping_calls=SPEC['iterations'] + 1,
                best_mapped_sha256=sha(folder / 'best-mapped.v'),
                equivalence_log_sha256=sha(folder / 'equivalence.log'),
                log_sha256=sha(folder / 'episodes/1/log.csv'))


def evaluate_task(group, circuit, seed):
    if group not in EVAL_GROUPS or circuit not in CIRCUITS or seed not in SEEDS:
        raise ValueError('Evaluation task is outside the fixed protocol.')
    digest, _ = identity()
    cfg = evaluation_config(circuit)
    source = checkpoint_for(group, circuit, seed)
    if not source.is_file():
        raise ValueError(f'Missing training checkpoint: {source}')
    checkpoint_hash = sha(source)
    if group == 'initial':
        old = training_folder('original', circuit, seed) / 'snapshots/0.pt'
        original = prior.old.load(old)
        candidate = prior.old.load(source)
        for key in ('network', 'optimizer', 'rng_state'):
            if prior.old.state_hash(original[key]) != prior.old.state_hash(candidate[key]):
                raise ValueError(f'Initial pair differs: {circuit}/{seed}/{key}')
    saved = prior.old.load(source)
    torch.set_num_threads(cfg['environment']['torch_threads'])
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(len(cfg['method']['features']),
                              len(cfg['protocol']['actions']),
                              cfg['method']['network'])
    network.load_state_dict(saved['network'])
    network.eval().requires_grad_(False)
    initial_network_hash = prior.old.state_hash(network.state_dict())
    folder = evaluation_folder(group, circuit, seed)
    output = folder / 'result.json'
    if output.exists():
        old = read(output)
        if old['status'] == 'complete' and old['fingerprint'] == digest and \
                old['checkpoint_sha256'] == checkpoint_hash:
            print(f'Already complete: {group}/{circuit}/{seed}', flush=True)
            return
        raise ValueError(f'Conflicting evaluation result: {output}')
    rows = []
    try:
        for eval_seed in evaluation_seeds(seed):
            destination = folder / f'rollout-{eval_seed}'
            record = destination / 'rollout.json'
            if record.exists():
                row = read(record)
                if row['status'] != 'complete' or row['fingerprint'] != digest or \
                        row['checkpoint_sha256'] != checkpoint_hash or \
                        row['evaluation_seed'] != eval_seed:
                    raise ValueError(f'Stale or incomplete rollout: {record}')
                if sha(destination / 'episodes/1/log.csv') != row['log_sha256'] or \
                        sha(destination / 'best-mapped.v') != row['best_mapped_sha256'] or \
                        sha(destination / 'equivalence.log') != row['equivalence_log_sha256']:
                    raise ValueError(f'Changed completed rollout: {record}')
            else:
                row = rollout(cfg, circuit, network, eval_seed, destination)
                row.update(status='complete', fingerprint=digest, group=group,
                           circuit=circuit, training_seed=seed,
                           checkpoint_sha256=checkpoint_hash)
                dump(record, row)
            rows.append(row)
            dump(folder / 'status.json',
                 dict(status='running', completed=len(rows), required=len(evaluation_seeds(seed))))
            if len(rows) % 10 == 0:
                print(f'{group}/{circuit}/seed-{seed}: {len(rows)}/30', flush=True)
        if prior.old.state_hash(network.state_dict()) != initial_network_hash or \
                sha(source) != checkpoint_hash:
            raise RuntimeError('Evaluation changed the network or checkpoint.')
        dump(output, dict(status='complete', fingerprint=digest, group=group,
                          circuit=circuit, training_seed=seed,
                          checkpoint_sha256=checkpoint_hash,
                          network_sha256=initial_network_hash,
                          network_unchanged=True, checkpoint_unchanged=True,
                          rollouts=rows))
        dump(folder / 'status.json', dict(status='complete', completed=len(rows)))
    except Exception as error:
        dump(folder / 'status.json',
             dict(status='failed', completed=len(rows), error=repr(error),
                  traceback=traceback.format_exc()))
        raise


def evaluate_all():
    digest, payload = identity()
    pair_path = OUT / 'pair-validation.json'
    training_path = OUT / 'train-manifest.json'
    if not pair_path.exists() or not training_path.exists():
        raise ValueError('Training pair validation is missing.')
    if read(pair_path)['status'] != 'pass' or read(pair_path)['fingerprint'] != digest or \
            read(training_path)['status'] != 'complete' or \
            read(training_path)['fingerprint'] != digest:
        raise ValueError('Training is incomplete or belongs to another fingerprint.')
    manifest_path = OUT / 'evaluation-manifest.json'
    if manifest_path.exists() and read(manifest_path)['fingerprint'] != digest:
        raise ValueError('Cannot resume evaluation with changed sources.')
    jobs = [(group, circuit, seed) for circuit in CIRCUITS
            for seed in SEEDS for group in EVAL_GROUPS]
    pending = []
    for group, circuit, seed in jobs:
        path = evaluation_folder(group, circuit, seed) / 'result.json'
        if path.exists():
            row = read(path)
            if row['status'] != 'complete' or row['fingerprint'] != digest:
                raise ValueError(f'Conflicting completed evaluation: {path}')
            continue
        pending.append((f'{group}-{circuit}-{seed}',
                        [sys.executable, '-B', '-u', str(HERE / 'evaluate.py'),
                         '--task', group, circuit, str(seed)]))
    with FileLock(str(OUT / '.evaluation.lock'), timeout=0):
        manifest = dict(status='running', fingerprint=digest,
                        source_payload=payload, jobs=[list(job) for job in jobs])
        dump(manifest_path, manifest)
        processes = prior.old.supervise(
            pending, OUT / 'logs/evaluation', workers=SPEC['workers'],
            timeout=SPEC['task_timeout_seconds'])
        missing = [list(job) for job in jobs
                   if not (evaluation_folder(*job) / 'result.json').exists()]
        manifest.update(
            status='complete' if not missing and all(p['exit_code'] == 0 for p in processes)
            else 'incomplete', processes=processes, missing=missing)
        dump(manifest_path, manifest)
        if manifest['status'] != 'complete':
            raise RuntimeError('Evaluation is incomplete; inspect manifest and task logs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', nargs=3, metavar=('GROUP', 'CIRCUIT', 'SEED'))
    args = parser.parse_args()
    if args.task:
        group, circuit, raw_seed = args.task
        evaluate_task(group, circuit, int(raw_seed))
    else:
        evaluate_all()


if __name__ == '__main__':
    main()

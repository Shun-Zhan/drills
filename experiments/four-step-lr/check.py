"""Verify full coverage, parity, frozen evaluation, metrics, and saved netlists."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import (ASSESSMENT, GROUPS, HERE, POLICIES, ActorCritic,
                    config, dump, evaluation_seeds, identity, load, now,
                    probe_path, read, sha, state_hash)
from evaluate import checkpoint_path


def network_diagnostics(cfg, network_state, probes):
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']),
                              cfg['method']['network'])
    network.load_state_dict(network_state)
    with torch.no_grad():
        logits, _ = network(torch.as_tensor(probes, dtype=torch.float32))
        logp = logits.log_softmax(-1)
        entropy = float((-logp.exp() * logp).sum(-1).mean())
    result = dict(entropy=entropy)
    for name in ('actor', 'critic'):
        weights = [p.detach().double().flatten() for key, p in network.state_dict().items()
                   if key.startswith(name + '.') and key.endswith('.weight')]
        result[name + '_weight_variance'] = float(torch.cat(weights).var(unbiased=False))
    return result


def generate():
    cfg = config()
    fingerprint, evidence = identity(cfg)
    root = Path(cfg['runtime']['output_dir'])
    training = read(root / 'experiment.json')
    evaluation = read(root / 'evaluation.json')
    errors = []
    if training['fingerprint'] != fingerprint or evaluation['training_fingerprint'] != fingerprint:
        errors.append('Experiment/source fingerprint mismatch.')
    if training['status'] != 'complete' or evaluation['status'] != 'complete':
        errors.append('Experiment phases are incomplete.')
    candidate_path = HERE.parent / 'ten-step-optimality/candidates.csv'
    with candidate_path.open() as stream:
        candidates = list(csv.DictReader(stream))
    observed_optima = {name: min(int(row['luts']) for row in candidates
                                  if row['circuit'] == name and row['feasible'] == 'True')
                       for name in cfg['protocol']['circuits']}
    if observed_optima != ASSESSMENT['optimum_luts']:
        errors.append('The known four-step optima differ from the enumerated evidence.')
    training_runs = evaluation_banks = evaluation_rollouts = log_rows = 0
    for circuit in cfg['protocol']['circuits']:
        probes_file = probe_path(cfg, circuit)
        probes = np.load(probes_file, allow_pickle=False)
        if sha(probes_file) != training['probes'][circuit] or probes.shape != (196, 9):
            errors.append(f'Fixed probe bank changed: {circuit}')
        for seed in cfg['protocol']['seeds']:
            initial_hashes = []
            first_logs = []
            for group in GROUPS:
                folder = root / 'training' / group / circuit / f'seed-{seed}'
                result = read(folder / 'result.json')
                checkpoint = folder / 'checkpoint.pt'
                if result['status'] != 'complete' or result['episodes_completed'] != 250:
                    errors.append(f'Incomplete training: {group}/{circuit}/{seed}')
                if sha(checkpoint) != result['checkpoint_sha256']:
                    errors.append(f'Training checkpoint hash mismatch: {group}/{circuit}/{seed}')
                if 'Networks are equivalent' not in (folder / 'equivalence.log').read_text():
                    errors.append(f'Training best netlist lacks CEC: {group}/{circuit}/{seed}')
                if read(folder / 'best.json') != result['best']:
                    errors.append(f'Training best export mismatch: {group}/{circuit}/{seed}')
                if result['best']['feasible'] and result['best']['luts'] < observed_optima[circuit]:
                    errors.append(f'Result beats certified four-step optimum: {group}/{circuit}/{seed}')
                for episode in range(1, 251):
                    path = folder / 'episodes' / str(episode) / 'log.csv'
                    with path.open() as stream:
                        rows = list(csv.DictReader(stream))
                    if len(rows) != 5 or [int(r['iteration']) for r in rows] != list(range(5)):
                        errors.append(f'Incorrect ABC search budget: {group}/{circuit}/{seed}/{episode}')
                    log_rows += len(rows)
                if group != 'uniform':
                    initial = load(folder / 'snapshots/0.pt')
                    final = load(folder / 'snapshots/250.pt')
                    initial_hashes.append((state_hash(initial['network']), state_hash(initial['rng_state'])))
                    first_logs.append((folder / 'episodes/1/log.csv').read_bytes())
                    metrics = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
                    if len(metrics) != 251 or [r['episode'] for r in metrics] != list(range(251)):
                        errors.append(f'Metric coverage error: {group}/{circuit}/{seed}')
                    else:
                        for episode, saved in ((0, initial), (250, final)):
                            recomputed = network_diagnostics(cfg, saved['network'], probes)
                            for key, value in recomputed.items():
                                if not math.isclose(metrics[episode][key], value, rel_tol=1e-8, abs_tol=1e-9):
                                    errors.append(f'Metric mismatch: {group}/{circuit}/{seed}/{episode}/{key}')
                        if any(r['probe_sha256'] != sha(probes_file) for r in metrics):
                            errors.append(f'Probe hash mismatch: {group}/{circuit}/{seed}')
                    if initial_hashes[-1][0] == state_hash(final['network']):
                        errors.append(f'Trained model did not change: {group}/{circuit}/{seed}')
                training_runs += 1
            if len(set(initial_hashes)) != 1 or len(set(first_logs)) != 1:
                errors.append(f'Learning-rate initial/first-episode parity failed: {circuit}/{seed}')
            for policy in POLICIES:
                folder = root / 'evaluation' / policy / circuit / f'seed-{seed}'
                result = read(folder / 'result.json')
                checkpoint = checkpoint_path(root, policy, circuit, seed)
                if result['status'] != 'complete' or len(result['rollouts']) != 10 or \
                        not result['network_unchanged'] or not result['checkpoint_unchanged'] or \
                        result['checkpoint_sha256'] != sha(checkpoint):
                    errors.append(f'Frozen evaluation failed: {policy}/{circuit}/{seed}')
                for expected_seed, rollout in zip(evaluation_seeds(seed), result['rollouts']):
                    destination = folder / f'rollout-{expected_seed}'
                    if rollout['evaluation_seed'] != expected_seed or rollout != read(destination / 'rollout.json'):
                        errors.append(f'Evaluation record mismatch: {policy}/{circuit}/{seed}/{expected_seed}')
                    if 'Networks are equivalent' not in (destination / 'equivalence.log').read_text():
                        errors.append(f'Evaluation best netlist lacks CEC: {policy}/{circuit}/{seed}/{expected_seed}')
                    if read(destination / 'best.json') != rollout['best']:
                        errors.append(f'Evaluation best export mismatch: {policy}/{circuit}/{seed}/{expected_seed}')
                    with (destination / 'episodes/1/log.csv').open() as stream:
                        rows = list(csv.DictReader(stream))
                    if len(rows) != 5 or [int(r['iteration']) for r in rows] != list(range(5)):
                        errors.append(f'Incorrect evaluation budget: {policy}/{circuit}/{seed}/{expected_seed}')
                    if rollout['best']['feasible'] and rollout['best']['luts'] < observed_optima[circuit]:
                        errors.append(f'Evaluation beats certified optimum: {policy}/{circuit}/{seed}/{expected_seed}')
                    evaluation_rollouts += 1
                evaluation_banks += 1
    if (training_runs, evaluation_banks, evaluation_rollouts, log_rows) != (90, 120, 1200, 112500):
        errors.append('Training/evaluation coverage or ABC mapping count is incorrect.')
    result = dict(status='verified' if not errors else 'failed', verified_at=now(),
                  fingerprint=fingerprint, known_optima=observed_optima,
                  training_runs=training_runs, training_abc_mappings=log_rows,
                  evaluation_banks=evaluation_banks, evaluation_rollouts=evaluation_rollouts,
                  evaluation_abc_mappings=evaluation_rollouts * 5,
                  uniform_banks_shared_across_training_seeds=False, errors=errors)
    dump(HERE / 'validation.json', result)
    if errors:
        raise ValueError(f'{len(errors)} verification errors; see validation.json')


if __name__ == '__main__':
    generate()

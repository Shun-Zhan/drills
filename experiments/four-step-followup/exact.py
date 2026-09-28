"""Exact frozen-policy evaluation over the certified four-step action tree."""
import argparse
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import torch

from support import (HERE, OLD_ROOT, OUT, ROOT, SEEDS, SNAPSHOTS, best_for_sequence,
                     candidates, dump, identity, old, old_config, read, sha)
from drills.features import extract_features

PREFIXES = tuple(seq for depth in range(4) for seq in itertools.product(range(7), repeat=depth))
LEAVES = tuple(itertools.product(range(7), repeat=4))
OPTIMA = old.ASSESSMENT['optimum_luts']


def structural_hash(path):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    if lines and lines[0].startswith(b'// Benchmark ') and b' written by ABC on ' in lines[0]:
        lines = lines[1:]
    import hashlib
    return hashlib.sha256(b''.join(lines)).hexdigest()


def build_feature_cache(cfg, old_fingerprint, table, circuit):
    path = OUT / 'prefix-states' / f'{circuit}.json'
    candidate_sha = sha(ROOT / 'experiments/ten-step-optimality/candidates.csv')
    if path.exists():
        saved = read(path)
        if (saved.get('old_fingerprint') == old_fingerprint and
                saved.get('candidates_sha256') == candidate_sha and
                len(saved.get('rows', [])) == len(PREFIXES)):
            raw = {tuple(row['actions']): np.asarray(row['features'], dtype=np.float32)
                   for row in saved['rows']}
            if set(raw) == set(PREFIXES) and all(x.shape == (9,) for x in raw.values()):
                return raw
        raise ValueError(f'Stale or malformed feature cache: {path}')
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    names = cfg['protocol']['actions']
    with tempfile.TemporaryDirectory(prefix='four-step-prefix-', dir=OUT) as temporary:
        folder = Path(temporary)
        unmapped, mapped = folder / 'current.v', folder / 'mapped.v'
        for index, sequence in enumerate(PREFIXES, 1):
            for target in (unmapped, mapped):
                target.unlink(missing_ok=True)
            operations = list(cfg['protocol']['initial_sequence']) + [names[a] for a in sequence]
            command = f'read "{cfg["protocol"]["circuits"][circuit]["file"]}"; '
            command += '; '.join(operations) + '; '
            command += f'write_verilog "{unmapped}"; if -K {cfg["protocol"]["lut_inputs"]}; '
            command += f'write_verilog "{mapped}"; print_stats;'
            output = subprocess.check_output([cfg['runtime']['abc_binary'], '-c', command], text=True)
            matches = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)
            if not unmapped.is_file() or not mapped.is_file() or not matches:
                raise RuntimeError(f'Prefix replay failed: {circuit}/{sequence}')
            luts, levels = map(int, matches[-1])
            expected = table[circuit, sequence]
            if ((luts, levels, structural_hash(unmapped), structural_hash(mapped)) !=
                    (expected['luts'], expected['levels'], expected['structural_sha256'],
                     expected['mapped_structural_sha256'])):
                raise ValueError(f'Prefix differs from the certified enumeration: {circuit}/{sequence}')
            features = extract_features(unmapped, cfg)
            if features.shape != (9,) or not np.isfinite(features).all():
                raise ValueError(f'Invalid features: {circuit}/{sequence}')
            rows.append(dict(actions=list(sequence), features=features.tolist()))
            if index % 100 == 0 or index == len(PREFIXES):
                print(f'{circuit} feature prefixes: {index}/{len(PREFIXES)}', flush=True)
    dump(path, dict(old_fingerprint=old_fingerprint, candidates_sha256=candidate_sha,
                    circuit=circuit, rows=rows))
    return {tuple(row['actions']): np.asarray(row['features'], dtype=np.float32) for row in rows}


def normalized_features(cfg, raw, circuit):
    result = {}
    for sequence in PREFIXES:
        first = raw[()]
        normalizer = old.Normalizer(len(first), cfg['method']['normalization'],
                                    cfg['method']['features'], first)
        for depth in range(len(sequence) + 1):
            value = normalizer.normalize(raw[sequence[:depth]])
        result[sequence] = value
    probes = np.load(old.probe_path(cfg, circuit), allow_pickle=False)
    expected = []
    for first in range(7):
        for second in range(7):
            path = (first, second, first)
            expected.extend(result[path[:depth]] for depth in range(4))
    if not np.array_equal(np.asarray(expected, dtype=np.float32), probes):
        raise ValueError(f'Cached normalized states differ from the original fixed probes: {circuit}')
    return result


def load_network(cfg, checkpoint, fingerprint):
    saved = old.load(checkpoint)
    if saved.get('experiment_fingerprint') != fingerprint:
        raise ValueError(f'Checkpoint fingerprint mismatch: {checkpoint}')
    with torch.random.fork_rng(devices=[]):
        network = old.ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']),
                                  cfg['method']['network'])
    network.load_state_dict(saved['network'])
    network.eval().requires_grad_(False)
    return network


def action_probabilities(network, states):
    if network is None:
        uniform = torch.full((7,), 1 / 7)
        return {seq: uniform for seq in PREFIXES}
    result = {}
    with torch.no_grad():
        for sequence in PREFIXES:
            logits, _ = network(torch.as_tensor(states[sequence], device='cpu'))
            result[sequence] = logits.softmax(-1).detach().clone()
    return result


def outcomes(table, circuit):
    result = {}
    optimum = OPTIMA[circuit]
    for sequence in LEAVES:
        best = best_for_sequence(table, circuit, sequence)
        result[sequence] = dict(luts=best['luts'], levels=best['levels'],
                                feasible=best['feasible'],
                                gap=best['luts'] - optimum if best['feasible'] else None,
                                hit=best['feasible'] and best['luts'] == optimum)
    return result


def evaluate(probs, outcome, circuit, seed, policy, episode, checkpoint=None):
    reach = {(): 1.0}
    entropy = 0.0
    for sequence in PREFIXES:
        values = probs[sequence]
        entropy += reach[sequence] * float((-values * values.log()).sum()) / 4
        for action in range(7):
            reach[sequence + (action,)] = reach[sequence] * float(values[action])
    total = sum(reach[sequence] for sequence in LEAVES)
    if abs(total - 1) > 2e-6:
        raise ValueError(f'Policy probability does not sum to one: {circuit}/{policy}/{seed}: {total}')
    feasible = sum(reach[seq] for seq in LEAVES if outcome[seq]['feasible'])
    hit = sum(reach[seq] for seq in LEAVES if outcome[seq]['hit'])
    weighted_gap = sum(reach[seq] * outcome[seq]['gap'] for seq in LEAVES
                       if outcome[seq]['feasible'])
    dist = {}
    for sequence in LEAVES:
        gap = outcome[sequence]['gap']
        if gap is not None:
            dist[gap] = dist.get(gap, 0.0) + reach[sequence]
    greedy = ()
    for _ in range(4):
        greedy += (int(torch.argmax(probs[greedy])),)
    greedy_score = outcome[greedy]
    curves = {}
    for k in range(1, 101):
        if feasible >= 1 - 2e-6:
            max_gap = max(dist)
            expected_best = sum(sum(prob for gap, prob in dist.items() if gap >= threshold) ** k
                                for threshold in range(1, max_gap + 1))
            curves[str(k)] = dict(expected_best_gap=expected_best)
        else:
            curves[str(k)] = dict(any_feasible=1 - (1 - feasible) ** k,
                                  any_optimal=1 - (1 - hit) ** k)
    return dict(circuit=circuit, seed=seed, policy=policy, episode=episode,
                checkpoint_sha256=sha(checkpoint) if checkpoint else None,
                expected_gap=weighted_gap if feasible >= 1 - 2e-6 else None,
                conditional_feasible_gap=weighted_gap / feasible if feasible > 0 else None,
                feasible_probability=feasible, optimal_probability=hit,
                occupancy_entropy=entropy, probability_sum=total,
                greedy_actions=list(greedy), greedy_best=greedy_score, curves=curves)


def verify_rollouts(probs, outcome, circuit, seed, policy):
    folder = OLD_ROOT / 'evaluation' / policy / circuit / f'seed-{seed}'
    files = sorted(folder.glob('rollout-*/rollout.json'))
    if len(files) != 10:
        raise ValueError(f'Expected ten original rollouts: {folder}')
    for path in files:
        record = read(path)
        generator = torch.Generator(device='cpu').manual_seed(record['evaluation_seed'])
        sequence = ()
        for expected_action in record['actions']:
            action = int(torch.multinomial(probs[sequence], 1, generator=generator))
            if action != expected_action:
                raise ValueError(f'Exact action replay mismatch: {path} at step {len(sequence)}')
            sequence += (action,)
        best = outcome[sequence]
        if (best['luts'], best['levels'], best['feasible']) != (
                record['best']['luts'], record['best']['levels'], record['best']['feasible']):
            raise ValueError(f'Exact best-result replay mismatch: {path}')
    return len(files)


def run_old(cfg, fingerprint, table, states, outcome):
    rows = []
    replayed = 0
    for circuit in cfg['protocol']['circuits']:
        uniform_probs = action_probabilities(None, states[circuit])
        rows.append(evaluate(uniform_probs, outcome[circuit], circuit, None, 'uniform', None))
        for seed in SEEDS:
            replayed += verify_rollouts(uniform_probs, outcome[circuit], circuit, seed, 'uniform')
            for group in ('lr001', 'lr010'):
                for episode in SNAPSHOTS:
                    checkpoint = (OLD_ROOT / 'training' / group / circuit /
                                  f'seed-{seed}/snapshots/{episode}.pt')
                    network = load_network(cfg, checkpoint, fingerprint)
                    probs = action_probabilities(network, states[circuit])
                    rows.append(evaluate(probs, outcome[circuit], circuit, seed, group, episode,
                                         checkpoint))
                    if episode == 0 and group == 'lr001':
                        replayed += verify_rollouts(probs, outcome[circuit], circuit, seed, 'initial')
                    if episode == 250:
                        replayed += verify_rollouts(probs, outcome[circuit], circuit, seed,
                                                    group + '-final')
        print(f'Exact old models and rollouts complete: {circuit}', flush=True)
    if replayed != 1200:
        raise ValueError(f'Expected 1200 exact action replays, got {replayed}.')
    return rows, replayed


def run_new(cfg, fingerprint, table, states, outcome):
    followup_fingerprint, _ = identity()
    rows = []
    circuits = []
    for circuit in cfg['protocol']['circuits']:
        folders = [OUT / 'training' / circuit / f'seed-{seed}' for seed in SEEDS]
        present = [folder / 'result.json' for folder in folders if (folder / 'result.json').exists()]
        if not present:
            continue
        if len(present) != len(SEEDS):
            raise ValueError(f'Partial training for {circuit}; exact evaluation requires all ten seeds.')
        circuits.append(circuit)
        for seed, folder in zip(SEEDS, folders):
            result = read(folder / 'result.json')
            if result['status'] != 'complete' or result['fingerprint'] != followup_fingerprint:
                raise ValueError(f'Invalid new training result: {folder}')
            for episode in SNAPSHOTS:
                checkpoint = folder / f'snapshots/{episode}.pt'
                network = load_network(cfg, checkpoint, followup_fingerprint)
                probs = action_probabilities(network, states[circuit])
                rows.append(evaluate(probs, outcome[circuit], circuit, seed, 'rolling8', episode,
                                     checkpoint))
        print(f'Exact rolling-window models complete: {circuit}', flush=True)
    if not circuits:
        raise ValueError('No completed new training circuit found.')
    return rows, circuits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('old', 'new'), required=True)
    args = parser.parse_args()
    cfg, old_fingerprint = old_config()
    torch.set_num_threads(cfg['environment']['torch_threads'])
    table = candidates()
    states = {}
    outcome = {}
    cache_hashes = {}
    for circuit in cfg['protocol']['circuits']:
        raw = build_feature_cache(cfg, old_fingerprint, table, circuit)
        states[circuit] = normalized_features(cfg, raw, circuit)
        outcome[circuit] = outcomes(table, circuit)
        cache_hashes[circuit] = sha(OUT / 'prefix-states' / f'{circuit}.json')
    if args.source == 'old':
        rows, replayed = run_old(cfg, old_fingerprint, table, states, outcome)
        extra = dict(replayed_rollouts=replayed)
    else:
        rows, circuits = run_new(cfg, old_fingerprint, table, states, outcome)
        extra = dict(circuits=circuits)
    digest, payload = identity()
    dump(OUT / f'{args.source}-exact.json', dict(status='complete', source=args.source,
         followup_fingerprint=digest, old_fingerprint=old_fingerprint,
         candidate_sha256=payload['candidates_sha256'], cache_sha256=cache_hashes,
         models=len(rows), **extra, rows=rows))
    print(f'Wrote {args.source} exact evaluation: {len(rows)} model rows.', flush=True)


if __name__ == '__main__':
    main()

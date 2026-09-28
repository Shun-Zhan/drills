"""Exact frozen-policy evaluation for each four-step reward and window snapshot."""
import importlib.util

import torch

from context import OUT, SEEDS4, SNAPSHOTS4, SPEC, dump, identity, prior, read, sha

_spec = importlib.util.spec_from_file_location('certified_four_step_exact', prior.HERE / 'exact.py')
exact = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exact)


def evaluate_all():
    digest, payload = identity()
    cfg, old_fingerprint = prior.old_config()
    torch.set_num_threads(cfg['environment']['torch_threads'])
    table = prior.candidates()
    initial = read(prior.OUT / 'old-exact.json')
    if initial['status'] != 'complete' or initial['followup_fingerprint'] != payload['followup_fingerprint']:
        raise ValueError('The certified old exact evaluation is stale.')
    old_rows = {(row['circuit'], row['seed'], row['policy'], row['episode']): row
                for row in initial['rows']}
    rows = []
    for circuit in SPEC['circuits']:
        raw = exact.build_feature_cache(cfg, old_fingerprint, table, circuit)
        states = exact.normalized_features(cfg, raw, circuit)
        outcomes = exact.outcomes(table, circuit)
        for group in ('window', 'reward'):
            for seed in SEEDS4:
                base = (prior.OUT / 'training' / circuit / f'seed-{seed}' if
                        group == 'window' and circuit == 'i2c' else
                        OUT / 'four-step' / group / circuit / f'seed-{seed}')
                result = read(base / 'result.json')
                if result['status'] != 'complete':
                    raise ValueError(f'Incomplete training result: {base}')
                fingerprint = (payload['followup_fingerprint'] if
                               group == 'window' and circuit == 'i2c' else digest)
                if result['fingerprint'] != fingerprint:
                    raise ValueError(f'Wrong training fingerprint: {base}')
                for episode in SNAPSHOTS4:
                    checkpoint = base / 'snapshots' / f'{episode}.pt'
                    network = exact.load_network(cfg, checkpoint, fingerprint)
                    probs = exact.action_probabilities(network, states)
                    row = exact.evaluate(probs, outcomes, circuit, seed, group, episode, checkpoint)
                    if episode == 0:
                        reference = old_rows[circuit, seed, 'lr001', 0]
                        for metric in ('expected_gap', 'feasible_probability', 'optimal_probability'):
                            value = row[metric]
                            baseline = reference[metric]
                            if value is None or baseline is None:
                                if value != baseline:
                                    raise ValueError(f'Initial policy mismatch: {circuit}/{seed}/{metric}')
                            elif abs(value - baseline) > 1e-9:
                                raise ValueError(f'Initial policy mismatch: {circuit}/{seed}/{metric}')
                    rows.append(row)
        print(f'Exact evaluation complete: {circuit}', flush=True)
    output = dict(status='complete', fingerprint=digest,
                  candidate_sha256=payload['candidate_sha256'],
                  cache_sha256=payload['prefix_sha256'], models=len(rows), rows=rows)
    dump(OUT / 'exact.json', output)
    return output


if __name__ == '__main__':
    result = evaluate_all()
    print('Exact model rows:', result['models'])

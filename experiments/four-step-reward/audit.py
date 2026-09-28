"""Audit both rewards against every certified four-action sequence."""
import itertools
from fractions import Fraction

from context import HERE, SPEC, dump, identity, prior
from reward import RewardTracker, original_reward


def audit_reward(table, cfg, circuit_name, reward_name, gamma):
    circuit = cfg['protocol']['circuits'][circuit_name]
    names = cfg['protocol']['actions']
    first = table[circuit_name, ()]
    optimum = prior.old.ASSESSMENT['optimum_luts'][circuit_name]
    scored = []
    for actions in itertools.product(range(len(names)), repeat=4):
        tracker = RewardTracker(first['luts'], first['levels'], circuit['max_levels'])
        previous = (first['luts'], first['levels'])
        best = first['luts'] if first['feasible'] else None
        value = Fraction(0)
        for index in range(1, 5):
            row = table[circuit_name, actions[:index]]
            current = row['luts'], row['levels']
            reward = (tracker.step_exact(*current) if reward_name == 'candidate' else
                      Fraction(original_reward(cfg, circuit, previous, current)))
            value += gamma ** (index - 1) * reward
            if row['feasible']:
                best = row['luts'] if best is None else min(best, row['luts'])
            previous = current
        scored.append((value, best == optimum, list(actions), best))
    highest = max(row[0] for row in scored)
    winners = [row for row in scored if row[0] == highest]
    runner_up = max(row[0] for row in scored if row[0] < highest)
    return dict(circuit=circuit_name, reward=reward_name, gamma=str(gamma),
                total_sequences=len(scored), top_score=float(highest),
                runner_up_score=float(runner_up), distinct_score_gap=float(highest-runner_up),
                tied_top=len(winners), optimal_top=sum(row[1] for row in winners),
                all_top_optimal=all(row[1] for row in winners),
                top_actions=[row[2] for row in winners[:20]],
                top_best_luts=sorted(set(row[3] for row in winners if row[3] is not None)))


def audit():
    cfg, _ = prior.old_config()
    table = prior.candidates()
    rows = [audit_reward(table, cfg, name, reward_name, gamma)
            for gamma in (Fraction(99, 100), Fraction(1))
            for reward_name in ('original', 'candidate')
            for name in ('int2float', 'i2c', 'max')]
    primary = [row for row in rows if row['gamma'] == '99/100']
    original = {row['circuit']: row for row in primary if row['reward'] == 'original'}
    candidate = {row['circuit']: row for row in primary if row['reward'] == 'candidate'}
    expected_original = {'int2float': (12, 2), 'i2c': (500, 4), 'max': (1, 0)}
    expected_candidate = {'int2float': 2, 'i2c': 2, 'max': 7}
    if any((original[name]['tied_top'], original[name]['optimal_top']) != target
           for name, target in expected_original.items()):
        raise ValueError('Original-reward audit does not reproduce the reference tie counts.')
    if any(candidate[name]['tied_top'] != target or not candidate[name]['all_top_optimal']
           for name, target in expected_candidate.items()):
        raise ValueError('Candidate reward does not rank only four-step optima highest.')
    if any(not row['all_top_optimal'] for row in rows if row['gamma'] == '1' and row['reward'] == 'candidate'):
        raise ValueError('Gamma-one candidate audit failed.')
    digest, evidence = identity()
    return dict(status='pass', fingerprint=digest,
                candidate_sha256=evidence['candidate_sha256'], rows=rows)


if __name__ == '__main__':
    result = audit()
    dump(HERE / 'audit.json', result)
    for row in result['rows']:
        if row['gamma'] == '99/100':
            print(row['circuit'], row['reward'], row['tied_top'], row['optimal_top'],
                  row['distinct_score_gap'])

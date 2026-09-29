"""Small deterministic checks for the fixed two-stage protocol and analysis."""
import copy
import unittest

from study import CIRCUITS, GROUPS, SEEDS, TEN, eval_seeds, training_config
from analyze import stage1_contrasts


class TwoStageTests(unittest.TestCase):
    def test_five_groups_change_only_prescribed_factors(self):
        self.assertEqual((CIRCUITS, SEEDS), (('int2float', 'i2c', 'max'), (0, 1, 2)))
        self.assertEqual(len(GROUPS), 5)
        legacy = training_config('legacy')
        self.assertEqual((legacy['protocol']['episodes'], legacy['protocol']['iterations']),
                         (100, 10))
        for group in GROUPS[1:]:
            cfg = training_config(group)
            self.assertEqual(cfg['protocol'], legacy['protocol'])
            self.assertEqual(cfg['environment'], legacy['environment'])
            expected = copy.deepcopy(legacy['method'])
            expected['optimizer']['learning_rate'] = TEN['learning_rates'][group[:4]]
            if group.endswith('_new'):
                expected['reward'] = dict(kind='best_feasible_v1', **TEN['reward'])
            self.assertEqual(cfg['method'], expected)

    def test_paired_evaluation_seed_banks_are_disjoint(self):
        banks = [eval_seeds(seed) for seed in SEEDS]
        self.assertEqual([len(bank) for bank in banks], [30, 30, 30])
        self.assertEqual([bank[0] for bank in banks], [40000, 40030, 40060])
        self.assertEqual(len(set(sum((list(bank) for bank in banks), []))), 90)

    def test_factorial_interaction_and_metric_direction(self):
        rows = []
        values = {'legacy': 12, 'w001_old': 11, 'w001_new': 8,
                  'w010_old': 10, 'w010_new': 5}
        for circuit in CIRCUITS:
            for seed in SEEDS:
                for group, score in values.items():
                    rows.append(dict(circuit=circuit, seed=seed, group=group,
                                     primary=score))
        contrasts = stage1_contrasts(rows)
        selected = {(r['circuit'], r['seed'], r['contrast']): r['improvement']
                    for r in contrasts}
        for circuit in CIRCUITS:
            for seed in SEEDS:
                self.assertEqual(selected[circuit, seed, 'final_minus_legacy'], -7 if circuit == 'max' else 7)
                self.assertEqual(selected[circuit, seed, 'interaction'], -2 if circuit == 'max' else 2)


if __name__ == '__main__':
    unittest.main()

"""Protocol, recovery, data-integrity, and statistical acceptance tests."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from analyze import (exact_sign_p, holm_adjust, parse_episode, statistics)
from context import (EVAL_GROUPS, SEEDS, SPEC, evaluation_seeds, prior,
                     training_config)
from evaluate import rollout
from drills.model import ActorCritic
from run import TenStepWindowA2C, check_config_isolation, first_episode


class ProtocolTests(unittest.TestCase):
    def test_reward_is_the_only_training_factor(self):
        check_config_isolation()

    def test_paired_disjoint_evaluation_banks(self):
        banks = [set(evaluation_seeds(seed)) for seed in SEEDS]
        self.assertTrue(all(len(bank) == 30 for bank in banks))
        for i, left in enumerate(banks):
            for right in banks[i + 1:]:
                self.assertFalse(left & right)

    def test_sign_test_holm_and_ties(self):
        self.assertEqual(exact_sign_p(9, 10), 11 / 1024)
        self.assertEqual(exact_sign_p(8, 10), 56 / 1024)
        self.assertEqual(holm_adjust([11 / 1024, 11 / 1024]),
                         [22 / 1024, 22 / 1024])
        self.assertEqual(holm_adjust([11 / 1024, 56 / 1024]),
                         [22 / 1024, 56 / 1024])

    def test_seed_level_decision_requires_nine_wins(self):
        def make_rows(i2c_wins, max_wins):
            rows = []
            for circuit in ('i2c', 'max'):
                for seed in SEEDS:
                    for group in EVAL_GROUPS:
                        for eval_seed in evaluation_seeds(seed):
                            if circuit == 'i2c':
                                lut = 320 if group != 'candidate' or \
                                    seed >= SEEDS[i2c_wins] else 319
                                feasible = 1
                            else:
                                feasible = int(group == 'candidate' and
                                               seed < SEEDS[max_wins])
                                lut = 770 if feasible else None
                            rows.append(dict(circuit=circuit, group=group,
                                             training_seed=seed,
                                             evaluation_seed=eval_seed,
                                             feasible=feasible,
                                             best_feasible_luts=lut))
            return rows

        paired, results = statistics(make_rows(9, 9))
        self.assertEqual(len(paired), 20)
        self.assertEqual([results[name]['conclusion'] for name in ('i2c', 'max')],
                         ['通过', '通过'])
        _, results = statistics(make_rows(8, 9))
        self.assertEqual(results['i2c']['conclusion'], '证据不足')
        self.assertEqual(results['max']['conclusion'], '通过')
        with self.assertRaises(ValueError):
            statistics(make_rows(9, 9)[:-1])
        with self.assertRaises(ValueError):
            rows = make_rows(9, 9)
            statistics(rows + rows[:1])

    def test_malformed_or_duplicate_mapping_rejected(self):
        cfg = training_config('i2c', 'original')
        circuit = cfg['protocol']['circuits']['i2c']
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'log.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['iteration', 'optimization', 'luts', 'levels', 'reward'])
                writer.writerow([0, 'strash', 365, 4, 0])
                for i in range(1, 11):
                    writer.writerow([i, 'rewrite', 365, 4, 0])
            parsed = parse_episode(path, circuit, cfg, 1, 'original')
            self.assertEqual(parsed.best['iteration'], 0)
            with path.open('a', newline='') as stream:
                csv.writer(stream).writerow([10, 'rewrite', 365, 4, 0])
            with self.assertRaises(ValueError):
                parse_episode(path, circuit, cfg, 1, 'original')


class ABCIntegrationTests(unittest.TestCase):
    def test_pair_window_resume_and_frozen_evaluation(self):
        original = training_config('i2c', 'original')
        candidate = training_config('i2c', 'candidate')
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            direct_path = root / 'direct'
            interrupted_path = root / 'interrupted'
            candidate_path = root / 'candidate'
            direct = TenStepWindowA2C(original, 'i2c', 20, direct_path, 'original')
            candidate_agent = TenStepWindowA2C(candidate, 'i2c', 20, candidate_path,
                                               'candidate')
            self.assertEqual(prior.old.state_hash(direct.network.state_dict()),
                             prior.old.state_hash(candidate_agent.network.state_dict()))
            self.assertEqual(prior.old.state_hash(direct.optimizer.state_dict()),
                             prior.old.state_hash(candidate_agent.optimizer.state_dict()))
            self.assertEqual(prior.old.state_hash(direct.rng.get_state()),
                             prior.old.state_hash(candidate_agent.rng.get_state()))
            direct.run_episode()
            candidate_agent.run_episode()
            self.assertEqual(first_episode(direct_path), first_episode(candidate_path))
            self.assertNotEqual(direct.rewards[0], candidate_agent.rewards[0])

            frozen = ActorCritic(len(original['method']['features']),
                                 len(original['protocol']['actions']),
                                 original['method']['network'])
            frozen.load_state_dict(prior.old.load(direct_path / 'snapshots/0.pt')['network'])
            frozen.eval().requires_grad_(False)
            before = prior.old.state_hash(frozen.state_dict())
            result = rollout(original, 'i2c', frozen, 90000, root / 'frozen')
            self.assertEqual(result['mapping_calls'], 11)
            self.assertEqual(before, prior.old.state_hash(frozen.state_dict()))

            for _ in range(8):
                direct.run_episode()
            interrupted = TenStepWindowA2C(original, 'i2c', 20, interrupted_path,
                                           'original')
            for _ in range(8):
                interrupted.run_episode()
            self.assertEqual(interrupted.episodes_completed, 8)
            del interrupted
            restored = TenStepWindowA2C(original, 'i2c', 20, interrupted_path,
                                        'original', resume=True)
            self.assertEqual(len(restored.return_window), 8)
            restored.run_episode()
            self.assertEqual(restored.episodes_completed, 9)
            for key in ('network', 'optimizer', 'rng_state', 'return_window', 'best'):
                a = prior.old.load(direct_path / 'checkpoint.pt')[key]
                b = prior.old.load(interrupted_path / 'checkpoint.pt')[key]
                self.assertEqual(prior.old.state_hash(a), prior.old.state_hash(b), key)
            self.assertEqual(direct.rewards, restored.rewards)
            diagnostics = [json.loads(line) for line in
                           (interrupted_path / 'return-diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(len(diagnostics), 9)
            self.assertEqual([row['mode'] for row in diagnostics],
                             ['within_episode_warmup'] * 8 + ['lagged_window'])
            self.assertTrue(all(len(row) == 6 for row in diagnostics))


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Checks for reward conservation, audited ranking, lookup continuation, and exact mass."""
import copy
from fractions import Fraction
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from audit import audit
from context import prior, training_config
from exact_eval import exact
from reward import RewardTracker
from run import TableWindowA2C, TenStepRewardA2C
from table import feature_states, table


class RewardExperimentTests(unittest.TestCase):
    def test_reward_conserves_best_feasible_improvement(self):
        tracker = RewardTracker(365, 4, 4)
        self.assertEqual(tracker.step_exact(350, 4), Fraction(1500, 365))
        self.assertEqual(tracker.step_exact(355, 4), 0)
        self.assertEqual(tracker.step_exact(340, 4), Fraction(1000, 365))
        self.assertEqual(tracker.step_exact(339, 5), 0)
        self.assertEqual(tracker.step_exact(340, 4), 0)
        self.assertEqual(tracker.best_luts, 340)

        max_tracker = RewardTracker(842, 56, 41)
        bonus = max_tracker.step_exact(840, 51)
        self.assertEqual(bonus, Fraction(10, 3))
        self.assertEqual(max_tracker.step_exact(830, 53), 0)
        first = max_tracker.step_exact(800, 41)
        self.assertEqual(bonus + first, Fraction(100) + Fraction(4200, 842))
        self.assertEqual(max_tracker.step_exact(810, 40), 0)
        self.assertEqual(max_tracker.step_exact(790, 41), Fraction(1000, 842))

    def test_audit_reproduces_reference_winners(self):
        result = audit()
        self.assertEqual(result['status'], 'pass')
        for circuit, expected in {'int2float': 2, 'i2c': 2, 'max': 7}.items():
            row = next(row for row in result['rows'] if row['circuit'] == circuit and
                       row['reward'] == 'candidate' and row['gamma'] == '99/100')
            self.assertEqual(row['tied_top'], expected)
            self.assertEqual(row['optimal_top'], expected)
            self.assertGreater(row['distinct_score_gap'], 0)

    def test_table_states_and_resume_are_bitwise_stable(self):
        cfg = training_config('i2c', 'window')
        states = feature_states('i2c')
        self.assertEqual(len(states), 400)
        self.assertEqual(len([key for key in table() if key[0] == 'i2c']), 2801)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = TableWindowA2C(cfg, 'i2c', 12, root / 'complete', 'window')
            interrupted = TableWindowA2C(cfg, 'i2c', 12, root / 'interrupted', 'window')
            for _ in range(12):
                complete.run_episode()
            for _ in range(5):
                interrupted.run_episode()
            restored = TableWindowA2C(cfg, 'i2c', 12, root / 'interrupted', 'window', resume=True)
            for _ in range(7):
                restored.run_episode()
            self.assertEqual(prior.old.state_hash(complete.network.state_dict()),
                             prior.old.state_hash(restored.network.state_dict()))
            self.assertEqual(prior.old.state_hash(complete.optimizer.state_dict()),
                             prior.old.state_hash(restored.optimizer.state_dict()))
            self.assertTrue(torch.equal(complete.rng.get_state(), restored.rng.get_state()))
            self.assertEqual(complete.return_window, restored.return_window)

    def test_exact_random_policy_probability_sums_to_one(self):
        values = exact.outcomes(table(), 'max')
        probs = exact.action_probabilities(None, {})
        result = exact.evaluate(probs, values, 'max', None, 'uniform', None)
        self.assertAlmostEqual(result['probability_sum'], 1.0, places=6)
        self.assertAlmostEqual(result['optimal_probability'], 19 / 2401, places=6)

    def test_ten_step_window_accepts_ten_returns(self):
        cfg = training_config('max', 'reward', ten_step=True)
        with tempfile.TemporaryDirectory() as temporary:
            agent = TenStepRewardA2C(cfg, 'max', 11, Path(temporary))
            states = [np.zeros(9, dtype=np.float32) for _ in range(10)]
            agent._update(states, [0] * 10, [0.0] * 10)
            self.assertEqual(len(agent.return_window), 1)
            self.assertEqual(len(agent.return_window[0]), 10)


if __name__ == '__main__':
    unittest.main()

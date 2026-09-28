"""Focused checks for the exact distribution and rolling-return continuation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from exact import PREFIXES, action_probabilities, evaluate, outcomes
from run import WindowA2C
from support import candidates, old, old_config


class FollowupTests(unittest.TestCase):
    def test_uniform_exact_distribution(self):
        table = candidates()
        uniform = action_probabilities(None, {})
        self.assertEqual(len(uniform), 400)
        max_score = evaluate(uniform, outcomes(table, 'max'), 'max', None, 'uniform', None)
        self.assertAlmostEqual(max_score['probability_sum'], 1, places=6)
        self.assertAlmostEqual(max_score['feasible_probability'], 26 / 2401, places=6)
        # Two paths reach the optimum before their fourth action.
        self.assertAlmostEqual(max_score['optimal_probability'], 19 / 2401, places=6)
        self.assertAlmostEqual(max_score['curves']['10']['any_feasible'],
                               1 - (1 - 26 / 2401) ** 10, places=6)

    def test_warmup_parity_and_exact_resume(self):
        cfg, _ = old_config()
        cfg = copy.deepcopy(cfg)
        cfg['runtime'].update(experiment_group='test', experiment_fingerprint='test')
        circuit = cfg['protocol']['circuits']['int2float']
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            control = old.A2C(cfg, circuit, 0, base / 'control')
            rolling = WindowA2C(cfg, circuit, 0, base / 'rolling')
            partial = WindowA2C(cfg, circuit, 0, base / 'partial')
            for episode in range(1, 9):
                control.run_episode()
                rolling.run_episode()
                if episode <= 4:
                    partial.run_episode()
                self.assertEqual(old.state_hash(control.network.state_dict()),
                                 old.state_hash(rolling.network.state_dict()))
                self.assertEqual(old.state_hash(control.optimizer.state_dict()),
                                 old.state_hash(rolling.optimizer.state_dict()))
                self.assertEqual((base / 'control/episodes' / str(episode) / 'log.csv').read_bytes(),
                                 (base / 'rolling/episodes' / str(episode) / 'log.csv').read_bytes())
            resumed = WindowA2C(cfg, circuit, 0, base / 'partial', resume=True)
            for _ in range(4):
                resumed.run_episode()
            self.assertEqual(old.state_hash(rolling.network.state_dict()),
                             old.state_hash(resumed.network.state_dict()))
            self.assertEqual(old.state_hash(rolling.optimizer.state_dict()),
                             old.state_hash(resumed.optimizer.state_dict()))
            self.assertEqual(rolling.return_window, resumed.return_window)
            control.run_episode()
            rolling.run_episode()
            resumed.run_episode()
            self.assertEqual((base / 'control/episodes/9/log.csv').read_bytes(),
                             (base / 'rolling/episodes/9/log.csv').read_bytes())
            self.assertEqual(old.state_hash(rolling.network.state_dict()),
                             old.state_hash(resumed.network.state_dict()))
            self.assertEqual(old.state_hash(rolling.optimizer.state_dict()),
                             old.state_hash(resumed.optimizer.state_dict()))
            self.assertEqual(rolling.return_window, resumed.return_window)
            metrics = [json.loads(line) for line in (base / 'rolling/return-diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(len(metrics), 9)
            self.assertTrue(all(row['mode'] == 'within_episode_warmup' for row in metrics[:8]))
            self.assertEqual(metrics[8]['mode'], 'lagged_window')


if __name__ == '__main__':
    unittest.main()

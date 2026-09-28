"""Focused parity and independent-evaluation checks using the installed ABC/Yosys."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from common import A2C, config, evaluation_seeds, load, state_hash
from evaluate import inference_rollout
from run import DiagnosticA2C


class FourStepTests(unittest.TestCase):
    def test_seed_banks_are_paired_within_seed_and_distinct_across_seeds(self):
        banks = [set(evaluation_seeds(seed)) for seed in range(10)]
        self.assertTrue(all(len(bank) == 10 for bank in banks))
        self.assertEqual(len(set.union(*banks)), 100)

    def test_diagnostics_preserve_original_update_and_evaluation_freezes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probes = root / 'probes.npy'
            np.save(probes, np.zeros((196, 9), dtype=np.float32), allow_pickle=False)
            for learning_rate in (0.001, 0.01):
                cfg = copy.deepcopy(config())
                cfg['protocol']['episodes'] = 2
                cfg['method']['optimizer']['learning_rate'] = learning_rate
                cfg['runtime'].update(experiment_group=f'lr-{learning_rate}', experiment_fingerprint='test')
                circuit = cfg['protocol']['circuits']['int2float']
                label = str(learning_rate)
                ordinary = A2C(cfg, circuit, 0, root / label / 'ordinary')
                diagnostic = DiagnosticA2C(cfg, circuit, 0, root / label / 'diagnostic', probes)
                self.assertEqual(state_hash(ordinary.network.state_dict()),
                                 state_hash(diagnostic.network.state_dict()))
                for episode in range(1, 3):
                    ordinary.run_episode()
                    diagnostic.run_episode()
                    self.assertEqual(state_hash(ordinary.network.state_dict()),
                                     state_hash(diagnostic.network.state_dict()))
                    self.assertEqual(state_hash(ordinary.optimizer.state_dict()),
                                     state_hash(diagnostic.optimizer.state_dict()))
                    self.assertTrue(torch.equal(ordinary.rng.get_state(), diagnostic.rng.get_state()))
                    self.assertEqual((root / label / 'ordinary/episodes' / str(episode) / 'log.csv').read_bytes(),
                                     (root / label / 'diagnostic/episodes' / str(episode) / 'log.csv').read_bytes())
                metrics = [json.loads(line) for line in diagnostic.metrics_path.read_text().splitlines()]
                self.assertEqual([row['episode'] for row in metrics], [0, 1, 2])
                self.assertEqual(len(metrics[2]['actor_layer_variance']), 3)
                self.assertEqual(len(metrics[2]['critic_layer_variance']), 2)
                self.assertTrue(np.isfinite([row['entropy'] for row in metrics]).all())
                before = state_hash(diagnostic.network.state_dict())
                first = inference_rollout(cfg, circuit, diagnostic.network, 40000,
                                          root / label / 'evaluation-a')
                second = inference_rollout(cfg, circuit, diagnostic.network, 40000,
                                           root / label / 'evaluation-b')
                self.assertEqual(first['actions'], second['actions'])
                self.assertEqual(first['best'], second['best'])
                self.assertEqual(before, state_hash(diagnostic.network.state_dict()))
                restored = DiagnosticA2C(cfg, circuit, 0, root / label / 'diagnostic', probes, resume=True)
                self.assertEqual(restored.episodes_completed, 2)
                self.assertEqual(before, state_hash(restored.network.state_dict()))


if __name__ == '__main__':
    unittest.main()

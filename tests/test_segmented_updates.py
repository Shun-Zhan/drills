"""Protocol checks with the real i2c circuit and ABC/Yosys toolchain.

Run from the repository root with the pinned Python:
  ../DRiLLS/.tools/conda-env/bin/python -B -m unittest discover -s tests -v
All experiment output goes to TemporaryDirectory; no scored study is launched.
"""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from drills.model import A2C, Normalizer
from drills.segmented_model import SegmentedA2C, nstep_returns, state_hash


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT.parent / 'DRiLLS' / '.tools' / 'conda-env' / 'bin'
STUDY_SPEC = importlib.util.spec_from_file_location('segmented_study',
    ROOT / 'experiments' / 'segmented-updates' / 'study.py')
STUDY = importlib.util.module_from_spec(STUDY_SPEC)
STUDY_SPEC.loader.exec_module(STUDY)


def experiment_config(group='D', horizon=6, interval=2, episodes=2):
    config = STUDY.group_config(STUDY.config(), group,
                               fingerprint='test-fixed-source-and-tool-fingerprint')
    config['protocol']['iterations'] = horizon
    config['protocol']['episodes'] = episodes
    method = config['method']
    method.update(learner='legacy' if group == 'A' else 'nstep',
                  update_interval=interval, learning_enabled=group not in ('F', 'U'),
                  uniform_actions=group == 'U')
    method['normalization']['returns'] = 'standardize' if group == 'A' else 'none'
    config['runtime'].update(abc_binary=str(TOOLS / 'yosys-abc'),
        yosys_binary=str(TOOLS / 'yosys'), experiment_group=group,
        experiment_fingerprint='test-fixed-source-and-tool-fingerprint')
    return config


def read_jsonl(folder, filename):
    path = Path(folder) / filename
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def read_csv(folder, episode):
    with (Path(folder) / 'episodes' / str(episode) / 'log.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def finish(agent):
    while agent.episodes_completed < agent.protocol['episodes']:
        agent.run_episode()
    return agent


def canonical_netlist(text):
    # ABC includes its wall-clock creation time in one header comment. The
    # Boolean structure must match exactly after removing only that metadata.
    return '\n'.join(line for line in text.splitlines() if not line.startswith('// Benchmark '))


class SegmentedUpdatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        absent = [str(path) for path in (TOOLS / 'yosys-abc', TOOLS / 'yosys') if not path.is_file()]
        if absent:
            raise RuntimeError('Required pinned toolchain is unavailable: ' + ', '.join(absent))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='drills-segmented-test-')
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def agent(self, name, group='D', horizon=6, interval=2, episodes=2, seed=10, resume=False):
        config = experiment_config(group, horizon, interval, episodes)
        return SegmentedA2C(config, config['protocol']['circuits']['i2c'], seed,
                            self.folder / name, resume=resume)

    def assert_agent_equal(self, left, right):
        for label in ('network', 'optimizer'):
            self.assertEqual(state_hash(getattr(left, label).state_dict()),
                             state_hash(getattr(right, label).state_dict()), label)
        self.assertTrue(torch.equal(left.rng.get_state(), right.rng.get_state()))
        for field in ('episodes_completed', 'candidates_completed', 'updates_completed',
                      'rewards', 'trajectory_rewards', 'active'):
            self.assertEqual(getattr(left, field), getattr(right, field), field)
        self.assertEqual(left.game.best, right.game.best)
        self.assertEqual({name: canonical_netlist(data) for name, data in left.game.best_netlists.items()},
                         {name: canonical_netlist(data) for name, data in right.game.best_netlists.items()})
        for filename in ('steps.jsonl', 'segments.jsonl'):
            self.assertEqual(read_jsonl(left.folder, filename), read_jsonl(right.folder, filename), filename)
        for episode in range(1, left.episodes_completed + 1):
            self.assertEqual(read_csv(left.folder, episode), read_csv(right.folder, episode))

    def test_nstep_targets_hand_calculated_reward_units(self):
        targets = nstep_returns([1, 2, 3], 0.5, 4)
        np.testing.assert_array_equal(targets, np.asarray([3.25, 4.5, 5], dtype=np.float32))
        self.assertEqual(targets.dtype, np.float32)
        np.testing.assert_array_equal(nstep_returns([1, 2, 3], 0.5, 0),
                                      np.asarray([2.75, 3.5, 3], dtype=np.float32))
        self.assertEqual(nstep_returns([], 0.99, 1).shape, (0,))

    def test_production_groups_locked_budgets_and_original_settings(self):
        config = STUDY.config()
        self.assertEqual(config['protocol']['seeds'], [10, 11, 12])
        self.assertEqual(config['experiment']['evaluation_seeds'], list(range(30000, 30030)))
        updates = {'A': 100, 'B': 100, 'C': 20, 'D': 100, 'F': 0, 'U': 0}
        for group in STUDY.GROUPS:
            local = STUDY.group_config(config, group)
            self.assertEqual(local['protocol']['iterations'] * local['protocol']['episodes'], 1000)
            estimated = 1000 // local['method']['update_interval'] if local['method']['learning_enabled'] else 0
            self.assertEqual(estimated, updates[group])
            self.assertEqual(local['method']['normalization']['state'], 'episode_welford')
            self.assertEqual(local['method']['normalization']['returns'],
                             'standardize' if group == 'A' else 'none')
            self.assertEqual(local['protocol']['lut_inputs'], 6)
            self.assertEqual(local['protocol']['circuits']['i2c']['max_levels'], 4)

    def test_state_hash_tracks_dtype_shape_and_values(self):
        first = {'b': [torch.tensor([1., 2.])], 'a': {'seed': 10}}
        self.assertEqual(state_hash(first), state_hash(copy.deepcopy(first)))
        self.assertEqual(state_hash(first), state_hash({'a': {'seed': 10}, 'b': first['b']}))
        second = copy.deepcopy(first)
        second['b'][0][0] = 3
        self.assertNotEqual(state_hash(first), state_hash(second))
        self.assertNotEqual(state_hash(first['b'][0]), state_hash(first['b'][0].double()))

    def test_legacy_exact_original_weights_adam_rng_and_episode_logs(self):
        config = experiment_config('A', horizon=10, interval=10, episodes=3)
        circuit = config['protocol']['circuits']['i2c']
        original = A2C(copy.deepcopy(config), circuit, 10, self.folder / 'original')
        segmented = SegmentedA2C(config, circuit, 10, self.folder / 'legacy')
        self.assertEqual(state_hash(original.network.state_dict()), segmented.initial_hash)
        for episode in range(1, 4):
            self.assertEqual(original.run_episode(), segmented.run_episode())
            for name in ('network', 'optimizer'):
                self.assertEqual(state_hash(getattr(original, name).state_dict()),
                                 state_hash(getattr(segmented, name).state_dict()), (episode, name))
            self.assertTrue(torch.equal(original.rng.get_state(), segmented.rng.get_state()))
            self.assertEqual(original.rewards, segmented.rewards)
            self.assertEqual(original.game.best, segmented.game.best)
            self.assertEqual(read_csv(self.folder / 'original', episode), read_csv(segmented.folder, episode))
        self.assertEqual(segmented.candidates_completed, 30)
        self.assertEqual(segmented.updates_completed, 3)

    def test_first_trajectory_pairing_and_equal_initial_weights(self):
        agents = {}
        for group in ('C', 'D', 'F'):
            agents[group] = self.agent(group, group, horizon=50,
                                       interval=10 if group == 'D' else 50, episodes=1)
        self.assertEqual(len({agent.initial_hash for agent in agents.values()}), 1)
        for agent in agents.values():
            agent.run_episode()
        steps = {group: read_jsonl(agent.folder, 'steps.jsonl') for group, agent in agents.items()}
        # Prefix comparisons include the observed state, policy probabilities,
        # RNG-sampled action and ABC mapping, not only the action labels.
        for field in ('action', 'raw_state', 'normalized_state', 'probabilities',
                      'luts', 'levels', 'reward', 'next_raw_state', 'normalizer_n'):
            self.assertEqual([row[field] for row in steps['C'][:10]],
                             [row[field] for row in steps['D'][:10]], field)
            self.assertEqual([row[field] for row in steps['C']],
                             [row[field] for row in steps['F']], field)
        self.assertEqual(agents['C'].updates_completed, 1)
        self.assertEqual(agents['D'].updates_completed, 5)
        self.assertEqual(agents['F'].updates_completed, 0)

    def test_boundary_bootstrap_terminal_zero_and_welford_once(self):
        agent = self.agent('boundary', horizon=5, interval=2, episodes=1)
        first = agent.run_segment()
        self.assertFalse(first['terminal'])
        self.assertEqual(first['end_iteration'], 2)
        self.assertEqual(first['normalizer_n'], 3)
        steps = read_jsonl(agent.folder, 'steps.jsonl')
        with torch.no_grad():
            # The checkpoint has updated parameters; use the pre-update value
            # from the next step only for cache equality, not for this target.
            self.assertEqual(first['next_normalized_state'], steps[-1]['next_normalized_state'])
        np.testing.assert_array_equal(np.asarray(first['targets'], dtype=np.float32),
            nstep_returns(first['rewards'], agent.method['gamma'], first['bootstrap']))
        next_segment = agent.run_segment()
        steps = read_jsonl(agent.folder, 'steps.jsonl')
        self.assertEqual(steps[2]['normalized_state'], first['next_normalized_state'])
        self.assertEqual(next_segment['normalizer_n'], 5)
        terminal = agent.run_segment()
        steps = read_jsonl(agent.folder, 'steps.jsonl')
        self.assertTrue(terminal['terminal'])
        self.assertEqual(terminal['bootstrap'], 0.0)
        self.assertIsNone(terminal['next_normalized_state'])
        self.assertEqual(terminal['normalizer_n'], 5)
        self.assertEqual(terminal['targets'][-1], terminal['rewards'][-1])
        self.assertEqual(agent.candidates_completed, 5)
        self.assertEqual(agent.updates_completed, 3)
        self.assertEqual(agent.game.iteration, 5)
        self.assertEqual(len(agent.game.sequence), len(agent.protocol['initial_sequence']) + 5)
        replay = Normalizer(len(agent.method['features']), agent.method['normalization'])
        for row in steps:
            np.testing.assert_array_equal(replay.normalize(np.asarray(row['raw_state'], dtype=np.float32)),
                                           np.asarray(row['normalized_state'], dtype=np.float32))
            self.assertEqual(row['normalizer_n'], min(row['iteration'] + 1, 5))
        self.assertEqual(replay.n, agent.normalizer.n)
        np.testing.assert_array_equal(replay.mean, agent.normalizer.mean)
        np.testing.assert_array_equal(replay.mean_diff, agent.normalizer.mean_diff)

    def test_bootstrap_uses_pre_update_critic_and_detached_target(self):
        agent = self.agent('bootstrap', horizon=4, interval=2, episodes=1)
        update = agent._update
        observed = {}

        def capture(states, actions, rewards, bootstrap):
            with torch.no_grad():
                observed['value'] = float(agent.network(torch.as_tensor(agent.state))[1])
            observed['bootstrap'] = bootstrap
            return update(states, actions, rewards, bootstrap)

        with patch.object(agent, '_update', side_effect=capture):
            first = agent.run_segment()
        self.assertEqual(observed['value'], observed['bootstrap'])
        self.assertEqual(first['bootstrap'], observed['value'])
        self.assertNotEqual(first['network_before'], first['network_after'])
        self.assertGreater(first['critic_gradient_norm'], 0)
        self.assertGreater(first['actor_gradient_norm'], 0)

    def test_actor_advantage_does_not_backpropagate_into_critic(self):
        # Disable each loss in turn. Detaching the actor advantage must keep
        # the critic fixed even though both heads participate in the forward.
        for actor_weight, critic_weight in ((1., 0.), (0., 1.)):
            with self.subTest(actor_weight=actor_weight):
                config = experiment_config('D', horizon=4, interval=2, episodes=1)
                config['method']['loss'].update(actor_weight=actor_weight, critic_weight=critic_weight)
                agent = SegmentedA2C(config, config['protocol']['circuits']['i2c'], 10,
                    self.folder / f'loss-{int(actor_weight)}')
                diagnostic = agent._update([np.zeros(9, dtype=np.float32), np.ones(9, dtype=np.float32)],
                                            [0, 1], [3, -1], 0)
                np.testing.assert_array_equal(np.asarray(diagnostic['targets'], dtype=np.float32),
                                               np.asarray([2.01, -1], dtype=np.float32))
                unchanged = 'critic' if critic_weight == 0 else 'actor'
                changed = 'actor' if actor_weight else 'critic'
                self.assertEqual(diagnostic[unchanged + '_gradient_norm'], 0)
                self.assertEqual(diagnostic[unchanged + '_update_rms'], 0)
                self.assertGreater(diagnostic[changed + '_gradient_norm'], 0)
                self.assertGreater(diagnostic[changed + '_update_rms'], 0)

    def test_frozen_and_uniform_network_adam_unchanged(self):
        for group in ('F', 'U'):
            with self.subTest(group=group):
                agent = finish(self.agent(group, group, horizon=6, interval=2, episodes=2))
                self.assertEqual(state_hash(agent.network.state_dict()), agent.initial_hash)
                self.assertEqual(state_hash(agent.optimizer.state_dict()), agent.initial_optimizer_hash)
                self.assertEqual(agent.updates_completed, 0)
                self.assertEqual(agent.candidates_completed, 12)
                self.assertEqual(agent.optimizer.state, {})
                for segment in read_jsonl(agent.folder, 'segments.jsonl'):
                    self.assertEqual(segment['actor_gradient_norm'], 0)
                    self.assertEqual(segment['critic_gradient_norm'], 0)
                    self.assertEqual(segment['actor_update_rms'], 0)
                    self.assertEqual(segment['critic_update_rms'], 0)
                if group == 'U':
                    expected = torch.full((len(agent.protocol['actions']),), 1 / len(agent.protocol['actions']))
                    for step in read_jsonl(agent.folder, 'steps.jsonl'):
                        self.assertEqual(step['probabilities'], expected.tolist())

    def test_real_tool_resume_between_segments_exact(self):
        uninterrupted = finish(self.agent('continuous'))
        interrupted = self.agent('between')
        interrupted.run_segment()
        committed = interrupted.candidates_completed
        recovered = self.agent('between', resume=True)
        self.assertEqual(recovered.candidates_completed, committed)
        self.assertEqual(recovered.restored_discarded_rows, 0)
        finish(recovered)
        self.assert_agent_equal(uninterrupted, recovered)

    def test_real_tool_resume_discards_failed_update_segment(self):
        uninterrupted = finish(self.agent('continuous'))
        interrupted = self.agent('update-failure')
        interrupted.run_segment()
        with patch.object(interrupted, '_update', side_effect=RuntimeError('injected before segment commit')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                interrupted.run_segment()
        self.assertEqual(len(read_jsonl(interrupted.folder, 'steps.jsonl')), 4)
        recovered = self.agent('update-failure', resume=True)
        self.assertEqual(recovered.candidates_completed, 2)
        self.assertEqual(recovered.restored_discarded_rows, 2)
        self.assertEqual(len(read_jsonl(recovered.folder, 'steps.jsonl')), 2)
        self.assertEqual(len(read_csv(recovered.folder, 1)), 3)
        finish(recovered)
        self.assert_agent_equal(uninterrupted, recovered)

    def test_real_tool_resume_discards_failure_mid_segment(self):
        uninterrupted = finish(self.agent('continuous'))
        interrupted = self.agent('step-failure')
        interrupted.run_segment()
        real_step = interrupted.game.step
        calls = 0

        def fail_second(action):
            nonlocal calls
            calls += 1
            result = real_step(action)
            if calls == 2:
                raise RuntimeError('injected after ABC before step logging')
            return result

        with patch.object(interrupted.game, 'step', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                interrupted.run_segment()
        recovered = self.agent('step-failure', resume=True)
        self.assertEqual(recovered.restored_discarded_rows, 1)
        self.assertEqual(recovered.game.iteration, 2)
        self.assertEqual(len(read_csv(recovered.folder, 1)), 3)
        finish(recovered)
        self.assert_agent_equal(uninterrupted, recovered)
        for filename in ('best.v', 'best-mapped.v'):
            output = subprocess.check_output([str(TOOLS / 'yosys-abc'), '-c',
                f'cec "{recovered.game.circuit["file"]}" "{recovered.folder / filename}";'], text=True)
            self.assertIn('Networks are equivalent', output)

    def test_real_tool_resume_after_first_uncommitted_segment(self):
        uninterrupted = finish(self.agent('continuous', horizon=4, interval=2, episodes=1))
        interrupted = self.agent('first-failure', horizon=4, interval=2, episodes=1)
        with patch.object(interrupted, '_update', side_effect=RuntimeError('injected first segment')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                interrupted.run_segment()
        recovered = self.agent('first-failure', horizon=4, interval=2, episodes=1, resume=True)
        self.assertEqual(recovered.candidates_completed, 0)
        self.assertEqual(recovered.restored_discarded_rows, 2)
        finish(recovered)
        self.assert_agent_equal(uninterrupted, recovered)

    def test_resume_rejects_group_fingerprint_seed_and_hyperparameter_changes(self):
        agent = self.agent('identity')
        agent.run_segment()
        changes = ('group', 'fingerprint', 'gamma', 'learning_rate', 'horizon', 'seed', 'actions')
        for change in changes:
            with self.subTest(change=change):
                config = copy.deepcopy(agent.config)
                seed = 10
                if change == 'group':
                    config['runtime']['experiment_group'] = 'C'
                elif change == 'fingerprint':
                    config['runtime']['experiment_fingerprint'] = 'changed-source'
                elif change == 'gamma':
                    config['method']['gamma'] = 0.5
                elif change == 'learning_rate':
                    config['method']['optimizer']['learning_rate'] = 0.01
                elif change == 'horizon':
                    config['protocol']['iterations'] = 8
                elif change == 'seed':
                    seed = 11
                elif change == 'actions':
                    config['protocol']['actions'] = list(reversed(config['protocol']['actions']))
                with self.assertRaisesRegex(ValueError, 'Checkpoint learner, protocol, seed or fingerprint changed'):
                    SegmentedA2C(config, config['protocol']['circuits']['i2c'], seed, agent.folder, resume=True)

    def test_complete_budget_cannot_extend_or_overwrite(self):
        agent = finish(self.agent('complete', horizon=4, interval=2, episodes=1))
        for action in (agent.run_segment, agent.run_episode):
            with self.assertRaisesRegex(RuntimeError, 'budget is complete'):
                action()
        restored = self.agent('complete', horizon=4, interval=2, episodes=1, resume=True)
        with self.assertRaisesRegex(RuntimeError, 'budget is complete'):
            restored.run_segment()
        with self.assertRaisesRegex(FileExistsError, 'Never overwrite'):
            self.agent('complete', horizon=4, interval=2, episodes=1)
        changed = copy.deepcopy(agent.config)
        changed['protocol']['episodes'] = 2
        with self.assertRaisesRegex(ValueError, 'Checkpoint learner'):
            SegmentedA2C(changed, changed['protocol']['circuits']['i2c'], 10, agent.folder, resume=True)

    def test_invalid_learner_settings_fail_before_running_tools(self):
        valid = experiment_config()
        variants = []
        for value in (0, -1, 7, True, 1.5):
            cfg = copy.deepcopy(valid)
            cfg['method']['update_interval'] = value
            variants.append(cfg)
        for changes in ({'learner': 'unknown'}, {'learner': 'legacy'},
                        {'uniform_actions': True}, {'learning_enabled': 'false'}):
            cfg = copy.deepcopy(valid)
            cfg['method'].update(changes)
            variants.append(cfg)
        cfg = copy.deepcopy(valid)
        cfg['method']['normalization']['returns'] = 'standardize'
        variants.append(cfg)
        for index, cfg in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                SegmentedA2C(cfg, cfg['protocol']['circuits']['i2c'], 10, self.folder / str(index))

    def test_independent_evaluation_reproducible_and_does_not_update(self):
        agent = finish(self.agent('trained', horizon=10, interval=10, episodes=1))
        weights = state_hash(agent.network.state_dict())
        optimizer = state_hash(agent.optimizer.state_dict())
        rng = agent.rng.get_state().clone()
        checkpoint = agent.checkpoint.read_bytes()
        config = copy.deepcopy(agent.config)
        config['protocol'].update(episodes=1, iterations=50)
        first = STUDY.inference_rollout(config, agent.network, 30000, self.folder / 'evaluation-first')
        second = STUDY.inference_rollout(config, agent.network, 30000, self.folder / 'evaluation-second')
        self.assertEqual(first, second)
        self.assertEqual(len(first['actions']), 50)
        self.assertEqual(set(first['prefixes']), {'10', '50'})
        self.assertEqual(first['prefixes']['50']['best'], first['best'])
        self.assertEqual(first['prefixes']['50']['terminal'], first['terminal'])
        self.assertEqual(state_hash(agent.network.state_dict()), weights)
        self.assertEqual(state_hash(agent.optimizer.state_dict()), optimizer)
        self.assertTrue(torch.equal(agent.rng.get_state(), rng))
        self.assertEqual(agent.checkpoint.read_bytes(), checkpoint)
        self.assertEqual(agent.updates_completed, 1)
        self.assertEqual(agent.candidates_completed, 10)
        config['protocol']['iterations'] = 10
        ten = STUDY.inference_rollout(config, agent.network, 30000, self.folder / 'evaluation-ten')
        self.assertEqual(first['actions'][:10], ten['actions'])
        self.assertEqual(first['rewards'][:10], ten['rewards'])
        self.assertEqual(first['steps'][:10], ten['steps'])
        self.assertEqual(first['prefixes']['10'], ten['prefixes']['10'])

    def test_independent_uniform_evaluation_has_uniform_probabilities(self):
        agent = self.agent('uniform-initial', 'U', horizon=4, interval=2, episodes=1)
        config = copy.deepcopy(agent.config)
        config['protocol']['iterations'] = 4
        before = state_hash(agent.network.state_dict())
        row = STUDY.inference_rollout(config, agent.network, 30000, self.folder / 'uniform-evaluation', True)
        expected = torch.full((7,), 1 / 7).tolist()
        self.assertEqual(len(row['actions']), 4)
        self.assertEqual(state_hash(agent.network.state_dict()), before)
        self.assertTrue(all(step['probabilities'] == expected for step in row['steps']))


if __name__ == '__main__':
    unittest.main(verbosity=2)

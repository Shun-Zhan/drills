"""Four-step lookup environment from certified prefixes and feature states."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from context import OUT, prior, read, sha
from reward import RewardTracker, original_reward
from drills.fpga_session import FPGASession


@lru_cache(maxsize=1)
def table():
    return prior.candidates()


@lru_cache(maxsize=3)
def feature_states(circuit):
    path = prior.OUT / 'prefix-states' / f'{circuit}.json'
    saved = read(path)
    cfg, old_fingerprint = prior.old_config()
    if (saved['old_fingerprint'] != old_fingerprint or
            saved['candidates_sha256'] != sha(prior.ROOT / 'experiments/ten-step-optimality/candidates.csv') or
            saved['circuit'] != circuit or len(saved['rows']) != 400):
        raise ValueError(f'Uncertified four-step feature cache: {path}')
    result = {tuple(row['actions']): np.asarray(row['features'], dtype=np.float32)
              for row in saved['rows']}
    if len(result) != 400 or any(value.shape != (9,) or not np.isfinite(value).all()
                                 for value in result.values()):
        raise ValueError(f'Invalid feature states: {path}')
    return result


class TableSession:
    def __init__(self, cfg, circuit_name, folder, group, completed=0, best=None):
        if cfg['protocol']['iterations'] != 4:
            raise ValueError('Lookup environment is certified only for four actions.')
        self.cfg, self.circuit_name = cfg, circuit_name
        self.circuit = cfg['protocol']['circuits'][circuit_name]
        self.folder = Path(folder)
        self.group = group
        self.episode = completed
        self.iteration = 0
        self.best = best
        # A2C's resume path exports these fields before the table game is swapped in.
        # Placeholder text is overwritten by the final ABC replay and CEC check.
        self.best_netlists = ({'best.v': '', 'best-mapped.v': ''} if best else {})
        self.rows = table()
        self.states = feature_states(circuit_name)
        self.trajectory_path = self.folder / 'trajectories.jsonl'
        if self.trajectory_path.exists():
            records = [json.loads(line) for line in self.trajectory_path.read_text().splitlines()]
            records = [row for row in records if row['episode'] <= completed]
            if len(records) != completed or [r['episode'] for r in records] != list(range(1, completed+1)):
                raise ValueError('Table trajectory log does not match committed checkpoint.')
            self.trajectory_path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
        elif completed:
            raise ValueError('Missing table trajectory log for resumed experiment.')

    def reset(self):
        self.episode += 1
        self.iteration = 0
        self.prefix = ()
        self.sequence = list(self.cfg['protocol']['initial_sequence'])
        initial = self.rows[self.circuit_name, ()]
        self.luts, self.levels = initial['luts'], initial['levels']
        self.tracker = RewardTracker(self.luts, self.levels, self.circuit['max_levels'])
        self.episode_steps = [dict(iteration=0, actions=[], luts=self.luts,
                                   levels=self.levels, reward=0)]
        self._consider_best()
        return self.states[()].copy()

    def _consider_best(self):
        row = dict(luts=self.luts, levels=self.levels,
                   feasible=self.levels <= self.circuit['max_levels'],
                   sequence=list(self.sequence), episode=self.episode,
                   iteration=self.iteration)
        eligible = self.iteration > 0 or self.cfg['protocol']['evaluation']['include_initial']
        if eligible and (self.best is None or FPGASession.rank(row) < FPGASession.rank(self.best)):
            self.best = row
            self.best_netlists = {'best.v': '', 'best-mapped.v': ''}

    def step(self, action):
        if not self.episode or self.iteration >= 4 or action not in range(7):
            raise RuntimeError('Reset before starting another four-step episode.')
        previous = (self.luts, self.levels)
        self.prefix += (int(action),)
        self.iteration += 1
        self.sequence.append(self.cfg['protocol']['actions'][action])
        row = self.rows[self.circuit_name, self.prefix]
        self.luts, self.levels = row['luts'], row['levels']
        reward = (original_reward(self.cfg, self.circuit, previous, (self.luts, self.levels))
                  if self.group == 'window' else self.tracker.step(self.luts, self.levels))
        self.episode_steps.append(dict(iteration=self.iteration, actions=list(self.prefix),
                                       luts=self.luts, levels=self.levels, reward=reward))
        self._consider_best()
        done = self.iteration == 4
        if done:
            with self.trajectory_path.open('a') as stream:
                stream.write(json.dumps(dict(episode=self.episode, steps=self.episode_steps),
                                        allow_nan=False) + '\n')
        return (self.states[self.prefix].copy() if not done else np.zeros(9, dtype=np.float32),
                reward, done)

    def export_best(self):
        destination = self.folder / 'best.json'
        if self.best is None:
            destination.unlink(missing_ok=True)
        else:
            destination.write_text(json.dumps(self.best, indent=2) + '\n')

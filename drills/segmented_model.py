"""On-policy segments with fixed reward units and resumable segment commits.

The supplied A2C and environment are unchanged. ``legacy`` delegates its
update to the supplied A2C, so the A group has the original numerical behavior.
"""
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .fpga_session import FPGASession
from .model import A2C, ActorCritic, Normalizer


def state_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def nstep_returns(rewards, gamma, bootstrap):
    """A float32 target in reward units; bootstrap is a detached scalar."""
    result = np.empty(len(rewards), dtype=np.float32)
    cumulative = float(bootstrap)
    for i in reversed(range(len(rewards))):
        cumulative = float(rewards[i]) + gamma * cumulative
        result[i] = cumulative
    return result


def flattened(module):
    return torch.cat([p.detach().flatten().clone() for p in module.parameters()])


class SegmentedA2C:
    def __init__(self, config, circuit, seed, directory, resume=False):
        self.config = copy.deepcopy(config)
        self.method, self.protocol = self.config['method'], self.config['protocol']
        self.rule = self.method['learner']
        self.K, self.H = self.method['update_interval'], self.protocol['iterations']
        if self.rule not in ('legacy', 'nstep'):
            raise ValueError('Unknown learner.')
        if any(type(v) is not int or v <= 0 for v in (self.K, self.H)) or self.K > self.H:
            raise ValueError('Require integer 1 <= K <= H.')
        if self.rule == 'legacy' and self.K != self.H:
            raise ValueError('The legacy learner requires K=H.')
        self.learning = self.method['learning_enabled']
        self.uniform = self.method.get('uniform_actions', False)
        if type(self.learning) is not bool or type(self.uniform) is not bool:
            raise ValueError('Learning and uniform switches must be boolean.')
        if self.uniform and self.learning:
            raise ValueError('Uniform actions cannot train an on-policy network.')
        if self.method['normalization']['state'] != 'episode_welford':
            raise ValueError('This learner preserves episode_welford normalization.')
        if self.rule == 'nstep' and self.method['normalization']['returns'] != 'none':
            raise ValueError('nstep critic targets must remain in fixed reward units.')
        torch.set_num_threads(config['environment']['torch_threads'])
        torch.manual_seed(seed)
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.folder = Path(directory)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.game = FPGASession(self.config, circuit, self.folder)
        self.network = ActorCritic(len(self.method['features']), len(self.protocol['actions']),
                                   self.method['network']).to('cpu')
        opt = self.method['optimizer']
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt['learning_rate'],
            betas=opt['betas'], eps=opt['epsilon'], weight_decay=opt['weight_decay'])
        self.checkpoint = self.folder / 'checkpoint.pt'
        self.identity = dict(seed=seed, method=self.method, protocol=self.protocol,
            group=config['runtime'].get('experiment_group'),
            fingerprint=config['runtime'].get('experiment_fingerprint'))
        self.episodes_completed, self.updates_completed, self.candidates_completed = 0, 0, 0
        self.rewards, self.trajectory_rewards = [], []
        self.training_seconds = 0.0
        self.active = False
        self.normalizer = None
        self.raw_state = self.state = None
        self.restored_discarded_rows = 0
        self.initial_hash = state_hash(self.network.state_dict())
        self.initial_optimizer_hash = state_hash(self.optimizer.state_dict())
        if resume:
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
            if saved.get('format_version') != 1 or saved['identity'] != self.identity:
                raise ValueError('Checkpoint learner, protocol, seed or fingerprint changed.')
            self.network.load_state_dict(saved['network'])
            self.optimizer.load_state_dict(saved['optimizer'])
            self.rng.set_state(saved['rng_state'])
            for field in ('episodes_completed', 'updates_completed', 'candidates_completed',
                          'rewards', 'trajectory_rewards', 'training_seconds', 'active',
                          'initial_hash', 'initial_optimizer_hash'):
                setattr(self, field, saved[field])
            self.game.best, self.game.best_netlists = saved['best'], saved['best_netlists']
            position = saved['environment']
            self.game.episode = position['episode']
            if self.active:
                for field in ('iteration', 'sequence', 'luts', 'levels'):
                    setattr(self.game, field, position[field])
                self.game.episode_dir = self.folder / 'episodes' / str(self.game.episode)
                self.game.log_file = self.game.episode_dir / 'log.csv'
                self.normalizer = Normalizer(len(self.method['features']), self.method['normalization'])
                for field in ('n', 'mean', 'mean_diff'):
                    value = saved['normalizer'][field]
                    setattr(self.normalizer, field, value if field == 'n' else np.asarray(value, dtype=np.float64))
                self.raw_state = np.asarray(saved['raw_state'], dtype=np.float32)
                self.state = np.asarray(saved['state'], dtype=np.float32)
            self._restore_logs(saved['log_positions'])
            self.game.export_best()
            with (self.folder / 'resume-events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(committed_candidates=self.candidates_completed,
                    discarded_step_rows=self.restored_discarded_rows)) + '\n')
        else:
            if self.checkpoint.exists() or (self.folder / 'steps.jsonl').exists():
                raise FileExistsError('Never overwrite an existing search.')
            self.save_model()
            (self.folder / 'initial.pt').write_bytes(self.checkpoint.read_bytes())

    def _restore_logs(self, positions):
        # A segment's files are durable before its checkpoint is atomically replaced.
        step_file = self.folder / 'steps.jsonl'
        if step_file.exists():
            data = step_file.read_bytes()
            self.restored_discarded_rows = data[positions.get('steps.jsonl', 0):].count(b'\n')
        for relative, size in positions.items():
            path = self.folder / relative
            if not path.exists() or path.stat().st_size < size:
                raise ValueError('Checkpoint log is missing or truncated: ' + relative)
            with path.open('r+b') as stream:
                stream.truncate(size)
        for name in ('steps.jsonl', 'segments.jsonl'):
            if name not in positions and (self.folder / name).exists():
                (self.folder / name).write_bytes(b'')
        # Logs from a not-yet-committed new episode are retained as orphan evidence;
        # reset will replace that episode log before it is used in the score.
        if self.active and self.game.iteration % self.K:
            raise ValueError('An active checkpoint must end at a complete segment.')

    def _start_trajectory(self):
        self.raw_state = self.game.reset()
        self.normalizer = Normalizer(len(self.raw_state), self.method['normalization'])
        self.state = self.normalizer.normalize(self.raw_state)
        self.trajectory_rewards = []
        self.active = True

    def _update(self, states, actions, rewards, bootstrap):
        targets = nstep_returns(rewards, self.method['gamma'], bootstrap)
        if self.rule == 'legacy':
            norm = self.method['normalization']
            targets = (targets - targets.mean()) / max(float(targets.std()), norm['returns_epsilon'])
        inputs = torch.tensor(np.asarray(states), device='cpu')
        with torch.no_grad():
            logits, values = self.network(inputs)
            advantages = torch.tensor(targets, device='cpu') - values
            logp = logits.log_softmax(-1).gather(1, torch.tensor(actions)[:, None]).squeeze(1)
            reduction = self.method['loss']['reduction']
            actor_loss = float(getattr(-logp * advantages, reduction)())
            critic_loss = float(getattr(advantages.square(), reduction)())
        before = {name: flattened(getattr(self.network, name)) for name in ('actor', 'critic')}
        before_network = state_hash(self.network.state_dict())
        before_optimizer = state_hash(self.optimizer.state_dict())
        if self.learning:
            if self.rule == 'legacy':
                A2C._update(self, states, actions, rewards)
            else:
                logits, values = self.network(inputs)
                advantage = torch.tensor(targets, device='cpu') - values
                logp = logits.log_softmax(-1).gather(1, torch.tensor(actions)[:, None]).squeeze(1)
                settings = self.method['loss']
                actor = getattr(-logp * advantage.detach(), settings['reduction'])()
                critic = getattr(advantage.square(), settings['reduction'])()
                loss = settings['actor_weight'] * actor + settings['critic_weight'] * critic
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.updates_completed += 1
        result = dict(targets=targets.tolist(), values=values.detach().tolist(),
            advantages=advantages.tolist(), actor_loss=actor_loss, critic_loss=critic_loss,
            network_before=before_network, network_after=state_hash(self.network.state_dict()),
            optimizer_before=before_optimizer, optimizer_after=state_hash(self.optimizer.state_dict()))
        for name in ('actor', 'critic'):
            module = getattr(self.network, name)
            gradient = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
            result[name + '_gradient_norm'] = float(torch.cat(gradient).norm()) if gradient else 0.0
            result[name + '_update_rms'] = float((flattened(module) - before[name]).double().square().mean().sqrt())
        if not all(torch.isfinite(p).all() for p in self.network.parameters()):
            raise FloatingPointError('Non-finite network parameters.')
        if not self.learning and (result['network_after'] != self.initial_hash or
                                  result['optimizer_after'] != self.initial_optimizer_hash):
            raise RuntimeError('Frozen network or Adam changed.')
        return result

    def run_segment(self):
        if self.episodes_completed >= self.protocol['episodes']:
            raise RuntimeError('The prescribed search budget is complete.')
        started = perf_counter()
        if not self.active:
            self._start_trajectory()
        episode = self.game.episode
        start_iteration = self.game.iteration
        states, actions, rewards = [], [], []
        done = False
        while not done and len(actions) < self.K:
            raw, state = self.raw_state.copy(), self.state.copy()
            with torch.no_grad():
                logits, value = self.network(torch.as_tensor(state, device='cpu'))
                probabilities = torch.full_like(logits, 1 / len(logits)) if self.uniform else logits.softmax(-1)
                action = torch.multinomial(probabilities, 1, generator=self.rng).item()
            states.append(state)
            actions.append(action)
            self.raw_state, reward, done = self.game.step(action)
            rewards.append(reward)
            self.trajectory_rewards.append(reward)
            # Consume s_(t+1) once. At a boundary, cache it for both bootstrap
            # and the next action. Do not consume a terminal state.
            self.state = None if done else self.normalizer.normalize(self.raw_state)
            step = dict(candidate=self.candidates_completed + len(actions), episode=episode,
                iteration=self.game.iteration, action=action, optimization=self.protocol['actions'][action],
                luts=self.game.luts, levels=self.game.levels, reward=reward,
                raw_state=raw.tolist(), normalized_state=state.tolist(),
                probabilities=probabilities.tolist(), value=float(value),
                next_raw_state=self.raw_state.tolist(),
                next_normalized_state=None if done else self.state.tolist(),
                normalizer_n=self.normalizer.n, updates_before=self.updates_completed)
            with (self.folder / 'steps.jsonl').open('a') as stream:
                stream.write(json.dumps(step, allow_nan=False) + '\n')
        bootstrap = 0.0
        if self.learning and self.rule == 'nstep' and not done:
            with torch.no_grad():
                bootstrap = float(self.network(torch.as_tensor(self.state, device='cpu'))[1])
        diagnostic = self._update(states, actions, rewards, bootstrap)
        self.candidates_completed += len(actions)
        diagnostic.update(episode=episode, start_iteration=start_iteration,
            end_iteration=self.game.iteration, candidate=self.candidates_completed,
            terminal=done, bootstrap=bootstrap, normalizer_n=self.normalizer.n,
            next_normalized_state=None if done else self.state.tolist(),
            actions=actions, rewards=rewards, learning=self.learning,
            updates_completed=self.updates_completed)
        with (self.folder / 'segments.jsonl').open('a') as stream:
            stream.write(json.dumps(diagnostic, allow_nan=False) + '\n')
        if done:
            self.episodes_completed += 1
            self.rewards.append(sum(self.trajectory_rewards))
            self.active = False
        self.training_seconds += perf_counter() - started
        self.save_model()
        return diagnostic

    def run_episode(self):
        current = self.episodes_completed
        while self.episodes_completed == current:
            self.run_segment()
        return self.rewards[-1]

    def save_model(self):
        positions = {name: (self.folder / name).stat().st_size
                     for name in ('steps.jsonl', 'segments.jsonl') if (self.folder / name).exists()}
        if self.game.episode:
            relative = f'episodes/{self.game.episode}/log.csv'
            positions[relative] = (self.folder / relative).stat().st_size
        environment = dict(episode=self.game.episode)
        if self.active:
            environment.update(iteration=self.game.iteration, sequence=self.game.sequence,
                               luts=self.game.luts, levels=self.game.levels)
        normalizer = None if self.normalizer is None else dict(n=self.normalizer.n,
            mean=self.normalizer.mean.tolist(), mean_diff=self.normalizer.mean_diff.tolist())
        saved = dict(format_version=1, identity=self.identity,
            network=self.network.state_dict(), optimizer=self.optimizer.state_dict(), rng_state=self.rng.get_state(),
            episodes_completed=self.episodes_completed, updates_completed=self.updates_completed,
            candidates_completed=self.candidates_completed, rewards=self.rewards,
            trajectory_rewards=self.trajectory_rewards, training_seconds=self.training_seconds,
            active=self.active, environment=environment, normalizer=normalizer,
            raw_state=None if self.raw_state is None else self.raw_state.tolist(),
            state=None if self.state is None else self.state.tolist(),
            initial_hash=self.initial_hash, initial_optimizer_hash=self.initial_optimizer_hash,
            best=self.game.best, best_netlists=self.game.best_netlists, log_positions=positions)
        temporary = self.checkpoint.with_suffix('.tmp')
        torch.save(saved, temporary)
        temporary.replace(self.checkpoint)
        self.game.export_best()

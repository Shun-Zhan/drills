"""Independent evidence checks and descriptive reporting for the locked study.

This module never trains or changes a checkpoint. CSV mappings, Welford state,
sampling, returns, losses and Adam updates are recomputed from retained evidence.
The six groups and three training seeds remain the unit of analysis throughout.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time

import numpy as np
import torch

from study import (HERE, ROOT, GROUPS, SEEDS, EVAL_SEEDS, ActorCritic,
                   dump, group_config, identity, load, now, read_rows, sha, state_hash)


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evidence_guard(cfg):
    """Accept a documented validation-only correction after the run was locked.

    The original fingerprint must still match after restoring *only* this
    verifier's original source hash. No learner, runner, protocol, tool,
    benchmark or dependency change is permitted for existing evidence.
    """
    root = Path(cfg['runtime']['output_dir'])
    manifest = read_json(root / 'experiment.json')
    current_fingerprint, payload = identity(cfg)
    if current_fingerprint == manifest['fingerprint']:
        return root, manifest
    amendment = read_json(root / 'validation-amendment.json')
    relative = str((HERE / 'results.py').relative_to(ROOT))
    require(amendment['file'] == relative and
            amendment['experiment_fingerprint'] == manifest['fingerprint'] and
            amendment['original_sha256'] == manifest['sources'][relative] and
            amendment['corrected_sha256'] == payload['sources'][relative] and
            amendment['scope'] == 'validation_and_report_only',
            'Validation amendment does not identify the verifier change')
    locked_payload = copy.deepcopy(payload)
    locked_payload['sources'][relative] = amendment['original_sha256']
    restored_fingerprint = hashlib.sha256(json.dumps(locked_payload, sort_keys=True).encode()).hexdigest()
    require(restored_fingerprint == manifest['fingerprint'],
            'Training or evaluation sources, protocol, tools, benchmark or environment changed')
    return root, manifest


def same_float(actual, expected, label, exact=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all() and
            np.isfinite(expected).all(), label + ': invalid shape/non-finite number')
    equal = np.array_equal(actual, expected) if exact else np.allclose(
        actual, expected, rtol=2e-6, atol=2e-6)
    require(equal, label + ': mismatch')


def rank(row):
    # Independent reproduction of feasible-first, LUT-first baseline ordering.
    return (0, row['luts'], row['levels']) if row['feasible'] else (1, row['levels'], row['luts'])


def reward(previous, current, method, max_levels):
    area = int(current['luts'] < previous['luts']) - int(current['luts'] > previous['luts'])
    if current['levels'] <= max_levels:
        return method['reward']['feasible'][area]
    depth = int(current['levels'] < previous['levels']) - int(current['levels'] > previous['levels'])
    return method['reward']['infeasible'][depth][area]


def mapping(row, sequence, episode, max_levels):
    luts, levels = int(row['luts']), int(row['levels'])
    require(luts > 0 and levels > 0, 'Mapping metrics must be positive integers')
    return dict(luts=luts, levels=levels, feasible=levels <= max_levels,
                sequence=list(sequence), episode=episode, iteration=int(row['iteration']))


class WelfordReplay:
    """Independent float64 accumulator, with float32 normalized observations."""
    def __init__(self, size, params):
        self.params, self.n = params, 0
        self.mean, self.mean_diff = np.zeros(size), np.zeros(size)

    def consume(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        require(np.isfinite(raw).all(), 'Non-finite raw features')
        self.n += 1
        delta = raw - self.mean
        self.mean = self.mean + delta / self.n
        self.mean_diff = self.mean_diff + delta * (raw - self.mean)
        variance = np.clip(self.mean_diff / self.n, self.params['variance_min'],
                           self.params['variance_max'])
        return ((raw - self.mean) / np.sqrt(variance)).astype(np.float32)


def network_and_optimizer(cfg, saved):
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']),
                              cfg['method']['network'])
    network.load_state_dict(saved['network'])
    settings = cfg['method']['optimizer']
    optimizer = torch.optim.Adam(network.parameters(), lr=settings['learning_rate'],
        betas=settings['betas'], eps=settings['epsilon'], weight_decay=settings['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    return network, optimizer


def independent_update(network, optimizer, states, actions, rewards, bootstrap, method, learning):
    targets = np.empty(len(rewards), dtype=np.float32)
    cumulative = float(bootstrap)
    for index in range(len(rewards) - 1, -1, -1):
        cumulative = float(rewards[index]) + method['gamma'] * cumulative
        targets[index] = cumulative
    if method['learner'] == 'legacy':
        targets = (targets - targets.mean()) / max(float(targets.std()),
                                                   method['normalization']['returns_epsilon'])
    before = {name: torch.cat([p.detach().flatten().clone() for p in getattr(network, name).parameters()])
              for name in ('actor', 'critic')}
    record = dict(network_before=state_hash(network.state_dict()),
                  optimizer_before=state_hash(optimizer.state_dict()))
    logits, values = network(torch.tensor(np.asarray(states), device='cpu'))
    advantages = torch.tensor(targets, device='cpu') - values
    logp = logits.log_softmax(-1).gather(1, torch.tensor(actions)[:, None]).squeeze(1)
    settings = method['loss']
    actor = getattr(-logp * advantages.detach(), settings['reduction'])()
    critic = getattr(advantages.square(), settings['reduction'])()
    record.update(targets=targets.tolist(), values=values.detach().tolist(),
        advantages=advantages.detach().tolist(), actor_loss=float(actor.detach()),
        critic_loss=float(critic.detach()))
    if learning:
        optimizer.zero_grad()
        (settings['actor_weight'] * actor + settings['critic_weight'] * critic).backward()
        optimizer.step()
    record.update(network_after=state_hash(network.state_dict()),
                  optimizer_after=state_hash(optimizer.state_dict()))
    for name in ('actor', 'critic'):
        module = getattr(network, name)
        gradients = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
        record[name + '_gradient_norm'] = float(torch.cat(gradients).norm()) if gradients else 0.0
        after = torch.cat([p.detach().flatten() for p in module.parameters()])
        record[name + '_update_rms'] = float((after - before[name]).double().square().mean().sqrt())
    return record


def training_replay(cfg, folder, group, seed, manifest):
    local = group_config(cfg, group, manifest['fingerprint'])
    method, protocol = local['method'], local['protocol']
    H, K = protocol['iterations'], method['update_interval']
    expected_candidates = cfg['experiment']['training_candidates_per_run']
    expected_updates = expected_candidates // K if method['learning_enabled'] else 0
    result, initial, final = read_json(folder / 'result.json'), load(folder / 'initial.pt'), load(folder / 'checkpoint.pt')
    require(result['status'] == 'complete' and result['group'] == group and result['seed'] == seed,
            'Incomplete/mislabeled training result')
    require(result['fingerprint'] == manifest['fingerprint'] and
            result['checkpoint_sha256'] == sha(folder / 'checkpoint.pt'), 'Training provenance mismatch')
    expected_identity = dict(seed=seed, method=method, protocol=protocol, group=group,
                             fingerprint=manifest['fingerprint'])
    for saved in (initial, final):
        require(saved['identity'] == expected_identity and saved['format_version'] == 1,
                'Checkpoint identity mismatch')
    require(initial['candidates_completed'] == 0 and initial['updates_completed'] == 0,
            'Initialization is not an untrained checkpoint')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        original_network = ActorCritic(len(method['features']), len(protocol['actions']), method['network'])
    require(state_hash(original_network.state_dict()) == state_hash(initial['network']),
            'Initial network does not match seed and original initialization')
    network, optimizer = network_and_optimizer(local, initial)
    generator = torch.Generator(device='cpu')
    generator.set_state(initial['rng_state'])
    require(torch.equal(initial['rng_state'], torch.Generator().manual_seed(seed).get_state()),
            'Initial sampling RNG mismatch')
    steps, segments = read_rows(folder / 'steps.jsonl'), read_rows(folder / 'segments.jsonl')
    require(len(steps) == expected_candidates and len(segments) == expected_candidates // K,
            'Step/segment candidate budget mismatch')
    best, rewards, cursor, segment_cursor, updates = None, [], 0, 0, 0
    infeasible = 0
    for episode in range(1, protocol['episodes'] + 1):
        rows = csv_rows(folder / 'episodes' / str(episode) / 'log.csv')
        require(len(rows) == H + 1, 'Episode CSV length mismatch')
        sequence = list(protocol['initial_sequence'])
        previous = mapping(rows[0], sequence, episode, 4)
        require(int(rows[0]['iteration']) == 0 and float(rows[0]['reward']) == 0 and
                rows[0]['optimization'] == sequence[-1], 'Initial mapping CSV mismatch')
        if protocol['evaluation']['include_initial'] and (best is None or rank(previous) < rank(best)):
            best = previous
        normalizer = WelfordReplay(len(method['features']), method['normalization'])
        episode_steps = steps[cursor:cursor + H]
        cached = normalizer.consume(episode_steps[0]['raw_state'])
        episode_rewards = []
        for start in range(0, H, K):
            states, actions, segment_rewards = [], [], []
            batch = episode_steps[start:start + K]
            for offset, step in enumerate(batch):
                iteration = start + offset + 1
                candidate = cursor + iteration
                row = rows[iteration]
                action = step['action']
                require(type(action) is int and 0 <= action < len(protocol['actions']), 'Invalid action')
                sequence.append(protocol['actions'][action])
                current = mapping(row, sequence, episode, 4)
                require(step['candidate'] == candidate and step['episode'] == episode and
                    step['iteration'] == iteration and int(row['iteration']) == iteration and
                    row['optimization'] == protocol['actions'][action] == step['optimization'] and
                    step['luts'] == current['luts'] and step['levels'] == current['levels'], 'Step/CSV identity mismatch')
                expected_reward = reward(previous, current, method, 4)
                require(float(row['reward']) == step['reward'] == expected_reward, 'Reward recomputation mismatch')
                require(step['updates_before'] == updates, 'Update happened inside a segment')
                same_float(step['normalized_state'], cached, 'Welford current state', exact=True)
                if iteration > 1:
                    same_float(step['raw_state'], episode_steps[iteration - 2]['next_raw_state'],
                               'Raw-state trajectory continuity', exact=True)
                with torch.no_grad():
                    logits, value = network(torch.as_tensor(cached))
                    probs = torch.full_like(logits, 1 / len(logits)) if method['uniform_actions'] else logits.softmax(-1)
                    sampled = torch.multinomial(probs, 1, generator=generator).item()
                same_float(step['probabilities'], probs.tolist(), 'Action probabilities', exact=True)
                same_float(step['value'], float(value), 'Critic before update', exact=True)
                require(action == sampled, 'Sampling RNG/action mismatch')
                states.append(cached.copy())
                actions.append(action)
                segment_rewards.append(expected_reward)
                episode_rewards.append(expected_reward)
                terminal = iteration == H
                cached = None if terminal else normalizer.consume(step['next_raw_state'])
                if terminal:
                    require(step['next_normalized_state'] is None, 'Terminal state was consumed')
                else:
                    same_float(step['next_normalized_state'], cached, 'Welford next-state cache', exact=True)
                require(step['normalizer_n'] == min(iteration + 1, H) == normalizer.n,
                        'State normalization was consumed more than once')
                infeasible += not current['feasible']
                if best is None or rank(current) < rank(best):
                    best = current
                previous = current
            segment = segments[segment_cursor]
            segment_cursor += 1
            terminal = start + len(batch) == H
            bootstrap = 0.0
            if method['learning_enabled'] and method['learner'] == 'nstep' and not terminal:
                with torch.no_grad():
                    bootstrap = float(network(torch.as_tensor(cached))[1])
            require(segment['episode'] == episode and segment['start_iteration'] == start and
                segment['end_iteration'] == start + len(batch) and
                segment['candidate'] == cursor + start + len(batch) and
                segment['terminal'] == terminal and segment['actions'] == actions and
                segment['rewards'] == segment_rewards and segment['learning'] == method['learning_enabled'] and
                segment['normalizer_n'] == normalizer.n, 'Segment boundary/continuity mismatch')
            same_float(segment['bootstrap'], bootstrap, 'Pre-update bootstrap', exact=True)
            if terminal:
                require(segment['next_normalized_state'] is None, 'Terminal bootstrap cache mismatch')
            else:
                same_float(segment['next_normalized_state'], cached, 'Segment cached state', exact=True)
            diagnostic = independent_update(network, optimizer, states, actions, segment_rewards,
                                            bootstrap, method, method['learning_enabled'])
            for name, value in diagnostic.items():
                if isinstance(value, str):
                    require(segment[name] == value, 'Independent update hash mismatch: ' + name)
                else:
                    same_float(segment[name], value, 'Independent update ' + name, exact=True)
            updates += int(method['learning_enabled'])
            require(segment['updates_completed'] == updates, 'Update counter mismatch')
        rewards.append(sum(episode_rewards))
        cursor += H
    require(updates == expected_updates and cursor == expected_candidates, 'Prescribed total mismatch')
    for saved in (result, final):
        require(saved['candidates_completed'] == expected_candidates and
            saved['updates_completed'] == expected_updates and saved['episodes_completed'] == protocol['episodes'] and
            saved['rewards'] == rewards and saved['best'] == best, 'Final search summary mismatch')
    require(not final['active'] and read_json(folder / 'best.json') == best, 'Final exported score mismatch')
    require(state_hash(network.state_dict()) == state_hash(final['network']) == result['final_network_hash'] and
        state_hash(optimizer.state_dict()) == state_hash(final['optimizer']) == result['final_optimizer_hash'] and
        torch.equal(generator.get_state(), final['rng_state']), 'Final network/Adam/RNG mismatch')
    require(state_hash(initial['network']) == result['initial_network_hash'] == final['initial_hash'] and
        state_hash(initial['optimizer']) == result['initial_optimizer_hash'] == final['initial_optimizer_hash'],
        'Initial network/Adam hash mismatch')
    require(final['normalizer']['n'] == H, 'Final normalizer counter mismatch')
    same_float(final['normalizer']['mean'], normalizer.mean, 'Final Welford mean', exact=True)
    same_float(final['normalizer']['mean_diff'], normalizer.mean_diff, 'Final Welford M2', exact=True)
    for name, contents in final['best_netlists'].items():
        require((folder / name).read_text() == contents, 'Checkpoint/export netlist mismatch')
    if not method['learning_enabled']:
        require(result['initial_network_hash'] == result['final_network_hash'] and
            result['initial_optimizer_hash'] == result['final_optimizer_hash'], 'Frozen model/Adam changed')
    return dict(group=group, seed=seed, status='pass', candidates=cursor, updates=updates,
                episodes=protocol['episodes'], infeasible_candidates=infeasible,
                normalized_states=protocol['episodes'] * H, best=best,
                initial_network_hash=result['initial_network_hash'], initial_optimizer_hash=result['initial_optimizer_hash'],
                final_network_hash=result['final_network_hash'], final_optimizer_hash=result['final_optimizer_hash'],
                checkpoint_sha256=result['checkpoint_sha256'])


def evaluation_replay(cfg, root, group, seed):
    folder = root / 'evaluation' / group / f'seed-{seed}'
    checkpoint = root / 'training' / group / f'seed-{seed}' / 'checkpoint.pt'
    source_digest = sha(checkpoint)
    result, saved = read_json(folder / 'result.json'), load(checkpoint)
    require(result['status'] == 'complete' and result['group'] == group and result['training_seed'] == seed and
        result['network_unchanged'] is True and result['checkpoint_unchanged'] is True and
        result['checkpoint_sha256'] == source_digest, 'Evaluation provenance mismatch')
    require(len(result['rollouts']) == len(EVAL_SEEDS), 'Evaluation rollout count mismatch')
    local = group_config(cfg, group)
    network, _ = network_and_optimizer(local, saved)
    network.eval().requires_grad_(False)
    before = state_hash(network.state_dict())
    observed_seeds, infeasible = [], 0
    for serialized in result['rollouts']:
        eval_seed = serialized['evaluation_seed']
        rollout_folder = folder / f'rollout-{eval_seed}'
        row = read_json(rollout_folder / 'rollout.json')
        require(row == serialized and row['status'] == 'complete' and row['group'] == group and
            row['training_seed'] == seed and row['checkpoint_sha256'] == source_digest and
            row['bank_id'] == f'{group}-{seed}-{eval_seed}', 'Evaluation rollout manifest mismatch')
        observed_seeds.append(eval_seed)
        actions, rewards, steps = row['actions'], row['rewards'], row['steps']
        require(len(actions) == len(rewards) == len(steps) == 50, 'Evaluation physical budget mismatch')
        mappings = csv_rows(rollout_folder / 'episodes' / '1' / 'log.csv')
        require(len(mappings) == 51, 'Evaluation CSV budget mismatch')
        sequence = list(cfg['protocol']['initial_sequence'])
        previous = mapping(mappings[0], sequence, 1, 4)
        require(int(mappings[0]['iteration']) == 0 and float(mappings[0]['reward']) == 0,
                'Evaluation initial mapping mismatch')
        best = previous if cfg['protocol']['evaluation']['include_initial'] else None
        normalizer = WelfordReplay(len(local['method']['features']), local['method']['normalization'])
        generator = torch.Generator().manual_seed(eval_seed)
        for iteration, (action, actual_reward, step) in enumerate(zip(actions, rewards, steps), 1):
            require(step['iteration'] == iteration and type(action) is int and 0 <= action < 7,
                    'Evaluation action/step identity mismatch')
            state = normalizer.consume(step['raw_state'])
            same_float(step['normalized_state'], state, 'Evaluation Welford state', exact=True)
            with torch.no_grad():
                logits, value = network(torch.as_tensor(state))
                probs = torch.full_like(logits, 1 / len(logits)) if group == 'U' else logits.softmax(-1)
                sampled = torch.multinomial(probs, 1, generator=generator).item()
            require(action == sampled, 'Evaluation action/RNG mismatch')
            same_float(step['probabilities'], probs.tolist(), 'Evaluation policy', exact=True)
            same_float(step['value'], float(value), 'Evaluation value', exact=True)
            sequence.append(cfg['protocol']['actions'][action])
            current = mapping(mappings[iteration], sequence, 1, 4)
            require(mappings[iteration]['optimization'] == sequence[-1] and
                int(mappings[iteration]['iteration']) == iteration and actual_reward ==
                float(mappings[iteration]['reward']) == reward(previous, current, local['method'], 4),
                'Evaluation mapping/reward mismatch')
            infeasible += not current['feasible']
            if best is None or rank(current) < rank(best):
                best = current
            if iteration in (10, 50):
                prefix = row['prefixes'][str(iteration)]
                terminal = {k: current[k] for k in ('luts', 'levels', 'feasible')}
                require(prefix == dict(best=best, terminal=terminal) and
                    read_json(rollout_folder / f'prefix-{iteration}' / 'best.json') == best,
                    'Dependent evaluation prefix score mismatch')
            previous = current
        require(row['best'] == best and row['terminal'] == row['prefixes']['50']['terminal'] and
            read_json(rollout_folder / 'best.json') == best, 'Evaluation final score mismatch')
        for name in ('best.v', 'best-mapped.v'):
            require((rollout_folder / name).read_bytes() ==
                (rollout_folder / 'prefix-50' / name).read_bytes(), 'Final evaluation export mismatch')
    require(observed_seeds == list(EVAL_SEEDS) and len(set(observed_seeds)) == 30,
            'Evaluation seed bank mismatch')
    require(state_hash(network.state_dict()) == before == state_hash(saved['network']) and
            sha(checkpoint) == source_digest, 'Evaluation changed model/checkpoint')
    return dict(group=group, training_seed=seed, status='pass', model_bank_id=f'{group}-{seed}',
                physical_rollouts=30, physical_candidates=1500, dependent_prefix_rows=60,
                infeasible_candidates=infeasible, checkpoint_sha256=source_digest,
                network_hash=before)


def netlist_check(cfg, root, directory, expected):
    record = dict(directory=str(directory.relative_to(root)), expected=expected,
                  mapped=None, unmapped=None, status='pass')
    for name, key, remap in (('best-mapped.v', 'mapped', False), ('best.v', 'unmapped', True)):
        netlist = directory / name
        log = directory / (name + '.verification.log')
        output = ''
        try:
            require(netlist.is_file(), 'Missing exported netlist: ' + str(netlist))
            command = f'read "{netlist}"; '
            if remap:
                command += f'if -K {cfg["protocol"]["lut_inputs"]}; '
            command += f'print_stats; cec "{cfg["protocol"]["circuits"]["i2c"]["file"]}" "{netlist}";'
            process = subprocess.run([cfg['runtime']['abc_binary'], '-c', command],
                capture_output=True, text=True, timeout=120)
            output = process.stdout + process.stderr
            log.write_text(output)
            matches = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', output)
            require(process.returncode == 0 and bool(matches), 'ABC metrics command failed')
            metrics = tuple(map(int, matches[-1]))
            # best.v is the unmapped network. A fresh LUT mapping may choose a
            # different cover, so only best-mapped.v must match the search score.
            if not remap:
                require(metrics == (expected['luts'], expected['levels']),
                        'Exported mapped metrics differ from the search score')
            require('Networks are equivalent' in output, 'CEC did not confirm equivalence')
            record[key] = dict(status='pass', sha256=sha(netlist), luts=metrics[0], levels=metrics[1],
                feasible=metrics[1] <= cfg['protocol']['circuits']['i2c']['max_levels'],
                matches_search_best=metrics == (expected['luts'], expected['levels']),
                cec=True, remapped=remap, log=str(log.relative_to(root)), log_sha256=sha(log))
        except Exception as error:
            record['status'] = 'fail'
            log.write_text(output + '\n' + type(error).__name__ + ': ' + str(error) + '\n')
            record[key] = dict(status='fail', error=type(error).__name__ + ': ' + str(error),
                log=str(log.relative_to(root)), log_sha256=sha(log))
    return record


def verify(cfg):
    root, manifest = evidence_guard(cfg)
    torch.set_num_threads(1)
    started = time.monotonic()
    checks, runs, banks, failures = [], [], [], []
    def attempt(label, action):
        try:
            value = action()
            checks.append(dict(check=label, status='pass'))
            return value
        except Exception as error:
            row = dict(check=label, status='fail', error=type(error).__name__ + ': ' + str(error))
            failures.append(row)
            checks.append(row)
            return None
    for group in GROUPS:
        for seed in SEEDS:
            directory = root / 'training' / group / f'seed-{seed}'
            row = attempt(f'training:{group}:{seed}', lambda d=directory, g=group, s=seed:
                          training_replay(cfg, d, g, s, manifest))
            if row:
                runs.append(row)
            print('Evidence replay:', group, seed, 'pass' if row else 'fail', flush=True)
    for seed in SEEDS:
        def pairing(s=seed):
            rows = [r for r in runs if r['seed'] == s]
            require(len(rows) == 6 and len({r['initial_network_hash'] for r in rows}) == 1 and
                    len({r['initial_optimizer_hash'] for r in rows}) == 1, 'Same-seed initialization differs')
            first = {g: read_rows(root / 'training' / g / f'seed-{s}' / 'steps.jsonl') for g in ('C', 'D', 'F')}
            fields = ('action', 'raw_state', 'normalized_state', 'probabilities', 'value',
                      'luts', 'levels', 'reward', 'next_raw_state', 'next_normalized_state', 'normalizer_n')
            for field in fields:
                require([r[field] for r in first['C'][:10]] == [r[field] for r in first['D'][:10]] ==
                        [r[field] for r in first['F'][:10]], 'C/D/F first ten steps differ: ' + field)
                require([r[field] for r in first['C'][:50]] == [r[field] for r in first['F'][:50]],
                        'C/F first fifty steps differ: ' + field)
            return True
        attempt(f'paired-initialization:{seed}', pairing)
    for group in GROUPS:
        for seed in ((10,) if group == 'U' else SEEDS):
            row = attempt(f'evaluation:{group}:{seed}', lambda g=group, s=seed: evaluation_replay(cfg, root, g, s))
            if row:
                banks.append(row)
            print('Evaluation replay:', group, seed, 'pass' if row else 'fail', flush=True)
    netlists = []
    for group in GROUPS:
        for seed in SEEDS:
            folder = root / 'training' / group / f'seed-{seed}'
            if (folder / 'best.json').exists():
                netlists.append((folder, read_json(folder / 'best.json')))
        for seed in ((10,) if group == 'U' else SEEDS):
            for evaluation_seed in EVAL_SEEDS:
                rollout = root / 'evaluation' / group / f'seed-{seed}' / f'rollout-{evaluation_seed}'
                for prefix in (10, 50):
                    folder = rollout / f'prefix-{prefix}'
                    if (folder / 'best.json').exists():
                        netlists.append((folder, read_json(folder / 'best.json')))
    with ThreadPoolExecutor(max_workers=3) as pool:
        netlist_results = list(pool.map(lambda item: netlist_check(cfg, root, *item), netlists))
    for row in netlist_results:
        if row['status'] != 'pass':
            failures.append(dict(check='netlist:' + row['directory'], status='fail', error='Netlist validation failed'))
    complete = len(runs) == 18 and len(banks) == 16 and len(netlist_results) == 978
    if not complete:
        failures.append(dict(check='all-required-evidence', status='fail',
            error=f'Required runs/banks/netlist pairs 18/16/978; observed {len(runs)}/{len(banks)}/{len(netlist_results)}'))
    value = dict(status='pass' if complete and not failures else 'fail', finished_at=now(),
        fingerprint=manifest['fingerprint'], duration_seconds=time.monotonic() - started,
        method='Independent CSV/rank/reward/Welford/RNG/model/Adam replay, exact hashes, ABC metrics and CEC',
        scope='All committed candidates and all 10/50-prefix exports; raw feature extraction is not rerun for every candidate',
        checks=checks, training=runs, evaluation=banks, netlists=netlist_results, failures=failures,
        training_candidates=sum(r['candidates'] for r in runs),
        evaluation_candidates=sum(r['physical_candidates'] for r in banks),
        netlist_pairs_required=978, netlist_pairs_checked=len(netlist_results),
        netlists_checked=2 * len(netlist_results))
    dump(root / 'verification.json', value)
    dump(HERE / 'verification.json', value)
    print('Verification:', value['status'], 'failures=', len(failures), flush=True)
    if failures:
        raise RuntimeError('Evidence verification failed; all failures preserved in verification.json')
    return value


def training_observations(cfg, root):
    runs, curves, hits = [], [], []
    for group in GROUPS:
        setting = cfg['experiment']['groups'][group]
        for seed in SEEDS:
            folder = root / 'training' / group / f'seed-{seed}'
            result = read_json(folder / 'result.json') if (folder / 'result.json').exists() else {}
            checkpoint = load(folder / 'checkpoint.pt') if (folder / 'checkpoint.pt').exists() else {}
            steps = read_rows(folder / 'steps.jsonl') if (folder / 'steps.jsonl').exists() else []
            completed = result.get('status') == 'complete' and result.get('candidates_completed') == 1000
            status = result.get('status', 'incomplete' if checkpoint else 'missing')
            best = result.get('best', checkpoint.get('best'))
            feasible = bool(best and best['feasible'])
            candidates = checkpoint.get('candidates_completed', result.get('candidates_completed', 0))
            steps = steps[:candidates]
            row = dict(group=group, training_seed=seed, status=status, complete=completed,
                H=setting['H'], K=setting['K'], candidates=candidates, candidate_budget=1000,
                episodes_completed=result.get('episodes_completed', checkpoint.get('episodes_completed', 0)),
                updates=result.get('updates_completed', checkpoint.get('updates_completed', 0)),
                best_luts=None if best is None else best['luts'], best_levels=None if best is None else best['levels'],
                feasible=feasible, best_feasible_luts=best['luts'] if feasible else None,
                best_episode=None if best is None else best['episode'], best_iteration=None if best is None else best['iteration'],
                training_seconds=result.get('training_seconds', checkpoint.get('training_seconds')),
                infeasible_candidates=sum(s['levels'] > 4 for s in steps),
                initial_network_hash=result.get('initial_network_hash', checkpoint.get('initial_hash')),
                final_network_hash=result.get('final_network_hash'),
                checkpoint_sha256=sha(folder / 'checkpoint.pt') if checkpoint else None)
            runs.append(row)
            running_best, initial_rows, first_hit = None, 0, None
            for episode in range(1, row['episodes_completed'] + int(checkpoint.get('active', False)) + 1):
                path = folder / 'episodes' / str(episode) / 'log.csv'
                if not path.exists():
                    continue
                rows = csv_rows(path)
                sequence = list(cfg['protocol']['initial_sequence'])
                if rows and cfg['protocol']['evaluation']['include_initial']:
                    initial_rows += 1
                    initial = mapping(rows[0], sequence, episode, 4)
                    if running_best is None or rank(initial) < rank(running_best):
                        running_best = initial
                    if episode == 1:
                        curves.append(dict(group=group, training_seed=seed, candidate=0,
                            best_feasible_luts=running_best['luts'] if running_best['feasible'] else None,
                            best_levels=running_best['levels'], feasible=running_best['feasible']))
                        if running_best['feasible'] and running_best['luts'] <= cfg['experiment']['target_luts']:
                            first_hit = 0
                for step in (s for s in steps if s['episode'] == episode):
                    sequence.append(step['optimization'])
                    current = dict(luts=step['luts'], levels=step['levels'], feasible=step['levels'] <= 4)
                    if running_best is None or rank(current) < rank(running_best):
                        running_best = current
                    if first_hit is None and running_best['feasible'] and running_best['luts'] <= cfg['experiment']['target_luts']:
                        first_hit = step['candidate']
                    curves.append(dict(group=group, training_seed=seed, candidate=step['candidate'],
                        best_feasible_luts=running_best['luts'] if running_best['feasible'] else None,
                        best_levels=running_best['levels'], feasible=running_best['feasible']))
            row['initial_mappings'] = initial_rows
            row['target_first_candidate'] = first_hit
            hits.append(dict(group=group, training_seed=seed, status=status, target_luts=300,
                hit=first_hit is not None, first_candidate=first_hit, candidates_observed=candidates,
                censored=first_hit is None))
    return runs, curves, hits


def evaluation_observations(root):
    rows, banks = [], []
    for group in GROUPS:
        for seed in ((10,) if group == 'U' else SEEDS):
            folder = root / 'evaluation' / group / f'seed-{seed}'
            final = read_json(folder / 'result.json') if (folder / 'result.json').exists() else {}
            bank_rows, physical = [], 0
            for evaluation_seed in EVAL_SEEDS:
                path = folder / f'rollout-{evaluation_seed}' / 'rollout.json'
                rollout = read_json(path) if path.exists() else {}
                complete = rollout.get('status') == 'complete' and len(rollout.get('actions', [])) == 50
                if complete:
                    physical += 1
                for prefix in (10, 50):
                    value = rollout.get('prefixes', {}).get(str(prefix), {})
                    best, terminal = value.get('best'), value.get('terminal')
                    feasible = bool(best and best['feasible'])
                    row = dict(group=group, training_seed=seed, model_bank_id=f'{group}-{seed}',
                        physical_rollout_id=f'{group}-{seed}-{evaluation_seed}', evaluation_seed=evaluation_seed,
                        prefix=prefix, status=rollout.get('status', 'missing'), complete=complete,
                        shared_reference=group == 'U', beyond_training_H=group in ('A', 'B') and prefix == 50,
                        best_luts=None if best is None else best['luts'], best_levels=None if best is None else best['levels'],
                        feasible=feasible, best_feasible_luts=best['luts'] if feasible else None,
                        terminal_luts=None if terminal is None else terminal['luts'],
                        terminal_levels=None if terminal is None else terminal['levels'],
                        terminal_feasible=bool(terminal and terminal['feasible']))
                    rows.append(row)
                    bank_rows.append(row)
            for prefix in (10, 50):
                selected = [r for r in bank_rows if r['prefix'] == prefix]
                valid = final.get('status') == 'complete' and len(selected) == 30 and all(
                    r['complete'] and r['feasible'] for r in selected)
                banks.append(dict(group=group, training_seed=seed, model_bank_id=f'{group}-{seed}',
                    prefix=prefix, physical_rollouts_completed=physical, physical_rollouts_required=30,
                    complete=valid, shared_reference=group == 'U',
                    beyond_training_H=group in ('A', 'B') and prefix == 50,
                    feasible_rollouts=sum(r['complete'] and r['feasible'] for r in selected),
                    mean_best_feasible_luts=float(np.mean([r['best_feasible_luts'] for r in selected])) if valid else None,
                    std_best_feasible_luts=float(np.std([r['best_feasible_luts'] for r in selected], ddof=0)) if valid else None,
                    mean_terminal_luts=float(np.mean([r['terminal_luts'] for r in selected])) if valid else None,
                    infeasible_terminals=sum(r['complete'] and not r['terminal_feasible'] for r in selected)))
    return rows, banks


def comparisons(runs, cfg, value='best_feasible_luts', prefix=None):
    output = []
    for treatment, control, interpretation in (('D', 'A', '工程收益'), ('D', 'C', '分段更新收益'),
            ('D', 'F', '相对冻结网络的学习增量'), ('D', 'U', '相对均匀随机的学习增量'), ('B', 'A', '尺度处理影响')):
        pairs = []
        for seed in SEEDS:
            treated = next((r for r in runs if r['group'] == treatment and r['training_seed'] == seed and
                            (prefix is None or r['prefix'] == prefix)), None)
            # One U evaluation bank is shared; training U has all three seeds.
            control_seed = 10 if prefix is not None and control == 'U' else seed
            reference = next((r for r in runs if r['group'] == control and r['training_seed'] == control_seed and
                              (prefix is None or r['prefix'] == prefix)), None)
            valid = bool(treated and reference and treated['complete'] and reference['complete'] and
                         treated.get(value) is not None and reference.get(value) is not None)
            pairs.append(dict(training_seed=seed, control_training_seed=control_seed, valid=valid,
                treatment=None if treated is None else treated.get(value),
                control=None if reference is None else reference.get(value),
                reduction=reference[value] - treated[value] if valid else None))
        valid = all(p['valid'] for p in pairs)
        reductions = [p['reduction'] for p in pairs] if valid else []
        mean = float(np.mean(reductions)) if valid else None
        wins = sum(v > 0 for v in reductions) if valid else None
        output.append(dict(treatment=treatment, control=control, interpretation=interpretation, prefix=prefix,
            pairs=pairs, complete=valid, mean_lut_reduction=mean, wins=wins,
            ties=sum(v == 0 for v in reductions) if valid else None,
            losses=sum(v < 0 for v in reductions) if valid else None,
            screening_pass=bool(valid and mean >= cfg['experiment']['minimum_mean_lut_reduction'] and
                                wins >= cfg['experiment']['minimum_seed_wins']),
            screening_applies_to='training' if prefix is None else 'descriptive evaluation comparison',
            shared_U_reference=prefix is not None and control == 'U'))
    return output


FALLACIES = (
    ('辛普森悖论', 'CAUTION', '展示三种子逐项差和均值；训练搜索与固定模型评估分别统计，不合并两类结果。'),
    ('生态谬误', 'CAUTION', '推断单位为三次训练种子；同一模型的30次评估和10/50前缀不升级为独立训练样本。'),
    ('伯克森悖论', 'NOTE', '种子和预算先固定，保留所有失败、不可行和未命中；仅i2c一个电路限制外推。'),
    ('碰撞变量偏差', 'NOTE', '不按训练收益、命中300或训练时长筛选模型，也不据这些后验量调整比较。'),
    ('忽视基准率', 'NOTE', '不是诊断分类任务；每组同时报告可行率、失败数和300 LUT命中数。'),
    ('回归均值', 'CAUTION', '重新运行全部预定种子，没有选择历史极端种子；本轮结果仍需外部实验确认。'),
    ('幸存者偏差', 'NOTE', '任一预定训练种子或评估轨迹缺失/不可行时，不对成功子集生成该组主均值。'),
    ('多重搜索效应', 'CAUTION', '完整报告预定D-A/D-C/D-F/D-U/B-A比较；阈值是探索性筛查，不是显著性检验。'),
    ('分析分叉路径', 'NOTE', '源码、协议、工具和电路哈希在运行前锁定；不增种子，不事后调参或改阈值。'),
    ('相关不等于因果', 'CAUTION', 'D-C隔离本方案中的更新间隔，D-F/U检查学习增量；D-A同时改变轨迹长度和回报尺度，不能单归因于分段更新。'),
    ('反向因果', 'NOTE', '算法和预算在测量前指定；不由结果改变组别。此设计不支持跨电路机制定论。'),
)


def fmt(value, digits=3):
    return '—' if value is None else (f'{value:.{digits}f}' if isinstance(value, float) else str(value))


def analyze(cfg):
    root, manifest = evidence_guard(cfg)
    runs, curves, hits = training_observations(cfg, root)
    evaluation, banks = evaluation_observations(root)
    aggregates = []
    for group in GROUPS:
        selected = [r for r in runs if r['group'] == group]
        valid = len(selected) == 3 and all(r['complete'] and r['feasible'] for r in selected)
        aggregates.append(dict(group=group, complete_runs=sum(r['complete'] for r in selected),
            required_runs=3, feasible_runs=sum(r['complete'] and r['feasible'] for r in selected),
            mean_best_feasible_luts=float(np.mean([r['best_feasible_luts'] for r in selected])) if valid else None,
            std_best_feasible_luts=float(np.std([r['best_feasible_luts'] for r in selected], ddof=0)) if valid else None,
            training_seconds_total=sum(r['training_seconds'] for r in selected) if all(
                r['training_seconds'] is not None for r in selected) else None,
            target_hits=sum(r['target_first_candidate'] is not None for r in selected)))
    training_comparisons = comparisons(runs, cfg)
    evaluation_comparisons = [comparison for prefix in (10, 50) for comparison in
        comparisons(banks, cfg, value='mean_best_feasible_luts', prefix=prefix)]
    verification = read_json(root / 'verification.json') if (root / 'verification.json').exists() else dict(status='not_run')
    process_rows = []
    for path in sorted((root / 'logs').glob('**/processes.json')):
        for row in read_json(path):
            process_rows.append(dict(**row, process_manifest=str(path.relative_to(root))))
    resume_events = []
    for path in sorted((root / 'training').glob('*/seed-*/resume-events.jsonl')):
        resume_events.extend(dict(path=str(path.relative_to(root)), **row) for row in read_rows(path))
    physical_eval = len([r for r in evaluation if r['prefix'] == 50 and r['complete']])
    cost = dict(training_candidates=sum(r['candidates'] for r in runs), training_candidate_budget=18000,
        evaluation_candidates=physical_eval * 50, evaluation_candidate_budget=24000,
        training_initial_mappings=sum(r['initial_mappings'] for r in runs),
        evaluation_initial_mappings=physical_eval, physical_evaluation_rollouts=physical_eval,
        physical_evaluation_rollouts_required=480, evaluation_model_banks=16,
        sum_training_segment_seconds=sum(r['training_seconds'] or 0 for r in runs),
        sum_supervised_process_seconds=sum(r['elapsed_seconds'] for r in process_rows),
        failed_processes=sum(r['exit_code'] != 0 for r in process_rows),
        timed_out_processes=sum(r['timed_out'] for r in process_rows),
        incomplete_training_runs=sum(not r['complete'] for r in runs),
        incomplete_evaluation_rollouts=480 - physical_eval,
        infeasible_training_candidates=sum(r['infeasible_candidates'] for r in runs),
        infeasible_evaluation_terminals=sum(r['complete'] and not r['terminal_feasible'] for r in evaluation if r['prefix'] == 50),
        discarded_committed_step_rows_on_resume=sum(r.get('discarded_step_rows', 0) for r in resume_events),
        resume_events=resume_events,
        timing_definition='Segment timers include trajectory reset, ABC/Yosys candidates and network updates; exclude checkpoint export. Supervision timers include each process and all recorded attempts, summed rather than wall-clock duration.',
        replay_cost_note='Discarded rows are observed logged work only; ABC calls before a crash or before logging cannot be counted exactly. Such extra wall time is retained in process attempts.')
    complete = (all(r['complete'] and r['feasible'] for r in runs) and
                all(b['complete'] for b in banks) and verification['status'] == 'pass')
    summary = dict(status='complete_verified' if complete else 'incomplete_or_unverified', generated_at=now(),
        fingerprint=manifest['fingerprint'], base_commit=manifest['base_commit'],
        implementation_commit=manifest['implementation_commit'], primary_unit='three prescribed training seeds',
        training=runs, training_groups=aggregates, training_comparisons=training_comparisons,
        evaluation_banks=banks, evaluation_comparisons=evaluation_comparisons, costs=cost,
        verification_status=verification['status'], verification_seconds=verification.get('duration_seconds'),
        confidence='CAUTION: exploratory, n=3, one circuit; no significance or stability claim',
        fallacy_scan=[dict(type=name, severity=severity, finding=finding) for name, severity, finding in FALLACIES],
        fallacy_coverage='11/11', process_attempts=process_rows)
    write_csv(HERE / 'training.csv', runs)
    write_csv(HERE / 'training-means.csv', aggregates)
    write_csv(HERE / 'evaluation.csv', evaluation)
    write_csv(HERE / 'evaluation-means.csv', banks)
    write_csv(HERE / 'search-curves.csv', curves)
    write_csv(HERE / 'target-hits.csv', hits)
    dump(HERE / 'summary.json', summary)
    dump(root / 'summary.json', summary)
    lines = ['## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run + validate', '- Origin Date: ' + now()[:10],
        '- Verification Status: ' + ('VERIFIED' if complete else 'UNVERIFIED'),
        '- Version Label: segmented_updates_result_v1', '', '# i2c 长轨迹分段更新实验报告', '',
        '本轮验证50步连续轨迹、每10步更新是否改善训练搜索的最佳可行LUT。'
        '所有预定组别与种子均在下表保留；主指标仅来自各训练搜索的1,000个动作候选，评估网表不会加入训练成绩。', '',
        f'分支 `codex/segmented-updates`，基点 `{manifest["base_commit"]}`，实现提交 `{manifest["implementation_commit"]}`。',
        f'实验指纹 `{manifest["fingerprint"]}`；完整性状态 `{summary["status"]}`，证据核验 `{verification["status"]}`。', '',
        '## 训练搜索结果', '', '|组别|种子|状态|候选|更新|最佳可行LUT|层数|首次≤300候选|耗时秒|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in runs:
        lines.append('| ' + ' | '.join(fmt(row[k]) for k in ('group', 'training_seed', 'status',
            'candidates', 'updates', 'best_feasible_luts', 'best_levels', 'target_first_candidate', 'training_seconds')) + ' |')
    lines += ['', '|组别|完成/预定|可行/预定|最佳可行LUT均值±标准差|300 LUT命中|', '|---|---|---|---:|---:|']
    for row in aggregates:
        lines.append(f'| {row["group"]} | {row["complete_runs"]}/3 | {row["feasible_runs"]}/3 | '
            f'{fmt(row["mean_best_feasible_luts"])} ± {fmt(row["std_best_feasible_luts"])} | {row["target_hits"]}/3 |')
    lines += ['', 'A为原版H10/K10；B为原奖励尺度H10/K10；C为H50/K50；D为目标H50/K10；'
        'F为初始化网络冻结H50，U为均匀随机H50。F/U每10步只提交检查点，没有更新。'
        '初始映射参加选优但不计动作预算；标准差使用ddof=0。缺失、不完整或不可行时不对成功子集求主均值。', '',
        '## 预定比较与效果判断', '', '正差=对照LUT−处理LUT；平均至少减少1 LUT且至少两个训练种子改善为预先固定的探索性筛查门槛。', '',
        '|比较|解释|种子10差|种子11差|种子12差|平均减少LUT|胜/平/负|筛查|', '|---|---|---:|---:|---:|---:|---|---|']
    for row in training_comparisons:
        label = '通过' if row['screening_pass'] else ('未通过' if row['complete'] else '证据不完整')
        lines.append(f'| {row["treatment"]}−{row["control"]} | {row["interpretation"]} | ' +
            ' | '.join(fmt(p['reduction']) for p in row['pairs']) +
            f' | {fmt(row["mean_lut_reduction"])} | {fmt(row["wins"])}/{fmt(row["ties"])}/{fmt(row["losses"])} | {label} |')
    engineering, segmentation, frozen, uniform, scale = training_comparisons
    for row, opening in ((engineering, '工程收益（D相对A）'), (segmentation, '分段更新收益（D相对C）'),
                         (frozen, '相对冻结网络的增量（D相对F）'), (uniform, '相对均匀随机的增量（D相对U）')):
        outcome = '达到预定筛查门槛' if row['screening_pass'] else ('未达到预定筛查门槛' if row['complete'] else '证据不完整，不能判断')
        lines += ['', f'{opening}：{outcome}；平均减少 {fmt(row["mean_lut_reduction"])} LUT，'
            f'三个种子胜/平/负为 {fmt(row["wins"])}/{fmt(row["ties"])}/{fmt(row["losses"])}。']
    lines += ['', 'D−A同时改变轨迹长度和回报尺度；它只支持整体工程比较。D−C针对同尺度学习器的更新间隔；'
        'D−F/U检查是否超出无学习搜索。长轨迹或随机搜索的收益不能直接归功于训练。三个种子仅支持本轮探索性描述，不能宣称稳定优越或统计显著。', '',
        '## 固定模型独立评估', '', '每个最终模型使用30000–30029共30个新评估种子，各执行一条不更新的50步物理轨迹。'
        'A/B/C/D/F共15个模型库，U只使用seed-10的一份共享随机参考，共16个库、480条轨迹。'
        '每条轨迹的10步和50步前缀相关，前缀行共960条，但样本数仍为480；U的30条参考不重复计数。'
        '每模型库均值描述策略采样变化，不能把30次评估当作30次独立训练。', '',
        '|组别|训练种子|前缀|可行/30|最佳可行LUT均值±标准差|不可行终点|超训练长度|', '|---|---:|---:|---:|---:|---:|---|']
    for row in banks:
        lines.append(f'| {row["group"]} | {row["training_seed"]} | {row["prefix"]} | {row["feasible_rollouts"]}/30 | '
            f'{fmt(row["mean_best_feasible_luts"])} ± {fmt(row["std_best_feasible_luts"])} | {row["infeasible_terminals"]} | '
            f'{"是" if row["beyond_training_H"] else "否"} |')
    lines += ['', 'A/B的50步评估明确为超出训练长度的测试；其10步前缀为训练长度内的参考。'
        'C/D/F/U的10步与50步前缀用于同条轨迹的预算比较。所有评估最佳值与终点值另见evaluation.csv。', '',
        '|前缀|比较|三个模型均值差的均值LUT|胜/平/负|共享U|', '|---:|---|---:|---|---|']
    for row in evaluation_comparisons:
        lines.append(f'| {row["prefix"]} | {row["treatment"]}−{row["control"]} | {fmt(row["mean_lut_reduction"])} | '
            f'{fmt(row["wins"])}/{fmt(row["ties"])}/{fmt(row["losses"])} | {"是" if row["shared_U_reference"] else "否"} |')
    lines += ['', '评估差额为逐训练模型的30轨迹均值之差；与共享U的三项差额共用相同参考，不能用来估计三份独立U样本的变异。', '',
        '## 成本、失败与核验', '',
        f'- 训练动作候选：{cost["training_candidates"]}/18,000；评估动作候选：{cost["evaluation_candidates"]}/24,000。',
        f'- 额外初始映射：训练{cost["training_initial_mappings"]}次、评估{cost["evaluation_initial_mappings"]}次；另有resyn2基线和最终网表核验调用。',
        f'- 训练段计时合计{cost["sum_training_segment_seconds"]:.3f}秒；所有监督进程计时合计{cost["sum_supervised_process_seconds"]:.3f}秒（并行进程耗时求和，非总墙钟时间）。',
        f'- 失败进程{cost["failed_processes"]}，硬超时{cost["timed_out_processes"]}，不完整训练{cost["incomplete_training_runs"]}，缺失评估轨迹{cost["incomplete_evaluation_rollouts"]}。',
        f'- 不可行训练候选{cost["infeasible_training_candidates"]}；评估50步不可行终点{cost["infeasible_evaluation_terminals"]}。',
        f'- 恢复时回滚的已记录动作行{cost["discarded_committed_step_rows_on_resume"]}；未写日志前的额外ABC调用不能精确追计，相关耗时留在原始进程尝试记录中。',
        '- 最多3个任务并行、每进程1个Torch线程、每任务30分钟硬超时；停止点为预定三种子，不增加种子或调参追加实验。',
        '- 独立核验逐步重算CSV奖励及最好值、Welford归一化、动作概率/采样RNG、段尾目标与网络/Adam更新哈希；核验全部交付最佳网表的映射指标和CEC。',
        '- 验证器修订见validation-amendment.json：未映射网表重新做LUT6映射时覆盖可能变化，分别报告其实测LUT/层数及CEC；已映射网表仍须与训练或评估得分精确一致。该修订未改变学习器或已完成轨迹。',
        '- 原始中间状态来自Yosys特征日志；没有再次运行全部42,000动作来重提取每个原始状态。真实工具连续/中断恢复精确回归由测试记录支持。', '',
        '## 统计风险核查（11/11）', '', '|风险|等级|本轮核查|', '|---|---|---|']
    lines += [f'| {name} | {severity} | {finding} |' for name, severity, finding in FALLACIES]
    lines += ['', '## 可复现证据', '', 'protocol.yml保存预定协议；experiment.json记录源码、工具、电路与环境哈希；'
        'verification.json记录独立回放和逐网表核验。完整初始/最终权重、动作日志、CSV、检查点、最佳网表及监督记录保存在results/segmented-updates。'
        'SHA-256清单、模型清单与本地证据压缩包由package入口生成，复现命令见README.md。', '',
        '![三个种子的训练搜索曲线](search-curves.png)', '', '![固定模型的10/50步评估](evaluation-comparison.png)', '']
    (HERE / 'report.md').write_text('\n'.join(lines))
    print('Report:', HERE / 'report.md', flush=True)
    return summary


def package(cfg):
    root, manifest = evidence_guard(cfg)
    require((HERE / 'summary.json').exists(), 'Run analyze before package')
    summary = read_json(HERE / 'summary.json')
    require(summary['fingerprint'] == manifest['fingerprint'], 'Stale report fingerprint')
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name not in ('evidence.tar.gz', '.lock', 'SHA256SUMS',
                'archive-validation.json') and path.name not in ('current.v', 'mapped.v') and path.suffix != '.tmp':
            files.append(path)
    models = []
    for group in GROUPS:
        for seed in SEEDS:
            folder = root / 'training' / group / f'seed-{seed}'
            for name in ('initial.pt', 'checkpoint.pt'):
                path = folder / name
                if path.exists():
                    saved = load(path)
                    models.append(dict(group=group, training_seed=seed, stage='initial' if name == 'initial.pt' else 'final',
                        file=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size,
                        network_hash=state_hash(saved['network']), optimizer_hash=state_hash(saved['optimizer']),
                        candidates=saved['candidates_completed'], updates=saved['updates_completed'],
                        fingerprint=saved['identity']['fingerprint']))
    dump(HERE / 'model-manifest.json', dict(fingerprint=manifest['fingerprint'], models=models))
    artifacts = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in
                 ('artifact-manifest.json', 'archive-validation.json') and p.suffix != '.tmp']
    source_manifest = {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in artifacts}
    dump(HERE / 'artifact-manifest.json', dict(fingerprint=manifest['fingerprint'],
        generated_at=now(), artifacts=source_manifest, report_status=summary['status']))
    artifacts.append(HERE / 'artifact-manifest.json')
    archive_files = files + artifacts
    entries = {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in archive_files}
    sums = root / 'SHA256SUMS'
    sums.write_text(''.join(value['sha256'] + '  ' + name + '\n' for name, value in sorted(entries.items())))
    archive_files.append(sums)
    entries[str(sums.relative_to(ROOT))] = dict(sha256=sha(sums), bytes=sums.stat().st_size)
    archive = root / 'evidence.tar.gz'
    temporary = archive.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as stream:
        for path in archive_files:
            stream.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    temporary.replace(archive)
    observed, failures = {}, []
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream.getmembers():
            require(member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts,
                    'Unexpected unsafe archive entry')
            require(member.name not in observed, 'Duplicate archive entry')
            fileobj = stream.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda: fileobj.read(1048576), b''):
                digest.update(block)
            observed[member.name] = dict(sha256=digest.hexdigest(), bytes=member.size)
            if entries.get(member.name) != observed[member.name]:
                failures.append(member.name)
    require(set(observed) == set(entries), 'Archive omitted an evidence file')
    validation = dict(status='pass' if not failures else 'fail', created_at=now(),
        fingerprint=manifest['fingerprint'], archive=str(archive.relative_to(ROOT)),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        checked_files=len(entries), failures=failures,
        exclusions='Only volatile current.v/mapped.v, .tmp files, .lock and the archive itself; all best/prefix netlists retained.',
        verification_status=summary['verification_status'], models=len(models))
    dump(root / 'archive-validation.json', validation)
    dump(HERE / 'archive-validation.json', validation)
    require(not failures, 'Archive hashes differ from live evidence')
    print('Evidence package:', archive, validation['archive_sha256'], flush=True)
    return validation

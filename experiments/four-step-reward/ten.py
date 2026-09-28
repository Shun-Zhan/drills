"""Conditional ten-step ABC search with verified old baselines and greedy policy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from context import (HERE, OUT, ROOT, SEEDS10, SPEC, dump, identity, prior,
                     read, sha, training_config)
from drills.experiment import verify_netlist
from drills.fpga_session import FPGASession
from drills.model import ActorCritic, Normalizer
from reward import RewardSession

OLD_TEN = ROOT / 'results/learning-effectiveness'


def _episodes(folder, circuit, cfg):
    """Read all eleven ABC mappings of each episode and reconstruct the best."""
    best = None
    records = []
    for episode in range(1, SPEC['ten_step']['episodes'] + 1):
        path = folder / 'episodes' / str(episode) / 'log.csv'
        with path.open(newline='') as stream:
            steps = list(csv.DictReader(stream))
        if len(steps) != 11 or [int(row['iteration']) for row in steps] != list(range(11)):
            raise ValueError(f'Incomplete ten-step ABC trajectory: {path}')
        if steps[0]['optimization'] != cfg['protocol']['initial_sequence'][0]:
            raise ValueError(f'Wrong initial operation: {path}')
        actions = [step['optimization'] for step in steps[1:]]
        if any(action not in cfg['protocol']['actions'] for action in actions):
            raise ValueError(f'Unknown action: {path}')
        episode_feasible = False
        for step in steps:
            iteration = int(step['iteration'])
            luts, levels = int(step['luts']), int(step['levels'])
            row = dict(luts=luts, levels=levels,
                       feasible=levels <= cfg['protocol']['circuits'][circuit]['max_levels'],
                       sequence=[cfg['protocol']['initial_sequence'][0], *actions[:iteration]],
                       episode=episode, iteration=iteration)
            episode_feasible |= row['feasible']
            if best is None or FPGASession.rank(row) < FPGASession.rank(best):
                best = row
        records.append(dict(episode=episode, calls=episode * 11,
                            best_luts=best['luts'] if best['feasible'] else None,
                            best_levels=best['levels'], best_feasible=best['feasible'],
                            episode_feasible=episode_feasible, actions=actions,
                            steps=[(int(r['luts']), int(r['levels'])) for r in steps]))
    return best, records


def _replay_first(folder, circuit, cfg, records, group, seed, historical):
    game = FPGASession(cfg, cfg['protocol']['circuits'][circuit], folder)
    torch.set_num_threads(cfg['environment']['torch_threads'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']),
                              cfg['method']['network'])
    original = prior.old.load(historical / 'snapshots/0.pt')
    if prior.old.state_hash(network.state_dict()) != prior.old.state_hash(original['network']):
        raise ValueError(f'Historical initial network differs: {group}/{circuit}/{seed}')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    state = game.reset()
    normalizer = Normalizer(len(state), cfg['method']['normalization'],
                            cfg['method']['features'], state)
    observed = [(game.luts, game.levels)]
    for expected in records[0]['actions']:
        normalized = normalizer.normalize(state)
        with torch.no_grad():
            logits, _ = network(torch.as_tensor(normalized, device='cpu'))
            probabilities = (torch.full_like(logits, 1 / len(logits)) if group == 'uniform'
                             else logits.softmax(-1))
            action = int(torch.multinomial(probabilities, 1, generator=generator))
        if cfg['protocol']['actions'][action] != expected:
            raise ValueError(f'Historical first-episode action differs: {group}/{circuit}/{seed}')
        state, _, _ = game.step(action)
        observed.append((game.luts, game.levels))
    if observed != records[0]['steps']:
        raise ValueError(f'Historical baseline first episode failed ABC replay: {circuit}')


def validate_old_baselines(circuits):
    digest, evidence = identity()
    manifest = read(OLD_TEN / 'experiment.json')
    old_cfg = manifest['config']
    current = training_config('i2c', 'frozen', ten_step=True)
    reasons = []
    if manifest['status'] != 'complete':
        reasons.append('Historical training manifest is incomplete.')
    for key in ('abc_binary', 'yosys_binary'):
        if manifest['tools'][key] != evidence['tools'][key]:
            reasons.append(f'{key} hash changed.')
    for circuit in circuits:
        if manifest['benchmarks'][circuit] != evidence['circuits'][circuit]:
            reasons.append(f'{circuit} benchmark hash changed.')
    source_drift = []
    for name, recorded in manifest['sources'].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != recorded:
            source_drift.append(name)
    old_protocol = old_cfg['protocol']
    new_protocol = current['protocol']
    for key in ('episodes', 'iterations', 'lut_inputs', 'initial_sequence', 'actions'):
        if old_protocol[key] != new_protocol[key]:
            reasons.append(f'Ten-step protocol mismatch: {key}')
    for circuit in circuits:
        if old_protocol['circuits'][circuit] != new_protocol['circuits'][circuit]:
            reasons.append(f'Ten-step circuit protocol mismatch: {circuit}')
    for key in ('features', 'network', 'normalization'):
        if old_cfg['method'][key] != current['method'][key]:
            reasons.append(f'Ten-step policy mismatch: {key}')
    reusable = []
    checks = []
    if not reasons:
        for circuit in circuits:
            cfg = training_config(circuit, 'frozen', ten_step=True)
            for group in ('frozen', 'uniform'):
                for seed in (0, 1, 2):
                    folder = OLD_TEN / 'training' / group / circuit / f'seed-{seed}'
                    result = read(folder / 'result.json')
                    if result['status'] != 'complete' or result['episodes_completed'] != 100:
                        reasons.append(f'Incomplete historical baseline: {group}/{circuit}/{seed}')
                        continue
                    reconstructed, records = _episodes(folder, circuit, cfg)
                    if reconstructed != result['best']:
                        reasons.append(f'Historical best mismatch: {group}/{circuit}/{seed}')
                        continue
                    _replay_first(OUT / 'ten-step' / 'baseline-replay' / group / circuit /
                                  f'seed-{seed}', circuit, cfg, records, group, seed, folder)
                    reusable.append((group, circuit, seed))
                    checks.append(dict(group=group, circuit=circuit, seed=seed,
                                       episodes=len(records), best=result['best'],
                                       first_episode_replayed=True))
    if reasons:
        reusable = []
    result = dict(status='complete', fingerprint=digest, reusable=[list(v) for v in reusable],
                  reasons=reasons, historical_source_drift=source_drift, checks=checks)
    dump(OUT / 'ten-baseline-validation.json', result)
    return set(reusable)


def _greedy_folder(circuit, seed):
    return OUT / 'ten-step' / 'greedy' / circuit / f'seed-{seed}'


def greedy_task(circuit, seed):
    cfg = training_config(circuit, 'reward', ten_step=True)
    digest, _ = identity()
    trained = OUT / 'ten-step' / 'reward' / circuit / f'seed-{seed}'
    result = read(trained / 'result.json')
    if result['status'] != 'complete' or result['fingerprint'] != digest:
        raise ValueError('Missing trained ten-step policy for greedy rollout.')
    trained_checkpoint = trained / 'checkpoint.pt'
    folder = _greedy_folder(circuit, seed)
    output = folder / 'result.json'
    if output.exists():
        old = read(output)
        if old['fingerprint'] == digest and old['status'] == 'complete':
            return
        raise ValueError(f'Conflicting greedy result: {output}')
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['environment']['torch_threads'])
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(len(cfg['method']['features']), len(cfg['protocol']['actions']),
                              cfg['method']['network'])
    saved = prior.old.load(trained_checkpoint)
    network.load_state_dict(saved['network'])
    network.eval().requires_grad_(False)
    network_hash = prior.old.state_hash(network.state_dict())
    game = RewardSession(cfg, cfg['protocol']['circuits'][circuit], folder)
    checkpoint = folder / 'checkpoint.pt'
    completed = 0
    if checkpoint.exists():
        previous = prior.old.load(checkpoint)
        if previous['fingerprint'] != digest or previous['trained_checkpoint_sha256'] != sha(trained_checkpoint):
            raise ValueError('Greedy resume fingerprint changed.')
        completed = previous['episodes_completed']
        game.episode = completed
        game.best = previous['best']
        game.best_netlists = previous['best_netlists']
        game.export_best()
    started = time.perf_counter()
    try:
        for episode in range(completed, SPEC['ten_step']['episodes']):
            state = game.reset()
            normalizer = Normalizer(len(state), cfg['method']['normalization'],
                                    cfg['method']['features'], state)
            done = False
            while not done:
                state = normalizer.normalize(state)
                with torch.no_grad():
                    logits, _ = network(torch.as_tensor(state, device='cpu'))
                    action = int(torch.argmax(logits))
                state, _, done = game.step(action)
            game.export_best()
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(dict(fingerprint=digest, trained_checkpoint_sha256=sha(trained_checkpoint),
                            episodes_completed=episode+1, best=game.best,
                            best_netlists=game.best_netlists), temporary)
            temporary.replace(checkpoint)
            if (episode+1) % 10 == 0:
                print(f'ten/greedy/{circuit}/seed-{seed}: {episode+1}/100', flush=True)
        verify_netlist(cfg, cfg['protocol']['circuits'][circuit], folder / 'best-mapped.v', game.best)
        dump(output, dict(status='complete', fingerprint=digest, group='greedy', circuit=circuit,
                          seed=seed, episodes_completed=100, mapping_calls=1100,
                          best=game.best, trained_checkpoint_sha256=sha(trained_checkpoint),
                          final_network_sha256=network_hash,
                          elapsed_seconds=time.perf_counter()-started))
    except Exception as error:
        dump(folder / 'status.json', dict(status='failed', episodes=game.episode,
                                        error=repr(error), traceback=traceback.format_exc()))
        raise


def _write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _summarize(circuits, reused):
    curves, final_rows, verdicts = [], [], {}
    for circuit in circuits:
        cfg = training_config(circuit, 'reward', ten_step=True)
        for group in ('reward', 'frozen', 'uniform', 'greedy'):
            for seed in SEEDS10:
                folder = (OLD_TEN / 'training' / group / circuit / f'seed-{seed}'
                          if (group, circuit, seed) in reused else
                          _greedy_folder(circuit, seed) if group == 'greedy' else
                          OUT / 'ten-step' / group / circuit / f'seed-{seed}')
                result = read(folder / 'result.json')
                reconstructed, records = _episodes(folder, circuit, cfg)
                if reconstructed != result['best']:
                    raise ValueError(f'Ten-step best differs from logs: {group}/{circuit}/{seed}')
                feasible_rate = sum(row['episode_feasible'] for row in records) / len(records)
                final_rows.append(dict(circuit=circuit, group=group, seed=seed,
                                       source='historical-verified' if (group,circuit,seed) in reused else 'new',
                                       best_luts=result['best']['luts'] if result['best']['feasible'] else '',
                                       best_levels=result['best']['levels'],
                                       best_feasible=result['best']['feasible'],
                                       episode_feasible_rate=feasible_rate,
                                       mapping_calls=records[-1]['calls']))
                for row in records:
                    curves.append(dict(circuit=circuit, group=group, seed=seed,
                                       calls=row['calls'],
                                       best_luts=row['best_luts'] if row['best_luts'] is not None else '',
                                       best_levels=row['best_levels'],
                                       best_feasible=row['best_feasible']))
        wins = []
        for seed in SEEDS10:
            scores = {row['group']: row for row in final_rows
                      if row['circuit'] == circuit and row['seed'] == seed}
            candidate = scores['reward']
            wins.append(bool(candidate['best_feasible'] and scores['frozen']['best_feasible'] and
                             scores['uniform']['best_feasible'] and
                             candidate['best_luts'] < scores['frozen']['best_luts'] and
                             candidate['best_luts'] < scores['uniform']['best_luts']))
        verdicts[circuit] = dict(seed_wins=sum(wins), required=SPEC['ten_step']['required_seed_wins'],
                                 passed=all(wins), paired_wins=wins)
    _write_csv(HERE / 'ten-results.csv', final_rows)
    _write_csv(HERE / 'ten-curves.csv', curves)
    digest, _ = identity()
    summary = dict(status='complete', fingerprint=digest, circuits=circuits, verdicts=verdicts,
                   baseline_reuse=[list(v) for v in sorted(reused)], rows=final_rows)
    dump(OUT / 'ten-summary.json', summary)
    return summary


def run_ten(supervise):
    gates = read(OUT / 'gates.json')
    digest, _ = identity()
    if gates['fingerprint'] != digest or gates['status'] != 'complete':
        raise ValueError('Four-step gates are incomplete or stale.')
    circuits = [name for name in SPEC['ten_step']['circuits_in_priority_order']
                if gates['gates'][name]['passed']]
    if not circuits:
        dump(OUT / 'ten-summary.json', dict(status='not_triggered', fingerprint=digest,
                                            circuits=[], verdicts={}))
        print('No circuit passed its four-step gate; ten-step stage not triggered.', flush=True)
        return
    reused = validate_old_baselines(circuits)
    for circuit in circuits:
        jobs = [('reward', circuit, seed) for seed in SEEDS10]
        jobs += [(group, circuit, seed) for group in ('frozen', 'uniform') for seed in SEEDS10
                 if (group, circuit, seed) not in reused]
        supervise('ten', jobs)
        (OUT / f'ten-{circuit}-manifest.json').write_bytes((OUT / 'ten-manifest.json').read_bytes())
        pending = []
        for seed in SEEDS10:
            path = _greedy_folder(circuit, seed) / 'result.json'
            if not path.exists():
                pending.append((f'greedy-{circuit}-{seed}',
                                [sys.executable, '-B', '-u', str(HERE / 'ten.py'),
                                 '--task', circuit, str(seed)]))
        processes = prior.old.supervise(pending, OUT / 'logs' / 'ten-greedy' / circuit,
                                        workers=SPEC['workers'], timeout=SPEC['task_timeout_seconds'])
        dump(OUT / f'ten-greedy-{circuit}-manifest.json',
             dict(fingerprint=digest, circuit=circuit, processes=processes,
                  status='complete' if all(row['exit_code'] == 0 for row in processes) else 'incomplete'))
        if any(row['exit_code'] != 0 for row in processes):
            raise RuntimeError(f'Greedy ten-step jobs incomplete: {circuit}')
    _summarize(circuits, reused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', nargs=2, metavar=('CIRCUIT', 'SEED'))
    args = parser.parse_args()
    if not args.task:
        parser.error('Use run.py --phase ten for the complete stage.')
    circuit, raw_seed = args.task
    seed = int(raw_seed)
    if circuit not in SPEC['ten_step']['circuits_in_priority_order'] or seed not in SEEDS10:
        raise ValueError('Greedy task outside fixed protocol.')
    greedy_task(circuit, seed)


if __name__ == '__main__':
    main()

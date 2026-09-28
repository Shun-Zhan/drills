"""Locked six-group study, independent inference and bounded supervision."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from drills.experiment import load_config, run_baselines
from drills.fpga_session import FPGASession
from drills.model import ActorCritic, Normalizer
from drills.segmented_model import SegmentedA2C, state_hash

BASE = 'adf7bef53cdf6c4161586c03e6e2caecffe26bc4'
GROUPS = ('A', 'B', 'C', 'D', 'F', 'U')
SEEDS = (10, 11, 12)
EVAL_SEEDS = tuple(range(30000, 30030))
GROUP_SETTINGS = {g: dict(learner='legacy' if g == 'A' else 'nstep',
    H=10 if g in ('A', 'B') else 50, K=50 if g == 'C' else 10,
    learning=g in ('A', 'B', 'C', 'D'), uniform=g == 'U') for g in GROUPS}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config():
    cfg = load_config(HERE / 'protocol.yml')
    expected = yaml.safe_load(subprocess.check_output(['git', 'show', BASE + ':params.yml'], cwd=ROOT))
    expected['protocol']['seeds'] = list(SEEDS)
    expected['protocol']['circuits'] = {'i2c': dict(file=str(ROOT / 'benchmarks/i2c.aig'), max_levels=4)}
    for key in ('protocol', 'method', 'environment'):
        if cfg[key] != expected[key]:
            raise ValueError('Settings differ from the locked supplied baseline: ' + key)
    if cfg['runtime']['workers'] != 3 or cfg['experiment'] != dict(groups=GROUP_SETTINGS,
        training_candidates_per_run=1000, evaluation_seeds=list(EVAL_SEEDS), evaluation_steps=50,
        timeout_seconds=1800, minimum_mean_lut_reduction=1, minimum_seed_wins=2, target_luts=300):
        raise ValueError('Group, sampling, execution or screening protocol changed.')
    return cfg


def group_config(cfg, group, fingerprint=None):
    local = copy.deepcopy(cfg)
    setting = local['experiment']['groups'][group]
    local['protocol'].update(iterations=setting['H'],
        episodes=local['experiment']['training_candidates_per_run'] // setting['H'])
    local['method'].update(learner=setting['learner'], update_interval=setting['K'],
                          learning_enabled=setting['learning'], uniform_actions=setting['uniform'])
    local['method']['normalization']['returns'] = 'standardize' if group == 'A' else 'none'
    local['runtime'].update(experiment_group=group, experiment_fingerprint=fingerprint)
    return local


def sources():
    paths = [ROOT / 'requirements.txt', *sorted((ROOT / 'drills').glob('*.py')),
             HERE / 'protocol.yml', *sorted(HERE.glob('*.py')),
             *sorted((ROOT / 'tests').glob('test_*.py'))]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def identity(cfg):
    payload = dict(sources=sources(), config=cfg,
        tools={k: sha(cfg['runtime'][k]) for k in ('abc_binary', 'yosys_binary')},
        benchmark=sha(cfg['protocol']['circuits']['i2c']['file']),
        python=sys.version, torch=torch.__version__, numpy=np.__version__)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def prepare(cfg):
    root = Path(cfg['runtime']['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    if any(p.name not in ('isolation-before.json', 'isolation-validation.json', '.lock') for p in root.iterdir()):
        raise FileExistsError('prepare never replaces existing experiment output.')
    for relative in ('drills/model.py', 'drills/fpga_session.py', 'drills/features.py',
                     'drills/experiment.py', 'params.yml', 'requirements.txt', 'benchmarks/i2c.aig'):
        original = subprocess.check_output(['git', 'show', BASE + ':' + relative], cwd=ROOT)
        if hashlib.sha256(original).hexdigest() != sha(ROOT / relative):
            raise ValueError('Original baseline file changed: ' + relative)
    fingerprint, payload = identity(cfg)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dump(root / 'experiment.json', dict(fingerprint=fingerprint, **payload,
        implementation_commit=commit, base_commit=BASE, started_at=now(),
        training_candidates=18000, evaluation_candidates=24000))
    run_baselines(cfg)
    print('Locked 18 independent searches and 480 evaluation trajectories.', flush=True)


def guard(cfg):
    root = Path(cfg['runtime']['output_dir'])
    manifest = json.loads((root / 'experiment.json').read_text())
    if identity(cfg)[0] != manifest['fingerprint']:
        raise ValueError('Source, protocol, tool, benchmark or environment changed after lock.')
    return root, manifest


def train_task(cfg, group, seed, resume=False):
    root, manifest = guard(cfg)
    folder = root / 'training' / group / f'seed-{seed}'
    local = group_config(cfg, group, manifest['fingerprint'])
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    if resume and (folder / 'result.json').exists():
        saved = json.loads((folder / 'result.json').read_text())
        if saved['status'] == 'complete' and saved['fingerprint'] == manifest['fingerprint']:
            print('Already complete:', group, seed, flush=True)
            return
    agent = SegmentedA2C(local, local['protocol']['circuits']['i2c'], seed, folder,
                         resume=resume and (folder / 'checkpoint.pt').exists())
    while agent.episodes_completed < local['protocol']['episodes']:
        segment = agent.run_segment()
        print(f'{group}/seed-{seed}: candidates {agent.candidates_completed}/1000; '
              f'updates={agent.updates_completed}; best LUT={agent.game.best["luts"]}', flush=True)
    dump(folder / 'result.json', dict(status='complete', group=group, seed=seed,
        fingerprint=manifest['fingerprint'], episodes_completed=agent.episodes_completed,
        candidates_completed=agent.candidates_completed, updates_completed=agent.updates_completed,
        rewards=agent.rewards, best=agent.game.best, training_seconds=agent.training_seconds,
        initial_network_hash=agent.initial_hash, final_network_hash=state_hash(agent.network.state_dict()),
        initial_optimizer_hash=agent.initial_optimizer_hash,
        final_optimizer_hash=state_hash(agent.optimizer.state_dict()),
        checkpoint_sha256=sha(agent.checkpoint)))


def inference_rollout(cfg, network, seed, folder, uniform=False):
    """One physical 50-step trajectory; 10/50 prefix metrics are dependent."""
    torch.set_num_threads(1)
    game = FPGASession(cfg, cfg['protocol']['circuits']['i2c'], folder)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    state = game.reset()
    normalizer = Normalizer(len(state), cfg['method']['normalization'])
    before = state_hash(network.state_dict())
    actions, rewards, steps, prefixes = [], [], [], {}
    for iteration in range(1, cfg['protocol']['iterations'] + 1):
        raw = state.copy()
        state = normalizer.normalize(state)
        with torch.no_grad():
            logits, value = network(torch.as_tensor(state, device='cpu'))
            probabilities = torch.full_like(logits, 1 / len(logits)) if uniform else logits.softmax(-1)
            action = torch.multinomial(probabilities, 1, generator=generator).item()
        steps.append(dict(iteration=iteration, raw_state=raw.tolist(), normalized_state=state.tolist(),
                          probabilities=probabilities.tolist(), value=float(value)))
        state, reward, done = game.step(action)
        actions.append(action)
        rewards.append(reward)
        if iteration in (10, 50) or done:
            prefixes[str(iteration)] = dict(best=copy.deepcopy(game.best),
                terminal=dict(luts=game.luts, levels=game.levels, feasible=game.levels <= 4))
            destination = Path(folder) / f'prefix-{iteration}'
            destination.mkdir(parents=True, exist_ok=True)
            dump(destination / 'best.json', game.best)
            for name, data in game.best_netlists.items():
                (destination / name).write_text(data)
    game.export_best()
    if before != state_hash(network.state_dict()):
        raise RuntimeError('Inference modified the model.')
    return dict(evaluation_seed=seed, actions=actions, rewards=rewards, steps=steps,
                prefixes=prefixes, best=game.best, terminal=prefixes[str(len(actions))]['terminal'])


def evaluate_task(cfg, group, training_seed, resume=False):
    root, manifest = guard(cfg)
    folder = root / 'evaluation' / group / f'seed-{training_seed}'
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise FileExistsError(folder)
    checkpoint = root / 'training' / group / f'seed-{training_seed}' / 'checkpoint.pt'
    saved = load(checkpoint)
    if saved['candidates_completed'] != 1000 or saved['active']:
        raise ValueError('Evaluation requires the prescribed final checkpoint.')
    local = group_config(cfg, group, manifest['fingerprint'])
    local['protocol'].update(episodes=1, iterations=50)
    digest = sha(checkpoint)
    with torch.random.fork_rng(devices=[]):
        network = ActorCritic(9, 7, local['method']['network'])
    network.load_state_dict(saved['network'])
    network.eval().requires_grad_(False)
    before = state_hash(network.state_dict())
    rows = []
    for seed in EVAL_SEEDS:
        destination = folder / f'rollout-{seed}'
        record = destination / 'rollout.json'
        if resume and record.exists():
            row = json.loads(record.read_text())
            if row['checkpoint_sha256'] != digest or row['status'] != 'complete':
                raise ValueError('Incompatible evaluation checkpoint.')
        else:
            row = inference_rollout(local, network, seed, destination, group == 'U')
            row.update(status='complete', group=group, training_seed=training_seed,
                       checkpoint_sha256=digest, bank_id=f'{group}-{training_seed}-{seed}')
            dump(record, row)
        rows.append(row)
        print(f'{group}/{training_seed}: rollout {len(rows)}/30, best={row["best"]["luts"]}', flush=True)
    if before != state_hash(network.state_dict()) or digest != sha(checkpoint):
        raise RuntimeError('Evaluation changed the source checkpoint.')
    dump(folder / 'result.json', dict(status='complete', group=group, training_seed=training_seed,
        checkpoint_sha256=digest, network_unchanged=True, checkpoint_unchanged=True,
        rollouts=rows))


def supervise(tasks, output, timeout=1800):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pending, active, records = list(tasks), [], []
    last_update = 0
    while pending or active:
        while pending and len(active) < 3:
            label, command = pending.pop(0)
            path = output / (label + '.log')
            stream = path.open('a')
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            active.append(dict(label=label, command=command, log=path, stream=stream,
                process=process, started=now(), start=time.monotonic(), timed_out=False))
        for job in list(active):
            process = job['process']
            if process.poll() is None and time.monotonic() - job['start'] > timeout:
                print('Hard timeout: stopping', job['label'], flush=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                job['timed_out'] = True
            code = process.poll()
            if code is not None:
                job['stream'].close()
                row = dict(label=job['label'], command=job['command'], started_at=job['started'],
                    finished_at=now(), elapsed_seconds=time.monotonic() - job['start'], exit_code=code,
                    timed_out=job['timed_out'], log=str(job['log'].relative_to(ROOT)),
                    log_sha256=sha(job['log']))
                records.append(row)
                active.remove(job)
                dump(output / 'processes.json', records)
                print(f'Finished {job["label"]}: exit={code}; {len(records)}/{len(tasks)}', flush=True)
        if time.monotonic() - last_update >= 30:
            row = dict(time=now(), completed=len(records), total=len(tasks), active=[dict(
                label=j['label'], pid=j['process'].pid, elapsed_seconds=time.monotonic() - j['start'],
                log_bytes=j['log'].stat().st_size) for j in active])
            with (output / 'monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print('Progress:', len(records), '/', len(tasks),
                  'active:', [j['label'] for j in active], flush=True)
            last_update = time.monotonic()
        if active:
            time.sleep(1)
    return records


def execute(cfg, phase, resume=False):
    root, _ = guard(cfg)
    tasks = []
    for group in GROUPS:
        for seed in ((10,) if phase == 'evaluate' and group == 'U' else SEEDS):
            command = [sys.executable, '-B', '-u', str(HERE / 'run.py'), phase, '--task', group, str(seed)]
            if resume:
                command.append('--resume')
            tasks.append((f'{group}-{seed}', command))
    folder = root / 'logs' / phase
    if (folder / 'processes.json').exists():
        if not resume:
            raise FileExistsError('Phase already attempted; explicit resume required.')
        attempt = 1
        while (folder / f'attempt-{attempt}').exists():
            attempt += 1
        folder = folder / f'attempt-{attempt}'
    records = supervise(tasks, folder, cfg['experiment']['timeout_seconds'])
    successful = all(row['exit_code'] == 0 for row in records)
    dump(root / (phase + '-provenance.json'), dict(processes=records, finished_at=now(),
         status='complete' if successful else 'incomplete'))
    if not successful:
        raise RuntimeError('Prescribed tasks failed; evidence retained, no automatic retry.')

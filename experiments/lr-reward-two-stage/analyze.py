"""Audit both experiments and produce compact evidence and the final report."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch

from study import (CIRCUITS, FOUR, GROUPS, HERE, OUT, ROOT, SEEDS, TEN,
                   eval_folder, eval_seeds, identity, read, sha, source_for,
                   training_config)
from drills.fpga_session import FPGASession

sys.path.insert(0, str(ROOT / 'experiments/four-step-reward'))
from reward import RewardTracker, original_reward  # noqa: E402
from table import table as lookup_table  # noqa: E402

LABELS = {'legacy': '原始实现', 'w001_old': '窗口/0.001/旧奖励',
          'w001_new': '窗口/0.001/新奖励', 'w010_old': '窗口/0.01/旧奖励',
          'w010_new': '窗口/0.01/新奖励'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def parse_episode(path, cfg, circuit, episode, reward_kind=None):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['iteration', 'optimization', 'luts', 'levels', 'reward'],
                f'Unexpected episode log header: {path}')
        raw = list(reader)
    require(len(raw) == 11, f'Incomplete episode: {path}')
    sequence = list(cfg['protocol']['initial_sequence'])
    limit = cfg['protocol']['circuits'][circuit]['max_levels']
    best, previous, tracker = None, None, None
    actions = []
    for index, entry in enumerate(raw):
        require(int(entry['iteration']) == index, f'Wrong iteration: {path}/{index}')
        luts, levels, reward = int(entry['luts']), int(entry['levels']), float(entry['reward'])
        require(luts > 0 and levels > 0 and math.isfinite(reward),
                f'Invalid mapped result: {path}/{index}')
        if index == 0:
            require(entry['optimization'] == sequence[-1] and reward == 0,
                    f'Invalid initial mapping: {path}')
            if reward_kind == 'new':
                tracker = RewardTracker(luts, levels, limit)
        else:
            action = entry['optimization']
            require(action in cfg['protocol']['actions'], f'Unknown action: {path}/{index}')
            sequence.append(action)
            actions.append(action)
            if reward_kind == 'new':
                expected = tracker.step(luts, levels)
            elif reward_kind == 'old':
                expected = original_reward(cfg, cfg['protocol']['circuits'][circuit],
                                           previous, (luts, levels))
            else:
                expected = None
            if expected is not None:
                require(math.isclose(reward, expected, rel_tol=1e-10, abs_tol=1e-10),
                        f'Reward mismatch: {path}/{index}')
        candidate = dict(luts=luts, levels=levels, feasible=levels <= limit,
                         sequence=list(sequence), episode=episode, iteration=index)
        if best is None or FPGASession.rank(candidate) < FPGASession.rank(best):
            best = candidate
        previous = (luts, levels)
    return best, actions


def check_proof(folder, best):
    proof = (folder / 'equivalence.log').read_text()
    require('Networks are equivalent' in proof, f'CEC failed: {folder}')
    values = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', proof)
    require(values and tuple(map(int, values[-1])) == (best['luts'], best['levels']),
            f'CEC metrics mismatch: {folder}')


def audit_training(group, circuit, seed):
    folder = source_for(group, circuit, seed)
    record = read(folder / 'result.json')
    cfg = training_config(group)
    require(record['status'] == 'complete' and record['episodes_completed'] == 100 and
            record['circuit'] == circuit and record['seed'] == seed,
            f'Incomplete training result: {folder}')
    best = None
    for episode in range(1, 101):
        kind = 'new' if group.endswith('_new') else 'old'
        local, _ = parse_episode(folder / f'episodes/{episode}/log.csv', cfg,
                                 circuit, episode, kind)
        if best is None or FPGASession.rank(local) < FPGASession.rank(best):
            best = local
    require(best == record['best'] and read(folder / 'best.json') == best,
            f'Training best disagrees with logs: {folder}')
    check_proof(folder, best)
    require((folder / 'best-mapped.v').is_file(), f'Missing best netlist: {folder}')
    return dict(circuit=circuit, group=group, seed=seed, mapping_calls=1100,
                best_feasible=int(best['feasible']), best_luts=best['luts'] if best['feasible'] else None,
                best_levels=best['levels'],
                source_checkpoint_path=str((folder / 'checkpoint.pt').relative_to(ROOT)),
                source_checkpoint_sha256=sha(folder / 'checkpoint.pt'),
                reused=int(folder != OUT / 'stage1/training' / group / circuit / f'seed-{seed}'))


def audit_evaluation(group, circuit, seed):
    folder = eval_folder(group, circuit, seed)
    result = read(folder / 'result.json')
    source = source_for(group, circuit, seed) / 'checkpoint.pt'
    digest, _ = identity(1)
    require(result['status'] == 'complete' and result['fingerprint'] == digest and
            result['checkpoint_sha256'] == sha(source) and
            result['group'] == group and result['circuit'] == circuit and
            result['training_seed'] == seed,
            f'Stale evaluation result: {folder}')
    rows = result['rollouts']
    require(len(rows) == 30 and [row['evaluation_seed'] for row in rows] == list(eval_seeds(seed)),
            f'Missing/duplicate evaluation rollout: {folder}')
    cfg = training_config('legacy')
    output = []
    for row in rows:
        destination = folder / f'rollout-{row["evaluation_seed"]}'
        require(read(destination / 'rollout.json') == row and row['status'] == 'complete' and
                row['fingerprint'] == digest and row['mapping_calls'] == 11 and
                row['log_sha256'] == sha(destination / 'episodes/1/log.csv') and
                row['best_mapped_sha256'] == sha(destination / 'best-mapped.v') and
                row['equivalence_log_sha256'] == sha(destination / 'equivalence.log'),
                f'Evaluation evidence changed: {destination}')
        best, actions = parse_episode(destination / 'episodes/1/log.csv', cfg, circuit, 1)
        require(best == row['best'] and actions == [cfg['protocol']['actions'][a]
                                                    for a in row['actions']],
                f'Evaluation sequence differs from log: {destination}')
        check_proof(destination, best)
        output.append(dict(circuit=circuit, group=group, training_seed=seed,
                           evaluation_seed=row['evaluation_seed'],
                           feasible=int(best['feasible']),
                           best_luts=best['luts'] if best['feasible'] else None,
                           best_levels=best['levels']))
    return output


def stage1_tables():
    digest, _ = identity(1)
    for name in ('train-manifest.json', 'pair-validation.json', 'evaluation-manifest.json'):
        manifest = read(OUT / 'stage1' / name)
        require(manifest['status'] in ('complete', 'pass') and manifest['fingerprint'] == digest,
                f'Incomplete stage-one manifest: {name}')
    training = [audit_training(group, circuit, seed)
                for circuit in CIRCUITS for seed in SEEDS for group in GROUPS]
    rollouts = [row for circuit in CIRCUITS for seed in SEEDS for group in GROUPS
                for row in audit_evaluation(group, circuit, seed)]
    per_seed = []
    for circuit in CIRCUITS:
        for seed in SEEDS:
            for group in GROUPS:
                subset = [row for row in rollouts if (row['circuit'], row['training_seed'],
                                                      row['group']) == (circuit, seed, group)]
                feasible = sum(row['feasible'] for row in subset)
                require(len(subset) == 30, 'Wrong rollout denominator.')
                if circuit != 'max':
                    require(feasible == 30, f'Unexpected infeasible trajectory: {circuit}/{group}/{seed}')
                    primary = sum(row['best_luts'] for row in subset) / 30
                else:
                    primary = feasible / 30
                per_seed.append(dict(circuit=circuit, seed=seed, group=group,
                                     primary=primary, feasible_rollouts=feasible,
                                     mean_luts_if_feasible=(sum(row['best_luts'] for row in subset
                                                                if row['feasible']) / feasible
                                                            if feasible else None)))
    return training, rollouts, per_seed


def score(circuit, row):
    return row['primary'] if circuit == 'max' else -row['primary']


def stage1_contrasts(per_seed):
    lookup = {(r['circuit'], r['seed'], r['group']): r for r in per_seed}
    rows = []
    for circuit in CIRCUITS:
        for seed in SEEDS:
            values = {g: score(circuit, lookup[circuit, seed, g]) for g in GROUPS}
            contrasts = {
                'final_minus_legacy': values['w010_new'] - values['legacy'],
                'window_at_001': values['w001_old'] - values['legacy'],
                'learning_rate_old_reward': values['w010_old'] - values['w001_old'],
                'learning_rate_new_reward': values['w010_new'] - values['w001_new'],
                'new_reward_at_001': values['w001_new'] - values['w001_old'],
                'new_reward_at_010': values['w010_new'] - values['w010_old'],
                'interaction': (values['w010_new'] - values['w010_old'] -
                                values['w001_new'] + values['w001_old']),
            }
            rows.extend(dict(circuit=circuit, seed=seed, contrast=name,
                             improvement=value) for name, value in contrasts.items())
    return rows


def stage2_tables():
    digest, _ = identity(2)
    manifest = read(OUT / 'stage2/train-manifest.json')
    exact = read(OUT / 'stage2/exact.json')
    require(manifest['status'] == 'complete' and manifest['fingerprint'] == digest and
            exact['status'] == 'complete' and exact['fingerprint'] == digest,
            'Stage-two training or exact evaluation is incomplete.')
    rows = exact['rows']
    mappings = lookup_table()
    cfg = training_config('legacy')
    actions = cfg['protocol']['actions']
    for circuit in FOUR['circuits']:
        for seed in FOUR['seeds']:
            folder = OUT / 'stage2/training' / circuit / f'seed-{seed}'
            result = read(folder / 'result.json')
            require(result['status'] == 'complete' and result['fingerprint'] == digest and
                    result['episodes_completed'] == 1000 and result['lookup_mappings'] == 5000 and
                    result['checkpoint_sha256'] == sha(folder / 'checkpoint.pt') and
                    result['best_mapped_sha256'] == sha(folder / 'best-mapped.v') and
                    result['equivalence_log_sha256'] == sha(folder / 'equivalence.log'),
                    f'Incomplete four-step training evidence: {folder}')
            best = None
            completed = 0
            with (folder / 'trajectories.jsonl').open() as stream:
                for episode, line in enumerate(stream, 1):
                    completed = episode
                    record = json.loads(line)
                    steps = record['steps']
                    require(record['episode'] == episode and len(steps) == 5,
                            f'Incomplete lookup episode: {folder}/{episode}')
                    for iteration, step in enumerate(steps):
                        prefix = tuple(step['actions'])
                        require(step['iteration'] == iteration and len(prefix) == iteration and
                                0 <= iteration <= 4 and all(a in range(7) for a in prefix),
                                f'Invalid lookup action: {folder}/{episode}/{iteration}')
                        mapped = mappings[circuit, prefix]
                        require((step['luts'], step['levels']) == (mapped['luts'], mapped['levels']),
                                f'Lookup mapping mismatch: {folder}/{episode}/{iteration}')
                        candidate = dict(luts=step['luts'], levels=step['levels'],
                                         feasible=step['levels'] <= cfg['protocol']['circuits'][circuit]['max_levels'],
                                         sequence=['strash', *[actions[a] for a in prefix]],
                                         episode=episode, iteration=iteration)
                        if best is None or FPGASession.rank(candidate) < FPGASession.rank(best):
                            best = candidate
            require(completed == 1000 and best == result['best'] and
                    read(folder / 'best.json') == best,
                    f'Four-step best differs from lookup logs: {folder}')
            check_proof(folder, best)
    expected = {(c, s, e) for c in FOUR['circuits'] for s in FOUR['seeds']
                for e in FOUR['snapshots']}
    actual = {(r['circuit'], r['seed'], r['episode']) for r in rows}
    require(len(rows) == len(expected) and actual == expected, 'Missing/duplicate exact result.')
    for row in rows:
        checkpoint = OUT / 'stage2/training' / row['circuit'] / f'seed-{row["seed"]}' / \
            'snapshots' / f'{row["episode"]}.pt'
        require(row['checkpoint_sha256'] == sha(checkpoint), f'Exact checkpoint changed: {checkpoint}')
        require(abs(row['probability_sum'] - 1) <= 2e-6, 'Exact probability not normalized.')
        # The prior exact evaluator accumulates float32 policy weights as Python
        # floats.  Its total can differ from one by ~2e-7; divide by that total
        # before publishing probabilities and unconditional expected gaps.
        total = row['probability_sum']
        row['feasible_probability'] /= total
        row['optimal_probability'] /= total
        if row['expected_gap'] is not None:
            row['expected_gap'] /= total
        require(0 <= row['feasible_probability'] <= 1 + 1e-12 and
                0 <= row['optimal_probability'] <= 1 + 1e-12,
                'Normalized exact probability outside [0, 1].')
    return rows


def _svg_header(width, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#1f2937} .small{font-size:12px}'
            '.title{font-size:18px;font-weight:bold}</style>']


def stage1_svg(per_seed):
    width, height = 960, 490
    lines = _svg_header(width, height)
    lines.append('<text x="30" y="34" class="title">100×10 冻结策略：各组平均表现</text>')
    lines.append('<text x="30" y="54" class="small">每点为三个训练种子的均值；各面板使用标明的局部纵轴</text>')
    colors = ['#64748b', '#38bdf8', '#0ea5e9', '#f59e0b', '#ef4444']
    for panel, circuit in enumerate(CIRCUITS):
        x0 = 65 + panel * 300
        y0, top = 376, 140
        vals = [sum(r['primary'] for r in per_seed if r['circuit'] == circuit and
                    r['group'] == g) / 3 for g in GROUPS]
        low, high = min(vals), max(vals)
        if high == low:
            high += 1
        pad = (high - low) * 0.12
        low -= pad
        high += pad
        lines.append(f'<text x="{x0-18}" y="88" font-size="16" font-weight="bold">{circuit}</text>')
        lines.append(f'<text x="{x0-18}" y="107" class="small">' +
                     ('可行率 ↑' if circuit == 'max' else '平均最佳可行 LUT ↓') + '</text>')
        lines.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#94a3b8"/>')
        for fraction in (0, 0.5, 1):
            y = y0 - fraction * (y0 - top)
            value = low + fraction * (high - low)
            lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+226}" y2="{y:.1f}" '
                         'stroke="#e2e8f0"/>')
            lines.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="small">'
                         f'{value:.2f}</text>')
        for i, value in enumerate(vals):
            x = x0 + 28 + i * 46
            y = y0 - (value - low) / (high - low) * (y0 - top)
            lines.append(f'<circle cx="{x}" cy="{y:.1f}" r="6" fill="{colors[i]}"/>')
            lines.append(f'<text x="{x}" y="{y-10:.1f}" text-anchor="middle" class="small">'
                         f'{value:.2f}</text>')
            lines.append(f'<text x="{x}" y="401" text-anchor="middle" class="small">{i+1}</text>')
    lines.append('<text x="35" y="447" class="small">1 原实现　2 窗口/0.001/旧　3 窗口/0.001/新　4 窗口/0.01/旧　5 窗口/0.01/新</text>')
    lines.append('</svg>')
    (HERE / 'ten-step.svg').write_text('\n'.join(lines) + '\n')


def stage2_svg(rows):
    width, height = 900, 470
    lines = _svg_header(width, height)
    lines.append('<text x="30" y="34" class="title">1000×4 冻结策略精确曲线：每点为 10 个种子均值</text>')
    lines.append('<text x="30" y="54" class="small">各面板纵轴为局部尺度；预设终点为 1000 回合</text>')
    for panel, circuit in enumerate(FOUR['circuits']):
        x0, y0, xspan, yspan = 52 + panel * 290, 365, 220, 250
        points = []
        for episode in FOUR['snapshots']:
            subset = [r for r in rows if r['circuit'] == circuit and r['episode'] == episode]
            values = [r['feasible_probability'] if circuit == 'max' else r['expected_gap']
                      for r in subset]
            require(all(v is not None for v in values), f'Undefined exact primary: {circuit}')
            points.append((episode, sum(values) / len(values)))
        low, high = min(v for _, v in points), max(v for _, v in points)
        if high == low:
            high += 1
        pad = (high - low) * 0.1
        low -= pad
        high += pad
        mapped = [(x0 + e / 1000 * xspan, y0 - (v-low)/(high-low)*yspan)
                  for e, v in points]
        lines.append(f'<text x="{x0}" y="77" font-size="16" font-weight="bold">{circuit}</text>')
        lines.append(f'<text x="{x0}" y="97" class="small">' +
                     ('可行概率 ↑' if circuit == 'max' else '期望最优差距 ↓') + '</text>')
        lines.append(f'<line x1="{x0}" y1="{y0-yspan}" x2="{x0}" y2="{y0}" stroke="#94a3b8"/>')
        for fraction in (0, 0.5, 1):
            y = y0 - fraction * yspan
            value = low + fraction * (high - low)
            lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+xspan}" y2="{y:.1f}" '
                         'stroke="#e2e8f0"/>')
            lines.append(f'<text x="{x0-5}" y="{y+4:.1f}" text-anchor="end" class="small">'
                         f'{value:.2f}</text>')
        lines.append('<polyline fill="none" stroke="#2563eb" stroke-width="2" points="' +
                     ' '.join(f'{x:.1f},{y:.1f}' for x, y in mapped) + '"/>')
        for (episode, value), (x, y) in zip(points, mapped):
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>')
            lines.append(f'<text x="{x-13:.1f}" y="{y-9:.1f}" class="small">{value:.3f}</text>')
            lines.append(f'<text x="{x-12:.1f}" y="385" class="small">{episode}</text>')
    lines.append('</svg>')
    (HERE / 'four-step.svg').write_text('\n'.join(lines) + '\n')


def report(training, per_seed, contrasts, exact_rows):
    lookup = {(r['circuit'], r['seed'], r['group']): r for r in per_seed}
    improvement = {(r['circuit'], r['seed'], r['contrast']): r['improvement'] for r in contrasts}
    lines = ['# 学习率、奖励与训练轮数：两步实验报告', '',
             '## 实验口径', '',
             '第一步在同样的 100 回合×10 步（每模型 1,100 次 ABC 映射）下比较原始实现与窗口回报的学习率×奖励四格。'
             '三个训练种子为原仓库的 0、1、2；每个冻结模型使用 30 条新的十步轨迹，同种子五组共享评估随机数。'
             '主指标是冻结策略表现，训练期最好解是次要搜索指标。', '',
             '第二步固定窗口回报、学习率 0.01 和新奖励，只把四步训练从 250 回合延长到 1000 回合。'
             '使用种子 0–9 和已认证查表环境；冻结策略对 2,401 条四步动作序列作精确评估。', '',
             '这两步都是对已探索过的设置继续研究；第一步只有三个独立训练种子。'
             '下文不使用显著性通过或普遍有效的表述。', '',
             '## 第一步：100×10 的冻结策略', '',
             '| 电路 | 配置 | seed 0 | seed 1 | seed 2 | 三种子均值 |',
             '|---|---|---:|---:|---:|---:|']
    for circuit in CIRCUITS:
        for group in GROUPS:
            vals = [lookup[circuit, seed, group]['primary'] for seed in SEEDS]
            lines.append(f'| {circuit} | {LABELS[group]} | ' +
                         ' | '.join(f'{v:.3f}' for v in vals) +
                         f' | {sum(vals)/3:.3f} |')
    lines += ['', 'i2c、int2float 的数值是平均最佳可行 LUT，越低越好；max 是可行轨迹比例，越高越好。'
              '各轨迹包含初始映射，不可行 max 轨迹没有被赋予虚构 LUT 罚分。',
              '', '![五组十步冻结策略表现](ten-step.svg)', '',
              '### 最终组合相对原始实现', '',
              '| 电路 | seed 0 改善 | seed 1 改善 | seed 2 改善 | 平均改善 | 改善种子 | 观察 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for circuit in CIRCUITS:
        vals = [improvement[circuit, seed, 'final_minus_legacy'] for seed in SEEDS]
        wins = sum(v > 0 for v in vals)
        label = ('三个种子均改善' if wins == 3 else
                 '三个种子均未改善' if wins == 0 else '种子方向不一致')
        lines.append(f'| {circuit} | ' + ' | '.join(f'{v:+.3f}' for v in vals) +
                     f' | {sum(vals)/3:+.3f} | {wins}/3 | {label} |')
    lines += ['', '正值表示最终组合较好；i2c、int2float 为减少的 LUT，max 为增加的可行率。'
              '三种子方向描述不等于统计检验通过。', '',
              '### 学习率、奖励和窗口对照', '',
              '| 电路 | 对照 | seed 0 | seed 1 | seed 2 | 均值 |',
              '|---|---|---:|---:|---:|---:|']
    names = [('window_at_001', '窗口效应：0.001＋旧奖励'),
             ('learning_rate_old_reward', '学习率：旧奖励'),
             ('learning_rate_new_reward', '学习率：新奖励'),
             ('new_reward_at_001', '奖励：0.001'),
             ('new_reward_at_010', '奖励：0.01'),
             ('interaction', '学习率×奖励交互')]
    for circuit in CIRCUITS:
        for key, label in names:
            vals = [improvement[circuit, seed, key] for seed in SEEDS]
            lines.append(f'| {circuit} | {label} | ' +
                         ' | '.join(f'{v:+.3f}' for v in vals) +
                         f' | {sum(vals)/3:+.3f} |')
    lines += ['', '上述对照为辅助解释。交互项是“0.01 下新奖励效应”减去“0.001 下新奖励效应”；'
              '不同电路的单位不同，不能跨电路合并。', '',
              '### 同预算训练搜索（次要）', '',
              '| 电路 | 配置 | seed 0 最佳 LUT | seed 1 最佳 LUT | seed 2 最佳 LUT |',
              '|---|---|---:|---:|---:|']
    train_lookup = {(r['circuit'], r['group'], r['seed']): r for r in training}
    for circuit in CIRCUITS:
        for group in GROUPS:
            vals = [train_lookup[circuit, group, seed]['best_luts'] for seed in SEEDS]
            lines.append(f'| {circuit} | {LABELS[group]} | ' +
                         ' | '.join(str(v) if v is not None else '未找到可行解' for v in vals) + ' |')
    lines += ['', '训练期最好值经过逐步日志重算，不能替代冻结策略独立评估。', '',
              '## 第二步：四步延长到 1000 回合', '',
              '| 电路 | 250 回合均值 | 1000 回合均值 | 配对平均改善 | 改善/持平/退步种子 | 观察 |',
              '|---|---:|---:|---:|---|---|']
    exact = {(r['circuit'], r['seed'], r['episode']): r for r in exact_rows}
    for circuit in FOUR['circuits']:
        start = [exact[circuit, seed, 250]['feasible_probability' if circuit == 'max'
                                          else 'expected_gap'] for seed in FOUR['seeds']]
        end = [exact[circuit, seed, 1000]['feasible_probability' if circuit == 'max'
                                         else 'expected_gap'] for seed in FOUR['seeds']]
        diffs = [(b-a if circuit == 'max' else a-b) for a, b in zip(start, end)]
        win, tie, loss = (sum(v > 1e-10 for v in diffs), sum(abs(v) <= 1e-10 for v in diffs),
                          sum(v < -1e-10 for v in diffs))
        observed = ('观察到改善' if sum(diffs)/10 > 0 and win >= 6 else
                    '观察到退步' if sum(diffs)/10 < 0 and loss >= 6 else '未见一致方向')
        lines.append(f'| {circuit} | {sum(start)/10:.4f} | {sum(end)/10:.4f} | '
                     f'{sum(diffs)/10:+.4f} | {win}/{tie}/{loss} | {observed} |')
    lines += ['', 'i2c、int2float 比较期望最优 LUT 差距（越低越好）；max 比较单次可行概率（越高越好）。'
              '精确评估的策略概率按其浮点总质量归一化。'
              '250 与 1000 的每种子完整结果见 `four-step-seeds.csv`；中间检查点只用于下图趋势。',
              '', '![四步训练长度曲线](four-step.svg)', '',
              '贪心动作、最优命中概率及训练期最好解随逐种子表提供。1000 回合是事先固定的终点，'
              '没有从中间检查点挑选最有利结果。', '',
              '## 完整性与边界', '',
              '- 第一步要求 45/45 个模型来源，其中 15 个沿用已核验检查点、30 个重新训练；'
              '来源路径及 SHA-256 见 `ten-step-training.csv`；'
              '冻结评估轨迹完成 1,350/1,350 条；'
              '每次训练均核对 100×11 个映射日志，并复核最佳网表的组合等价证据。',
              '- 第二步要求 30/30 次 1000 回合查表训练；第 250 回合的网络、优化器、随机状态、'
              '回报窗口和最佳结果逐种子与旧实验一致，最终最佳网表经真实 ABC 重放及组合等价检查。',
              '- 四步延长只检验既有三电路和既有种子下的训练长度；不能把 1000×4 的结果替换为'
              '原始 100×10 同预算比较，也不证明在其他电路上泛化。',
              '- 原始日志与检查点保存在 Git 忽略的 `results/lr-reward-two-stage/`；'
              '提交的 CSV、图和哈希清单支持复核。', '']
    (HERE / 'report.md').write_text('\n'.join(lines).rstrip('\n') + '\n')


def main():
    training, rollouts, per_seed = stage1_tables()
    contrasts = stage1_contrasts(per_seed)
    exact_rows = stage2_tables()
    write_csv(HERE / 'ten-step-training.csv', training,
              ['circuit', 'group', 'seed', 'mapping_calls', 'best_feasible', 'best_luts',
               'best_levels', 'source_checkpoint_path', 'source_checkpoint_sha256', 'reused'])
    write_csv(HERE / 'ten-step-rollouts.csv', rollouts,
              ['circuit', 'group', 'training_seed', 'evaluation_seed', 'feasible',
               'best_luts', 'best_levels'])
    write_csv(HERE / 'ten-step-seeds.csv', per_seed,
              ['circuit', 'seed', 'group', 'primary', 'feasible_rollouts',
               'mean_luts_if_feasible'])
    write_csv(HERE / 'ten-step-contrasts.csv', contrasts,
              ['circuit', 'seed', 'contrast', 'improvement'])
    stage1_svg(per_seed)
    compact = []
    for row in exact_rows:
        result = read(OUT / 'stage2/training' / row['circuit'] /
                      f'seed-{row["seed"]}' / 'result.json')
        checkpoint = OUT / 'stage2/training' / row['circuit'] / f'seed-{row["seed"]}' / \
            'snapshots' / f'{row["episode"]}.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        training_best = saved['best']
        if row['episode'] == 1000:
            require(training_best == result['best'], f'Final best differs from snapshot: {checkpoint}')
        compact.append(dict(circuit=row['circuit'], seed=row['seed'], episode=row['episode'],
                            expected_gap=row['expected_gap'],
                            feasible_probability=row['feasible_probability'],
                            optimal_probability=row['optimal_probability'],
                            greedy_actions=json.dumps(row['greedy_actions']),
                            greedy_best_luts=row['greedy_best']['luts'],
                            greedy_best_feasible=int(row['greedy_best']['feasible']),
                            training_best_luts=training_best['luts']
                            if training_best and training_best['feasible'] else None,
                            checkpoint_sha256=row['checkpoint_sha256']))
    write_csv(HERE / 'four-step-seeds.csv', compact,
              ['circuit', 'seed', 'episode', 'expected_gap', 'feasible_probability',
               'optimal_probability', 'greedy_actions', 'greedy_best_luts',
               'greedy_best_feasible', 'training_best_luts', 'checkpoint_sha256'])
    stage2_svg(exact_rows)
    report(training, per_seed, contrasts, exact_rows)
    files = ['ten-step-training.csv', 'ten-step-rollouts.csv', 'ten-step-seeds.csv',
             'ten-step-contrasts.csv', 'four-step-seeds.csv', 'ten-step.svg',
             'four-step.svg']
    evidence = dict(stage1_fingerprint=identity(1)[0], stage2_fingerprint=identity(2)[0],
                    files={name: sha(HERE / name) for name in files},
                    raw_manifests={str(path.relative_to(ROOT)): sha(path) for path in
                                   (OUT / 'stage1/preflight.json', OUT / 'stage1/train-manifest.json',
                                    OUT / 'stage1/evaluation-manifest.json',
                                    OUT / 'stage2/preflight.json', OUT / 'stage2/train-manifest.json',
                                    OUT / 'stage2/exact.json')})
    (HERE / 'evidence-hashes.json').write_text(json.dumps(evidence, indent=2,
                                                         ensure_ascii=False) + '\n')
    print('Final report and evidence complete:', HERE / 'report.md')


if __name__ == '__main__':
    main()

"""Report exact policy quality and the predeclared i2c expansion decision."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from support import HERE, OUT, SEEDS, SNAPSHOTS, SPEC, dump, identity, read, sha

T9_975 = 2.2621571627409915


def interval(differences):
    values = np.asarray(differences, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError('A paired interval requires ten finite seed differences.')
    rng = np.random.default_rng(SPEC['bootstrap_seed'])
    draws = values[rng.integers(0, len(values), size=(SPEC['bootstrap_draws'], len(values)))].mean(axis=1)
    bootstrap = [float(x) for x in np.quantile(draws, (0.025, 0.975))]
    mean = float(values.mean())
    margin = T9_975 * float(values.std(ddof=1)) / math.sqrt(10)
    return dict(mean=mean, bootstrap95=bootstrap, paired_t95=[mean - margin, mean + margin])


def old_index(exact):
    return {(row['circuit'], row['seed'], row['policy'], row['episode']): row
            for row in exact['rows']}


def new_index(exact):
    return {(row['circuit'], row['seed'], row['episode']): row
            for row in exact['rows']}


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def fmt(value, digits=3):
    return '未定义' if value is None else f'{value:.{digits}f}'


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def draw_curves(old_rows, new_rows):
    series = []
    for circuit, measure, title in [('i2c', 'expected_best_gap', 'i2c：k 次取最好，期望 LUT 差距'),
                                    ('max', 'any_feasible', 'max：k 次内至少一次可行的概率')]:
        for label, rows in old_rows.items():
            relevant = [row for row in rows if row['circuit'] == circuit]
            if relevant:
                series.append((circuit, label, [float(np.mean([r['curves'][str(k)][measure]
                                                         for r in relevant])) for k in range(1, 101)]))
        relevant = [row for row in new_rows if row['circuit'] == circuit]
        if relevant:
            series.append((circuit, '滑动窗口', [float(np.mean([r['curves'][str(k)][measure]
                                                      for r in relevant])) for k in range(1, 101)]))
    if not series:
        return
    colors = {'旧 0.01': '#dc2626', '初始网络': '#64748b', '均匀随机': '#16a34a',
              '滑动窗口': '#2563eb'}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 560">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172033;font-size:14px}</style>']
    for panel, (circuit, measure, title) in enumerate([
            ('i2c', 'expected_best_gap', 'i2c: expected best LUT gap'),
            ('max', 'any_feasible', 'max: probability of at least one feasible run')]):
        x0, x1 = 72 + panel * 500, 455 + panel * 500
        y0, y1 = 70, 450
        selected = [(name, values) for c, name, values in series if c == circuit]
        if not selected:
            continue
        low = min(min(v) for _, v in selected)
        high = max(max(v) for _, v in selected)
        if high == low:
            high = low + 1
        parts.append(f'<text x="{x0}" y="35" font-size="17">{title}</text>')
        parts.append(f'<path d="M{x0} {y0}V{y1}H{x1}" fill="none" stroke="#667085"/>')
        for name, values in selected:
            points = ' '.join(f'{x0 + i / 99 * (x1 - x0):.1f},'
                              f'{y1 - (v - low) / (high - low) * (y1 - y0):.1f}'
                              for i, v in enumerate(values))
            parts.append(f'<polyline points="{points}" fill="none" '
                         f'stroke="{colors[name]}" stroke-width="2.5"/>')
        parts.append(f'<text x="{x0}" y="470">k = 1</text><text x="{x1}" y="470" '
                     f'text-anchor="end">k = 100</text>')
        parts.append(f'<text x="{x0-8}" y="{y0+5}" text-anchor="end">{high:.2f}</text>')
        parts.append(f'<text x="{x0-8}" y="{y1+5}" text-anchor="end">{low:.2f}</text>')
    for index, (name, color) in enumerate(colors.items()):
        x = 75 + index * 225
        parts.append(f'<line x1="{x}" x2="{x+27}" y1="520" y2="520" '
                     f'stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+34}" y="525">{name}</text>')
    parts.append('</svg>')
    (HERE / 'best-of-k.svg').write_text('\n'.join(parts) + '\n')


def main():
    digest, _ = identity()
    old_exact = read(OUT / 'old-exact.json')
    if old_exact['status'] != 'complete' or old_exact['followup_fingerprint'] != digest:
        raise ValueError('Old exact evaluation does not match the current follow-up sources.')
    original = old_index(old_exact)
    new_path = OUT / 'new-exact.json'
    new_exact = read(new_path) if new_path.exists() else None
    if new_exact and (new_exact['status'] != 'complete' or
                      new_exact['followup_fingerprint'] != digest):
        raise ValueError('New exact evaluation does not match the current follow-up sources.')
    updated = new_index(new_exact) if new_exact else {}
    lines = ['# 四步环境：精确复评与跨回合回报实验', '',
             '基于完整的 7⁴ 条动作序列计算冻结策略的期望结果；训练期累计最好值另作辅助指标。',
             '最优 LUT 为 int2float=44、i2c=312、max=781。', '',
             '## 原有模型的精确复评', '',
             '| 电路 | 旧 0.01 | 初始网络 | 均匀随机 |',
             '|---|---:|---:|---:|']
    old_groups = {'旧 0.01': [], '初始网络': [], '均匀随机': []}
    for circuit in ('int2float', 'i2c', 'max'):
        high = [original[circuit, seed, 'lr010', 250] for seed in SEEDS]
        initial = [original[circuit, seed, 'lr001', 0] for seed in SEEDS]
        uniform = [original[circuit, None, 'uniform', None]]
        old_groups['旧 0.01'].extend(high)
        old_groups['初始网络'].extend(initial)
        old_groups['均匀随机'].extend(uniform)
        field = 'feasible_probability' if circuit == 'max' else 'expected_gap'
        digits = 6 if circuit == 'max' else 3
        lines.append(f'| {circuit} ({"单次可行率" if circuit == "max" else "单次期望差距"}) | '
                     f'{fmt(mean(high, field), digits)} | {fmt(mean(initial, field), digits)} | '
                     f'{fmt(uniform[0][field], digits)} |')
    lines += ['', '均匀随机在精确口径下每个电路只有一个理论值；它没有训练种子波动。', '',
              '### 旧模型训练快照', '',
              '| 电路 | 学习率 | 0 | 50 | 100 | 150 | 200 | 250 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for circuit in ('int2float', 'i2c', 'max'):
        metric = 'feasible_probability' if circuit == 'max' else 'expected_gap'
        for policy in ('lr001', 'lr010'):
            values = [mean([original[circuit, seed, policy, episode] for seed in SEEDS], metric)
                      for episode in SNAPSHOTS]
            digits = 6 if circuit == 'max' else 3
            lines.append(f'| {circuit} | {policy} | ' +
                         ' | '.join(fmt(v, digits) for v in values) + ' |')
    old_i2c = [original['i2c', seed, 'lr010', 250]['expected_gap'] for seed in SEEDS]
    old_initial = [original['i2c', seed, 'lr001', 0]['expected_gap'] for seed in SEEDS]
    old_estimate = interval([a - b for a, b in zip(old_initial, old_i2c)])
    lines += ['', 'i2c 旧 0.01 相对初始网络的配对平均改善为 '
              f'{fmt(old_estimate["mean"])} LUT；bootstrap 95% 区间 '
              f'[{fmt(old_estimate["bootstrap95"][0])}, '
              f'{fmt(old_estimate["bootstrap95"][1])}]，仍应作为探索性观察。', '']
    lines += ['', 'i2c 旧 0.01 的种子 7、9 逐快照单次期望差距：', '',
              '| 种子 | 0 | 50 | 100 | 150 | 200 | 250 |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for seed in (7, 9):
        values = [original['i2c', seed, 'lr010', episode]['expected_gap']
                  for episode in SNAPSHOTS]
        lines.append(f'| {seed} | ' + ' | '.join(fmt(v) for v in values) + ' |')

    gate = None
    paired_rows = []
    final_new = []
    if updated:
        lines += ['', '## 滑动窗口回报：冻结策略结果', '',
                  '新方法先在 i2c 运行十个配对训练种子。扩展门槛只决定是否运行另外两个电路，不是统计显著性检验。', '']
        for circuit in ('i2c', 'int2float', 'max'):
            rows = [updated[circuit, seed, 250] for seed in SEEDS
                    if (circuit, seed, 250) in updated]
            if not rows:
                continue
            if len(rows) != 10:
                raise ValueError(f'Incomplete exact final model coverage: {circuit}')
            final_new.extend(rows)
            baseline = [original[circuit, seed, 'lr010', 250] for seed in SEEDS]
            field = 'feasible_probability' if circuit == 'max' else 'expected_gap'
            lines.append(f'### {circuit}')
            lines.append('')
            digits = 6 if circuit == 'max' else 3
            lines.append(f'新方法 {"单次可行率" if circuit == "max" else "单次期望差距"}均值 '
                         f'{fmt(mean(rows, field), digits)}；旧 0.01 为 '
                         f'{fmt(mean(baseline, field), digits)}。')
            lines.append('')
            new_search = [read(OUT / 'training' / circuit / f'seed-{seed}/result.json')['best']
                          for seed in SEEDS]
            old_search = [read(OUT.parent / 'four-step-lr/training/lr010' / circuit /
                               f'seed-{seed}/result.json')['best'] for seed in SEEDS]
            new_feasible = [row['luts'] for row in new_search if row['feasible']]
            old_feasible = [row['luts'] for row in old_search if row['feasible']]
            lines += [f'训练期间累计最好值（辅助）：新方法可行种子 {len(new_feasible)}/10，'
                      f'旧 0.01 为 {len(old_feasible)}/10；'
                      f'全部可行时平均 LUT 分别为 '
                      f'{fmt(float(np.mean(new_feasible)) if len(new_feasible) == 10 else None)}、'
                      f'{fmt(float(np.mean(old_feasible)) if len(old_feasible) == 10 else None)}。', '']
            if circuit == 'i2c':
                initial = [original[circuit, seed, 'lr001', 0] for seed in SEEDS]
                diffs = [a['expected_gap'] - b['expected_gap']
                         for a, b in zip(baseline, rows)]
                estimate = interval(diffs)
                improved = sum(delta > 0 for delta in diffs)
                gate = (estimate['mean'] > 0 and improved >=
                        SPEC['expansion_gate']['minimum_improved_seeds'] and
                        mean(rows, 'expected_gap') <= mean(initial, 'expected_gap'))
                lines += [f'主比较（旧 0.01 减新方法）：平均 {fmt(estimate["mean"])} LUT；'
                          f'{improved}/10 个种子改善；bootstrap 95% 区间 '
                          f'[{fmt(estimate["bootstrap95"][0])}, {fmt(estimate["bootstrap95"][1])}]；'
                          f'配对 t 区间 [{fmt(estimate["paired_t95"][0])}, '
                          f'{fmt(estimate["paired_t95"][1])}]。', '',
                          f'扩展门槛：{"通过" if gate else "未通过"}。', '']
                uniform_gap = original[circuit, None, 'uniform', None]['expected_gap']
                lines += [f'新方法均值 {fmt(mean(rows, "expected_gap"))}；初始网络 '
                          f'{fmt(mean(initial, "expected_gap"))}；均匀随机 '
                          f'{fmt(uniform_gap)}。这些是独立比较，不与主比较混作一个结论。', '']
                if estimate['bootstrap95'][0] > 0 and estimate['paired_t95'][0] > 0:
                    lines += ['两个探索性区间均支持改善；样本单位仍只有十个训练种子。', '']
                else:
                    lines += ['区间未一致支持改善，方向性结论为不确定。', '']
                lines += ['| 种子 | 旧 0.01 差距 | 新方法差距 | 初始差距 | 旧减新 | 新方法贪心序列 | 贪心差距 |',
                          '|---:|---:|---:|---:|---:|---|---:|']
                for seed, base, candidate, first, delta in zip(SEEDS, baseline, rows, initial, diffs):
                    paired_rows.append(dict(seed=seed, old_gap=base['expected_gap'],
                                            new_gap=candidate['expected_gap'],
                                            initial_gap=first['expected_gap'], old_minus_new=delta,
                                            greedy_actions=candidate['greedy_actions'],
                                            greedy_gap=candidate['greedy_best']['gap']))
                    lines.append(f'| {seed} | {fmt(base["expected_gap"])} | '
                                 f'{fmt(candidate["expected_gap"])} | {fmt(first["expected_gap"])} | '
                                 f'{fmt(delta)} | {candidate["greedy_actions"]} | '
                                 f'{fmt(candidate["greedy_best"]["gap"])} |')
                lines.append('')
            else:
                lines += ['该电路作为探索性扩展，不参与 i2c 扩展门槛。', '']
        diagnostics = []
        for seed in SEEDS:
            path = OUT / 'training/i2c' / f'seed-{seed}/return-diagnostics.jsonl'
            if path.exists():
                diagnostics.extend(row for row in
                                   (json.loads(line) for line in path.read_text().splitlines())
                                   if row['episode'] > 200)
        if diagnostics:
            old_update = []
            for seed in SEEDS:
                path = OUT.parent / 'four-step-lr/training/lr010/i2c' / f'seed-{seed}/metrics.jsonl'
                old_update.extend(row['actor_update_rms'] for row in
                                  (json.loads(line) for line in path.read_text().splitlines())
                                  if row['episode'] > 200)
            lines += ['### 更新幅度诊断', '',
                      f'新方法最后 50 轮平均 |优势|={fmt(mean(diagnostics, "mean_abs_advantage"))}，'
                      f'Actor 更新 RMS={fmt(mean(diagnostics, "actor_update_rms"), 6)}；'
                      f'旧 0.01 的 Actor 更新 RMS={fmt(float(np.mean(old_update)), 6)}。'
                      '旧实验没有逐轮保存 |优势|，因此不作这一项的直接配对比较。', '']
    else:
        lines += ['', '新训练尚未完成；此报告仅包含旧模型的精确复评。', '']
    lines += ['![精确 k 次取最好曲线](best-of-k.svg)', '']
    lines += ['## 复核与局限', '',
              f'旧评估的 {old_exact["replayed_rollouts"]} 条轨迹已逐动作与最好结果核对。'
              '穷举只覆盖固定四步动作空间。精确评估消除了冻结轨迹抽样噪声，'
              '没有消除训练种子之间的不确定性。',
              '训练期间见过的最好 LUT 是相同搜索预算下的辅助指标，'
              '不能替代冻结策略的单次表现。',
              '优于旧 0.01 不自动等于优于初始网络或均匀随机。', '']
    (HERE / 'report.md').write_text('\n'.join(lines))
    write_csv(HERE / 'i2c-paired.csv', paired_rows)
    draw_curves({name: [row for row in rows if row['episode'] in (None, 0, 250)]
                 for name, rows in old_groups.items()}, final_new)
    summary = dict(status='complete', gate=gate, i2c_paired=paired_rows,
                   old_exact_sha256=sha(OUT / 'old-exact.json'),
                   new_exact_sha256=sha(new_path) if new_exact else None,
                   report_sha256=sha(HERE / 'report.md'))
    dump(OUT / 'summary.json', summary)
    print(f'Wrote report; i2c expansion gate = {gate}', flush=True)


if __name__ == '__main__':
    main()

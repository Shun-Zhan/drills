"""Generate paired four-step decisions and the final Chinese evidence report."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from context import (HERE, OUT, SEEDS4, SNAPSHOTS4, SPEC, dump, identity,
                     prior, read, sha)

T9_975 = 2.2621571627409915


def paired_interval(differences):
    values = np.asarray(differences, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError('Expected ten finite paired differences.')
    rng = np.random.default_rng(SPEC['four_step_gate']['bootstrap_seed'])
    count = SPEC['four_step_gate']['bootstrap_draws']
    draws = values[rng.integers(0, 10, size=(count, 10))].mean(axis=1)
    mean = float(values.mean())
    margin = T9_975 * float(values.std(ddof=1)) / math.sqrt(10)
    return dict(mean=mean, bootstrap95=[float(x) for x in np.quantile(draws, (0.025, 0.975))],
                paired_t95=[mean-margin, mean+margin])


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    return '未定义' if value is None or value == '' else f'{float(value):.{digits}f}'


def _chart(path, panels):
    """Small dependency-free SVG for experiment curves."""
    width, height = 1080, 390 * len(panels)
    colors = {'window': '#2563eb', 'reward': '#dc2626', 'frozen': '#64748b',
              'uniform': '#16a34a', 'greedy': '#9333ea'}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172033;font-size:14px}</style>']
    for panel_index, panel in enumerate(panels):
        top = panel_index * 390
        x0, x1, y0, y1 = 95, 1020, top + 55, top + 300
        series = panel['series']
        valid = [v for item in series.values() for v in item if v is not None]
        if not valid:
            continue
        low, high = min(valid), max(valid)
        if high == low:
            high += 1
        pad = (high - low) * 0.08
        low -= pad
        high += pad
        parts += [f'<text x="{x0}" y="{top+27}" font-size="18">{panel["title"]}</text>',
                  f'<path d="M{x0} {y0}V{y1}H{x1}" fill="none" stroke="#667085"/>',
                  f'<text x="{x0-10}" y="{y0+5}" text-anchor="end">{high:.3f}</text>',
                  f'<text x="{x0-10}" y="{y1+5}" text-anchor="end">{low:.3f}</text>',
                  f'<text x="{x0}" y="{y1+22}">{panel["x_label_start"]}</text>',
                  f'<text x="{x1}" y="{y1+22}" text-anchor="end">{panel["x_label_end"]}</text>']
        for name, values in series.items():
            points = []
            for i, value in enumerate(values):
                if value is None:
                    if len(points) >= 2:
                        parts.append(f'<polyline points="{" ".join(points)}" fill="none" '
                                     f'stroke="{colors[name]}" stroke-width="3"/>')
                    points = []
                    continue
                x = x0 + i / max(len(values)-1, 1) * (x1-x0)
                y = y1 - (value-low)/(high-low)*(y1-y0)
                points.append(f'{x:.1f},{y:.1f}')
            if len(points) >= 2:
                parts.append(f'<polyline points="{" ".join(points)}" fill="none" '
                             f'stroke="{colors[name]}" stroke-width="3"/>')
        for j, name in enumerate(series):
            x = x0 + j * 220
            parts.append(f'<line x1="{x}" x2="{x+28}" y1="{top+350}" y2="{top+350}" '
                         f'stroke="{colors[name]}" stroke-width="3"/>')
            parts.append(f'<text x="{x+35}" y="{top+355}">{name}</text>')
    parts.append('</svg>')
    Path(path).write_text('\n'.join(parts) + '\n')


def analyze():
    digest, evidence = identity()
    exact = read(OUT / 'exact.json')
    if exact['status'] != 'complete' or exact['fingerprint'] != digest:
        raise ValueError('Four-step exact evaluation is incomplete or stale.')
    audit = read(HERE / 'audit.json')
    pilot = read(OUT / 'pilot.json')
    parity = read(OUT / 'equivalence.json')
    old = read(prior.OUT / 'old-exact.json')
    current = {(r['circuit'], r['seed'], r['policy'], r['episode']): r for r in exact['rows']}
    reference = {(r['circuit'], r['seed'], r['policy'], r['episode']): r for r in old['rows']}
    comparisons, paired_rows, snapshot_rows, training_rows = {}, [], [], []
    charts = []
    lines = ['# 四步奖励实验与条件性十步验证', '',
             '## 实验条件与前置核验', '',
             '在固定四步动作空间中仅更换奖励；滑动窗口长度 8、学习率 0.01、'
             '训练种子 0–9、每种子 250×4 与原窗口实验一致。i2c、max 分别设主比较，'
             'int2float 为次要观察。最优 LUT：int2float 44、i2c 312、max 781。', '',
             f'- 查表环境：{"10/10 个 i2c 最终网络哈希一致" if parity["table_equivalent"] else "未通过哈希等价，正式训练回退 ABC"}。',
             f'- 20 轮预检：i2c 窗口阶段 mean|A|={fmt(next(r for r in pilot["rows"] if r["circuit"] == "i2c")["mean_abs_advantage"])}；'
             f'参考值 0.434；状态 {pilot["status"]}。',
             f'- 候选奖励审计：{audit["status"]}；γ=0.99 下三电路最高分序列全部最优。', '',
             '| 电路 | 原奖励并列最高 | 其中最优 | 新奖励并列最高 | 其中最优 | 新奖励第一、二档分差 |',
             '|---|---:|---:|---:|---:|---:|']
    for circuit in ('int2float', 'i2c', 'max'):
        a = next(r for r in audit['rows'] if r['circuit'] == circuit and
                 r['gamma'] == '99/100' and r['reward'] == 'original')
        b = next(r for r in audit['rows'] if r['circuit'] == circuit and
                 r['gamma'] == '99/100' and r['reward'] == 'candidate')
        lines.append(f'| {circuit} | {a["tied_top"]} | {a["optimal_top"]} | '
                     f'{b["tied_top"]} | {b["optimal_top"]} | {b["distinct_score_gap"]:.6f} |')
    lines += ['', '## 四步精确复评', '',
              '每个冻结模型遍历全部 2,401 条四步动作序列；随机基线是理论概率，'
              '训练种子而非动作路径是配对比较的独立单位。正的“改善”表示新奖励更好。', '',
              '| 电路及主指标 | 窗口版均值 | 新奖励均值 | 平均配对改善 | 改善种子 | 探索性 bootstrap 95% 区间 | 门槛 |',
              '|---|---:|---:|---:|---:|---|---|']
    for circuit in ('i2c', 'max', 'int2float'):
        metric = 'feasible_probability' if circuit == 'max' else 'expected_gap'
        window_rows = [current[circuit, seed, 'window', 250] for seed in SEEDS4]
        reward_rows = [current[circuit, seed, 'reward', 250] for seed in SEEDS4]
        initial_rows = [reference[circuit, seed, 'lr001', 0] for seed in SEEDS4]
        random_row = reference[circuit, None, 'uniform', None]
        diffs = [(r[metric]-w[metric] if circuit == 'max' else w[metric]-r[metric])
                 for w, r in zip(window_rows, reward_rows)]
        estimate = paired_interval(diffs)
        improved = sum(x > 0 for x in diffs)
        window_mean = float(np.mean([r[metric] for r in window_rows]))
        reward_mean = float(np.mean([r[metric] for r in reward_rows]))
        initial_mean = float(np.mean([r[metric] for r in initial_rows]))
        passed = (estimate['mean'] > 0 and improved >= SPEC['four_step_gate']['improved_seeds']
                  and (circuit != 'max' or reward_mean > initial_mean)) if circuit != 'int2float' else None
        comparisons[circuit] = dict(metric=metric, window_mean=window_mean,
                                    reward_mean=reward_mean, initial_mean=initial_mean,
                                    uniform_theory=random_row[metric],
                                    uniform_optimal_theory=random_row['optimal_probability'],
                                    improved_seeds=improved, interval=estimate,
                                    passed=passed)
        digits = 6 if circuit == 'max' else 3
        lines.append(f'| {circuit}（{"单次可行率" if circuit == "max" else "单次期望差距"}） | '
                     f'{fmt(window_mean,digits)} | {fmt(reward_mean,digits)} | '
                     f'{fmt(estimate["mean"],digits)} | {improved}/10 | '
                     f'[{fmt(estimate["bootstrap95"][0],digits)}, '
                     f'{fmt(estimate["bootstrap95"][1],digits)}] | '
                     f'{"次要观察" if passed is None else "通过" if passed else "未通过"} |')
        for seed, w, r, first, diff in zip(SEEDS4, window_rows, reward_rows, initial_rows, diffs):
            paired_rows.append(dict(circuit=circuit, seed=seed, metric=metric,
                                    window=w[metric], reward=r[metric], initial=first[metric],
                                    improvement=diff,
                                    reward_optimal_probability=r['optimal_probability'],
                                    window_optimal_probability=w['optimal_probability'],
                                    reward_greedy_actions=json.dumps(r['greedy_actions']),
                                    reward_greedy_best_luts=r['greedy_best']['luts'],
                                    reward_greedy_feasible=r['greedy_best']['feasible']))
        for episode in SNAPSHOTS4:
            record = dict(circuit=circuit, episode=episode, metric=metric)
            for group in ('window', 'reward'):
                record[group] = float(np.mean([current[circuit, seed, group, episode][metric]
                                               for seed in SEEDS4]))
            snapshot_rows.append(record)
        charts.append(dict(title=f'{circuit}: {metric}',
                           series={group: [row[group] for row in snapshot_rows
                                           if row['circuit'] == circuit] for group in ('window', 'reward')},
                           x_label_start='第 0 轮', x_label_end='第 250 轮'))
        for group in ('window', 'reward'):
            for seed in SEEDS4:
                path = (prior.OUT / 'training' / circuit / f'seed-{seed}' / 'result.json'
                        if group == 'window' and circuit == 'i2c' else
                        OUT / 'four-step' / group / circuit / f'seed-{seed}' / 'result.json')
                result = read(path)
                training_rows.append(dict(circuit=circuit, group=group, seed=seed,
                                          best_luts=result['best']['luts'],
                                          best_levels=result['best']['levels'],
                                          best_feasible=result['best']['feasible']))
    gates = {name: dict(passed=comparisons[name]['passed'],
                        improved_seeds=comparisons[name]['improved_seeds'],
                        metric=comparisons[name]['metric']) for name in ('i2c', 'max')}
    dump(OUT / 'gates.json', dict(status='complete', fingerprint=digest, gates=gates))
    lines += ['', 'max 的门槛另外要求新奖励平均可行率高于初始网络；'
              '以上门槛用于决定十步扩展，不作统计显著性判断。', '',
              '### 理论基线、贪心与训练搜索', '',
              '| 电路 | 初始网络主指标 | 均匀随机主指标 | 均匀随机最优命中概率 | 新奖励贪心最优种子 | 新奖励训练期最好 LUT 均值 |',
              '|---|---:|---:|---:|---:|---:|']
    for circuit in ('i2c', 'max', 'int2float'):
        info = comparisons[circuit]
        final = [current[circuit, seed, 'reward', 250] for seed in SEEDS4]
        optimal = prior.old.ASSESSMENT['optimum_luts'][circuit]
        greedy_wins = sum(row['greedy_best']['feasible'] and row['greedy_best']['luts'] == optimal
                          for row in final)
        best = [row['best_luts'] for row in training_rows
                if row['circuit'] == circuit and row['group'] == 'reward' and row['best_feasible']]
        lines.append(f'| {circuit} | {fmt(info["initial_mean"], 6 if circuit == "max" else 3)} | '
                     f'{fmt(info["uniform_theory"], 6 if circuit == "max" else 3)} | '
                     f'{fmt(info["uniform_optimal_theory"],6)} | {greedy_wins}/10 | '
                     f'{fmt(np.mean(best) if len(best) == 10 else None)} |')
    lines += ['', '逐种子单次指标、最优命中概率及贪心序列见 paired.csv；'
              '训练期间累计最好值仅是辅助搜索指标。', '',
              '### 第 0/50/100/150/200/250 轮快照', '',
              '| 电路 | 策略 | 0 | 50 | 100 | 150 | 200 | 250 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for circuit in ('i2c', 'max', 'int2float'):
        for group in ('window', 'reward'):
            records = [row for row in snapshot_rows if row['circuit'] == circuit]
            digits = 6 if circuit == 'max' else 3
            lines.append(f'| {circuit} | {group} | ' +
                         ' | '.join(fmt(row[group], digits) for row in records) + ' |')
    lines += ['', '![四步精确快照曲线](snapshots.svg)', '']
    write_csv(HERE / 'paired.csv', paired_rows)
    write_csv(HERE / 'snapshots.csv', snapshot_rows)
    write_csv(HERE / 'training-best.csv', training_rows)
    _chart(HERE / 'snapshots.svg', charts)
    ten_path = OUT / 'ten-summary.json'
    ten = read(ten_path) if ten_path.exists() else None
    lines += ['## 条件性十步验证', '']
    if ten is None:
        lines += ['四步门槛已判定；十步阶段尚未完成。', '']
    elif ten['status'] == 'not_triggered':
        lines += ['i2c 与 max 均未通过各自的四步门槛，按预定规则不启动十步训练。', '']
    elif ten['status'] == 'complete':
        lines += ['十步使用真实 ABC，每种子 100×10 动作、1,100 次映射；'
                  '已核验的旧冻结及均匀随机轨迹才复用。以下只陈述观测到的可行性和最佳 LUT，'
                  '不推断十步全局最优。', '',
                  '| 电路 | 种子 | 新奖励最好 LUT | 冻结最好 LUT | 随机最好 LUT | 贪心最好 LUT | 新奖励胜过两基线 |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for circuit in ten['circuits']:
            for seed in SPEC['ten_step']['seeds']:
                rows = {row['group']: row for row in ten['rows']
                        if row['circuit'] == circuit and row['seed'] == seed}
                cand = rows['reward']
                win = bool(cand['best_feasible'] and rows['frozen']['best_feasible'] and
                           rows['uniform']['best_feasible'] and
                           cand['best_luts'] < rows['frozen']['best_luts'] and
                           cand['best_luts'] < rows['uniform']['best_luts'])
                lines.append(f'| {circuit} | {seed} | {cand["best_luts"]} | '
                             f'{rows["frozen"]["best_luts"]} | {rows["uniform"]["best_luts"]} | '
                             f'{rows["greedy"]["best_luts"]} | {"是" if win else "否"} |')
            verdict = ten['verdicts'][circuit]
            lines += ['', f'{circuit} 十步 5/5 门槛：{verdict["seed_wins"]}/5，'
                      f'{"通过" if verdict["passed"] else "未通过"}。', '']
        lines += ['逐种子可行回合率和等调用预算曲线见 ten-results.csv、ten-curves.csv。',
                  '![十步等调用预算搜索曲线](ten-curves.svg)', '']
        curve_rows = list(csv.DictReader((HERE / 'ten-curves.csv').open(newline='')))
        panels = []
        for circuit in ten['circuits']:
            series = {}
            for group in ('reward', 'frozen', 'uniform', 'greedy'):
                values = []
                for episode in range(1, 101):
                    rows = [row for row in curve_rows if row['circuit'] == circuit and
                            row['group'] == group and int(row['calls']) == episode*11]
                    numeric = [float(row['best_luts']) for row in rows if row['best_luts']]
                    values.append(float(np.mean(numeric)) if len(numeric) == 5 else None)
                series[group] = values
            panels.append(dict(title=f'{circuit}: 同等映射调用预算下的最佳可行 LUT',
                               series=series, x_label_start='11 次', x_label_end='1,100 次'))
        _chart(HERE / 'ten-curves.svg', panels)
    lines += ['## 解释与复现边界', '',
              '四步精确评估消除了冻结策略抽样噪声，仍只有十个独立训练种子；'
              '区间和 6/10 门槛不构成显著性声明。max 不可行路径不赋任意 LUT 罚分。'
              '查表只用于四步，十步结果来自真实 ABC 调用。原始模型、逐步日志和完整状态'
              '位于本机 Git 忽略的 results/four-step-reward/；Git 保存协议、代码、精简表、图及哈希清单。',
              '源代码、工具、电路、穷举表、特征缓存的 SHA-256 和运行状态见本实验的'
              '证据清单与本机运行 manifest。', '']
    (HERE / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    summary = dict(status='complete' if ten is not None else 'four_step_complete',
                   fingerprint=digest, four_step=comparisons, gates=gates,
                   ten_step=ten['status'] if ten else 'pending')
    dump(HERE / 'summary.json', summary)
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name not in ('artifact-manifest.json',))
    dump(HERE / 'artifact-manifest.json', dict(fingerprint=digest,
        artifacts={p.name: sha(p) for p in files}, source_payload=evidence))
    return summary


if __name__ == '__main__':
    result = analyze()
    print('Four-step gates:', result['gates'])

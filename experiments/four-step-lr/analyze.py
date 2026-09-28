"""Rebuild tables, paired estimates, chart, and Chinese report from recorded evidence."""
import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

from common import (ASSESSMENT, GROUPS, HERE, POLICIES, config, dump,
                    evaluation_seeds, identity, now, read, sha)

METRICS = ('actor_weight_variance', 'critic_weight_variance', 'entropy')
LABELS = {'actor_weight_variance': 'Actor 权重方差',
          'critic_weight_variance': 'Critic 权重方差', 'entropy': '策略熵'}


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'No rows for {path}')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def bootstrap_ci(differences, repetitions=10000, seed=20260928):
    values = np.asarray(differences, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        return None
    if np.ptp(values) == 0:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def fmt(value, digits=3):
    return '未定义' if value is None else f'{value:.{digits}f}'


def pair_summary(rows, left, right, metric):
    diffs = [b[metric] - a[metric] for a, b in zip(left, right)]
    if any(value is None for value in diffs):
        return dict(mean=None, ci95=None)
    return dict(mean=float(np.mean(diffs)), ci95=bootstrap_ci(diffs))


def chart(metric_means):
    width, height = 1350, 870
    panel_w, panel_h = 405, 245
    left, top = 80, 70
    colors = {'lr001': '#2563eb', 'lr010': '#dc2626'}
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font-family:Arial, sans-serif;fill:#172033} .axis{stroke:#9aa4b2;stroke-width:1} '
              '.grid{stroke:#e5e7eb;stroke-width:1} .line{fill:none;stroke-width:2.5}</style>',
              '<text x="80" y="35" font-size="21" font-weight="bold">四步学习率实验：权重方差与策略熵</text>',
              '<line x1="925" y1="29" x2="955" y2="29" stroke="#2563eb" stroke-width="3"/>',
              '<text x="963" y="34" font-size="14">学习率 0.001</text>',
              '<line x1="1120" y1="29" x2="1150" y2="29" stroke="#dc2626" stroke-width="3"/>',
              '<text x="1158" y="34" font-size="14">学习率 0.01</text>']
    for ci, circuit in enumerate(('int2float', 'i2c', 'max')):
        for mi, metric in enumerate(METRICS):
            x0, y0 = left + mi * 430, top + ci * 265
            series = {group: [r for r in metric_means if r['circuit'] == circuit and r['group'] == group]
                      for group in colors}
            values = [r[metric] for rows in series.values() for r in rows]
            low, high = min(values), max(values)
            padding = max((high - low) * 0.12, 0.0005)
            low -= padding
            high += padding
            plot_x0, plot_x1 = x0 + 54, x0 + panel_w - 15
            plot_y0, plot_y1 = y0 + 28, y0 + panel_h - 30
            pieces.append(f'<text x="{x0 + 4}" y="{y0 + 15}" font-size="15" font-weight="bold">'
                          f'{html.escape(circuit)} · {html.escape(LABELS[metric])}</text>')
            for tick in range(3):
                y = plot_y1 - tick * (plot_y1 - plot_y0) / 2
                value = low + tick * (high - low) / 2
                pieces.append(f'<line class="grid" x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}"/>')
                pieces.append(f'<text x="{plot_x0 - 5}" y="{y + 4:.1f}" text-anchor="end" font-size="11">'
                              f'{value:.4f}</text>')
            pieces.append(f'<line class="axis" x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}"/>')
            for tick in (0, 125, 250):
                x = plot_x0 + tick / 250 * (plot_x1 - plot_x0)
                pieces.append(f'<text x="{x:.1f}" y="{plot_y1 + 17}" text-anchor="middle" font-size="11">{tick}</text>')
            for group, rows in series.items():
                rows.sort(key=lambda row: row['episode'])
                points = ' '.join(f'{plot_x0 + row["episode"] / 250 * (plot_x1 - plot_x0):.1f},'
                                  f'{plot_y1 - (row[metric] - low) / (high - low) * (plot_y1 - plot_y0):.1f}'
                                  for row in rows)
                pieces.append(f'<polyline class="line" stroke="{colors[group]}" points="{points}"/>')
    pieces.append('</svg>')
    (HERE / 'diagnostics.svg').write_text('\n'.join(pieces))


def generate():
    cfg = config()
    fingerprint, _ = identity(cfg)
    root = Path(cfg['runtime']['output_dir'])
    training_manifest = read(root / 'experiment.json')
    evaluation_manifest = read(root / 'evaluation.json')
    if any(row['status'] != 'complete' for row in (training_manifest, evaluation_manifest)) or \
            training_manifest['fingerprint'] != fingerprint or evaluation_manifest['training_fingerprint'] != fingerprint:
        raise ValueError('Both phases must be complete under the current source fingerprint.')
    training, evaluations, diagnostics = [], [], []
    for circuit in cfg['protocol']['circuits']:
        target = ASSESSMENT['optimum_luts'][circuit]
        for seed in cfg['protocol']['seeds']:
            for group in GROUPS:
                folder = root / 'training' / group / circuit / f'seed-{seed}'
                value = read(folder / 'result.json')
                best = value['best']
                feasible = bool(best['feasible'])
                training.append(dict(circuit=circuit, seed=seed, group=group,
                                     status=value['status'], episodes=value['episodes_completed'],
                                     best_luts=best['luts'], best_levels=best['levels'], feasible=feasible,
                                     gap=best['luts'] - target if feasible else None,
                                     hit=feasible and best['luts'] == target,
                                     checkpoint_sha256=value['checkpoint_sha256']))
                if group != 'uniform':
                    lines = (folder / 'metrics.jsonl').read_text().splitlines()
                    if len(lines) != 251:
                        raise ValueError(f'Missing per-episode metrics: {folder}')
                    for episode, line in enumerate(lines):
                        metric = json.loads(line)
                        if metric['episode'] != episode:
                            raise ValueError(f'Misaligned metric: {folder}/{episode}')
                        diagnostics.append(dict(circuit=circuit, seed=seed, group=group, **metric))
            for policy in POLICIES:
                folder = root / 'evaluation' / policy / circuit / f'seed-{seed}'
                result = read(folder / 'result.json')
                if result['status'] != 'complete' or len(result['rollouts']) != 10:
                    raise ValueError(f'Incomplete independent policy bank: {folder}')
                for expected_seed, rollout in zip(evaluation_seeds(seed), result['rollouts']):
                    if rollout['evaluation_seed'] != expected_seed:
                        raise ValueError(f'Evaluation seed mismatch: {folder}')
                    best = rollout['best']
                    feasible = bool(best['feasible'])
                    evaluations.append(dict(circuit=circuit, seed=seed, policy=policy,
                                            evaluation_seed=expected_seed, best_luts=best['luts'],
                                            best_levels=best['levels'], feasible=feasible,
                                            gap=best['luts'] - target if feasible else None,
                                            hit=feasible and best['luts'] == target,
                                            checkpoint_sha256=rollout['checkpoint_sha256']))
    training_summary, evaluation_summary, per_seed, comparisons = [], [], [], []
    for circuit in cfg['protocol']['circuits']:
        by_training = {(r['group'], r['seed']): r for r in training if r['circuit'] == circuit}
        for group in GROUPS:
            bank = [by_training[(group, seed)] for seed in cfg['protocol']['seeds']]
            gaps = [r['gap'] for r in bank]
            training_summary.append(dict(circuit=circuit, group=group, feasible=sum(r['feasible'] for r in bank),
                                         hits=sum(r['hit'] for r in bank),
                                         mean_gap=float(np.mean(gaps)) if all(v is not None for v in gaps) else None,
                                         complete_gap=all(v is not None for v in gaps)))
        for group in ('lr001', 'lr010'):
            left = [by_training[('uniform', seed)] for seed in cfg['protocol']['seeds']]
            right = [by_training[(group, seed)] for seed in cfg['protocol']['seeds']]
            valid = all(a['gap'] is not None and b['gap'] is not None for a, b in zip(left, right))
            comparisons.append(dict(circuit=circuit, group=group, baseline='uniform',
                                    pass_rule=(float(np.mean([b['gap'] for b in right])) <
                                               float(np.mean([a['gap'] for a in left])) and
                                               sum(b['gap'] <= a['gap'] for a, b in zip(left, right)) >= 8)
                                    if valid else None,
                                    nonworse=sum(b['gap'] <= a['gap'] for a, b in zip(left, right)) if valid else None))
        for seed in cfg['protocol']['seeds']:
            row = dict(circuit=circuit, seed=seed)
            for policy in POLICIES:
                bank = [r for r in evaluations if (r['circuit'], r['seed'], r['policy']) == (circuit, seed, policy)]
                feasible = [r for r in bank if r['feasible']]
                row[policy + '_feasible'] = len(feasible)
                row[policy + '_hits'] = sum(r['hit'] for r in bank)
                row[policy + '_best_luts'] = min((r['best_luts'] for r in feasible), default=None)
                row[policy + '_best_gap'] = min((r['gap'] for r in feasible), default=None)
            per_seed.append(row)
        for policy in POLICIES:
            bank = [r for r in evaluations if r['circuit'] == circuit and r['policy'] == policy]
            seed_bank = [r for r in per_seed if r['circuit'] == circuit]
            best_gaps = [r[policy + '_best_gap'] for r in seed_bank]
            evaluation_summary.append(dict(circuit=circuit, policy=policy, attempts=len(bank),
                                           feasible=sum(r['feasible'] for r in bank), hits=sum(r['hit'] for r in bank),
                                           seeds_with_feasible=sum(v is not None for v in best_gaps),
                                           mean_seed_best_gap=float(np.mean(best_gaps))
                                           if all(v is not None for v in best_gaps) else None))
    metric_means, metric_pairs, final_metric_summary = [], [], []
    metric_lookup = {(r['circuit'], r['group'], r['seed'], r['episode']): r for r in diagnostics}
    for circuit in cfg['protocol']['circuits']:
        for episode in range(251):
            for group in ('lr001', 'lr010'):
                bank = [metric_lookup[(circuit, group, seed, episode)] for seed in cfg['protocol']['seeds']]
                metric_means.append(dict(circuit=circuit, group=group, episode=episode,
                                         **{metric: float(np.mean([row[metric] for row in bank])) for metric in METRICS}))
            for seed in cfg['protocol']['seeds']:
                low = metric_lookup[(circuit, 'lr001', seed, episode)]
                high = metric_lookup[(circuit, 'lr010', seed, episode)]
                metric_pairs.append(dict(circuit=circuit, seed=seed, episode=episode,
                                         **{metric + '_delta': high[metric] - low[metric] for metric in METRICS}))
        for metric in METRICS:
            low = [metric_lookup[(circuit, 'lr001', seed, 250)][metric] for seed in cfg['protocol']['seeds']]
            high = [metric_lookup[(circuit, 'lr010', seed, 250)][metric] for seed in cfg['protocol']['seeds']]
            initial = [metric_lookup[(circuit, 'lr001', seed, 0)][metric] for seed in cfg['protocol']['seeds']]
            differences = [b - a for a, b in zip(low, high)]
            final_metric_summary.append(dict(circuit=circuit, metric=metric,
                                             initial_mean=float(np.mean(initial)), lr001_mean=float(np.mean(low)),
                                             lr010_mean=float(np.mean(high)), paired_delta_mean=float(np.mean(differences)),
                                             paired_delta_ci95=bootstrap_ci(differences)))
    write_csv(HERE / 'training.csv', training)
    write_csv(HERE / 'evaluation.csv', evaluations)
    write_csv(HERE / 'evaluation-by-seed.csv', per_seed)
    write_csv(HERE / 'diagnostics.csv', diagnostics)
    write_csv(HERE / 'diagnostic-means.csv', metric_means)
    write_csv(HERE / 'diagnostic-pairs.csv', metric_pairs)
    chart(metric_means)
    summary = dict(status='complete', fingerprint=fingerprint, generated_at=now(),
                   training=training_summary, training_rule=comparisons,
                   evaluation=evaluation_summary, evaluation_by_seed=per_seed,
                   final_diagnostics=final_metric_summary,
                   bootstrap=dict(unit='ten independent training seeds', repetitions=10000, seed=20260928))
    dump(HERE / 'summary.json', summary)
    lines = ['# 四步环境学习率对照与独立策略评估', '',
             '## 实验条件', '',
             'int2float、i2c、max；训练种子0–9；每组250轮×4动作；原A2C每轮更新一次。'
             '学习率分别为0.001、0.01；均匀随机组没有权重更新。其余网络、奖励、归一化、动作、'
             'LUT6和层数限制与原参数一致。四步最优可行LUT依次为44、312、781。', '',
             '每个最终模型、对应初始权重和独立随机组各在10个新随机种子上执行完整四步序列。'
             '同一电路和训练种子内四组共用评估种子；不同训练种子有各自的随机组评估bank。'
             '每条轨迹从原电路重新开始，策略冻结，初始映射可成为最佳候选。', '',
             '## 训练期间搜索（250轮，含每轮初始映射）', '',
             '| 电路 | 组别 | 可行种子/10 | 命中四步最优/10 | 平均LUT差距 |',
             '|---|---|---:|---:|---:|']
    for row in training_summary:
        lines.append(f'| {row["circuit"]} | {row["group"]} | {row["feasible"]}/10 | {row["hits"]}/10 | '
                     f'{fmt(row["mean_gap"])} |')
    lines += ['', '| 电路 | A2C学习率 | 相对随机组符合预定标准 | 不更差的种子/10 |',
              '|---|---:|---|---:|']
    for row in comparisons:
        verdict = '无法判定（有种子未找到可行解）' if row['pass_rule'] is None else ('是' if row['pass_rule'] else '否')
        lines.append(f'| {row["circuit"]} | {row["group"]} | {verdict} | '
                     f'{row["nonworse"] if row["nonworse"] is not None else "未定义"} |')
    lines += ['', '这里的累计最好值衡量相同搜索预算下找到的解；不单独证明冻结后的策略有所改善。', '',
              '## 训练后独立策略评估（每模型10次）', '',
              '| 电路 | 策略 | 可行尝试/100 | 最优命中/100 | 有可行解种子/10 | 种子最佳LUT平均差距 |',
              '|---|---|---:|---:|---:|---:|']
    for row in evaluation_summary:
        lines.append(f'| {row["circuit"]} | {row["policy"]} | {row["feasible"]}/100 | '
                     f'{row["hits"]}/100 | {row["seeds_with_feasible"]}/10 | '
                     f'{fmt(row["mean_seed_best_gap"])} |')
    lines += ['', '十次尝试中若没有可行解，该种子的最佳LUT差距未定义；汇总均值仅在全部十个种子'
              '都有可行解时给出。尤其是max，十次四步评估对可行率和最优命中的分辨力很低。', '',
              '### 逐种子独立评估', '',
              '| 电路 | seed | 0.001最终 可行/命中/最佳LUT | 0.01最终 可行/命中/最佳LUT | '
              '初始权重 可行/命中/最佳LUT | 均匀随机 可行/命中/最佳LUT |',
              '|---|---:|---|---|---|---|']
    for row in per_seed:
        def cell(policy):
            best = row[policy + '_best_luts']
            return f'{row[policy + "_feasible"]}/10 / {row[policy + "_hits"]}/10 / {best if best is not None else "无可行解"}'
        lines.append(f'| {row["circuit"]} | {row["seed"]} | '
                     f'{cell("lr001-final")} | {cell("lr010-final")} | '
                     f'{cell("initial")} | {cell("uniform")} |')
    lines += ['', '## 权重方差和策略熵', '',
              '权重方差为各Linear权重矩阵元素的总体方差，Actor和Critic分开计算；不含偏置。'
              '熵为同一电路固定状态探针上七动作分布的平均熵，均匀策略的理论值为ln(7)≈1.946。'
              '下表区间是按十个配对训练种子重采样的探索性95%区间。', '',
              '| 电路 | 指标 | 初始均值 | 0.001最终 | 0.01最终 | 0.01−0.001配对差 | 探索性95%区间 |',
              '|---|---|---:|---:|---:|---:|---|']
    for row in final_metric_summary:
        ci = row['paired_delta_ci95']
        lines.append(f'| {row["circuit"]} | {LABELS[row["metric"]]} | {fmt(row["initial_mean"], 6)} | '
                     f'{fmt(row["lr001_mean"], 6)} | {fmt(row["lr010_mean"], 6)} | '
                     f'{fmt(row["paired_delta_mean"], 6)} | [{fmt(ci[0], 6)}, {fmt(ci[1], 6)}] |')
    lines += ['', '![逐轮权重方差与策略熵](diagnostics.svg)', '',
              '参数方差、漂移和熵的变化描述训练动态，不能单独判定策略质量。'
              '两档学习率的每轮明细见diagnostics.csv，逐种子配对差见diagnostic-pairs.csv。', '',
              '## 复核与局限', '',
              '训练、评估和工具指纹见results/four-step-lr/中的experiment.json及evaluation.json；'
              '逐条评估见evaluation.csv。每次独立评估的模型与检查点在评估前后均核对哈希；'
              '最佳映射网表通过ABC组合等价性检查。测试与完整性复核结果见validation.json。', '',
              '每模型仅10条独立轨迹，因此独立评估属于小样本观察；不从不可行轨迹筛选LUT后'
              '声称平均收益。当前结论仅覆盖这三个电路、固定四步环境及本次训练预算。', '',
              '本实验使用AI辅助编写实验代码、检查统计口径和起草报告；全部数字由保存的运行记录生成。', '']
    (HERE / 'report.md').write_text('\n'.join(lines))
    dump(HERE / 'analysis-provenance.json', dict(fingerprint=fingerprint, generated_at=now(),
         source_sha256=sha(__file__), inputs={'training_manifest': sha(root / 'experiment.json'),
                                              'evaluation_manifest': sha(root / 'evaluation.json')},
         outputs={path.name: sha(path) for path in (HERE / 'training.csv', HERE / 'evaluation.csv',
             HERE / 'evaluation-by-seed.csv', HERE / 'diagnostics.csv', HERE / 'diagnostic-means.csv',
             HERE / 'diagnostic-pairs.csv', HERE / 'diagnostics.svg', HERE / 'summary.json', HERE / 'report.md')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    generate()


if __name__ == '__main__':
    main()

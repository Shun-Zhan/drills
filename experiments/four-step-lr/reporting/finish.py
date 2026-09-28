"""Generate the report, including intervals for ten independent attempts.

This read-only reporting layer sits outside the frozen training-source fingerprint.
It can be rerun after all training and evaluation records have been saved.
"""
import csv
from math import sqrt
from pathlib import Path
import sys

import numpy as np

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE))
import analyze  # noqa: E402
from common import dump, now, read, sha  # noqa: E402

Z95 = 1.959963984540054
POLICIES = ('lr001-final', 'lr010-final', 'initial', 'uniform')


def wilson(count, total=10):
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    if not 0 <= count <= total:
        raise ValueError((count, total))
    p = count / total
    denominator = 1 + Z95 ** 2 / total
    center = (p + Z95 ** 2 / (2 * total)) / denominator
    half_width = Z95 * sqrt(p * (1 - p) / total + Z95 ** 2 / (4 * total ** 2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    analyze.generate()
    summary = read(SUITE / 'summary.json')
    banks = summary['evaluation_by_seed']
    with (SUITE.parent / 'ten-step-optimality/candidates.csv').open() as stream:
        max_four_step = [row for row in csv.DictReader(stream)
                         if row['circuit'] == 'max' and row['depth'] == '4']
    feasible_sequences = sum(row['feasible'] == 'True' for row in max_four_step)
    if (len(max_four_step), feasible_sequences) != (2401, 26):
        raise ValueError('The enumerated max four-step baseline changed.')
    random_any_feasible = 1 - (1 - feasible_sequences / len(max_four_step)) ** 10
    intervals = []
    for bank in banks:
        for policy in POLICIES:
            feasible = bank[policy + '_feasible']
            hits = bank[policy + '_hits']
            feasible_ci = wilson(feasible)
            hit_ci = wilson(hits)
            intervals.append(dict(circuit=bank['circuit'], seed=bank['seed'], policy=policy,
                                  attempts=10, feasible=feasible, feasible_ci_low=feasible_ci[0],
                                  feasible_ci_high=feasible_ci[1], optimum_hits=hits,
                                  optimum_ci_low=hit_ci[0], optimum_ci_high=hit_ci[1]))
    write_csv(SUITE / 'evaluation-intervals.csv', intervals)

    comparisons = (('lr001-final', 'initial'), ('lr001-final', 'uniform'),
                   ('lr010-final', 'initial'), ('lr010-final', 'uniform'),
                   ('lr010-final', 'lr001-final'))
    pairs = []
    rng = np.random.default_rng(20260928)
    for circuit in ('int2float', 'i2c', 'max'):
        subset = sorted((row for row in banks if row['circuit'] == circuit), key=lambda row: row['seed'])
        if len(subset) != 10:
            raise ValueError(f'Expected ten training seeds for {circuit}')
        for policy, comparator in comparisons:
            for metric in ('feasible', 'hits'):
                differences = np.asarray([(row[policy + '_' + metric] - row[comparator + '_' + metric]) / 10
                                          for row in subset], dtype=float)
                draws = differences[rng.integers(0, 10, size=(10000, 10))].mean(axis=1)
                low, high = np.quantile(draws, (0.025, 0.975))
                pairs.append(dict(circuit=circuit, policy=policy, comparator=comparator,
                                  metric=metric, mean_difference=float(differences.mean()),
                                  bootstrap_ci_low=float(low), bootstrap_ci_high=float(high),
                                  unit='training_seed', seeds=10, attempts_per_seed=10))
    write_csv(SUITE / 'evaluation-pairs.csv', pairs)

    report_path = SUITE / 'report.md'
    report = report_path.read_text()
    training_by = {(row['circuit'], row['group']): row for row in summary['training']}
    evaluation_by = {(row['circuit'], row['policy']): row for row in summary['evaluation']}
    diagnostics_by = {(row['circuit'], row['metric']): row for row in summary['final_diagnostics']}
    variance_increased = all(diagnostics_by[circuit, metric]['paired_delta_mean'] > 0
                             for circuit in ('int2float', 'i2c', 'max')
                             for metric in ('actor_weight_variance', 'critic_weight_variance'))
    entropy_decreased = all(diagnostics_by[circuit, 'entropy']['paired_delta_mean'] < 0
                            for circuit in ('int2float', 'i2c', 'max'))
    observations = [
        '## 主要观察', '',
        ('三个电路中，0.01 相对 0.001 的最终 Actor/Critic 权重方差配对均值均上升，'
         '固定探针策略熵均下降。' if variance_increased and entropy_decreased else
         '两档学习率的最终权重方差和固定探针策略熵见下方逐电路配对数据。') +
        '这些参数变化本身不能证明策略表现改善。', '',
        f'int2float 的训练搜索平均 LUT 差距在 0.001 和 0.01 下分别为 '
        f'{training_by["int2float", "lr001"]["mean_gap"]:.1f} 与 '
        f'{training_by["int2float", "lr010"]["mean_gap"]:.1f}；独立评估的'
        f'种子最佳 LUT 平均差距分别为 '
        f'{evaluation_by["int2float", "lr001-final"]["mean_seed_best_gap"]:.1f} 与 '
        f'{evaluation_by["int2float", "lr010-final"]["mean_seed_best_gap"]:.1f}，'
        '两档最终策略都没有命中四步最优。', '',
        f'训练搜索中，i2c 的平均最优 LUT 差距从 '
        f'{training_by["i2c", "lr001"]["mean_gap"]:.1f} 降到 '
        f'{training_by["i2c", "lr010"]["mean_gap"]:.1f}，最优命中从 '
        f'{training_by["i2c", "lr001"]["hits"]}/10 变为 '
        f'{training_by["i2c", "lr010"]["hits"]}/10；但冻结策略每模型10次评估的'
        f'种子最佳 LUT 平均差距从 '
        f'{evaluation_by["i2c", "lr001-final"]["mean_seed_best_gap"]:.1f} 变为 '
        f'{evaluation_by["i2c", "lr010-final"]["mean_seed_best_gap"]:.1f}，'
        '两档都没有命中四步最优。训练期搜索收益未在这组独立尝试中得到支持。', '',
        f'max 的训练搜索中，0.001 有 {training_by["max", "lr001"]["feasible"]}/10 '
        f'个可行种子、{training_by["max", "lr001"]["hits"]}/10 个命中最优；'
        f'0.01 分别为 {training_by["max", "lr010"]["feasible"]}/10 和 '
        f'{training_by["max", "lr010"]["hits"]}/10。冻结评估中两档最终策略'
        f'分别只有 {evaluation_by["max", "lr001-final"]["feasible"]}/100 和 '
        f'{evaluation_by["max", "lr010-final"]["feasible"]}/100 次可行，'
        f'初始权重与随机策略各为 {evaluation_by["max", "initial"]["feasible"]}/100 和 '
        f'{evaluation_by["max", "uniform"]["feasible"]}/100。'
        '在这十次尝试的分辨率下，不能把训练时命中最优解释为冻结策略已稳定掌握可行序列。', '',
    ]
    training_marker = '## 训练期间搜索（250轮，含每轮初始映射）'
    if report.count(training_marker) != 1:
        raise ValueError('Training section insertion point is missing or duplicated.')
    report = report.replace(training_marker, '\n'.join(observations) + '\n' + training_marker)
    marker = '## 复核与局限'
    if report.count(marker) != 1:
        raise ValueError('Report insertion point is missing or duplicated.')
    zero = wilson(0)
    full = wilson(10)
    section = [
        '## 十次独立尝试的不确定性', '',
        '每个固定模型的可行率和最优命中率按10次尝试计算双侧 Wilson 95% 区间；'
        '全部逐种子区间见 [evaluation-intervals.csv](evaluation-intervals.csv)。'
        f'即使观察到 0/10，区间仍为 [{zero[0]:.3f}, {zero[1]:.3f}]；'
        f'观察到 10/10 时为 [{full[0]:.3f}, {full[1]:.3f}]。'
        '这些区间只描述固定模型下重复抽取四步动作的精度，不包含训练种子之间的变动。', '',
        f'四步穷举数据中，max 的 {len(max_four_step)} 条均匀动作序列仅有 '
        f'{feasible_sequences} 条可行；抽取10条时至少一次可行的概率约为 '
        f'{random_any_feasible:.1%}。因此 0/10 可行仍是常见的随机波动。', '',
        '下表按同一训练种子配对，展示每种策略相对对照组的可行率差；'
        '区间从10个训练种子重采样10,000次得到。逐组最优命中率差及区间另见 '
        '[evaluation-pairs.csv](evaluation-pairs.csv)。区间属于探索性估计，不能据此宣称普遍收益。', '',
        '| 电路 | 策略 − 对照 | 平均可行率差 | 探索性95%区间 |',
        '|---|---|---:|---|',
    ]
    for row in pairs:
        if row['metric'] == 'feasible':
            section.append(f'| {row["circuit"]} | {row["policy"]} − {row["comparator"]} | '
                           f'{row["mean_difference"]:+.3f} | '
                           f'[{row["bootstrap_ci_low"]:+.3f}, {row["bootstrap_ci_high"]:+.3f}] |')
    section += ['', marker]
    report_path.write_text(report.replace(marker, '\n'.join(section)))
    provenance_path = SUITE / 'analysis-provenance.json'
    provenance = read(provenance_path)
    provenance['generated_at'] = now()
    provenance['reporting_extension_sha256'] = sha(__file__)
    provenance['outputs']['evaluation-intervals.csv'] = sha(SUITE / 'evaluation-intervals.csv')
    provenance['outputs']['evaluation-pairs.csv'] = sha(SUITE / 'evaluation-pairs.csv')
    provenance['outputs']['report.md'] = sha(report_path)
    dump(provenance_path, provenance)


if __name__ == '__main__':
    main()

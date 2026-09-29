"""Audit every ABC trajectory and apply the preregistered seed-level tests."""
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from context import (ARCHIVE, CIRCUITS, EVAL_GROUPS, GROUPS, HERE, OUT, SEEDS, SPEC,
                     evaluation_folder, evaluation_seeds, identity, prior, read,
                     sha, training_config, training_folder)
from evaluate import checkpoint_for
from reward import RewardTracker, original_reward


@dataclass
class Episode:
    rows: list
    best: dict


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_episode(path, circuit, cfg, episode, group=None):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['iteration', 'optimization', 'luts', 'levels',
                                     'reward'], f'Malformed log header: {path}')
        raw = list(reader)
    require(len(raw) == SPEC['iterations'] + 1, f'Missing or duplicate mapping: {path}')
    sequence = list(cfg['protocol']['initial_sequence'])
    actions = set(cfg['protocol']['actions'])
    rows, best = [], None
    tracker = None
    previous = None
    for index, entry in enumerate(raw):
        require(None not in entry and None not in entry.values(),
                f'Malformed mapping row: {path}/{index}')
        try:
            actual_index = int(entry['iteration'])
            luts, levels = int(entry['luts']), int(entry['levels'])
            reward = float(entry['reward'])
        except (TypeError, ValueError) as error:
            raise ValueError(f'Malformed mapping value: {path}/{index}') from error
        require(actual_index == index and luts > 0 and levels > 0 and math.isfinite(reward),
                f'Invalid mapping order or value: {path}/{index}')
        action = entry['optimization']
        if index == 0:
            require(action == sequence[-1] and reward == 0,
                    f'Invalid initial mapping: {path}')
            if group == 'candidate':
                tracker = RewardTracker(luts, levels, circuit['max_levels'])
        else:
            require(action in actions, f'Unknown action: {path}/{index}')
            sequence.append(action)
            if group == 'candidate':
                expected_reward = tracker.step(luts, levels)
            elif group == 'original':
                expected_reward = original_reward(cfg, circuit, previous, (luts, levels))
            else:
                expected_reward = None
            if expected_reward is not None:
                require(math.isclose(reward, expected_reward, rel_tol=1e-10,
                                     abs_tol=1e-10),
                        f'Reward mismatch: {path}/{index}')
        result = dict(luts=luts, levels=levels,
                      feasible=levels <= circuit['max_levels'],
                      sequence=list(sequence), episode=episode, iteration=index)
        if index > 0 or cfg['protocol']['evaluation']['include_initial']:
            if best is None or prior.old.FPGASession.rank(result) < \
                    prior.old.FPGASession.rank(best):
                best = result
        rows.append(dict(iteration=index, action=action, luts=luts, levels=levels,
                         reward=reward, best=best))
        previous = (luts, levels)
    require(best is not None, f'No eligible mapping: {path}')
    return Episode(rows, best)


def check_best_files(folder, best, recorded):
    folder = Path(folder)
    require(best == recorded['best'] == read(folder / 'best.json'),
            f'Recorded best disagrees with mapping logs: {folder}')
    for name, key in [('best-mapped.v', 'best_mapped_sha256'),
                      ('equivalence.log', 'equivalence_log_sha256')]:
        require(sha(folder / name) == recorded[key], f'Changed evidence: {folder / name}')
    proof = (folder / 'equivalence.log').read_text()
    require('Networks are equivalent' in proof,
            f'CEC does not certify equivalence: {folder}')
    import re
    metrics = re.findall(r'\bnd\s*=\s*(\d+)[^\n]*?\blev\s*=\s*(\d+)', proof)
    require(metrics and tuple(map(int, metrics[-1])) == (best['luts'], best['levels']),
            f'CEC LUT/depth disagrees with best: {folder}')


def audit_training(group, circuit, seed, digest, cfg, curve_writer):
    folder = training_folder(group, circuit, seed)
    record = read(folder / 'result.json')
    require(record['status'] == 'complete' and record['fingerprint'] == digest and
            record['group'] == group and record['circuit'] == circuit and
            record['seed'] == seed and record['episodes_completed'] == SPEC['episodes'] and
            record['mapping_calls'] == SPEC['episodes'] * (SPEC['iterations'] + 1),
            f'Incomplete training: {folder}')
    checkpoint = prior.old.load(folder / 'checkpoint.pt')
    initial = prior.old.load(folder / 'snapshots/0.pt')
    require(record['checkpoint_sha256'] == sha(folder / 'checkpoint.pt') and
            record['final_network_sha256'] == prior.old.state_hash(checkpoint['network']) and
            checkpoint['episodes_completed'] == SPEC['episodes'] and
            len(checkpoint['return_window']) == SPEC['return_window'] and
            len(checkpoint['rewards']) == SPEC['episodes'],
            f'Checkpoint mismatch: {folder}')
    require(record['initial_hashes'] ==
            {key: prior.old.state_hash(initial[key])
             for key in ('network', 'optimizer', 'rng_state')},
            f'Initial checkpoint mismatch: {folder}')
    require(checkpoint['best'] == record['best'], f'Checkpoint best mismatch: {folder}')
    diagnostics = [json.loads(line) for line in
                   (folder / 'return-diagnostics.jsonl').read_text().splitlines()]
    require(len(diagnostics) == SPEC['episodes'] and
            [row['episode'] for row in diagnostics] == list(range(1, SPEC['episodes'] + 1)) and
            all(row['mode'] == ('within_episode_warmup'
                                if row['episode'] <= SPEC['warmup_episodes']
                                else 'lagged_window') for row in diagnostics),
            f'Window diagnostics mismatch: {folder}')
    best = None
    mapping_calls = 0
    feasible_episodes = 0
    for episode in range(1, SPEC['episodes'] + 1):
        parsed = parse_episode(folder / 'episodes' / str(episode) / 'log.csv',
                               cfg['protocol']['circuits'][circuit], cfg, episode, group)
        if parsed.best['feasible']:
            feasible_episodes += 1
        for item in parsed.rows:
            mapping_calls += 1
            candidate = item['best']
            if best is None or prior.old.FPGASession.rank(candidate) < \
                    prior.old.FPGASession.rank(best):
                best = candidate
            curve_writer.writerow(dict(circuit=circuit, group=group, seed=seed,
                                       mapping_calls=mapping_calls,
                                       feasible_so_far=int(best['feasible']),
                                       best_feasible_luts=(best['luts'] if best['feasible'] else ''),
                                       best_levels=best['levels']))
    require(mapping_calls == record['mapping_calls'], f'Mapping budget mismatch: {folder}')
    check_best_files(folder, best, record)
    return dict(circuit=circuit, group=group, seed=seed,
                mapping_calls=mapping_calls, feasible_episodes=feasible_episodes,
                training_best_feasible=int(best['feasible']),
                training_best_luts=best['luts'] if best['feasible'] else None,
                training_best_levels=best['levels'],
                training_seconds=record['training_seconds'])


def audit_evaluation(group, circuit, seed, digest, cfg):
    folder = evaluation_folder(group, circuit, seed)
    record = read(folder / 'result.json')
    checkpoint = checkpoint_for(group, circuit, seed)
    saved = prior.old.load(checkpoint)
    require(record['status'] == 'complete' and record['fingerprint'] == digest and
            record['group'] == group and record['circuit'] == circuit and
            record['training_seed'] == seed and
            record['checkpoint_sha256'] == sha(checkpoint) and
            record['network_sha256'] == prior.old.state_hash(saved['network']) and
            record['network_unchanged'] is True and
            record['checkpoint_unchanged'] is True,
            f'Frozen checkpoint mismatch: {folder}')
    rows = record['rollouts']
    seeds = evaluation_seeds(seed)
    require(len(rows) == len(seeds) and
            [row['evaluation_seed'] for row in rows] == list(seeds),
            f'Missing, duplicate, or reordered evaluation trajectory: {folder}')
    output = []
    for row, eval_seed in zip(rows, seeds):
        destination = folder / f'rollout-{eval_seed}'
        on_disk = read(destination / 'rollout.json')
        require(row == on_disk and row['status'] == 'complete' and
                row['fingerprint'] == digest and row['group'] == group and
                row['circuit'] == circuit and row['training_seed'] == seed and
                row['checkpoint_sha256'] == record['checkpoint_sha256'] and
                row['mapping_calls'] == SPEC['iterations'] + 1 and
                row['log_sha256'] == sha(destination / 'episodes/1/log.csv'),
                f'Changed or duplicate rollout: {destination}')
        parsed = parse_episode(destination / 'episodes/1/log.csv',
                               cfg['protocol']['circuits'][circuit], cfg, 1)
        require([item['action'] for item in parsed.rows[1:]] ==
                [cfg['protocol']['actions'][index] for index in row['actions']],
                f'Action log differs from evaluation record: {destination}')
        check_best_files(destination, parsed.best, row)
        output.append(dict(circuit=circuit, group=group, training_seed=seed,
                           evaluation_seed=eval_seed, mapping_calls=row['mapping_calls'],
                           feasible=int(parsed.best['feasible']),
                           best_feasible_luts=(parsed.best['luts']
                                               if parsed.best['feasible'] else None),
                           best_levels=parsed.best['levels']))
    return output


def exact_sign_p(wins, total):
    require(0 <= wins <= total, 'Invalid sign-test win count.')
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2 ** total


def holm_adjust(pvalues):
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [None] * len(pvalues)
    floor = 0.0
    for rank, (index, pvalue) in enumerate(indexed):
        floor = max(floor, (len(pvalues) - rank) * pvalue)
        adjusted[index] = min(floor, 1.0)
    return adjusted


def paired_bootstrap(differences, draws, seed):
    values = np.asarray(differences, dtype=float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
            'Bootstrap differences must be finite and nonempty.')
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def statistics(evaluation_rows):
    by_key = {(row['circuit'], row['group'], row['training_seed'],
               row['evaluation_seed']): row for row in evaluation_rows}
    require(len(by_key) == len(evaluation_rows), 'Duplicate evaluation rows.')
    required = {(circuit, group, seed, evaluation_seed)
                for circuit in CIRCUITS for group in EVAL_GROUPS for seed in SEEDS
                for evaluation_seed in evaluation_seeds(seed)}
    require(set(by_key) == required, 'Missing or unexpected evaluation trajectory.')
    paired, tests = [], {}
    for circuit in CIRCUITS:
        improvements = []
        for seed in SEEDS:
            bank = evaluation_seeds(seed)
            data = {group: [by_key[circuit, group, seed, evaluation_seed]
                            for evaluation_seed in bank] for group in EVAL_GROUPS}
            original, candidate = data['original'], data['candidate']
            old_feasible = sum(row['feasible'] for row in original)
            new_feasible = sum(row['feasible'] for row in candidate)
            baseline_feasible = sum(row['feasible'] for row in data['initial'])
            if circuit == 'i2c':
                require(old_feasible == len(bank) and new_feasible == len(bank),
                        f'i2c feasible-LUT primary metric is undefined: seed {seed}')
                old_primary = sum(row['best_feasible_luts'] for row in original) / len(bank)
                new_primary = sum(row['best_feasible_luts'] for row in candidate) / len(bank)
                difference_numerator = sum(row['best_feasible_luts'] for row in original) - \
                    sum(row['best_feasible_luts'] for row in candidate)
                difference = difference_numerator / len(bank)
            else:
                old_primary, new_primary = old_feasible / len(bank), new_feasible / len(bank)
                difference_numerator = new_feasible - old_feasible
                difference = difference_numerator / len(bank)
            improvements.append(difference)
            paired.append(dict(circuit=circuit, seed=seed, original=old_primary,
                               candidate=new_primary, improvement=difference,
                               improved=int(difference_numerator > 0),
                               original_feasible=old_feasible,
                               candidate_feasible=new_feasible,
                               initial_feasible=baseline_feasible,
                               initial_primary=(sum(row['best_feasible_luts']
                                                    for row in data['initial']) / len(bank)
                                                if circuit == 'i2c' else
                                                baseline_feasible / len(bank)),
                               original_conditional_luts=conditional_luts(original),
                               candidate_conditional_luts=conditional_luts(candidate),
                               initial_conditional_luts=conditional_luts(data['initial'])))
        wins = sum(row['improved'] for row in paired if row['circuit'] == circuit)
        tests[circuit] = dict(wins=wins, seeds=len(SEEDS),
                              mean_improvement=float(np.mean(improvements)),
                              median_improvement=float(np.median(improvements)),
                              bootstrap_95=paired_bootstrap(
                                  improvements, SPEC['statistics']['bootstrap_draws'],
                                  SPEC['statistics']['bootstrap_seed'] + CIRCUITS.index(circuit)),
                              p_one_sided=exact_sign_p(wins, len(SEEDS)))
    adjusted = holm_adjust([tests[circuit]['p_one_sided'] for circuit in CIRCUITS])
    for circuit, pvalue in zip(CIRCUITS, adjusted):
        tests[circuit]['p_holm'] = pvalue
        tests[circuit]['conclusion'] = (
            '通过' if pvalue <= SPEC['statistics']['alpha'] and
            tests[circuit]['mean_improvement'] > 0 else '证据不足')
    return paired, tests


def conditional_luts(rows):
    values = [row['best_feasible_luts'] for row in rows if row['feasible']]
    return float(np.mean(values)) if values else None


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(digest, training_rows, evaluation_rows, paired, tests):
    def fmt(value, digits=3):
        return '不可用' if value is None else f'{value:.{digits}f}'

    lines = [
        '# 十步新奖励验证', '',
        f'运行指纹：`{digest}`。协议在执行前固定；种子 20–29 为新训练种子，'
        '历史种子 0–4 不参与检验。两组仅奖励函数不同，均为 100 轮×10 步、'
        '每种子 1,100 次真实 ABC 映射。', '',
        '## 冻结策略主检验', '',
        '每个最终模型独立抽取 30 条轨迹，种子内两组共用评估随机种子，'
        '种子间评估组不重叠。i2c 的主指标为每种子 30 条轨迹的平均最佳可行 LUT'
        '（越低越好）；max 为可行轨迹比例（越高越好）。'
        '以 10 个训练种子为独立单位，持平计为未改善。', '',
        '| 电路 | 改善种子 | 平均改善 | 中位改善 | 配对 bootstrap 95% 区间 | 单侧精确符号检验 p | Holm p | 判定 |',
        '|---|---:|---:|---:|---|---:|---:|---|',
    ]
    for circuit in CIRCUITS:
        row = tests[circuit]
        lo, hi = row['bootstrap_95']
        lines.append(f'| {circuit} | {row["wins"]}/10 | {fmt(row["mean_improvement"])} | '
                     f'{fmt(row["median_improvement"])} | [{fmt(lo)}, {fmt(hi)}] | '
                     f'{row["p_one_sided"]:.6f} | {row["p_holm"]:.6f} | '
                     f'{row["conclusion"]} |')
    lines += ['', 'Holm 方法控制两项预先指定检验的总体错误率为 0.05；'
              '每个通过的电路须至少 9/10 个种子改善且平均方向正确。'
              'bootstrap 区间只描述效应量，不替代主判据。未通过表示证据不足，'
              '不表示两奖励等效。', '',
              '| 电路 | 种子 | 原奖励主指标 | 新奖励主指标 | 改善（正值为好） | 原可行/30 | 新可行/30 | 初始可行/30 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in paired:
        lines.append(f'| {row["circuit"]} | {row["seed"]} | {fmt(row["original"])} | '
                     f'{fmt(row["candidate"])} | {fmt(row["improvement"])} | '
                     f'{row["original_feasible"]} | {row["candidate_feasible"]} | '
                     f'{row["initial_feasible"]} |')
    lines += ['', '初始网络只作参照，不参与主检验。max 的 LUT 只在可行轨迹中汇总；'
              '不可行轨迹没有虚构的 LUT 罚分。', '',
              '## 同预算训练搜索（次要）', '',
              '| 电路 | 奖励 | 训练种子找到可行解 | 可行回合/1000 | 可行种子的最佳 LUT 均值 |',
              '|---|---|---:|---:|---:|']
    for circuit in CIRCUITS:
        for group in GROUPS:
            subset = [row for row in training_rows if row['circuit'] == circuit and
                      row['group'] == group]
            feasible = [row['training_best_luts'] for row in subset
                        if row['training_best_feasible']]
            lines.append(f'| {circuit} | {group} | {len(feasible)}/10 | '
                         f'{sum(row["feasible_episodes"] for row in subset)}/1000 | '
                         f'{fmt(float(np.mean(feasible)) if feasible else None)} |')
    lines += ['', '逐调用预算曲线在 `training-curves.csv`：从每回合初始映射起，'
              '按每种子相同的 ABC 调用数统计累计最好结果。逐轨迹结果见'
              ' `evaluation-rollouts.csv`，逐种子比较见 `paired-seeds.csv`。', '',
              '## 完整性', '',
              f'- 训练：{len(training_rows)}/40 个运行，'
              f'{sum(row["mapping_calls"] for row in training_rows):,} 次 ABC 映射。',
              f'- 冻结评估：{len(evaluation_rows):,}/1,800 条独立轨迹，'
              f'{sum(row["mapping_calls"] for row in evaluation_rows):,} 次 ABC 映射。',
              '- 所有逐步日志均重建最佳结果；输出哈希、CEC 等价结论及最佳网表'
              ' LUT/层数已逐项核对；冻结评估的网络和检查点哈希未变。',
              '- 源码、协议、ABC/Yosys、基准电路的 SHA-256 及环境版本见'
              ' `results/ten-step-reward-validation/preflight.json`；旧实验快照的'
              ' SHA-256 见仓库外 `experiment-archives/four-step-reward-682ee92.sha256`。',
              '', '## 解释边界', '',
              '30 条评估轨迹估计的是每个冻结策略在该抽样方案下的表现，'
              '而非十步动作空间的穷举最优。十个种子适合检验方向一致的大效应；'
              '不能从证据不足推断等效，不按结果改变门槛或追加种子重新宣称通过。',
              '', 'Origin Skill: academic-research-suite/experiment-agent  '
              'Origin Mode: run+validate  Origin Date: 2026-09-29  '
              'Verification Status: VERIFIED', '']
    (HERE / 'report.md').write_text('\n'.join(lines))


def main():
    digest, _ = identity()
    require(sha(ARCHIVE) == SPEC['legacy_snapshot_sha256'],
            'The legacy results archive changed.')
    for name, status in [('preflight.json', 'pass'), ('train-manifest.json', 'complete'),
                         ('pair-validation.json', 'pass'),
                         ('evaluation-manifest.json', 'complete')]:
        row = read(OUT / name)
        require(row['status'] == status and row['fingerprint'] == digest,
                f'Missing, stale, or incomplete manifest: {name}')
    training, evaluation = [], []
    curves = HERE / 'training-curves.csv'
    with curves.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['circuit', 'group', 'seed',
                                                    'mapping_calls', 'feasible_so_far',
                                                    'best_feasible_luts', 'best_levels'])
        writer.writeheader()
        for circuit in CIRCUITS:
            for seed in SEEDS:
                for group in GROUPS:
                    training.append(audit_training(group, circuit, seed, digest,
                                                   training_config(circuit, group), writer))
    require(len(training) == 40, 'Missing or duplicate training result.')
    for circuit in CIRCUITS:
        cfg = training_config(circuit, 'original')
        for seed in SEEDS:
            for group in EVAL_GROUPS:
                evaluation.extend(audit_evaluation(group, circuit, seed, digest, cfg))
    require(len(evaluation) == 1800, 'Missing or duplicate evaluation trajectory.')
    paired, tests = statistics(evaluation)
    from context import dump
    evidence = dict(status='verified', fingerprint=digest,
                    legacy_snapshot_sha256=sha(ARCHIVE),
                    training={f'{circuit}/{group}/{seed}':
                              sha(training_folder(group, circuit, seed) / 'result.json')
                              for circuit in CIRCUITS for seed in SEEDS for group in GROUPS},
                    evaluation={f'{circuit}/{group}/{seed}':
                                sha(evaluation_folder(group, circuit, seed) / 'result.json')
                                for circuit in CIRCUITS for seed in SEEDS for group in EVAL_GROUPS},
                    rollouts={f'{circuit}/{group}/{seed}/{eval_seed}':
                              sha(evaluation_folder(group, circuit, seed) /
                                  f'rollout-{eval_seed}/rollout.json')
                              for circuit in CIRCUITS for seed in SEEDS
                              for group in EVAL_GROUPS
                              for eval_seed in evaluation_seeds(seed)})
    dump(OUT / 'evidence-hashes.json', evidence)
    write_csv(HERE / 'training-seeds.csv', training,
              list(training[0]))
    write_csv(HERE / 'evaluation-rollouts.csv', evaluation,
              list(evaluation[0]))
    write_csv(HERE / 'paired-seeds.csv', paired, list(paired[0]))
    dump(OUT / 'statistics.json', dict(status='verified', fingerprint=digest,
                                       tests=tests, training_runs=len(training),
                                       evaluation_rollouts=len(evaluation)))
    report(digest, training, evaluation, paired, tests)
    print(f'Verified report: {HERE / "report.md"}', flush=True)


if __name__ == '__main__':
    main()

#!/bin/sh
# Run after analyze.py to add reviewed tables and commit-ready evidence hashes.
set -eu
python3 - "$0" <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

here = Path(sys.argv[1]).resolve().parent
root = here.parents[1]
raw = root / "results/four-step-reward"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_run(rows, phase, group, circuit, seed, folder, source, episodes, calls):
    result = read(folder / "result.json")
    assert result["status"] == "complete"
    assert result["episodes_completed"] == episodes
    mapped = folder / "best-mapped.v"
    log = folder / "equivalence.log"
    assert "Networks are equivalent" in log.read_text()
    rows.append(dict(
        phase=phase, group=group, circuit=circuit, seed=seed, source=source,
        result_sha256=sha(folder / "result.json"),
        best_mapped_sha256=sha(mapped), equivalence_log_sha256=sha(log),
        final_network_sha256=result.get("final_network_sha256"),
        best=result["best"], episodes=episodes, mapping_calls=calls,
    ))


rows = []
for circuit in ("i2c", "max", "int2float"):
    for group in ("window", "reward"):
        for seed in range(10):
            old = group == "window" and circuit == "i2c"
            folder = (root / "results/four-step-followup/training/i2c" / f"seed-{seed}"
                      if old else raw / "four-step" / group / circuit / f"seed-{seed}")
            add_run(rows, "four", group, circuit, seed, folder,
                    "historical-verified" if old else "new", 250, 1250)
for circuit in ("max", "i2c"):
    for group in ("reward", "frozen", "uniform", "greedy"):
        for seed in range(5):
            old = group in ("frozen", "uniform") and seed < 3
            folder = (root / "results/learning-effectiveness/training" / group /
                      circuit / f"seed-{seed}" if old else
                      raw / "ten-step" / group / circuit / f"seed-{seed}")
            add_run(rows, "ten", group, circuit, seed, folder,
                    "historical-verified" if old else "new", 100, 1100)

parity = read(raw / "equivalence.json")
pilot = read(raw / "pilot.json")
baseline = read(raw / "ten-baseline-validation.json")
exact = read(raw / "exact.json")
ten = read(raw / "ten-summary.json")
assert len(rows) == 100 and parity["table_equivalent"] and len(parity["rows"]) == 10
assert pilot["status"] == "pass" and len(exact["rows"]) == 360
assert len(baseline["reusable"]) == 12
assert all(check["first_episode_replayed"] for check in baseline["checks"])
assert len(ten["rows"]) == 40
evidence = dict(
    fingerprint=ten["fingerprint"],
    checks=dict(
        parity_seed_matches=len(parity["rows"]), pilot=pilot["status"],
        four_exact_rows=len(exact["rows"]),
        reused_ten_baselines=len(baseline["reusable"]),
        historical_best_netlists_cec=22, ten_rows=len(ten["rows"]),
        best_netlists_cec=len(rows),
        historical_source_drift=baseline["historical_source_drift"],
    ),
    runs=rows,
)
(here / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")

paired = list(csv.DictReader((here / "paired.csv").open()))
ten_rows = list(csv.DictReader((here / "ten-results.csv").open()))
assert len(paired) == 30 and len(ten_rows) == 40
report = here / "report.md"
text = report.read_text()
assert "### 逐种子配对结果" not in text, "Run analyze.py before finalizing."
text = text.replace("最优 LUT：int2float 44、i2c 312、max 781。",
                    "四步穷举最优 LUT：int2float 44、i2c 312、max 781。")
table = [
    "### 逐种子配对结果", "",
    "下表 i2c 与 int2float 为单次期望 LUT 差距，越低越好；max 为单次可行率，越高越好。配对改善均以正数为好。", "",
    "| 电路 | 种子 | 窗口版 | 新奖励 | 配对改善 |",
    "|---|---:|---:|---:|---:|",
]
for row in paired:
    number = "{:.6g}" if row["circuit"] == "max" else "{:.3f}"
    table.append("| {} | {} | {} | {} | {} |".format(
        row["circuit"], row["seed"], number.format(float(row["window"])),
        number.format(float(row["reward"])),
        number.format(float(row["improvement"]))))
table += ["", "最优命中概率与四步贪心动作序列见 paired.csv。", "", ""]
anchor = "### 理论基线、贪心与训练搜索"
assert text.count(anchor) == 1
text = text.replace(anchor, "\n".join(table) + anchor)
text = text.replace(
    "逐种子单次指标、最优命中概率及贪心序列见 paired.csv；训练期间累计最好值仅是辅助搜索指标。",
    "训练期间累计最好值仅是辅助搜索指标。max 两组均仅 8/10 个种子在训练抽样中找到可行网表，因此不报告十种子最好 LUT 均值。")
header = ("| 电路 | 种子 | 新奖励最好 LUT | 冻结最好 LUT | 随机最好 LUT | "
          "贪心最好 LUT | 新奖励胜过两基线 |\n"
          "|---|---:|---:|---:|---:|---:|---|")
assert text.count(header) == 1
assert text.count("\n| i2c | 0 | 298 |") == 1
text = text.replace("\n| i2c | 0 | 298 |", "\n" + header + "\n| i2c | 0 | 298 |")
for seed in (0, 3, 4):
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines)
            if line.startswith(f"| max | {seed} |") and " |  | " in line]
    assert len(hits) == 1
    lines[hits[0]] = lines[hits[0]].replace(" |  | ", " | 不可行 | ")
    text = "\n".join(lines) + "\n"
rates = [
    "### 观测到的可行回合率", "",
    "每个策略的数值是五个种子各 100 回合的可行回合率均值；逐种子值见 ten-results.csv。", "",
    "| 电路 | 新奖励 | 冻结 | 均匀随机 | 贪心 |",
    "|---|---:|---:|---:|---:|",
]
for circuit in ("max", "i2c"):
    values = []
    for group in ("reward", "frozen", "uniform", "greedy"):
        rate = statistics.mean(
            float(row["episode_feasible_rate"]) for row in ten_rows
            if row["circuit"] == circuit and row["group"] == group)
        values.append(f"{rate:.1%}")
    rates.append("| {} | {} |".format(circuit, " | ".join(values)))
rates.extend(["", ""])
anchor = "逐种子可行回合率和等调用预算曲线见 ten-results.csv、ten-curves.csv。"
assert text.count(anchor) == 1
text = text.replace(anchor, "\n".join(rates) + anchor)
text = text.replace(
    "见 ten-results.csv、ten-curves.csv。\n![十步",
    "见 ten-results.csv、ten-curves.csv。\n\n![十步")
anchor = "- 候选奖励审计：pass；γ=0.99 下三电路最高分序列全部最优。"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + "\n- 最佳网表：100/100 份均经 ABC 组合等价检查与 LUT/层数复核，哈希见 evidence.json；其中复用历史网表 22 份。")
anchor = "十步使用真实 ABC，每种子 100×10 动作、1,100 次映射；已核验的旧冻结及均匀随机轨迹才复用。以下只陈述观测到的可行性和最佳 LUT，不推断十步全局最优。"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + "\n\n历史轨迹的 `drills/model.py` 文件哈希与当前版本不同。复用前已核对协议、工具和电路哈希，逐回合重建轨迹与最佳结果，并在当前 ABC 环境中复演首回合动作及映射；12 条历史轨迹通过这些检查。")
text = text.replace(
    "源代码、工具、电路、穷举表、特征缓存的 SHA-256 和运行状态见本实验的证据清单与本机运行 manifest。",
    "源代码、工具、电路、穷举表、特征缓存的 SHA-256 见 artifact-manifest.json；逐运行最佳网表、CEC 日志和结果哈希见 evidence.json；完整运行 manifest 保存在本机 results/。")
report.write_text(text)

manifest_path = here / "artifact-manifest.json"
manifest = read(manifest_path)
assert manifest["fingerprint"] == evidence["fingerprint"]
manifest["artifacts"] = {
    path.name: sha(path) for path in sorted(here.iterdir())
    if path.is_file() and path.name != manifest_path.name
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(f"Final report ready; {len(rows)} netlists and {len(manifest['artifacts'])} artifacts hashed.")
PY

# 四步精确复评与滑动窗口回报实验

本套件读取原四步学习率实验的冻结模型、轨迹和 0–4 步完整穷举表。它不修改
`drills/` 或 `experiments/four-step-lr/` 中的文件，因此保留旧实验的源码指纹。
原始模型和轨迹位于本机 Git 忽略的 `results/four-step-lr/`，在其他机器运行前需复制它们。

## 协议

- 精确评估复算 0–3 步的 400 个决策前缀状态；每个电路的 2,401 条四步路径用于计算单次期望表现。
- 新训练仅将回报标准化改为滑动窗口：前八轮沿用逐回合标准化；以后用此前八轮同一步位的回报均值和标准差，标准差下限为 1；完成本轮更新后窗口才前进。
- 学习率为 0.01，训练种子 0–9，每种子 250 轮 × 4 动作，每轮更新一次。旧 0.01 模型是配对对照。
- i2c 是唯一主比较。其十种子平均单次期望 LUT 差距小于旧 0.01、至少六个种子改善、且均值不差于初始网络时，才扩展 int2float 和 max。该门槛只用于安排实验，不是显著性判断。

## 运行

从仓库根目录执行：

```bash
.tools/conda-env/bin/python -B -m unittest discover -s experiments/four-step-followup -p 'test_*.py'
.tools/conda-env/bin/python -B -u experiments/four-step-followup/exact.py --source old
.tools/conda-env/bin/python -B -u experiments/four-step-followup/run.py --circuit i2c
.tools/conda-env/bin/python -B -u experiments/four-step-followup/exact.py --source new
.tools/conda-env/bin/python -B experiments/four-step-followup/analyze.py
```

查看 `results/four-step-followup/summary.json` 的 `gate`。只有 `true` 才运行：

```bash
.tools/conda-env/bin/python -B -u experiments/four-step-followup/run.py --circuit int2float
.tools/conda-env/bin/python -B -u experiments/four-step-followup/run.py --circuit max
.tools/conda-env/bin/python -B -u experiments/four-step-followup/exact.py --source new
.tools/conda-env/bin/python -B experiments/four-step-followup/analyze.py
```

训练中断后，仅当源码、工具及协议不变时对相应 `run.py --circuit ...` 命令追加 `--resume`。
检查点同时保存模型、优化器、随机数状态和回报窗口。三个 worker，每任务 30 分钟硬超时。
失败与缺失任务保留在状态、日志和 manifest 中；不以中途快照替代最终模型。

## 结果口径

`old-exact.json` 与 `new-exact.json` 位于 `results/four-step-followup/`，包含每个模型的精确单次指标、
贪心序列和 k=1–100 的取最好曲线。旧评估的 1,200 条轨迹要逐动作严格重放。`report.md`、
`i2c-paired.csv` 和 `best-of-k.svg` 是可复核的结论与图。max 以单次可行概率为主；
可行轨迹之外不定义 LUT 差距。bootstrap 和配对 t 区间都按十个训练种子计算，并标为探索性。

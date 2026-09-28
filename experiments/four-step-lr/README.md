# 四步环境学习率对照

依据 [FPGA 实验评审第 3.1 节](https://github.com/EpsilonLh/drills/blob/main/docs/fpga-experiment-review.md)，
本套件只比较原学习率 `0.001` 与 `0.01`，另设均匀随机组。每组在三个电路、十个种子上运行 250 轮、
每轮四个优化动作。其余 A2C 方法参数与原始 `adf7bef` 一致。

训练期间累计最好值和训练后独立策略评估分开报告。独立评估对每个最终模型、对应初始权重和均匀随机
策略分别执行十条新四步轨迹；同一电路和训练种子内共用评估随机种子，不同训练种子的随机组独立运行。
评估全程无梯度及优化器更新，检查点和模型状态在评估前后校验哈希。

## 运行

在仓库根目录执行；需先按仓库现有方式准备 `.tools/conda-env` 中的 Python、ABC 和 Yosys。
首次运行需要干净的 `results/four-step-lr/`，中断后只在源代码、协议及工具指纹相同的情况下使用 `--resume`。

```bash
.tools/conda-env/bin/python -B -m unittest discover -s experiments/four-step-lr -p 'test_*.py'
.tools/conda-env/bin/python -B -u experiments/four-step-lr/run.py
.tools/conda-env/bin/python -B -u experiments/four-step-lr/evaluate.py
.tools/conda-env/bin/python -B experiments/four-step-lr/analyze.py
.tools/conda-env/bin/python -B experiments/four-step-lr/check.py
```

`run.py --resume` 和 `evaluate.py --resume` 继续相同指纹的未完成任务。三个 worker 同时运行，
每个任务 30 分钟硬超时；失败记录保留，分析不会把缺失任务写成完整结果。

## 交付口径

- `report.md`：中文结论、全部十个训练种子的独立评估结果、适用范围和局限。
- `training.csv`、`evaluation.csv`、`evaluation-by-seed.csv`：搜索与独立策略评估原始汇总。
- `diagnostics.csv`、`diagnostic-pairs.csv`、`diagnostics.svg`：逐轮权重方差及固定探针策略熵。
- `summary.json`、`validation.json`、`analysis-provenance.json`：统计、验证和哈希。

原始模型、逐轨迹日志、最佳网表及过程日志保存在 Git 忽略的 `results/four-step-lr/`。在另一台机器
复现冻结模型评估需要复制这些模型文件或重新训练。比较规则是：逐电路十种子平均最优 LUT 差距小于
均匀随机，且至少八个配对种子不更差；若有种子未找到可行解，该差距规则不判定为通过。

权重方差只统计 Actor/Critic 线性层权重矩阵，不含偏置。策略熵在每个电路固定的 196 个归一化
状态探针上计算，探针来自 49 种有序动作对的四步重复序列，与训练采样随机数隔离。

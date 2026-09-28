# 四步奖励实验及条件性十步验证

本目录独立于旧四步源码，复用已认证的穷举表、400 个前缀状态和旧模型证据。
源码来自提交 `cfd580f`，方案见
[滑动窗口回报实验之后：审计奖励，再只改奖励](https://github.com/EpsilonLh/drills/blob/main/docs/four-step-window-return-review.md)。

从仓库根目录运行：

```bash
.tools/conda-env/bin/python -B -m unittest discover -s experiments/four-step-reward -p 'test_*.py' -v
.tools/conda-env/bin/python -B experiments/four-step-reward/audit.py
.tools/conda-env/bin/python -B -u experiments/four-step-reward/run.py --phase parity
.tools/conda-env/bin/python -B -u experiments/four-step-reward/run.py --phase pilot
.tools/conda-env/bin/python -B -u experiments/four-step-reward/run.py --phase four
.tools/conda-env/bin/python -B -u experiments/four-step-reward/exact_eval.py
.tools/conda-env/bin/python -B experiments/four-step-reward/analyze.py
.tools/conda-env/bin/python -B -u experiments/four-step-reward/run.py --phase ten
.tools/conda-env/bin/python -B experiments/four-step-reward/analyze.py
sh experiments/four-step-reward/finalize-report.sh
```

`parity` 对旧窗口版 i2c 的十个最终网络逐一验证哈希；未通过则后续自动回退到
ABC 环境。`pilot` 的 20 轮使用种子 10，与正式种子 0–9 分离。四步训练使用
250×4，保存第 0/50/100/150/200/250 轮模型。若 i2c 或 max 四步门槛通过，
`ten` 才对相应电路用真实 ABC 运行 5 种子、100×10 的等调用预算对照。

原始权重、日志和执行 manifest 保存在 Git 忽略的
`results/four-step-reward/`。本目录提交协议、审计、紧凑 CSV、SVG、摘要、报告和
SHA-256 清单。`evidence.json` 列出 100 份最佳网表、对应 CEC 日志及运行结果的
哈希；`artifact-manifest.json` 列出提交文件和输入环境的哈希。`report.md` 中的
逐种子表和可行回合率表来自 `paired.csv` 与 `ten-results.csv`。

中断后使用相同命令即可从同一指纹的检查点续跑；源码或数据指纹
改变时必须使用新的结果目录，旧数据不被覆盖。

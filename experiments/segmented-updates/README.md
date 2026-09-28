# 长轨迹搜索、分段更新：i2c 六组实验

从 `adf7bef53cdf6c4161586c03e6e2caecffe26bc4` 独立分支。原版源文件和默认入口保持原样；新增 `drills/segmented_model.py`，实验只用本目录入口。

训练种子 10/11/12，六组各 1,000 候选：A 原版 100×10；B 固定奖励尺度 100×10；C 同 B 的 20×50 整轮更新；D 同 B 的 20×50 每十步更新；F 初始化冻结 20×50；U 均匀随机 20×50。D 为目标方案。原 Welford 状态归一化和原奖励不变。

B/C/D 的 Critic 使用未标准化的多步目标，Actor 使用未标准化且 detach 的优势。非终点使用更新前 Critic 的段尾价值，终点取零；不加熵损失、梯度裁剪或新学习率。保留原求和损失。K 同时影响多步目标长度与更新节奏，不能把结果归因于更新次数单一因素。

每段更新不重置环境或归一化器，下一状态只消费一次。段提交为恢复边界，保存网络/Adam/RNG/前缀/奖励/归一化/缓存状态/最佳网表/日志位置。中断后恢复到上个段提交；重放未提交段不重复计入已提交预算，额外执行记录见恢复事件和进程日志。

最终模型用 30000–30029 动作种子进行五十步无更新评估；A/B 另报告同轨迹十步前缀。15 个网络模型加一份共享均匀随机，共 480 条真实轨迹、24,000 候选。训练预算另计 18,000。固定模型评估不改训练成绩；三个训练种子是独立分析单位。

主比较为 D 对 A/C/F/U 和 B 对 A。正差定义为对照 LUT 减目标 LUT。各项筛查门槛：均值减少至少 1 LUT，且至少两种子改善。仅作探索性筛查；无论结果如何都在三种子预算收口，不追加确认阶段。固定共同目标为最佳可行 LUT≤300。

## 运行

在新 worktree 根目录执行。训练 Python 使用原仓库已经安装的运行时，不更改其环境。换机器时需在 protocol.yml 的 runtime 中配置对应 ABC/Yosys 路径，并在首次锁定前完成设置。

```bash
/Users/zhan/Downloads/test/DRiLLS/.tools/conda-env/bin/python -B experiments/segmented-updates/check.py
/Users/zhan/Downloads/test/DRiLLS/.tools/conda-env/bin/python -B -u experiments/segmented-updates/run.py all
```

也可分阶段执行 `prepare`、`train`、`evaluate`、`verify`、`analyze`、`package`。只有中断后明确续跑才使用 `train --resume` 或 `evaluate --resume`。指纹覆盖源码、测试、协议、工具、电路及版本；锁定后不得调参、换种子或改源码续跑。三进程、每进程单 Torch 线程，每任务 1,800 秒硬超时；失败保留，不自动重复。新实验目录和结果不会覆盖旧实验。

绘图沿用 ReportLab/PyMuPDF，默认使用 Codex 内置文档 Python；其他机器通过 `--plot-python /对应/python` 指定独立绘图运行时。该选项不改变训练依赖。

## 交付与证据

本目录保存中文 report.md、CSV、JSON、SVG/PNG、校验与模型哈希清单。新 worktree 的 `results/segmented-updates/` 保存完整模型、逐步记录、最佳网表及本地 evidence.tar.gz；结果目录不进入 Git。仅两次本地中文提交，无远端推送。

分支隔离记录会核对原工作区状态、已有分支指针、历史实验及结果文件哈希。托管工具因聊天位于仓库外层返回 Not a git repository，本次使用 Git worktree fallback；隔离机制相同，但它不是 Codex 托管附件。

算法参考：[Mnih 等，Actor-Critic 多步目标](https://arxiv.org/pdf/1602.01783)。AI 用于实现、测试、核验、分析和报告撰写。

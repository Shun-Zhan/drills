# DRiLLS

基于 [官方 DRiLLS](https://github.com/scale-lab/DRiLLS) 的 FPGA 版本，使用 PyTorch CPU 训练 A2C，搜索满足深度约束且 LUT 数较少的 ABC 综合序列。

## 环境

- Linux（已在 Ubuntu 20.04 x86_64 验证），无需 GPU。
- Python 3.9。
- PyTorch 2.8.0+cpu、NumPy、PyYAML、filelock，由 `requirements.txt` 安装。
- [ABC](https://github.com/berkeley-abc/abc) 和 [Yosys](https://github.com/YosysHQ/yosys)（本次实验使用 ABC 1.01、Yosys 0.44；安装方法见各仓库 README）。

使用任意 Python 3.9 环境（如 Conda、venv 或已有环境），在项目目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

ABC 和 Yosys 的安装方式及编译依赖见上面的官方仓库链接。安装完成后配置可执行文件路径。

在 `params.yml` 中设置工具路径：

```yaml
runtime:
  abc_binary: /你的工具目录/bin/yosys-abc
  yosys_binary: /你的工具目录/bin/yosys
  output_dir: results
  workers: 3
```

如果工具已在 PATH 中，可直接填写 `yosys-abc` 和 `yosys`。

## 配置与运行

所有实验参数均在 `params.yml` 中，可自行修改。默认运行 int2float、i2c、max，使用种子 0、1、2，每次训练 100 轮、每轮 10 步。输入电路已包含在 `benchmarks/` 中。

```bash
# 运行 resyn2 基线和全部训练搜索
python drills.py train fpga

# 续训至 params.yml 中设置的总轮数
python drills.py train fpga --resume

# 仅运行 resyn2 基线
python drills.py baseline fpga

# 重新生成结果汇总
python drills.py report fpga
```

使用其他配置时，在 `fpga` 后追加 YAML 路径。相对路径以配置文件所在目录为基准。

结果默认保存到 `results/`：`results.md` 查看各个种子的 LUTs、levels 和训练耗时；各电路的 `seed-*` 目录保存检查点、最优网表和逐步日志。

## 参考基线

采用 LUT6 映射；参考运行使用 100 episodes × 10 iterations、种子 0/1/2。参考结果取三个种子各自训练搜索最优可行解的均值。

| 电路 | resyn2 levels | resyn2 LUTs | 参考结果 levels | 参考结果 LUTs |
|---|---:|---:|---:|---:|
| int2float | 3 | 48 | 3 | 44.00 |
| i2c | 4 | 322 | 4 | 303.67 |
| max | 41 | 777 | 41 | 772.33 |

## 实验评审

参考结果是训练搜索中见过的最好可行解，不是独立测试成绩。设计是否可靠、这些数字说明什么、下一步改什么，见 [docs/fpga-experiment-review.md](docs/fpga-experiment-review.md)。

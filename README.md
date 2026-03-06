# DreamerV3 Torch Lite（精简版）

这是一个从原始版本改造的 **PyTorch 精简实现**，目标是：

- 依赖少，环境容易搭建
- 结构清晰，便于二次开发
- 只使用 Conda 构建环境，不使用 Docker

当前默认在 `CartPole-v1` 上训练，可直接运行。

## 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate dreamerv3-torch
```

如果你更新了 `requirements.txt`，可执行：

```bash
conda env update -f environment.yml --prune
```

## 2. 安装项目

```bash
pip install -e .
```

## 3. 开始训练

```bash
python -m dreamerv3.main --env CartPole-v1 --steps 5000
```

## 4. 查看可用环境

```bash
python -m dreamerv3.main --list-envs
```

当前精简版已验证：

- `CartPole-v1`
- `Acrobot-v1`
- `MountainCar-v0`

注意：

- 当前实现仅支持 `Box` 观测 + `Discrete` 动作。
- `Minecraft/MineRL` 不在本精简版支持范围内（需要额外环境封装与大规模依赖）。

## 5. 训练输出

默认日志目录：

```text
logs/时间戳/
```

会保存：

- `config.yaml`：训练配置
- `dreamer_lite.pt`：模型权重
- `metrics.csv`：训练全量指标（画图输入）

## 6. 画曲线

```bash
python plot.py --metrics logs/时间戳/metrics.csv
```

可选参数：

```bash
python plot.py \
  --metrics logs/时间戳/metrics.csv \
  --out logs/时间戳/curves.png \
  --smooth 20
```

默认会输出 4 个子图：

- Episode Reward（原始+平滑）
- Train Loss（total/world/actor/value）
- Runtime Metrics（episode_len/fps/buffer_size）
- Optimization Signals（entropy/grad_norm/adv_mean/target_mean/value_mean/reward_mean）

## 7. 核心目录

```text
dreamerv3/
  main.py      # 训练入口 + metrics.csv 输出
  agent.py     # 智能体与优化逻辑
  rssm.py      # 世界模型与回放池
  configs.yaml # 简化配置
plot.py        # 训练曲线绘图脚本
```

## 8. 说明

- 本项目环境构建方式为 Conda。
- 已移除 Docker 相关文件，统一使用 Conda。
- 不依赖 JAX、Ninjax、Optax。

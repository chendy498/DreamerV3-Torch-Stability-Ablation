# DreamerV3 Torch Lite（精简版）

这是一个基于 PyTorch 的 DreamerLite 实现，已精简为单一 Conda 工作流，支持 Gym、Atari57、MuJoCo。

## 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate dreamerv3-torch
```

## 2. 安装项目

```bash
pip install -e .
```

## 3. 训练

```bash
python -m dreamerv3.main --env CartPole-v1 --steps 5000
```

## 4. 查看支持环境

```bash
python -m dreamerv3.main --list-envs
```

当前包含三类环境：

- Gym 常用环境（如 `CartPole-v1`、`Pendulum-v1`）
- Atari57（`ALE/*-v5` 共 57 个）
- MuJoCo 常用环境（如 `HalfCheetah-v4`、`Walker2d-v4`）

说明：

- 观测空间需为 `Box`。
- 动作空间支持 `Discrete` 和 `Box`（连续动作会自动映射）。
- Atari 首次使用前建议执行一次：`AutoROM --accept-license`。
- `Minecraft/MineRL` 仍不在当前精简版支持范围内。

## 5. 稳定性模块

当前 Torch 版本已接入以下稳定性工程：

- `KL balance`（动力学 KL 与表示 KL 按比例混合）
- `free bits`（KL 下限裁剪）
- `dynamics/representation KL` 拆分监控
- 离散策略 `unimix` 混合，降低策略塌缩风险

对应可调参数在 `dreamerv3/configs.yaml`：

- `kl_balance`
- `kl_free_nats`
- `kl_scale`
- `kl_dyn_scale`
- `kl_rep_scale`
- `unimix_ratio`

## 6. 训练输出

默认日志目录：

```text
logs/时间戳/
```

每次训练会保存：

- `config.yaml`：配置快照
- `dreamer_lite.pt`：模型参数
- `metrics.csv`：可视化输入

## 7. 画图

```bash
python plot.py --metrics logs/时间戳/metrics.csv
```

也可指定输出与平滑窗口：

```bash
python plot.py \
  --metrics logs/时间戳/metrics.csv \
  --out logs/时间戳/curves.png \
  --smooth 20
```

默认输出：

- Episode Reward
- Train Losses（total/world/reward/actor/value）
- KL Stabilization（kl_loss/kl_dyn/kl_rep/free_bits 后 KL）
- Runtime Metrics（episode_len/fps/buffer_size）
- Optimization Signals（entropy/grad_norm/adv_mean/target_mean/value_mean/reward_mean）

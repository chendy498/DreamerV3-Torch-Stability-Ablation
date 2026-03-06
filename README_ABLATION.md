# DreamerV3 Torch 稳定性消融对比实验

项目名建议：**DreamerV3-Torch-Stability-Ablation**

本项目的目标是做一个可复现的**消融对比实验**：

- 对照组：关闭稳定性模块（No-Stable）
- 实验组：开启稳定性模块（Stable）

用于观察 `KL balance / free bits / KL拆分 / unimix` 对训练稳定性和收敛表现的影响。

## 1. 环境准备

```bash
conda env create -f environment.yml
conda activate dreamerv3-torch
pip install -e .
```

## 2. 运行对照组（不加稳定性）

```bash
python -m dreamerv3.main --config dreamerv3/configs_no_stable.yaml --env CartPole-v1 --steps 3000 --device cpu --logdir runs_ablation/no_stable
```

## 3. 运行实验组（加稳定性）

```bash
python -m dreamerv3.main --config dreamerv3/configs_stable.yaml --env CartPole-v1 --steps 3000 --device cpu --logdir runs_ablation/stable
```

## 4. 画图（分别生成）

### 4.1 对照组曲线

```powershell
$run=(Get-ChildItem .\runs_ablation\no_stable | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName; python .\plot.py --metrics "$run\metrics.csv" --out "$run\curves.png" --smooth 20; Write-Host "No-Stable 结果: $run"
```

### 4.2 实验组曲线

```powershell
$run=(Get-ChildItem .\runs_ablation\stable | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName; python .\plot.py --metrics "$run\metrics.csv" --out "$run\curves.png" --smooth 20; Write-Host "Stable 结果: $run"
```

## 5. 重点对比哪些指标

- `Episode Reward`：收敛速度、后期上限、波动幅度
- `Train Losses`：是否出现尖峰爆炸
- `KL Stabilization`：`kl_dyn / kl_rep / free_bits 后 KL` 是否更平滑
- `Optimization Signals`：`grad_norm` 是否出现大规模尖峰

## 6. 配置文件说明

- 稳定性开启配置：`dreamerv3/configs_stable.yaml`
- 稳定性关闭配置：`dreamerv3/configs_no_stable.yaml`

两者除稳定性参数外保持一致，保证对比公平。

## 7. 实验结论模板（可直接填）

- 在相同 `env/seed/steps` 下，Stable 组相较 No-Stable：
- `reward`：更快 / 更慢 / 无明显差异
- `loss`：更稳 / 更抖
- `grad_norm`：尖峰更少 / 更多
- 结论：稳定性模块对该任务是正收益 / 中性 / 负收益

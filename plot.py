"""最小可视化脚本：读取 metrics.csv 并生成奖励/损失曲线。"""

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="绘制 DreamerLite 训练曲线")
    parser.add_argument("--metrics", type=str, required=True, help="metrics.csv 路径")
    parser.add_argument("--out", type=str, default=None, help="输出图片路径，默认同目录 curves.png")
    parser.add_argument("--smooth", type=int, default=20, help="滑动平均窗口")
    return parser.parse_args()


def to_float(x: str) -> float:
    """把字符串安全转换为 float。"""
    if x is None or x == "":
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """计算滑动平均，便于查看趋势。"""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_metrics(path: pathlib.Path):
    """读取 CSV 并拆分 episode 与 train 两类数据。"""
    episode_steps, episode_rewards, episode_lens = [], [], []
    train_steps = []
    train_metrics = {
        "loss": [],
        "world_loss": [],
        "reward_loss": [],
        "kl_loss": [],
        "kl_dyn": [],
        "kl_rep": [],
        "kl_dyn_free": [],
        "kl_rep_free": [],
        "value_loss": [],
        "actor_loss": [],
        "entropy": [],
        "grad_norm": [],
        "adv_mean": [],
        "target_mean": [],
        "value_mean": [],
        "reward_mean": [],
        "fps": [],
        "buffer_size": [],
    }

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = row.get("event", "")
            step = int(float(row["step"])) if row.get("step") else 0

            if event == "episode":
                episode_steps.append(step)
                episode_rewards.append(to_float(row.get("episode_reward", "")))
                episode_lens.append(to_float(row.get("episode_len", "")))
            elif event == "train":
                train_steps.append(step)
                for key in train_metrics:
                    train_metrics[key].append(to_float(row.get(key, "")))

    return (
        np.array(episode_steps),
        np.array(episode_rewards),
        np.array(episode_lens),
        np.array(train_steps),
        {k: np.array(v) for k, v in train_metrics.items()},
    )


def main() -> None:
    """绘图主函数。"""
    args = parse_args()
    metrics_path = pathlib.Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到 metrics 文件: {metrics_path}")

    out = pathlib.Path(args.out) if args.out else metrics_path.parent / "curves.png"

    ep_steps, ep_rewards, ep_lens, tr_steps, tr = load_metrics(metrics_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    axes[0, 0].plot(ep_steps, ep_rewards, alpha=0.35, label="raw_reward")
    if len(ep_rewards) > 0:
        axes[0, 0].plot(ep_steps, moving_average(ep_rewards, args.smooth), linewidth=2, label=f"smoothed_reward({args.smooth})")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(tr_steps, tr["loss"], label="total")
    axes[0, 1].plot(tr_steps, tr["world_loss"], label="world")
    axes[0, 1].plot(tr_steps, tr["reward_loss"], label="reward")
    axes[0, 1].plot(tr_steps, tr["actor_loss"], label="actor")
    axes[0, 1].plot(tr_steps, tr["value_loss"], label="value")
    axes[0, 1].set_title("Train Losses")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].plot(tr_steps, tr["kl_loss"], label="kl_loss")
    axes[0, 2].plot(tr_steps, tr["kl_dyn"], label="kl_dyn")
    axes[0, 2].plot(tr_steps, tr["kl_rep"], label="kl_rep")
    axes[0, 2].plot(tr_steps, tr["kl_dyn_free"], label="kl_dyn_free")
    axes[0, 2].plot(tr_steps, tr["kl_rep_free"], label="kl_rep_free")
    axes[0, 2].set_title("KL Stabilization")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("KL")
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].legend()

    axes[1, 0].plot(ep_steps, ep_lens, label="episode_len")
    axes[1, 0].plot(tr_steps, tr["fps"], label="fps")
    axes[1, 0].plot(tr_steps, tr["buffer_size"], label="buffer_size")
    axes[1, 0].set_title("Runtime Metrics")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(tr_steps, tr["entropy"], label="entropy")
    axes[1, 1].plot(tr_steps, tr["grad_norm"], label="grad_norm")
    axes[1, 1].plot(tr_steps, tr["adv_mean"], label="adv_mean")
    axes[1, 1].plot(tr_steps, tr["target_mean"], label="target_mean")
    axes[1, 1].plot(tr_steps, tr["value_mean"], label="value_mean")
    axes[1, 1].plot(tr_steps, tr["reward_mean"], label="reward_mean")
    axes[1, 1].set_title("Optimization Signals")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"曲线图已保存到: {out}")


if __name__ == "__main__":
    main()

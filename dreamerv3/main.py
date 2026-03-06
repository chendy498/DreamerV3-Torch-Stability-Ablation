"""PyTorch 版本的简化 DreamerV3 入口。"""

import argparse
import csv
import pathlib
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium import spaces

from .agent import DreamerLiteAgent
from .rssm import ReplayBuffer

# 当前精简实现已验证可运行的环境（来自 gymnasium[classic-control]）。
SUPPORTED_ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
]


class MetricsLogger:
    """把训练过程中的关键指标写入 CSV，方便后续画图与对比。"""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.fieldnames = [
            "time_sec",
            "step",
            "episode",
            "buffer_size",
            "fps",
            "event",
            "episode_reward",
            "episode_len",
            "loss",
            "world_loss",
            "value_loss",
            "actor_loss",
            "entropy",
            "grad_norm",
            "target_mean",
            "value_mean",
            "adv_mean",
            "reward_mean",
        ]
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})


def load_config(config_path: pathlib.Path) -> dict:
    """读取 YAML 配置文件。"""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="简化版 DreamerV3 (PyTorch)")
    parser.add_argument("--config", type=str, default="dreamerv3/configs.yaml", help="配置文件路径")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium 环境名")
    parser.add_argument("--steps", type=int, default=None, help="总训练步数，覆盖配置")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，覆盖配置")
    parser.add_argument("--logdir", type=str, default="logs", help="日志输出目录")
    parser.add_argument("--device", type=str, default=None, help="设备，cuda 或 cpu")
    parser.add_argument("--list-envs", action="store_true", help="打印当前已验证可运行的环境")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子，保证实验可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_env(env) -> tuple[int, int]:
    """检查环境是否符合当前最小实现要求并返回维度。"""
    if not isinstance(env.observation_space, spaces.Box):
        raise ValueError("当前仅支持 Box 观测空间。")
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError("当前仅支持 Discrete 动作空间。")

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    return obs_dim, act_dim


def main() -> None:
    """执行训练主循环。"""
    args = parse_args()
    if args.list_envs:
        print("当前已验证可运行环境：")
        for name in SUPPORTED_ENVS:
            print(f"- {name}")
        print("说明：Minecraft/MineRL 不在当前精简版支持范围内。")
        return

    config = load_config(pathlib.Path(args.config))

    if args.steps is not None:
        config["train_steps"] = args.steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device

    seed = int(config["seed"])
    set_seed(seed)

    env = gym.make(args.env)
    env.action_space.seed(seed)

    obs_dim, act_dim = validate_env(env)

    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DreamerLiteAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        lr=float(config["lr"]),
        gamma=float(config["gamma"]),
        world_coef=float(config["world_coef"]),
        value_coef=float(config["value_coef"]),
        actor_coef=float(config["actor_coef"]),
        ent_coef=float(config["ent_coef"]),
        hidden_dim=int(config["hidden_dim"]),
        feature_dim=int(config["feature_dim"]),
    )

    buffer = ReplayBuffer(
        capacity=int(config["buffer_size"]),
        obs_dim=obs_dim,
        device=device,
    )

    logdir = pathlib.Path(args.logdir) / time.strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)

    with (logdir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    metrics_logger = MetricsLogger(logdir / "metrics.csv")

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    start_time = time.time()

    for step in range(1, int(config["train_steps"]) + 1):
        action = agent.select_action(obs, explore=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        buffer.add(obs, action, reward, done, next_obs)

        obs = next_obs
        episode_reward += float(reward)
        episode_steps += 1

        elapsed = max(time.time() - start_time, 1e-6)
        fps = step / elapsed

        if done:
            episode_count += 1
            metrics_logger.write(
                {
                    "time_sec": round(elapsed, 3),
                    "step": step,
                    "episode": episode_count,
                    "buffer_size": len(buffer),
                    "fps": round(fps, 2),
                    "event": "episode",
                    "episode_reward": round(episode_reward, 6),
                    "episode_len": episode_steps,
                }
            )
            if episode_count % int(config["log_episode_every"]) == 0:
                print(
                    f"[Episode {episode_count}] "
                    f"step={step} reward={episode_reward:.2f} len={episode_steps} fps={fps:.1f}"
                )
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        if (
            step >= int(config["warmup_steps"])
            and step % int(config["update_every"]) == 0
            and len(buffer) >= int(config["batch_size"])
        ):
            metrics = agent.update(buffer, batch_size=int(config["batch_size"]))
            metrics_logger.write(
                {
                    "time_sec": round(elapsed, 3),
                    "step": step,
                    "episode": episode_count,
                    "buffer_size": len(buffer),
                    "fps": round(fps, 2),
                    "event": "train",
                    **{k: round(v, 6) for k, v in metrics.items()},
                }
            )
            if step % int(config["log_train_every"]) == 0:
                print(
                    "[Train] "
                    f"step={step} "
                    f"loss={metrics['loss']:.4f} "
                    f"world={metrics['world_loss']:.4f} "
                    f"actor={metrics['actor_loss']:.4f} "
                    f"value={metrics['value_loss']:.4f} "
                    f"fps={fps:.1f}"
                )

    model_path = logdir / "dreamer_lite.pt"
    torch.save(
        {
            "agent": asdict(agent.export_config()),
            "state_dict": agent.state_dict(),
        },
        model_path,
    )
    print(f"训练结束，模型已保存到: {model_path}")
    print(f"训练指标已保存到: {logdir / 'metrics.csv'}")

    env.close()


if __name__ == "__main__":
    main()

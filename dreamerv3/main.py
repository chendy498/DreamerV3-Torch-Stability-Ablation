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

# Gym 常用环境（当前实现可直接使用）。
SUPPORTED_GYM_ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
]

# Atari57 基准环境（Gymnasium Atari + ALE v5 命名）。
ATARI57_ENVS = [
    "ALE/Alien-v5",
    "ALE/Amidar-v5",
    "ALE/Assault-v5",
    "ALE/Asterix-v5",
    "ALE/Asteroids-v5",
    "ALE/Atlantis-v5",
    "ALE/BankHeist-v5",
    "ALE/BattleZone-v5",
    "ALE/BeamRider-v5",
    "ALE/Berzerk-v5",
    "ALE/Bowling-v5",
    "ALE/Boxing-v5",
    "ALE/Breakout-v5",
    "ALE/Centipede-v5",
    "ALE/ChopperCommand-v5",
    "ALE/CrazyClimber-v5",
    "ALE/Defender-v5",
    "ALE/DemonAttack-v5",
    "ALE/DoubleDunk-v5",
    "ALE/Enduro-v5",
    "ALE/FishingDerby-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Gopher-v5",
    "ALE/Gravitar-v5",
    "ALE/Hero-v5",
    "ALE/IceHockey-v5",
    "ALE/Jamesbond-v5",
    "ALE/Kangaroo-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MontezumaRevenge-v5",
    "ALE/MsPacman-v5",
    "ALE/NameThisGame-v5",
    "ALE/Phoenix-v5",
    "ALE/Pitfall-v5",
    "ALE/Pooyan-v5",
    "ALE/Pong-v5",
    "ALE/PrivateEye-v5",
    "ALE/Qbert-v5",
    "ALE/Riverraid-v5",
    "ALE/RoadRunner-v5",
    "ALE/Robotank-v5",
    "ALE/Seaquest-v5",
    "ALE/Skiing-v5",
    "ALE/Solaris-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/StarGunner-v5",
    "ALE/Tennis-v5",
    "ALE/TimePilot-v5",
    "ALE/Tutankham-v5",
    "ALE/UpNDown-v5",
    "ALE/Venture-v5",
    "ALE/VideoPinball-v5",
    "ALE/WizardOfWor-v5",
    "ALE/YarsRevenge-v5",
    "ALE/Zaxxon-v5",
]

# MuJoCo 常见任务（Gymnasium MuJoCo 套件）。
MUJOCO_ENVS = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "HumanoidStandup-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    "Pusher-v4",
    "Reacher-v4",
    "Swimmer-v4",
    "Walker2d-v4",
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
            "reward_loss",
            "kl_loss",
            "kl_dyn",
            "kl_rep",
            "kl_dyn_free",
            "kl_rep_free",
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
    parser.add_argument("--list-envs", action="store_true", help="打印当前支持环境列表")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子，保证实验可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """统一预处理观测：拉平并转换为 float32。"""
    obs = np.asarray(obs)
    if obs.dtype == np.uint8:
        obs = obs.astype(np.float32) / 255.0
    else:
        obs = obs.astype(np.float32)
    return obs.reshape(-1)


def denorm_action(norm_action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """把 [-1, 1] 范围动作映射回环境动作空间。"""
    return low + (norm_action + 1.0) * 0.5 * (high - low)


def validate_env(env) -> tuple[int, int, str]:
    """检查环境是否符合当前实现要求并返回维度与动作类型。"""
    if not isinstance(env.observation_space, spaces.Box):
        raise ValueError("当前仅支持 Box 观测空间。")

    obs_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, spaces.Discrete):
        return obs_dim, int(env.action_space.n), "discrete"
    if isinstance(env.action_space, spaces.Box):
        return obs_dim, int(np.prod(env.action_space.shape)), "continuous"
    raise ValueError("当前仅支持 Discrete 或 Box 动作空间。")


def print_supported_envs() -> None:
    """打印环境清单。"""
    print("Gym 常用环境：")
    for name in SUPPORTED_GYM_ENVS:
        print(f"- {name}")
    print(f"\nAtari57 环境（共 {len(ATARI57_ENVS)} 个）：")
    for name in ATARI57_ENVS:
        print(f"- {name}")
    print("\nMuJoCo 常用环境：")
    for name in MUJOCO_ENVS:
        print(f"- {name}")
    print("\n说明：Minecraft/MineRL 不在当前精简版支持范围内。")


def create_run_dir(base_dir: pathlib.Path) -> pathlib.Path:
    """创建不会冲突的日志目录。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / stamp
    idx = 1
    while run_dir.exists():
        run_dir = base_dir / f"{stamp}-{idx}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    """执行训练主循环。"""
    args = parse_args()
    if args.list_envs:
        print_supported_envs()
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

    obs_dim, act_dim, action_type = validate_env(env)

    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DreamerLiteAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_type=action_type,
        device=device,
        lr=float(config["lr"]),
        gamma=float(config["gamma"]),
        world_coef=float(config["world_coef"]),
        value_coef=float(config["value_coef"]),
        actor_coef=float(config["actor_coef"]),
        ent_coef=float(config["ent_coef"]),
        kl_balance=float(config["kl_balance"]),
        kl_free_nats=float(config["kl_free_nats"]),
        kl_scale=float(config["kl_scale"]),
        kl_dyn_scale=float(config["kl_dyn_scale"]),
        kl_rep_scale=float(config["kl_rep_scale"]),
        unimix_ratio=float(config["unimix_ratio"]),
        hidden_dim=int(config["hidden_dim"]),
        feature_dim=int(config["feature_dim"]),
    )

    buffer = ReplayBuffer(
        capacity=int(config["buffer_size"]),
        obs_dim=obs_dim,
        action_type=action_type,
        act_dim=act_dim,
        device=device,
    )

    logdir = create_run_dir(pathlib.Path(args.logdir))

    with (logdir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({**config, "action_type": action_type}, f, allow_unicode=True)

    metrics_logger = MetricsLogger(logdir / "metrics.csv")

    raw_obs, _ = env.reset(seed=seed)
    obs = preprocess_obs(raw_obs)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    start_time = time.time()

    action_low = None
    action_high = None
    if action_type == "continuous":
        action_low = env.action_space.low.reshape(-1).astype(np.float32)
        action_high = env.action_space.high.reshape(-1).astype(np.float32)

    for step in range(1, int(config["train_steps"]) + 1):
        model_action = agent.select_action(obs, explore=True)
        if action_type == "discrete":
            env_action = model_action
            buffer_action = model_action
        else:
            env_action = denorm_action(model_action, action_low, action_high).reshape(env.action_space.shape)
            env_action = np.clip(env_action, env.action_space.low, env.action_space.high)
            buffer_action = model_action

        next_raw_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = bool(terminated or truncated)
        next_obs = preprocess_obs(next_raw_obs)

        buffer.add(obs, buffer_action, reward, done, next_obs)

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
            raw_obs, _ = env.reset()
            obs = preprocess_obs(raw_obs)
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
                    f"kl={metrics['kl_loss']:.4f} "
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

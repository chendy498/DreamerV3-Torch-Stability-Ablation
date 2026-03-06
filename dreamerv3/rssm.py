"""简化版世界模型与经验回放。"""

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """基础 MLP 模块。"""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModel(nn.Module):
    """世界模型与策略价值头。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, feature_dim: int, action_type: str):
        super().__init__()
        self.act_dim = act_dim
        self.feature_dim = feature_dim
        self.action_type = action_type

        # 后验网络 q(z_t | o_t)
        self.posterior_net = MLP(obs_dim, feature_dim * 2, hidden_dim)
        # 先验网络 p(z_{t+1} | z_t, a_t)
        self.prior_net = MLP(feature_dim + act_dim, feature_dim * 2, hidden_dim)

        self.reward_head = MLP(feature_dim + act_dim, 1, hidden_dim)
        actor_out_dim = act_dim * 2 if action_type == "continuous" else act_dim
        self.actor_head = MLP(feature_dim, actor_out_dim, hidden_dim)
        self.value_head = MLP(feature_dim, 1, hidden_dim)

    def _split_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把网络输出拆成高斯均值和对数方差。"""
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        return mean, log_std

    def posterior_stats(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算后验分布参数。"""
        return self._split_stats(self.posterior_net(obs))

    def _merge_feat_action(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """将动作向量与特征拼接。"""
        if self.action_type == "discrete":
            action_vec = torch.nn.functional.one_hot(action.long(), num_classes=self.act_dim).float()
        else:
            action_vec = action.float()
        return torch.cat([feat, action_vec], dim=-1)

    def prior_stats(self, feat: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算先验分布参数。"""
        x = self._merge_feat_action(feat, action)
        return self._split_stats(self.prior_net(x))

    def sample_latent(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """使用重参数化技巧采样潜变量。"""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, obs: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """把观测编码到潜变量空间。"""
        mean, log_std = self.posterior_stats(obs)
        if sample:
            return self.sample_latent(mean, log_std)
        return mean

    def predict_reward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测即时奖励。"""
        x = self._merge_feat_action(feat, action)
        return self.reward_head(x).squeeze(-1)

    def actor(self, feat: torch.Tensor) -> torch.Tensor:
        """输出策略网络原始结果。"""
        return self.actor_head(feat)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        """输出状态价值。"""
        return self.value_head(feat)


class ReplayBuffer:
    """固定容量经验回放池。"""

    def __init__(self, capacity: int, obs_dim: int, action_type: str, act_dim: int, device: str):
        self.capacity = capacity
        self.action_type = action_type
        self.act_dim = act_dim
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        if action_type == "discrete":
            self.action = np.zeros((capacity,), dtype=np.int64)
        else:
            self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, done, next_obs) -> None:
        """写入一条转移。"""
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.done[self.pos] = float(done)
        self.next_obs[self.pos] = next_obs
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int):
        """随机采样一个批次并转换到目标设备。"""
        size = len(self)
        indices = np.random.randint(0, size, size=batch_size)
        obs = torch.tensor(self.obs[indices], dtype=torch.float32, device=self.device)
        action_dtype = torch.int64 if self.action_type == "discrete" else torch.float32
        action = torch.tensor(self.action[indices], dtype=action_dtype, device=self.device)
        reward = torch.tensor(self.reward[indices], dtype=torch.float32, device=self.device)
        done = torch.tensor(self.done[indices], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(self.next_obs[indices], dtype=torch.float32, device=self.device)
        return obs, action, reward, done, next_obs

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
    """同时包含世界模型与策略价值头。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.act_dim = act_dim

        self.encoder = MLP(obs_dim, feature_dim, hidden_dim)
        self.transition = MLP(feature_dim + act_dim, feature_dim, hidden_dim)
        self.reward = MLP(feature_dim + act_dim, 1, hidden_dim)
        self.actor_head = MLP(feature_dim, act_dim, hidden_dim)
        self.value_head = MLP(feature_dim, 1, hidden_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """把观测编码到特征空间。"""
        return self.encoder(obs)

    def _merge_feat_action(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """将离散动作 one-hot 后与特征拼接。"""
        one_hot = torch.nn.functional.one_hot(action.long(), num_classes=self.act_dim).float()
        return torch.cat([feat, one_hot], dim=-1)

    def predict_next(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一步特征。"""
        x = self._merge_feat_action(feat, action)
        return self.transition(x)

    def predict_reward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测即时奖励。"""
        x = self._merge_feat_action(feat, action)
        return self.reward(x).squeeze(-1)

    def actor(self, feat: torch.Tensor) -> torch.Tensor:
        """输出动作 logits。"""
        return self.actor_head(feat)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        """输出状态价值。"""
        return self.value_head(feat)


class ReplayBuffer:
    """固定容量经验回放池。"""

    def __init__(self, capacity: int, obs_dim: int, device: str):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
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
        action = torch.tensor(self.action[indices], dtype=torch.int64, device=self.device)
        reward = torch.tensor(self.reward[indices], dtype=torch.float32, device=self.device)
        done = torch.tensor(self.done[indices], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(self.next_obs[indices], dtype=torch.float32, device=self.device)
        return obs, action, reward, done, next_obs

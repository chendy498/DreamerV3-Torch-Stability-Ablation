"""简化版 PyTorch 智能体实现。"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .rssm import WorldModel


@dataclass
class AgentConfig:
    """模型超参数。"""

    obs_dim: int
    act_dim: int
    hidden_dim: int
    feature_dim: int
    gamma: float
    world_coef: float
    value_coef: float
    actor_coef: float
    ent_coef: float


class DreamerLiteAgent(nn.Module):
    """简化版 Dreamer 智能体。"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: str,
        lr: float,
        gamma: float,
        world_coef: float,
        value_coef: float,
        actor_coef: float,
        ent_coef: float,
        hidden_dim: int = 256,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.config = AgentConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            gamma=gamma,
            world_coef=world_coef,
            value_coef=value_coef,
            actor_coef=actor_coef,
            ent_coef=ent_coef,
        )
        self.device = torch.device(device)

        self.world = WorldModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
        )

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def export_config(self) -> AgentConfig:
        """导出配置，便于存档。"""
        return self.config

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """根据当前观测选择动作。"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        feature = self.world.encode(obs_t)
        logits = self.world.actor(feature)
        probs = torch.softmax(logits, dim=-1)
        if explore:
            action = torch.multinomial(probs, num_samples=1)
        else:
            action = torch.argmax(probs, dim=-1, keepdim=True)
        return int(action.item())

    def update(self, replay, batch_size: int) -> dict:
        """执行一次参数更新。"""
        obs, action, reward, done, next_obs = replay.sample(batch_size)

        feature = self.world.encode(obs)
        next_feature = self.world.encode(next_obs)
        next_feature_detached = next_feature.detach()

        pred_next_feature = self.world.predict_next(feature, action)
        pred_reward = self.world.predict_reward(feature, action)

        world_loss = F.mse_loss(pred_next_feature, next_feature_detached)
        world_loss = world_loss + F.mse_loss(pred_reward, reward)

        value = self.world.value(feature).squeeze(-1)
        with torch.no_grad():
            next_value = self.world.value(next_feature_detached).squeeze(-1)
            target = reward + self.config.gamma * (1.0 - done) * next_value

        value_loss = F.mse_loss(value, target)

        logits = self.world.actor(feature)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        chosen_log_prob = log_probs.gather(1, action.long().unsqueeze(-1)).squeeze(-1)

        advantage = (target - value).detach()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        actor_loss = -(chosen_log_prob * advantage).mean() - self.config.ent_coef * entropy

        total_loss = (
            self.config.world_coef * world_loss
            + self.config.value_coef * value_loss
            + self.config.actor_coef * actor_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "world_loss": float(world_loss.item()),
            "value_loss": float(value_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
            "target_mean": float(target.mean().item()),
            "value_mean": float(value.mean().item()),
            "adv_mean": float(advantage.mean().item()),
            "reward_mean": float(reward.mean().item()),
        }

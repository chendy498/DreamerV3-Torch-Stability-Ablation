"""简化版 PyTorch 智能体实现。"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from .rssm import WorldModel


@dataclass
class AgentConfig:
    """模型超参数。"""

    obs_dim: int
    act_dim: int
    action_type: str
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
        action_type: str,
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
            action_type=action_type,
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
            action_type=action_type,
        )

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def export_config(self) -> AgentConfig:
        """导出配置，便于存档。"""
        return self.config

    def _continuous_dist(self, actor_out: torch.Tensor) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        """构建连续动作分布。"""
        mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        return Normal(mean, std), mean, log_std

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True):
        """根据当前观测选择动作。"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        feature = self.world.encode(obs_t)
        actor_out = self.world.actor(feature)

        if self.config.action_type == "discrete":
            probs = torch.softmax(actor_out, dim=-1)
            if explore:
                action = torch.multinomial(probs, num_samples=1)
            else:
                action = torch.argmax(probs, dim=-1, keepdim=True)
            return int(action.item())

        dist, mean, _ = self._continuous_dist(actor_out)
        raw_action = dist.sample() if explore else mean
        # 连续动作统一压到 [-1, 1]，由主循环再映射到环境动作范围。
        norm_action = torch.tanh(raw_action)
        return norm_action.squeeze(0).cpu().numpy().astype(np.float32)

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

        actor_out = self.world.actor(feature)
        if self.config.action_type == "discrete":
            log_probs = torch.log_softmax(actor_out, dim=-1)
            probs = torch.softmax(actor_out, dim=-1)
            chosen_log_prob = log_probs.gather(1, action.long().unsqueeze(-1)).squeeze(-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        else:
            dist, mean, log_std = self._continuous_dist(actor_out)
            chosen_log_prob = dist.log_prob(action).sum(dim=-1)
            # 高斯分布熵：每一维的熵累加。
            entropy = (0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std).sum(dim=-1).mean()

        advantage = (target - value).detach()
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

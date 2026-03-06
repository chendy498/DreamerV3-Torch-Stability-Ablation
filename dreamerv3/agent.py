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
    kl_balance: float
    kl_free_nats: float
    kl_scale: float
    kl_dyn_scale: float
    kl_rep_scale: float
    unimix_ratio: float


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
        kl_balance: float,
        kl_free_nats: float,
        kl_scale: float,
        kl_dyn_scale: float,
        kl_rep_scale: float,
        unimix_ratio: float,
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
            kl_balance=kl_balance,
            kl_free_nats=kl_free_nats,
            kl_scale=kl_scale,
            kl_dyn_scale=kl_dyn_scale,
            kl_rep_scale=kl_rep_scale,
            unimix_ratio=unimix_ratio,
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

    def _unimix_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """对离散策略做均匀混合，避免过早塌缩。"""
        mix = float(self.config.unimix_ratio)
        if mix <= 0.0:
            return probs
        uniform = torch.full_like(probs, fill_value=1.0 / probs.shape[-1])
        return (1.0 - mix) * probs + mix * uniform

    def _diag_kl(
        self,
        mean_p: torch.Tensor,
        log_std_p: torch.Tensor,
        mean_q: torch.Tensor,
        log_std_q: torch.Tensor,
    ) -> torch.Tensor:
        """计算 KL(N_p || N_q) 并在特征维做求和。"""
        var_p = torch.exp(2.0 * log_std_p)
        var_q = torch.exp(2.0 * log_std_q)
        kl = log_std_q - log_std_p + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q) - 0.5
        return kl.sum(dim=-1)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True):
        """根据当前观测选择动作。"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        feature = self.world.encode(obs_t, sample=False)
        actor_out = self.world.actor(feature)

        if self.config.action_type == "discrete":
            probs = torch.softmax(actor_out, dim=-1)
            probs = self._unimix_probs(probs)
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

        feat = self.world.encode(obs, sample=False)
        next_post_mean, next_post_log_std = self.world.posterior_stats(next_obs)
        next_feat = self.world.sample_latent(next_post_mean, next_post_log_std)

        prior_mean, prior_log_std = self.world.prior_stats(feat, action)
        pred_reward = self.world.predict_reward(feat, action)

        reward_loss = F.mse_loss(pred_reward, reward)

        dyn_kl = self._diag_kl(
            next_post_mean.detach(),
            next_post_log_std.detach(),
            prior_mean,
            prior_log_std,
        ).mean()
        rep_kl = self._diag_kl(
            next_post_mean,
            next_post_log_std,
            prior_mean.detach(),
            prior_log_std.detach(),
        ).mean()

        dyn_kl_free = torch.clamp(dyn_kl, min=self.config.kl_free_nats)
        rep_kl_free = torch.clamp(rep_kl, min=self.config.kl_free_nats)

        balanced_kl = (
            self.config.kl_balance * self.config.kl_dyn_scale * dyn_kl_free
            + (1.0 - self.config.kl_balance) * self.config.kl_rep_scale * rep_kl_free
        )
        kl_loss = self.config.kl_scale * balanced_kl
        world_loss = reward_loss + kl_loss

        value = self.world.value(feat).squeeze(-1)
        with torch.no_grad():
            next_value = self.world.value(next_feat).squeeze(-1)
            target = reward + self.config.gamma * (1.0 - done) * next_value

        value_loss = F.mse_loss(value, target)

        actor_out = self.world.actor(feat)
        if self.config.action_type == "discrete":
            probs = torch.softmax(actor_out, dim=-1)
            probs = self._unimix_probs(probs)
            log_probs = torch.log(probs + 1e-8)
            chosen_log_prob = log_probs.gather(1, action.long().unsqueeze(-1)).squeeze(-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        else:
            dist, mean, log_std = self._continuous_dist(actor_out)
            chosen_log_prob = dist.log_prob(action).sum(dim=-1)
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
            "reward_loss": float(reward_loss.item()),
            "kl_loss": float(kl_loss.item()),
            "kl_dyn": float(dyn_kl.item()),
            "kl_rep": float(rep_kl.item()),
            "kl_dyn_free": float(dyn_kl_free.item()),
            "kl_rep_free": float(rep_kl_free.item()),
            "value_loss": float(value_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
            "target_mean": float(target.mean().item()),
            "value_mean": float(value.mean().item()),
            "adv_mean": float(advantage.mean().item()),
            "reward_mean": float(reward.mean().item()),
        }

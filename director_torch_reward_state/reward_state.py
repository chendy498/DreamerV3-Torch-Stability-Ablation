"""Reward-state auxiliary heads for Director PyTorch experiments.

The auxiliary is intentionally orthogonal to Director's control loop. It adds
future-return and future-event prediction losses on top of RSSM posterior states,
so A/B/C/D experiments can isolate representation learning from policy changes.
"""

from __future__ import annotations

import torch

from . import torchnets as nets
from . import torchutils


def _cfg(config, key, default=None):
  return getattr(config, key, default) if config is not None else default


class RewardStateAux(torchutils.Module):
  """Predict future reward semantics from latent RSSM states.

  B1 adds only the two losses below to the world-model objective:
    z_t -> H-step discounted return
    z_t -> whether a reward/event occurs in the next H steps
  """

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.horizon = int(_cfg(config, 'horizon', 8))
    self.discount = float(_cfg(config, 'discount', 0.99))
    self.event_threshold = float(_cfg(config, 'event_threshold', 1e-6))
    inputs = list(_cfg(config, 'inputs', ['deter', 'stoch']))
    return_head = dict(_cfg(config, 'return_head', {}))
    event_head = dict(_cfg(config, 'event_head', {}))
    return_head = {'layers': 2, 'units': 256, 'act': 'elu', 'norm': 'layer',
                   'dist': 'symlog', 'outscale': 0.1, 'inputs': inputs, **return_head}
    event_head = {'layers': 2, 'units': 256, 'act': 'elu', 'norm': 'layer',
                  'dist': 'binary', 'outscale': 0.1, 'inputs': inputs, **event_head}
    self.return_head = nets.MLP((), **return_head)
    self.event_head = nets.MLP((), **event_head)

  def targets(self, reward: torch.Tensor, cont: torch.Tensor):
    """Build H-step return and reward-event targets."""
    reward = reward.float()
    cont = cont.float()
    b, _ = reward.shape[:2]
    future_return = torch.zeros_like(reward)
    future_event = torch.zeros_like(reward)
    weight = torch.ones_like(reward)
    for k in range(self.horizon):
      if k == 0:
        rew_k = reward
        cont_k = cont
      else:
        pad = torch.zeros(b, k, device=reward.device, dtype=reward.dtype)
        rew_k = torch.cat([reward[:, k:], pad], 1)
        cont_k = torch.cat([cont[:, k:], pad], 1)
      future_return = future_return + (self.discount ** k) * weight * rew_k
      future_event = torch.maximum(future_event, (rew_k.abs() > self.event_threshold).float())
      weight = weight * cont_k
    return future_return.detach(), future_event.detach()

  def loss(self, state, data):
    target_return, target_event = self.targets(data['reward'], data['cont'])
    ret_dist = self.return_head(state)
    evt_dist = self.event_head(state)
    return_loss = -ret_dist.log_prob(target_return)
    event_loss = -evt_dist.log_prob(target_event)
    metrics = {
      'rs_return_target_mean': target_return.mean().detach(),
      'rs_return_target_std': target_return.std().detach(),
      'rs_event_rate': target_event.mean().detach(),
      'rs_return_pred_mean': ret_dist.mean().mean().detach(),
      'rs_event_pred_mean': evt_dist.mean().mean().detach(),
      'rs_return_loss_mean': return_loss.mean().detach(),
      'rs_event_loss_mean': event_loss.mean().detach(),
    }
    return {'rs_return': return_loss, 'rs_event': event_loss}, metrics

  def score_state(self, state):
    """Return a scalar task-semantics score for goal reranking/adaptation."""
    ret = self.return_head(state).mean()
    evt = self.event_head(state).mean()
    alpha = float(_cfg(self.config, 'goal_return_scale', 1.0))
    beta = float(_cfg(self.config, 'goal_event_scale', 1.0))
    return alpha * ret + beta * evt

  def event_prob(self, state):
    return self.event_head(state).mean()

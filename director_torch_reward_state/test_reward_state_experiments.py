from types import SimpleNamespace

import torch
import yaml

import reward_state


def cfg(**kw):
  base = dict(
    horizon=3,
    discount=0.9,
    event_threshold=0.01,
    return_scale=0.1,
    event_scale=0.1,
    goal_return_scale=1.0,
    goal_event_scale=1.0,
    inputs=['deter', 'stoch'],
    return_head={'layers': 1, 'units': 8, 'act': 'elu', 'norm': 'none', 'dist': 'symlog', 'inputs': ['deter', 'stoch']},
    event_head={'layers': 1, 'units': 8, 'act': 'elu', 'norm': 'none', 'dist': 'binary', 'inputs': ['deter', 'stoch']},
  )
  base.update(kw)
  return SimpleNamespace(**base)


def test_reward_state_targets_shape_and_event():
  aux = reward_state.RewardStateAux(cfg())
  reward = torch.tensor([[0.0, 1.0, 0.0, 2.0]])
  cont = torch.ones_like(reward)
  ret, event = aux.targets(reward, cont)
  assert ret.shape == reward.shape
  assert event.shape == reward.shape
  assert event[0, 0] == 1
  assert event[0, -1] == 1


def test_experiment_configs_are_staged():
  with open('configs_torch.yaml') as f:
    data = yaml.safe_load(f)
  assert data['torch_baseline']['reward_state']['enabled'] is False
  assert data['rs_aux']['reward_state']['enabled'] is True
  assert data['rs_aux']['rs_goal']['enabled'] is False
  assert data['rs_goal']['rs_goal']['enabled'] is True
  assert data['rs_adapt']['rs_adapt']['enabled'] is True

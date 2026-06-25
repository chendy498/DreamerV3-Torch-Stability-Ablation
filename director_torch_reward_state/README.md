# Director Torch Reward-State Port

This folder is reserved for the Director PyTorch + reward-state experiment line.

The implementation is organized as staged A/B/C/D variants:

| Variant | Config overlay | Meaning |
| --- | --- | --- |
| A | `torch_baseline` | PyTorch Director baseline: world model + manager + worker. |
| B | `torch_baseline,rs_aux` | Adds reward-state auxiliary heads only. |
| C | `torch_baseline,rs_aux,rs_goal` | Adds reward-state goal candidate reranking. |
| D | `torch_baseline,rs_aux,rs_goal,rs_adapt` | Adds adaptive goal refresh based on reward-state signals. |

The full code package was generated as `director_torch_reward_state_port.zip` in the ChatGPT artifact output. Apply it at the root of a fork/copy of `danijar/director` so that files land under `embodied/agents/director/`.

Key new modules:

- `agent_torch.py`
- `torchagent.py`
- `torchnets.py`
- `torchutils.py`
- `hierarchy_torch.py`
- `reward_state.py`
- `behaviors_torch.py`
- `train_torch.py`
- `configs_torch.yaml`
- `EXPERIMENTS.md`

Debug commands after applying the package:

```bash
python -m embodied.agents.director.train_torch --configs debug,torch_baseline
python -m embodied.agents.director.train_torch --configs debug,torch_baseline,rs_aux
python -m embodied.agents.director.train_torch --configs debug,torch_baseline,rs_aux,rs_goal
python -m embodied.agents.director.train_torch --configs debug,torch_baseline,rs_aux,rs_goal,rs_adapt
```

B is the clean first experiment: it changes only the world-model representation objective, not the policy, rewards, manager, worker, or action space.

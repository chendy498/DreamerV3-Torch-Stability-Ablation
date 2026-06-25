# Director PyTorch A/B/C/D experiment plan

This port is structured so each variant changes exactly one thing.

| Variant | Configs | Difference from previous |
| --- | --- | --- |
| A | `torch_baseline` | PyTorch Director baseline: world model + manager + worker. |
| B | `torch_baseline,rs_aux` | Adds reward-state auxiliary heads to the world model only. |
| C | `torch_baseline,rs_aux,rs_goal` | Uses reward-state scores to rerank manager goal candidates. |
| D | `torch_baseline,rs_aux,rs_goal,rs_adapt` | Refreshes goals early when reward-state/event signals jump. |

Recommended debug commands:

```bash
python -m embodied.agents.director.train_torch --configs debug,torch_baseline --logdir ~/logdir/director_torch/A
python -m embodied.agents.director.train_torch --configs debug,torch_baseline,rs_aux --logdir ~/logdir/director_torch/B
python -m embodied.agents.director.train_torch --configs debug,torch_baseline,rs_aux,rs_goal --logdir ~/logdir/director_torch/C
python -m embodied.agents.director.train_torch --configs debug,torch_baseline,rs_aux,rs_goal,rs_adapt --logdir ~/logdir/director_torch/D
```

For paper experiments, keep all seeds, tasks, replay settings, and training budgets identical across A/B/C/D. The clean first claim is B vs A:

```text
B changes representation learning only; policy, reward, manager, worker, and action space remain unchanged.
```

Important metrics added by B:

- `rs_return_loss_mean`
- `rs_event_loss_mean`
- `rs_return_target_mean/std`
- `rs_event_rate`
- `rs_return_pred_mean`
- `rs_event_pred_mean`

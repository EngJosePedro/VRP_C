
import torch
import math

def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = []
    for group in param_groups:
        params = [p for p in group['params'] if p.grad is not None]
        if len(params) == 0:
            grad_norms.append(torch.tensor(0.0))
            continue
        g = torch.nn.utils.clip_grad_norm_(
            params,
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        grad_norms.append(g)

    if max_norm > 0:
        grad_norms_clipped = [min(float(g), max_norm) for g in grad_norms]
    else:
        grad_norms_clipped = [float(g) for g in grad_norms]

    return grad_norms, grad_norms_clipped

def grad_prop(model):
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(name, p.grad.norm().item())


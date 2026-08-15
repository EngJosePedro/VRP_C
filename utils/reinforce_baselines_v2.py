# utils/reinforce_baselines_v2.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel


class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, *args, **kwargs):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert len(self.dataset) == len(self.baseline)

    def __getitem__(self, item):
        return {
            "data": self.dataset[item],
            "baseline": self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)


class _SimpleCriticMLP(nn.Module):
    """
    Crítico simples e robusto: agrega estatísticas da instância e prevê V(s).
    Entrada:
      depots:   (B, nd, 2)
      customers:(B, nc, C)  (assume coords em [:,:2], demanda opcional em [:, :, 2])
      edge:     (B, N, N) ou None
    Saída:
      v: (B,)
    """
    def __init__(self, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, depots, customers, edge=None):
        # coords
        d_xy = depots.float()
        c_xy = customers[..., :2].float()

        # stats
        d_mean = d_xy.mean(dim=1)                         # (B,2)
        c_mean = c_xy.mean(dim=1)                         # (B,2)
        c_std  = c_xy.std(dim=1, unbiased=False)          # (B,2)

        # demanda (se existir)
        if customers.size(-1) >= 3:
            dem = customers[..., 2].float()
            dem_mean = dem.mean(dim=1, keepdim=True)      # (B,1)
            dem_std  = dem.std(dim=1, unbiased=False, keepdim=True)
        else:
            dem_mean = torch.zeros(d_xy.size(0), 1, device=customers.device)
            dem_std  = torch.zeros(d_xy.size(0), 1, device=customers.device)

        feat = torch.cat([d_mean, c_mean, c_std, dem_mean, dem_std], dim=-1)  # (B,8)
        v = self.mlp(feat).squeeze(-1)
        return v


class CriticBaseline(Baseline):
    """
    Actor-Critic baseline: V_phi(s).
    Diferente do RolloutBaseline, NÃO precisa wrap_dataset nem unwrap_batch.
    """
    def __init__(self, critic=None, critic_hidden=256, device=None):
        super(Baseline, self).__init__()
        self.critic = critic if critic is not None else _SimpleCriticMLP(hidden=critic_hidden)
        if device is not None:
            self.critic = self.critic.to(device)

    def eval(self, depots, customers, edge=None):
        v = self.critic(depots, customers, edge)          # (B,)
        return v.detach(), v

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def state_dict(self):
        return {"critic": self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        critic_state = state_dict.get("critic", {})
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state})


class ExponentialBaseline(Baseline):
    def __init__(self, beta):
        super(Baseline, self).__init__()
        self.beta = beta
        self.v = None

    def eval(self, x, c):
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()
        self.v = v.detach()
        return self.v, 0

    def state_dict(self):
        return {"v": self.v}

    def load_state_dict(self, state_dict):
        self.v = state_dict["v"]


class WarmupBaseline(Baseline):
    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        super(Baseline, self).__init__()
        self.baseline = baseline
        assert n_epochs > 0
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        self.warmup_baseline.rollout = self.baseline.rollout
        self.warmup_baseline.model = self.baseline.model
        self.warmup_baseline.opts = self.baseline.opts
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        self.baseline.load_state_dict(state_dict)


class RolloutBaseline(Baseline):

    def __init__(self, model, problem, opts, rollout, val_dataset, epoch=0):
        super(Baseline, self).__init__()
        self.rollout = rollout
        self.problem = problem
        self.opts = opts
        self.val_dataset = val_dataset
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, candidate_vals=None):
        self.model = copy.deepcopy(model)
        if candidate_vals is not None:
            self.bl_vals = candidate_vals
        else:
            print("Evaluating baseline model on evaluation dataset")
            self.bl_vals = self.rollout(self.model, self.val_dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, dataset):
        return self.rollout(self.model, dataset, self.opts, batch=dataset)

    def wrap_dataset(self, train_dataset):
        print("Evaluating baseline on dataset...")
        BL = BaselineDataset(train_dataset, self.rollout(self.model, train_dataset, self.opts).view(-1, 1))
        print(f"Avg. Baseline Costs on Training Dataset: {BL.baseline.mean()}")
        return BL

    def unwrap_batch(self, batch):
        return batch["data"], batch["baseline"].view(-1)

    def epoch_callback(self, model, epoch):
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = self.rollout(model, self.val_dataset, self.opts).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        print(
            f"Epoch {epoch} candidate mean {candidate_mean}, "
            f"baseline epoch {self.epoch} mean {self.mean}, "
            f"difference {candidate_mean - self.mean}"
        )

        if candidate_mean - self.mean < 0:
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2
            assert t < 0
            print("p-value: {}".format(p_val))
            if p_val < 0.05:
                print("Update baseline")
                self._update_model(model, epoch, candidate_vals)

    def state_dict(self):
        return {"model": self.model, "dataset": self.val_dataset, "epoch": self.epoch}
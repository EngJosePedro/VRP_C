
import os
import copy
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from utils.tools import load_model2 as load_model, load_modelFer
from utils.gradientes_tools import clip_grad_norms
from utils.log_utils import log_values
from utils.memory import mem

import math

def cosine_decay(epoch, total_epochs, start, end=0.0, warmup=0):
    if epoch < warmup:
        return start

    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    progress = min(max(progress, 0.0), 1.0)

    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

def grad_norm_by_prefix(model):
    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        prefix = name.split(".")[0]
        g = p.grad.detach().norm(2)

        if prefix not in out:
            out[prefix] = 0.0

        out[prefix] += g.item() ** 2

    return {k: v ** 0.5 for k, v in out.items()}


def get_baseline(baseline):
    return {
        'rollout': RolloutBaseline,
        'pomo': POMO,
        'a2c': ActorCritic,
        'coma': ActorCritic_COMA,
    }.get(baseline, None)


class Baseline(object):

    def baseline_functions(self, *args, **kwargs):
        return None

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, inputs):
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
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)

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

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']

    def wrap_dataset(self, train_dataset):
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        
        # Calcular Custos totais do baseline para o dataset de treino
        
        
        BL = BaselineDataset(train_dataset, self.rollout(self.model, train_dataset, self.opts).view(-1, 1))
        print(f"Avg. Baseline Costs on Training Dataset: {BL.baseline.mean()}")
        return BL

class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, ):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
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
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class RolloutBaseline(Baseline):

    def __init__(self, model, problem, opts, rollout, val_dataset, epoch=0):
        super(Baseline, self).__init__()

        self.rollout = rollout
        self.problem = problem
        self.opts = opts

        self.val_dataset = val_dataset
        self._update_model(model, epoch)

    def get_optimizer(self, model):
        return optim.Adam([
            {'params': model.parameters(), 'lr': self.opts.lr_model},
        ])
    
    def _update_model(self, model, epoch, candidate_vals = None):

        # candidate_vals -> Caso já esteja calculado!
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        if candidate_vals is not None:
            self.bl_vals = candidate_vals
        else:
            print("Evaluating baseline model on evaluation dataset")
            self.bl_vals = self.rollout(self.model, self.val_dataset, self.opts).cpu().numpy()

        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, dataset):
        #print("Evaluating baseline model on evaluation dataset")
        return self.rollout(self.model, dataset, self.opts, batch = dataset)

    def wrap_dataset(self, train_dataset, show_tqdm = True):
        if show_tqdm: print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        
        # Calcular Custos totais do baseline para o dataset de treino
        
        
        BL = BaselineDataset(train_dataset, self.rollout(self.model, train_dataset, self.opts, show_tqdm = show_tqdm).view(-1, 1))
        if show_tqdm: print(f"Avg. Baseline Costs on Training Dataset: {BL.baseline.mean()}")
        return BL

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D
    

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")

        #print(f"Verificação de tipo {type(model) is type(self.model)}")

        #sdM = model.state_dict()
        #sdN = self.model.state_dict()

        #missing_in_N = sdM.keys() - sdN.keys()
        #missing_in_M = sdN.keys() - sdM.keys()

        #print("missing_in_N:", missing_in_N)
        #print("missing_in_M:", missing_in_M)
        import torch
        def models_exact_equal(M, N):
            sdM = M.state_dict()
            sdN = N.state_dict()
            if sdM.keys() != sdN.keys():
                return False, "Different keys"

            for k in sdM.keys():
                a, b = sdM[k], sdN[k]
                if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                    return False, f"Mismatch meta at {k}"
                if not torch.equal(a, b):  # bitwise equal
                    # acha o primeiro ponto que difere
                    diff = (a != b)
                    idx = diff.nonzero(as_tuple=False)[0].tolist() if diff.any() else None
                    return False, f"Value mismatch at {k}, first diff idx={idx}"
            return True, "All state_dict tensors exactly equal"

        #ok, msg = models_exact_equal(model, self.model)
        #print(ok, msg)

        def models_allclose(M, N, rtol=1e-6, atol=1e-8):
            sdM, sdN = M.state_dict(), N.state_dict()
            if sdM.keys() != sdN.keys():
                return False, "Different keys"
            for k in sdM:
                if not torch.allclose(sdM[k], sdN[k], rtol=rtol, atol=atol):
                    max_abs = (sdM[k] - sdN[k]).abs().max().item()
                    return False, f"{k} not close, max_abs={max_abs}"
            return True, "All tensors allclose"

        #ok, msg = models_allclose(model, self.model)
        #print(ok, msg)



        candidate_vals = self.rollout(model, self.val_dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch} mean {self.mean}, difference {candidate_mean - self.mean}")

        if candidate_mean - self.mean < 0:
            # Calc p value
            #print(candidate_vals, self.bl_vals)
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05: #self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch, candidate_vals)

    
    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.val_dataset,
            'epoch': self.epoch
        }
    
    def train_batch(
        self, batch, model, 
        optimizer, step, n_steps, epoch, opts, device
    ):
        batch, bl_val = self.unwrap_batch(batch)
        bl_val = bl_val.to(device = device)
        bl_val = bl_val[:, None]

        depots, customers = batch
        depots = depots.to(device=device)
        customers = customers.to(device=device)
        
        with torch.no_grad():
            #mem("after data to gpu")
            edge_w = self.model.Problem.calc_energy(depots, customers)
        
        logp, seq, costs, entropy = model(depots, customers, edge_w)
        
        bl_val = costs.mean(dim = 1, keepdim=True) # (B, P)

        # Loss por layers
        with torch.no_grad():
            adv = ((costs - bl_val) / (bl_val.abs().clamp_min(1e-8) )).detach()
            adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-8))
            adv = adv.clamp(-5.0, 5.0)                      # clip

        loss = 1.0 * (adv * (logp)).mean()  - 0.00 * entropy.mean()
            
        # Perform backward pass and optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        grad_norms = clip_grad_norms(optimizer.param_groups, 1.0)
       
        optimizer.step()
        
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, "clear_cache"):
            m.clear_cache()

        # Logging
        if (step + 1) % int(opts.log_step) == 0 or step == n_steps - 1:
            
            with torch.no_grad():
                log_values(costs, grad_norms, epoch, step+1)
                best_costs, _ = costs.min(dim=1)
                str_r = f"Best: {best_costs.mean().item():.2f} "
                str_r += f"Baseline: {bl_val.mean().item():.2f} "
                str_r += f"LogP_L: {logp.mean().item():.2f} "
                str_r += f"entr: {entropy.mean().item():.3f} "
                
                print(str_r)

                mem("GPU Memory after train")
                #print(torch.cuda.memory_summary())
                print(grad_norm_by_prefix(model))

                def optimizer_state_gb(optimizer):
                    total = 0
                    for state in optimizer.state.values():
                        for v in state.values():
                            if torch.is_tensor(v) and v.is_cuda:
                                total += v.numel() * v.element_size()
                    return total / 1024**3

                print("OPT STATE GB:", optimizer_state_gb(optimizer))


    def train_epoch(
            self,dataloader,model,optimizer,epoch,opts,device,freeze_model_epoch = -1
        ):

        # Manter Modelo Congelado por n epocas
        if epoch <= freeze_model_epoch:
            model.freeze()
        elif epoch == freeze_model_epoch:
            model.unfreeze()

        model.train()
        model.set_decode_type("sampling")

        n_steps = len(dataloader)
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            self.train_batch(batch, model, optimizer, step, n_steps, epoch, opts, device)

            if opts.sync:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def save_model(self, model, optimizer, epoch):
        print('Saving model and state...')
        
        torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "baseline_state_dict": self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
            )
        

class POMO(Baseline):

    def __init__(self, model, problem, opts, rollout, val_dataset, epoch=0, use_rollout = False):
        super(Baseline, self).__init__()

        self.rollout = rollout
        self.problem = problem
        self.opts = opts

        self.val_dataset = val_dataset
        self._update_model(model, epoch)

        self.device = torch.device("cuda:0" if opts.use_cuda else "cpu")# Set the device

    def get_optimizer(self, model):
        return optim.Adam([
            {'params': model.parameters(), 'lr': self.opts.lr_model},
        ])

    def baseline_functions(self, *args, **kwargs):
        return None

    def _update_model(self, model, epoch, candidate_vals = None):

        # candidate_vals -> Caso já esteja calculado!
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        if candidate_vals is not None:
            self.bl_vals = candidate_vals
        else:
            print("Evaluating baseline model on evaluation dataset")
            self.bl_vals = self.rollout(self.model, self.val_dataset, self.opts).cpu().numpy()

        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, dataset):
        #print("Evaluating baseline model on evaluation dataset")
        return self.rollout(self.model, dataset, self.opts, batch = dataset)

    def wrap_dataset(self, train_dataset, show_tqdm = True):
        return train_dataset

    def unwrap_batch(self, batch):
        return batch
    
    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """

        pass

        
        print("Evaluating candidate model on evaluation dataset")
       
        candidate_vals = self.rollout(model, self.val_dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch} mean {self.mean}, difference {candidate_mean - self.mean}")

        
        if candidate_mean - self.mean < 0:
            # Calc p value
            #print(candidate_vals, self.bl_vals)
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05: #self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch, candidate_vals)
        
    
    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.val_dataset,
            'epoch': self.epoch
        }
    
    def train_batch(
        self, batch, model, 
        optimizer, step, n_steps, epoch, opts, device
    ):
        depots, sats, customers = batch
        depots = depots.to(device=device)
        sats = sats.to(device=device)
        customers = customers.to(device=device)
        
        with torch.no_grad():
            #mem("after data to gpu")
            edge_1, edge_2 = model.Problem.calc_energy(depots, sats, customers)
        
        resp = model(depots, sats, customers, edge_1, edge_2)
        if self.opts.strategy == "N":
            logp_L1, logp_L2, seq_1, seq_2, costs_1, costs_2, entropy_1, entropy_2 = resp
        elif opts.strategy == "JCF":
            logp_L1, logp_L2, logp_cluster, seq_1, seq_2, costs_1, costs_2, entropy_1, entropy_2, baseline_resp_1, baseline_resp_1 = resp
        else:
            logp_cluster, logp_L1, logp_L2, seq_1, seq_2, costs_1, costs_2, entropy_1, entropy_2, baseline_resp_1, baseline_resp_1 = resp
            #logp_cluster, logp_L1, logp_L2, seq_1, seq_2, costs_1, costs_2, entropy_1, entropy_2, entropy_cluster, _, _, _, entropy_sat, balance_sat, entropy_cust, balance_cust = resp
        if epoch < 5:
            costs = costs_2
        else:
            costs = costs_1 + costs_2
               

        bl_val_1 = costs_1.mean(dim = 1, keepdim=True) # (B, P)
        bl_val_2 = costs_2.mean(dim = 1, keepdim=True) # (B, P)
        bl_val = costs.mean(dim = 1, keepdim=True) # (B, P)

        # Loss por layers
        with torch.no_grad():
            adv_1 = ((costs_1 - bl_val_1) / (bl_val_1.abs().clamp_min(1e-8) )).detach()
            adv_1 = (adv_1 - adv_1.mean()) / (adv_1.std().clamp_min(1e-8))
            adv_1 = adv_1.clamp(-5.0, 5.0)                      # clip

            adv_2 = ((costs_2 - bl_val_2) / (bl_val_2.abs().clamp_min(1e-8) )).detach()
            adv_2 = (adv_2 - adv_2.mean()) / (adv_2.std().clamp_min(1e-8))
            adv_2 = adv_2.clamp(-5.0, 5.0)                      # clip

            adv = ((costs - bl_val) / (bl_val.abs().clamp_min(1e-8) )).detach()
            adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-8))
            adv = adv.clamp(-5.0, 5.0)                      # clip

        #with torch.no_grad():
        #    adv = ((costs - bl_val) / (bl_val.abs().clamp_min(1e-8) )).detach()
        #    adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-8))
        #    adv = adv.clamp(-5.0, 5.0)                      # clip
        
        delta = 1.0 if epoch < 5 else 0.0
        if self.opts.strategy == "N":
            loss = 1.0 * (adv * (logp_L1)).mean()  + 1.0 * (adv * (logp_L2)).mean() - 0.00 * entropy_1.mean() - 0.00 * entropy_2.mean()
        elif opts.strategy == "JCF":
            loss = 1.0 * (adv * (logp_cluster)).mean() + 1.0 * (adv_1 * (logp_L1)).mean()  + 0.0 * (adv * (logp_L2)).mean() - 0.00 * entropy_1.mean() - 0.00 * entropy_2.mean()
        else:
            #loss_rl = (adv * (logp_cluster)).mean() + 1.0 * (adv_1 * (logp_L1)).mean()  + 1.0 * (adv_2 * (logp_L2)).mean() - 0.00 * entropy_cluster.mean() - 0.00 * entropy_1.mean() - 0.00 * entropy_2.mean()
            loss_rl = (adv * (logp_cluster)).mean() + 1.0 * (adv_1 * (logp_L1)).mean()  + 1.0 * (adv_2 * (logp_L2)).mean() 

            eta_sat = 0.00
            lambda_balance = 0.000

            eta_base = cosine_decay(
                epoch,
                opts.n_epochs,
                start=eta_sat,
                end=0.0,
                warmup=2
            )

            lambda_base = cosine_decay(
                epoch,
                opts.n_epochs,
                start=lambda_balance,
                end=0.0,
                warmup=2
            )
            with torch.no_grad():
                adv_strength = adv.abs().mean().clamp(0.0, 5.0)
                adv_factor = (1.0 / (1.0 + adv_strength)).item()
                eta_sat = eta_base * adv_factor
                lambda_balance = lambda_base * adv_factor

            loss = loss_rl
            #loss = (
            #    loss_rl
            #    - eta_sat * entropy_sat.mean()
            #    - eta_sat * entropy_cust.mean()
            #    + lambda_balance * balance_sat
            #    + lambda_balance * balance_cust
            #)
            
        # Perform backward pass and optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        grad_norms = clip_grad_norms(optimizer.param_groups, 1.0)
       
        optimizer.step()
        
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, "clear_cache"):
            m.clear_cache()

        # Logging
        if (step + 1) % int(opts.log_step) == 0 or step == n_steps - 1:
            
            with torch.no_grad():
                log_values(costs, grad_norms, epoch, step+1)
                best_costs, _ = costs.min(dim=1)
                str_r = f"Best: {best_costs.mean().item():.2f} "
                str_r += f"Baseline: {bl_val.mean().item():.2f} "
                str_r += f"LogP_C: {logp_cluster.mean().item():.2f} " if self.opts.strategy != "N" else ""
                str_r += f"LogP_L1: {logp_L1.mean().item():.2f} " if self.opts.strategy != "CF" else ""
                str_r += f"LogP_L2: {logp_L2.mean().item():.2f} "
                #str_r += f"entr_c: {entropy_cluster.mean().item():.3f} " if self.opts.strategy != "N" else ""
                #str_r += f"entr_L1: {entropy_1.mean().item():.3f} " if self.opts.strategy == "JCF" else ""
                #str_r += f"entr_L2: {entropy_2.mean().item():.3f} "
                if self.opts.strategy in ["N", "JCF"]:
                    str_r += f"entr_L: ({entropy_1.mean().item():.3f}, {entropy_2.mean().item():.3f}) "
                #else:
                #    str_r += f"entr_L2: ({entropy_cust.mean().item():.3f}, {entropy_sat.mean().item():.3f}) "
                #    str_r += f"balance_L2: ({balance_sat.mean().item():.3f}, {balance_cust.mean().item():.3f}) "
                #    str_r += f"I: ({adv_factor:.3f}, {eta_sat:.3f}, {lambda_balance:.3f})"

                print(str_r)

                mem("GPU Memory after train")
                #print(torch.cuda.memory_summary())
                print(grad_norm_by_prefix(model))

                def optimizer_state_gb(optimizer):
                    total = 0
                    for state in optimizer.state.values():
                        for v in state.values():
                            if torch.is_tensor(v) and v.is_cuda:
                                total += v.numel() * v.element_size()
                    return total / 1024**3

                print("OPT STATE GB:", optimizer_state_gb(optimizer))

    def train_epoch(
            self,dataloader,model,optimizer,epoch,opts,device,freeze_model_epoch = -1
        ):

        def optimizer_state_gb(optimizer):
                    total = 0
                    for state in optimizer.state.values():
                        for v in state.values():
                            if torch.is_tensor(v) and v.is_cuda:
                                total += v.numel() * v.element_size()
                    return total / 1024**3
        
        # Manter Modelo Congelado por n epocas
        if epoch <= freeze_model_epoch:
            model.freeze()
        elif epoch == freeze_model_epoch:
            model.unfreeze()

        model.train()
        model.set_decode_type("sampling")

        n_steps = len(dataloader)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem(f"START epoch {epoch}")
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            self.train_batch(batch, model, optimizer, step, n_steps, epoch, opts, device)
            """if step % 50 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                mem(f"AFTER train_batch returned step={step}")
                print("OPT STATE GB:", optimizer_state_gb(optimizer))
            """

            """if self.opts.strategy == "CF":
                # Treinar cluster
                #model.ClusterFirst.unfreeze()
                #model.NodeEncoder.unfreeze()
                #model.decoder.freeze()
                self.train_batch(batch, model, optimizer, step, n_steps, epoch, opts, device)

                #model.ClusterFirst.freeze()
                #model.NodeEncoder.freeze()
                #model.decoder.unfreeze()
                #self.train_batch(batch, model, optimizer, step, n_steps, epoch, opts, device)
                
            else:
                self.train_batch(batch, model, optimizer, step, n_steps, epoch, opts, device)"""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem(f"END epoch {epoch}")

    def save_model(self, model, optimizer, epoch):

        print('Saving model and state...')
        torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "baseline_state_dict": self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
            )


class ActorCritic(Baseline):

    def __init__(self, model, problem, opts, rollout, val_dataset, epoch=0):
        super(Baseline, self).__init__()

        self.rollout = rollout
        self.problem = problem
        self.opts = opts

        self.val_dataset = val_dataset
        self._update_model(model, epoch)

        self.device = torch.device("cuda:0" if opts.use_cuda else "cpu")# Set the device
        self.critic = Critic_POMO(in_dim = 3 * opts.embedding_dim, hidden_dim=opts.embedding_dim).to(self.device)

    def get_optimizer(self, model):
        return optim.Adam([
            {'params': model.parameters(), 'lr': self.opts.lr_model},
            {'params': self.critic.parameters(), 'lr': self.opts.lr_model * 0.5},
        ])

    def _update_model(self, model, epoch, candidate_vals = None):

        # candidate_vals -> Caso já esteja calculado!
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        if candidate_vals is not None:
            self.bl_vals = candidate_vals
        else:
            print("Evaluating baseline model on evaluation dataset")
            self.bl_vals = self.rollout(self.model, self.val_dataset, self.opts).cpu().numpy()

        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, dataset):
        #print("Evaluating baseline model on evaluation dataset")
        return self.rollout(self.model, dataset, self.opts, batch = dataset)

    def wrap_dataset(self, train_dataset, show_tqdm = True):
        return train_dataset

    def unwrap_batch(self, batch):
        return batch

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """

        pass

        
        print("Evaluating candidate model on evaluation dataset")
       
        candidate_vals = self.rollout(model, self.val_dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch} mean {self.mean}, difference {candidate_mean - self.mean}")

        
        if candidate_mean - self.mean < 0:
            # Calc p value
            #print(candidate_vals, self.bl_vals)
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05: #self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch, candidate_vals)
        
    
    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.val_dataset,
            'epoch': self.epoch
        }
    
    
    def train_batch(
        self, batch, model, 
        optimizer, step, n_steps, epoch, opts, device
    ):
        # Aqui o baseline rollout não é necessário.
        # Porém, para reaproveitar seu DataLoader atual com BaselineDataset,
        # aceitamos tanto batch normal quanto batch embrulhado.

        if isinstance(batch, dict) and "data" in batch:
            batch = batch["data"]

        depots, customers = batch
        depots = depots.to(device=device)
        customers = customers.to(device=device)

        with torch.no_grad():
            edge = model.Problem.calc_energy(depots, customers)

        logp_steps, future_cost, value_steps, logp_acc, costs, entropy = model.get_decoder_critic_sample(depots, customers, edge, self.critic)
        
        # logp_steps, future_cost, value_steps (T, B, P)
        # Aplicar bootstrap <Rt​=rt​+γrt+1​+...+γnV(st+n​)> após validar melhor o modelo!!!

        adv = future_cost.detach() - value_steps.detach()
        
        actor_loss = (adv * logp_steps).mean()
        critic_loss = F.mse_loss(value_steps, future_cost.detach())
        
        beta = 0.5
        eta = 0.01
        loss = actor_loss + beta * critic_loss - eta * entropy.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norms = clip_grad_norms(optimizer.param_groups, 1.0)
        optimizer.step()

        if (step + 1) % int(opts.log_step) == 0 or step == n_steps - 1:
            with torch.no_grad():
                log_values(costs, grad_norms, epoch, step + 1)
                best_costs, _ = costs.min(dim=1)

                print(
                    f"Best: {best_costs.mean().item():.2f} "
                    f"Costs: {costs.mean().item():.2f} "
                    f"F: {future_cost[0, :, :].mean().item():.2f} "
                    f"V: {value_steps[0, :, :].mean().item():.2f} "
                    f"AcLoss: {actor_loss.item():.3f} "
                    f"CrLoss: {critic_loss.item():.3f} "
                    f"AMean: {adv.mean().item():.4f} "
                    f"AStd: {adv.std().item():.4f} "
                    f"LogP: {logp_acc.mean().item():.2f} "
                    f"ent.: {entropy.mean().item():.3f}"
                )

                mem("GPU Memory after train")
                print(grad_norm_by_prefix(model))

    def train_epoch(
            self,dataloader,model,optimizer,epoch,opts,device,freeze_model_epoch = -1
        ):

        # Manter Modelo Congelado por n epocas
        if epoch == 0 and freeze_model_epoch >= 0:
            model.freeze()
        elif epoch <= freeze_model_epoch:
            model.unfreeze()

        n_steps = len(dataloader)
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            self.train_batch(batch, model, optimizer, step, n_steps, epoch, opts, device)

    def save_model(self, model, optimizer, epoch):

        print('Saving model and state...')
        """torch.save(
            {
                'model': get_inner_model(self.Model).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': self.baseline.state_dict()
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )"""
        torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
            )
    
class ActorCritic_COMA(Baseline):

    def __init__(self, model, problem, opts, rollout, val_dataset, epoch=0, use_rollout = False):
        super(Baseline, self).__init__()

        self.rollout = rollout
        self.problem = problem
        self.opts = opts

        self.val_dataset = val_dataset
        self._update_model(model, epoch)

        self.device = torch.device("cuda:0" if opts.use_cuda else "cpu")# Set the device
        self.critic = QCriticPOMO(in_dim = 3 * opts.embedding_dim, embed_dim=opts.embedding_dim).to(self.device)
        self.critic_Cluster = QCriticComaCluster(in_dim = 2 * opts.embedding_dim, embed_dim=opts.embedding_dim).to(self.device)

        self.use_rollout = use_rollout

    def get_optimizer(self, model):
        return optim.Adam([
            {'params': model.parameters(), 'lr': self.opts.lr_model},
            {'params': self.critic.parameters(), 'lr': self.opts.lr_model * 0.5},
            {'params': self.critic_Cluster.parameters(), 'lr': self.opts.lr_model * 0.5},
        ])

    def baseline_functions(self, *args, **kwargs):
        if kwargs["strategy"] == "cluster":
            state_embedding = kwargs["state_embedding"] # B,n_c,E
            node_emb = kwargs["node_emb"] # B,n_s,E
            logp = kwargs["logp"] # # (B,n_c,n_sats)
            idx = kwargs["idx"]
            selected = kwargs["selected"] # (B,P,nc)
            P = kwargs["pomo"]
            future_cost  = kwargs["future_cost"] # (B, P, nc)

            B = node_emb.shape[0]
            NC = state_embedding.shape[1]
            NS = node_emb.shape[1]

            Q_t = self.critic_Cluster(state_embedding, node_emb) # B, n_c, n_s
            q_selected = Q_t[:, None,:,:].expand(B, P, NC, NS).gather(idx, selected.unsqueeze(-1)).squeeze(-1)  # (B,P,n_c)
            cf_baseline = (logp.exp().detach() *  Q_t).sum(dim=-1)  # (B,n_c)
            
            logp_nc = logp[:, None, :, :].expand(B, P, NC, NS).gather(idx, selected.unsqueeze(-1)).squeeze(-1)  # (B,P,n_c)
            
            return q_selected, cf_baseline, logp_nc, future_cost
        if kwargs["strategy"] == "cluster_seq":
            state_embedding = kwargs["state_embedding"] # B,P,2E
            node_emb = kwargs["node_emb"] # B,n_s,E
            logp = kwargs["logp"] # # (B,n_c,n_sats)
            idx = kwargs["idx"]
            selected = kwargs["selected"] # (B,P,nc)
            P = kwargs["pomo"]
            future_cost  = kwargs["future_cost"] # (B, P, nc)

            B = node_emb.shape[0]
            NC = state_embedding.shape[1]
            NS = node_emb.shape[1]

            Q_t = self.critic_Cluster(state_embedding, node_emb) # B, n_c, n_s
            q_selected = Q_t[:, None,:,:].expand(B, P, NC, NS).gather(idx, selected.unsqueeze(-1)).squeeze(-1)  # (B,P,n_c)
            cf_baseline = (logp.exp().detach() *  Q_t).sum(dim=-1)  # (B,n_c)
            
            logp_nc = logp[:, None, :, :].expand(B, P, NC, NS).gather(idx, selected.unsqueeze(-1)).squeeze(-1)  # (B,P,n_c)
            
            return q_selected, cf_baseline, logp_nc, future_cost
        if kwargs["strategy"] == "L1":
            state_embedding = kwargs["state_embedding"] # B,P,E
            node_emb = kwargs["node_emb"] # B,N,E
            logp = kwargs["logp"] # # (B,P,N)
            idx = kwargs["idx"]
            selected = kwargs["selected"] # (B,P,nc)
            P = kwargs["pomo"]

            B = node_emb.shape[0]
            NC = state_embedding.shape[1]
            NS = node_emb.shape[1]

            Q_t = self.critic(state_embedding, node_emb) # B, P, N
            q_selected = Q_t.gather(idx, selected.unsqueeze(-1)).squeeze(-1)  # (B,P,N)
            cf_baseline = (logp.exp().detach() *  Q_t).sum(dim=-1)  # (B,P)
            
            return q_selected, cf_baseline
        else:
            return None

    def _update_model(self, model, epoch, candidate_vals = None):

        # candidate_vals -> Caso já esteja calculado!
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        if candidate_vals is not None:
            self.bl_vals = candidate_vals
        else:
            print("Evaluating baseline model on evaluation dataset")
            self.bl_vals = self.rollout(self.model, self.val_dataset, self.opts).cpu().numpy()

        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, dataset):
        #print("Evaluating baseline model on evaluation dataset")
        return self.rollout(self.model, dataset, self.opts, batch = dataset)

    def wrap_dataset(self, train_dataset, show_tqdm = True):
        if self.use_rollout:
            if show_tqdm: print("Evaluating baseline on dataset...")
            # Need to convert baseline to 2D to prevent converting to double, see
            # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
            
            # Calcular Custos totais do baseline para o dataset de treino
            
            
            BL = BaselineDataset(train_dataset, self.rollout(self.model, train_dataset, self.opts, show_tqdm = show_tqdm).view(-1, 1))
            if show_tqdm: print(f"Avg. Baseline Costs on Training Dataset: {BL.baseline.mean()}")
            return BL
        return train_dataset

    def unwrap_batch(self, batch):
        if self.use_rollout:
            return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D
        return batch
    
    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """

        pass

        
        print("Evaluating candidate model on evaluation dataset")
       
        candidate_vals = self.rollout(model, self.val_dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch} mean {self.mean}, difference {candidate_mean - self.mean}")

        
        if candidate_mean - self.mean < 0:
            # Calc p value
            #print(candidate_vals, self.bl_vals)
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05: #self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch, candidate_vals)
        
    
    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.val_dataset,
            'epoch': self.epoch
        }
    
    
    def train_batch(
        self, batch, model, 
        optimizer, step, n_steps, epoch, opts, device
    ):
        if self.use_rollout:
            batch, bl_val = self.unwrap_batch(batch)
            bl_val = bl_val.to(device)
        else:
            if isinstance(batch, dict) and "data" in batch:
                batch = batch["data"]
        
        depots, sats, customers = batch
        depots = depots.to(device=device)
        sats = sats.to(device=device)
        customers = customers.to(device=device)
        
        with torch.no_grad():
            #mem("after data to gpu")
            edge = model.Problem.calc_energy(depots, sats, customers)

        resp = model(depots, sats, customers, edge)
        if self.opts.strategy == "N":
            logp_L1, logp_L2, seq_1, seq_2, costs_1, costs_2, entropy_1, entropy_2, baseline_resp_cluster = resp
            entropy_cluster = torch.zeros_like(entropy_1)
        else:
            logp_cluster, logp_L1, logp_L2, seq_1, seq_2, costs_1, costs_2, entropy_1, entropy_2, entropy_cluster, baseline_resp_cluster, baseline_resp_1, baseline_resp_2 = resp
            q_selected_cluster, cf_baseline_cluster, logp_cluster_nc, future_cost = baseline_resp_cluster # (B,P,n_c), (B,n_c), (B, P, n_c), (B, P, n_c)
            adv_Cluster = q_selected_cluster.detach() - cf_baseline_cluster[:, None,:].detach() # B, P, n_c
            #adv_Cluster = (adv_Cluster - adv_Cluster.mean()) / (adv_Cluster.std() + 1e-8)
            #adv_Cluster = adv_Cluster.clamp(-5.0, 5.0)                      # clip

            actor_cluster_loss = (adv_Cluster * logp_cluster_nc).mean()
            critic_cluster_loss = F.mse_loss(q_selected_cluster, future_cost.detach())
            
            beta = 0.5
            loss_clusyet_local = actor_cluster_loss + beta * critic_cluster_loss

            loss = loss_clusyet_local

            # L1
            q_selected_step_1, cf_baseline_step_1, logp_step_1, future_cost_step_1 = baseline_resp_1 # T, B, P
            adv_L1 = q_selected_step_1.detach() - cf_baseline_step_1.detach() # T, B, P
            actor_L1_loss = (adv_L1 * logp_step_1).mean()
            critic_L1_loss = F.mse_loss(q_selected_step_1, future_cost_step_1.detach())
            
            beta = 0.5
            loss_L1 = actor_L1_loss + beta * critic_L1_loss

            #print(q_selected_step_1.shape, cf_baseline_step_1.shape, logp_step_1.shape, future_cost_step_1.shape)
            loss = loss + 0.0 * loss_L1
        
            costs = costs_1 + costs_2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norms = clip_grad_norms(optimizer.param_groups, 1.0)
            optimizer.step()

            if (step + 1) % int(opts.log_step) == 0 or step == n_steps - 1:
                with torch.no_grad():
                    log_values(costs, grad_norms, epoch, step + 1)
                    best_costs, _ = costs.min(dim=1)

                    print(
                        f"Costs: ({best_costs.mean().item():.2f}, {costs.mean().item():.2f}) "
                        f"C_L: ({costs_1.mean().item():.2f}, {costs_2.mean().item():.2f}) "
                        f"FQ: ({future_cost.mean().item():.2f}<=>{q_selected_cluster.mean().item():.2f}), ({future_cost_step_1[0].mean().item():.2f}<=>{q_selected_step_1[0].mean().item():.2f}) "
                        f"AC: ({actor_cluster_loss.item():.3f}, {critic_cluster_loss.item():.3f}, {actor_L1_loss.item():.3f}, {critic_L1_loss.item():.3f}) "
                        f"LP: ({logp_cluster_nc.mean().item():.2f}, {logp_step_1.mean().item():.2f}) "
                        f"ent.: ({entropy_cluster.mean().item():.3f}, {entropy_1.mean().item():.3f})"
                    )

                    mem("GPU Memory after train")
                    print(grad_norm_by_prefix(model))

        
        
        #adv = future_cost.detach() - cf_baseline_steps.detach()
        #adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        #adv = adv.clamp(-5.0, 5.0)                      # clip
        
        #if self.use_rollout:
        #    gamma = 0.05
        #    adv_global = (costs - bl_val[:, None]) / (bl_val[:, None].abs() + 1e-8).detach()
        #    adv_global = (adv_global - adv_global.mean()) / (adv_global.std() + 1e-8)
        #    adv_global = adv_global.clamp(-5.0, 5.0)                      # clip
        #    loss_global = (adv_global * logp_acc).mean()
        #    loss = loss + gamma * loss_global
        
        

    def train_epoch(
            self,dataloader,model,optimizer,epoch,opts,device,freeze_model_epoch = -1
        ):

        # Manter Modelo Congelado por n epocas
        if epoch == 0 and freeze_model_epoch >= 0:
            model.freeze()
        elif epoch <= freeze_model_epoch:
            model.unfreeze()

        n_steps = len(dataloader)
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            self.train_batch(batch, model, optimizer, step, n_steps, epoch, opts, device)

    def save_model(self, model, optimizer, epoch):

        print('Saving model and state...')
        """torch.save(
            {
                'model': get_inner_model(self.Model).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': self.baseline.state_dict()
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )"""
        torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
            )


class Critic_POMO(nn.Module):
    """
    Critic simples.

    Entrada:
        embedding:    (B, P, E)

    Saída:
          (B, P)
    """

    def __init__(self, in_dim = 128, hidden_dim=128):
        super().__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embedding):
        # embedding B, P, E

        V = self.head(embedding)           # (B, P, 1)

        return V.squeeze(-1)

class QCriticComaCluster(nn.Module):
    def __init__(self, in_dim, embed_dim):
        # Q utilizado pelo CounterFactual Multi Agent
        super().__init__()

        self.state_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.node_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, state_embedding, node_emb, mask=None):
        """
        state_embedding: (B,n_c,E)
        node_emb:        (B,n_s,E)
        mask:            (B,P,N), True = inviável

        return:
            Q: (B,n_c, P,N)
        """

        h_s = self.state_proj(state_embedding)      # (B,n_c,E)
        h_n = self.node_proj(node_emb)              # (B,n_s,E)

        Q = torch.bmm(h_s, h_n.transpose(1, 2))     # (B,n_c,n_s)

        if mask is not None:
            Q = Q.masked_fill(mask, 0.0)

        return Q

class QCriticPOMO(nn.Module):
    def __init__(self, in_dim, embed_dim):
        # Q utilizado pelo CounterFactual Multi Agent
        super().__init__()

        self.state_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.node_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, state_embedding, node_emb, mask=None):
        """
        state_embedding: (B,P,3E)
        node_emb:        (B,N,E)
        mask:            (B,P,N), True = inviável

        return:
            Q: (B,P,N)
        """

        h_s = self.state_proj(state_embedding)      # (B,P,E)
        h_n = self.node_proj(node_emb)              # (B,N,E)

        Q = torch.bmm(h_s, h_n.transpose(1, 2))     # (B,P,N)

        if mask is not None:
            Q = Q.masked_fill(mask, 0.0)

        return Q







class RolloutBaseline_ResidualCritic(Baseline):

    def __init__(self, model, problem, opts, rollout, val_dataset, epoch=0):
        super(Baseline, self).__init__()

        self.rollout = rollout
        self.problem = problem
        self.opts = opts

        self.val_dataset = val_dataset
        self._update_model(model, epoch)

        self.device = torch.device("cuda:0" if opts.use_cuda else "cpu")# Set the device
        self.residual_critic = Critic(hidden_dim=opts.embedding_dim).to(self.device)

    def get_optimizer(self, model):
        return optim.Adam([
            {'params': model.parameters(), 'lr': self.opts.lr_model},
            {'params': self.residual_critic.parameters(), 'lr': self.opts.lr_model * 0.5},
        ])

    def _update_model(self, model, epoch, candidate_vals = None):

        # candidate_vals -> Caso já esteja calculado!
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        if candidate_vals is not None:
            self.bl_vals = candidate_vals
        else:
            print("Evaluating baseline model on evaluation dataset")
            self.bl_vals = self.rollout(self.model, self.val_dataset, self.opts).cpu().numpy()

        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, dataset):
        #print("Evaluating baseline model on evaluation dataset")
        return self.rollout(self.model, dataset, self.opts, batch = dataset)

    def wrap_dataset(self, train_dataset, show_tqdm = True):
        if show_tqdm: print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        
        # Calcular Custos totais do baseline para o dataset de treino
        
        
        BL = BaselineDataset(train_dataset, self.rollout(self.model, train_dataset, self.opts, show_tqdm = show_tqdm).view(-1, 1))
        if show_tqdm: print(f"Avg. Baseline Costs on Training Dataset: {BL.baseline.mean()}")
        return BL

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D
    """
    def eval(self, x, c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)

        # There is no loss
        return v, 0

    """

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")

        #print(f"Verificação de tipo {type(model) is type(self.model)}")

        #sdM = model.state_dict()
        #sdN = self.model.state_dict()

        #missing_in_N = sdM.keys() - sdN.keys()
        #missing_in_M = sdN.keys() - sdM.keys()

        #print("missing_in_N:", missing_in_N)
        #print("missing_in_M:", missing_in_M)
        import torch
        def models_exact_equal(M, N):
            sdM = M.state_dict()
            sdN = N.state_dict()
            if sdM.keys() != sdN.keys():
                return False, "Different keys"

            for k in sdM.keys():
                a, b = sdM[k], sdN[k]
                if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                    return False, f"Mismatch meta at {k}"
                if not torch.equal(a, b):  # bitwise equal
                    # acha o primeiro ponto que difere
                    diff = (a != b)
                    idx = diff.nonzero(as_tuple=False)[0].tolist() if diff.any() else None
                    return False, f"Value mismatch at {k}, first diff idx={idx}"
            return True, "All state_dict tensors exactly equal"

        #ok, msg = models_exact_equal(model, self.model)
        #print(ok, msg)

        def models_allclose(M, N, rtol=1e-6, atol=1e-8):
            sdM, sdN = M.state_dict(), N.state_dict()
            if sdM.keys() != sdN.keys():
                return False, "Different keys"
            for k in sdM:
                if not torch.allclose(sdM[k], sdN[k], rtol=rtol, atol=atol):
                    max_abs = (sdM[k] - sdN[k]).abs().max().item()
                    return False, f"{k} not close, max_abs={max_abs}"
            return True, "All tensors allclose"

        #ok, msg = models_allclose(model, self.model)
        #print(ok, msg)



        candidate_vals = self.rollout(model, self.val_dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch} mean {self.mean}, difference {candidate_mean - self.mean}")

        if candidate_mean - self.mean < 0:
            # Calc p value
            #print(candidate_vals, self.bl_vals)
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05: #self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch, candidate_vals)

    
    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.val_dataset,
            'epoch': self.epoch
        }
    
    
    def train_batch(
        self, batch, model, 
        optimizer, step, n_steps, epoch, opts, device
    ):
        batch, bl_val = self.unwrap_batch(batch)

        depots, customers = batch
        depots = depots.to(device=device)
        customers = customers.to(device=device)

        with torch.no_grad():
            edge = model.Problem.calc_energy(depots, customers)

        curr_confif = model.return_refined_embedding
        model.return_refined_embedding = True
        logp_refine, embedding, R = model(depots, customers, edge)
        model.return_refined_embedding = curr_confif
        log_probs, _, costs = R

        bl_val = bl_val.to(device=device)               # (B,)
        bl = bl_val[:, None]                            # (B,1)

        # ------------------------------------------------------------
        # 1) Critic residual
        # ------------------------------------------------------------
        residual_pred = self.residual_critic(embedding)  # (B,1)
        residual_pred = residual_pred.expand_as(costs)   # (B, P)

        # O target residual é calculado contra as soluções amostradas.
        # Como costs é (B,P), usamos a média das soluções POMO para treinar o critic.
        with torch.no_grad():
            target_residual_all = costs.detach() - bl             # (B,P)
            target_residual = target_residual_all#.mean(dim=1, keepdim=True)

            scale = bl.abs().clamp_min(1e-8)

        # Critic aprende resíduo normalizado
        residual_pred_norm = residual_pred / scale
        target_residual_norm = target_residual / scale

        critic_loss = F.smooth_l1_loss(
            residual_pred_norm,
            target_residual_norm
        )

        # ------------------------------------------------------------
        # 2) Advantage residual
        # ------------------------------------------------------------
        with torch.no_grad():
            baseline_hat = bl + residual_pred.detach()             # (B,1)
            adv = (costs - baseline_hat) / scale                   # (B,P)

            # Normalização local por instância, melhor que global
            #adv = adv - adv.mean(dim=1, keepdim=True)
            #adv = adv / adv.std(dim=1, keepdim=True)#.clamp_min(1e-8)
            #adv = (costs - bl_val[:, None]) / (bl_val[:, None].abs() + 1e-8).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.clamp(-5.0, 5.0)

        logp_total = log_probs + logp_refine  # (B,P)
        actor_loss = (adv * logp_total).mean()

        # ------------------------------------------------------------
        # 3) Loss total
        # ------------------------------------------------------------
        critic_coef = getattr(opts, "critic_coef", 0.2)
        loss = actor_loss + critic_coef * critic_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norms = clip_grad_norms(optimizer.param_groups, 1.0)
        optimizer.step()

        if (step + 1) % int(opts.log_step) == 0 or step == n_steps - 1:
            with torch.no_grad():
                log_values(costs, grad_norms, epoch, step + 1)
                best_costs, _ = costs.min(dim=1)

                print(
                    f"BestCosts: {best_costs.mean().item():.2f} "
                    f"Costs: {costs.mean().item():.2f} "
                    f"Baseline: {bl_val.mean().item():.2f} "
                    f"ResidualPred: {residual_pred.mean().item():.4f} "
                    f"TargetResidual: {target_residual.mean().item():.4f} "
                    f"ActorLoss: {actor_loss.item():.4f} "
                    f"CriticLoss: {critic_loss.item():.4f}"
                    f"log_likelihood: {log_probs.mean().item():.2f}"
                    f"log_likelihood_ref: {logp_refine.mean().item():.2f}"
                )

                mem("GPU Memory after train")
                print(grad_norm_by_prefix(model))

    

class ActorCritic_FER(Baseline):

    def __init__(self, model, problem, opts, rollout, val_dataset, epoch=0):
        super(Baseline, self).__init__()

        self.rollout = rollout
        self.problem = problem
        self.opts = opts

        self.val_dataset = val_dataset
        self._update_model(model, epoch)

        self.device = torch.device("cuda:0" if opts.use_cuda else "cpu")# Set the device
        self.critic = Critic(hidden_dim=opts.embedding_dim).to(self.device)

    def get_optimizer(self, model):
        return optim.Adam([
            {'params': model.parameters(), 'lr': self.opts.lr_model},
            {'params': self.critic.parameters(), 'lr': self.opts.lr_model * 0.5},
        ])

    def _update_model(self, model, epoch, candidate_vals = None):

        # candidate_vals -> Caso já esteja calculado!
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        if candidate_vals is not None:
            self.bl_vals = candidate_vals
        else:
            print("Evaluating baseline model on evaluation dataset")
            self.bl_vals = self.rollout(self.model, self.val_dataset, self.opts).cpu().numpy()

        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def eval(self, dataset):
        #print("Evaluating baseline model on evaluation dataset")
        return self.rollout(self.model, dataset, self.opts, batch = dataset)

    def wrap_dataset(self, train_dataset, show_tqdm = True):
        if show_tqdm: print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        
        # Calcular Custos totais do baseline para o dataset de treino
        
        
        #BL = BaselineDataset(train_dataset, self.rollout(self.model, train_dataset, self.opts, show_tqdm = show_tqdm).view(-1, 1))
        #if show_tqdm: print(f"Avg. Baseline Costs on Training Dataset: {BL.baseline.mean()}")
        return train_dataset

    def unwrap_batch(self, batch):
        return batch #batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D
    """
    def eval(self, x, c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)

        # There is no loss
        return v, 0

    """

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """

        pass

        
        print("Evaluating candidate model on evaluation dataset")

        #print(f"Verificação de tipo {type(model) is type(self.model)}")

        #sdM = model.state_dict()
        #sdN = self.model.state_dict()

        #missing_in_N = sdM.keys() - sdN.keys()
        #missing_in_M = sdN.keys() - sdM.keys()

        #print("missing_in_N:", missing_in_N)
        #print("missing_in_M:", missing_in_M)
        import torch
        def models_exact_equal(M, N):
            sdM = M.state_dict()
            sdN = N.state_dict()
            if sdM.keys() != sdN.keys():
                return False, "Different keys"

            for k in sdM.keys():
                a, b = sdM[k], sdN[k]
                if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                    return False, f"Mismatch meta at {k}"
                if not torch.equal(a, b):  # bitwise equal
                    # acha o primeiro ponto que difere
                    diff = (a != b)
                    idx = diff.nonzero(as_tuple=False)[0].tolist() if diff.any() else None
                    return False, f"Value mismatch at {k}, first diff idx={idx}"
            return True, "All state_dict tensors exactly equal"

        #ok, msg = models_exact_equal(model, self.model)
        #print(ok, msg)

        def models_allclose(M, N, rtol=1e-6, atol=1e-8):
            sdM, sdN = M.state_dict(), N.state_dict()
            if sdM.keys() != sdN.keys():
                return False, "Different keys"
            for k in sdM:
                if not torch.allclose(sdM[k], sdN[k], rtol=rtol, atol=atol):
                    max_abs = (sdM[k] - sdN[k]).abs().max().item()
                    return False, f"{k} not close, max_abs={max_abs}"
            return True, "All tensors allclose"

        #ok, msg = models_allclose(model, self.model)
        #print(ok, msg)



        candidate_vals = self.rollout(model, self.val_dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch} mean {self.mean}, difference {candidate_mean - self.mean}")

        
        if candidate_mean - self.mean < 0:
            # Calc p value
            #print(candidate_vals, self.bl_vals)
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05: #self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch, candidate_vals)
        
    
    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.val_dataset,
            'epoch': self.epoch
        }
    
    
    def train_batch(
        self, batch, model, 
        optimizer, step, n_steps, epoch, opts, device
    ):
        # Aqui o baseline rollout não é necessário.
        # Porém, para reaproveitar seu DataLoader atual com BaselineDataset,
        # aceitamos tanto batch normal quanto batch embrulhado.

        if isinstance(batch, dict) and "data" in batch:
            batch = batch["data"]

        depots, customers = batch
        depots = depots.to(device=device)
        customers = customers.to(device=device)

        with torch.no_grad():
            edge = model.Problem.calc_energy(depots, customers)

        curr_confif = model.return_refined_embedding
        model.return_refined_embedding = True
        logp_refine, embedding, R = model(depots, customers, edge)
        model.return_refined_embedding = curr_confif
        log_probs, _, costs = R

        # costs:     (B, P)
        # log_probs: (B, P)

        value = self.critic(embedding)          # (B,1)
        value = value.expand_as(costs)

        # ------------------------------------------------------------
        # 1) Critic target
        # ------------------------------------------------------------
        with torch.no_grad():
            # O critic padrão aprende o custo esperado da política atual.
            # Como há P soluções POMO por instância, usamos média POMO.
            target_value = costs.detach()#.mean(dim=1, keepdim=True)  # (B,1)

        # Normalização por escala para estabilizar regressão
        #scale = target_value.abs().detach().clamp_min(1e-8)

        critic_loss = F.smooth_l1_loss(
            value, # / scale
            target_value# / scale
        )

        # ------------------------------------------------------------
        # 2) Advantage
        # ------------------------------------------------------------
        with torch.no_grad():
            adv = costs - value.detach()                 # (B,P)
            #adv = adv / scale                            # (B,P)

            # Normalização local por instância/POMO
            #adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True).clamp_min(1e-8))
            #adv = adv.clamp(-5.0, 5.0)

        logp_total = log_probs + logp_refine  # (B,P)
        actor_loss = (adv * logp_total).mean()

        # ------------------------------------------------------------
        # 3) Entropia opcional
        # ------------------------------------------------------------
        entropy_coef = getattr(opts, "entropy_coef", 0.0)

        # Se log_probs é log-likelihood acumulado da rota, isto NÃO é entropia real.
        # Use apenas como regularizador fraco se desejar.
        entropy_proxy = -log_probs.mean()

        # ------------------------------------------------------------
        # 4) Loss total
        # ------------------------------------------------------------
        critic_coef = getattr(opts, "critic_coef", 0.5)

        loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy_proxy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norms = clip_grad_norms(optimizer.param_groups, 1.0)
        optimizer.step()

        if (step + 1) % int(opts.log_step) == 0 or step == n_steps - 1:
            with torch.no_grad():
                log_values(costs, grad_norms, epoch, step + 1)
                best_costs, _ = costs.min(dim=1)

                print(
                    f"BestCosts: {best_costs.mean().item():.2f} "
                    f"Costs: {costs.mean().item():.2f} "
                    f"Value: {value.mean().item():.2f} "
                    f"TargetValue: {target_value.mean().item():.2f} "
                    f"ActorLoss: {actor_loss.item():.4f} "
                    f"CriticLoss: {critic_loss.item():.4f} "
                    f"AdvMean: {adv.mean().item():.4f} "
                    f"AdvStd: {adv.std().item():.4f} "
                    f"LogLikelihood: {log_probs.mean().item():.2f}"
                    f"LogLikelihood_Ref: {logp_refine.mean().item():.2f}"
                )

                mem("GPU Memory after train")
                print(grad_norm_by_prefix(model))


class PPOBaseline(Baseline):
    """
    PPO para seu AttentionModel.

    A política antiga é representada por:
        old_logp
        old_values
        tours

    Não precisa deepcopy do actor.
    """

    def __init__(self, model, problem, opts, rollout, val_dataset, epoch=0):
        super().__init__()
        self.problem = problem
        self.opts = opts
        self.rollout = rollout
        self.val_dataset = val_dataset
        self.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

        self.critic = Critic(hidden_dim=opts.embedding_dim).to(self.device)

        # mantém um rollout baseline apenas para validação/challenge opcional
        self.model = copy.deepcopy(model)
        self.epoch = epoch

    def wrap_dataset(self, train_dataset, show_tqdm=True):
        # PPO não precisa pré-computar baseline por instância
        return train_dataset

    def unwrap_batch(self, batch):
        if isinstance(batch, dict) and "data" in batch:
            return batch["data"]
        return batch

    def get_optimizer(self, model):
        return optim.Adam([
            {"params": model.parameters(), "lr": self.opts.lr_model},
            {"params": self.critic.parameters(), "lr": self.opts.lr_model * getattr(self.opts, "ppo_critic_lr_mult", 0.5)},
        ])

    def epoch_callback(self, model, epoch):
        # PPO não precisa atualizar baseline a cada época.
        # Se quiser manter avaliação tipo RolloutBaseline, pode colocar aqui.
        pass
    
    @torch.no_grad()
    def collect_batch(self, batch, model, device):
        batch = self.unwrap_batch(batch)
        depots, customers = batch

        depots = depots.to(device=device)
        customers = customers.to(device=device)

        edge = model.Problem.calc_energy(depots, customers)

        model.eval()
        model.set_decode_type("sampling")

        old_logp, tours, costs, entropy = model.sample_for_ppo(
            depots,
            customers,
            edge
        )

        # critic usa embedding atual
        node_emb = model.get_encoder(depots, customers)
        old_values = self.critic(node_emb).detach()  # (B,1)

        returns = -costs.detach()                   # maximização de reward negativo

        return {
            "depots": depots.detach(),
            "customers": customers.detach(),
            "edge": edge.detach(),
            "tours": tours.detach(),
            "old_logp": old_logp.detach(),
            "old_values": old_values.detach(),
            "returns": returns.detach(),
            "old_costs": costs.detach(),
            "old_entropy": entropy.detach(),
        }

    def train_ppo_minibatch(self, data, model, optimizer, opts):
        model.train()

        depots = data["depots"]
        customers = data["customers"]
        edge = data["edge"]
        tours = data["tours"]

        old_logp = data["old_logp"]
        old_values = data["old_values"]
        returns = data["returns"]

        new_logp, entropy, _ = model.evaluate_actions(
            depots,
            customers,
            edge,
            tours
        )

        node_emb = model.get_encoder(depots, customers)
        values = self.critic(node_emb)

        loss, stats = ppo_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            values=values,
            old_values=old_values,
            returns=returns,
            entropy=entropy,
            clip_eps=getattr(opts, "ppo_clip_eps", 0.2),
            vf_coef=getattr(opts, "ppo_vf_coef", 0.5),
            ent_coef=getattr(opts, "ppo_ent_coef", 0.01),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norms = clip_grad_norms(
            optimizer.param_groups,
            getattr(opts, "max_grad_norm", 1.0)
        )

        optimizer.step()

        return stats, grad_norms

    def train_epoch(self, dataloader, model, optimizer, epoch, opts, device):
        """
        PPO correto:
        1. coleta buffer com política antiga;
        2. otimiza várias vezes sobre esse buffer;
        3. descarta buffer.
        """

        ppo_collect_batches = getattr(opts, "ppo_collect_batches", 8)
        ppo_epochs = getattr(opts, "ppo_epochs", 4)

        iterator = iter(dataloader)
        global_step = 0

        while True:
            buffer = []

            # ------------------------------------------------------------
            # 1) Coleta
            # ------------------------------------------------------------
            for _ in range(ppo_collect_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break

                data = self.collect_batch(batch, model, device)
                buffer.append(data)
                global_step += 1

            if len(buffer) == 0:
                break

            # ------------------------------------------------------------
            # 2) Otimização PPO
            # ------------------------------------------------------------
            for ppo_ep in range(ppo_epochs):
                perm = torch.randperm(len(buffer)).tolist()

                for idx in perm:
                    stats, grad_norms = self.train_ppo_minibatch(
                        buffer[idx],
                        model,
                        optimizer,
                        opts
                    )

            # ------------------------------------------------------------
            # 3) Log
            # ------------------------------------------------------------
            with torch.no_grad():
                costs = torch.cat([b["old_costs"] for b in buffer], dim=0)
                best_costs, _ = costs.min(dim=1)

                print(
                    f"[PPO] Epoch {epoch} "
                    f"Step {global_step} "
                    f"BestCosts: {best_costs.mean().item():.2f} "
                    f"Costs: {costs.mean().item():.2f} "
                    f"ActorLoss: {stats['actor_loss'].item():.4f} "
                    f"CriticLoss: {stats['critic_loss'].item():.4f} "
                    f"Entropy: {stats['entropy'].item():.4f} "
                    f"KL: {stats['approx_kl'].item():.6f} "
                    f"ClipFrac: {stats['clipfrac'].item():.4f} "
                    f"Ratio: {stats['ratio_mean'].item():.4f}"
                )

                mem("GPU Memory after PPO train")
                print(grad_norm_by_prefix(model))

            # segurança: se KL explodir, reduzir epochs/coleta
            if stats["approx_kl"].item() > getattr(opts, "ppo_target_kl", 0.05):
                print("[PPO] Early stop: KL acima do limite.")

def ppo_loss(
    new_logp,
    old_logp,
    values,
    old_values,
    returns,
    entropy,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
):
    """
    new_logp:   (B,P)
    old_logp:   (B,P)
    values:     (B,1) ou (B,P)
    old_values: (B,1) ou (B,P)
    returns:    (B,P)
    entropy:    (B,P) ou escalar
    """

    if values.size(1) == 1:
        values = values.expand_as(returns)

    if old_values.size(1) == 1:
        old_values = old_values.expand_as(returns)

    with torch.no_grad():
        adv = returns - old_values

        # normalização local por instância/POMO
        adv = adv - adv.mean(dim=1, keepdim=True)
        adv = adv / adv.std(dim=1, keepdim=True).clamp_min(1e-8)
        adv = adv.clamp(-5.0, 5.0)

    ratio = torch.exp(new_logp - old_logp)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv

    actor_loss = -torch.min(surr1, surr2).mean()

    value_clipped = old_values + (values - old_values).clamp(
        -clip_eps,
        clip_eps
    )

    vf_loss1 = (values - returns).pow(2)
    vf_loss2 = (value_clipped - returns).pow(2)

    critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

    entropy_loss = entropy.mean()

    loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy_loss

    with torch.no_grad():
        approx_kl = 0.5 * (new_logp - old_logp).pow(2).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_eps).float().mean()

    stats = {
        "loss": loss.detach(),
        "actor_loss": actor_loss.detach(),
        "critic_loss": critic_loss.detach(),
        "entropy": entropy_loss.detach(),
        "approx_kl": approx_kl.detach(),
        "clipfrac": clipfrac.detach(),
        "ratio_mean": ratio.mean().detach(),
        "adv_mean": adv.mean().detach(),
        "adv_std": adv.std().detach(),
    }

    return loss, stats

class Critic(nn.Module):
    """
    Critic simples para CVRP.

    Entrada:
        depots:    (B, 1, 2) ou (B, nd, 2)
        customers: (B, N, 3)  -> x, y, demand
        edge:      (B, Ntot, Ntot)

    Saída:
        residual:  (B, 1)
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embedding):
        # embedding B, N, E

        graph_h = embedding.mean(dim=1)                 # (B, E)
        residual = self.head(graph_h)           # (B, 1)

        return residual
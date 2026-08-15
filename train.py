

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor")

#CUDA_LAUNCH_BLOCKING=1

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn as nn


def set_dropout(model, p: float):
    """
    Altera dropout de todos os módulos Dropout do modelo.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p

        # SDPA custom
        if hasattr(module, "attn_dropout"):
            module.attn_dropout = p
        if hasattr(module, "dropout"):
            try:
                module.dropout = p
            except:
                module.dropout.p = p
                
                


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
       
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from Problems import load_problem

from nets.AttentionModel import AttentionModel
from nets.utils.reinforce_baselines import get_baseline

from utils.log_utils import log_values, model_resume
from utils.gradientes_tools import clip_grad_norms
from utils.tools import load_model, load_model2,  setup_save_dir

from utils.memory import mem

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class treinner():
    def __init__(self, opts):
        """
        Docstring for __init__
        
        :param self: Description
        :param opts: Description
        """
        
        super(treinner, self).__init__()
        
        self.opts = opts

        modes = ["train", "resume"]
        mode = opts.mode
        assert mode in modes, f"Modo de treino [{mode}] deve ser {" or ".join(modes)} 'train' or 'eval' "

        problem_type = opts.problem
        self.Problem = load_problem(problem_type)
        self.Problem.set_dist_type(opts.dist_type)

        # -- Set de parametros
        self.device = torch.device("cuda:0" if opts.use_cuda else "cpu")# Set the device
        self.mode = mode

        self.ModelType = "attention"
        self.class_Model = {"attention": AttentionModel}.get(self.ModelType)
        
        self.Model = self.class_Model(self.Problem, self.opts).to(device=self.device)
        
        if self.opts.model != "":
            # Continuar algum treinamento
            self.Model = self.class_Model.load_pretreinandeModel(self.opts.model).to(device=self.device)
            
        if self.opts.resume: model_resume(self.Model)

        if opts.use_cuda and torch.cuda.device_count() > 1:
            self.Model = torch.nn.DataParallel(self.Model)
            torch.cuda.reset_peak_memory_stats()
        
        if self.opts.epoch_epsodes > 0:
            self.epoch_epsodes = self.opts.epoch_epsodes
            self.mode = "train_epsodes"

        # Start the validation dataset
        self.val_dataset = self.Problem.make_dataset(num_samples = self.opts.val_size, n_cust = self.opts.graph_size)
        
        self.str_baseline_model = getattr(self.opts, "baseline_type", "rollout")
        base = get_baseline(self.str_baseline_model)
        assert base is not None, f"Baseline {self.str_baseline_model} inválido!!!"

        self.baseline = base(
                model = self.Model, 
                problem = self.Problem, 
                opts = self.opts, 
                rollout = lambda model, dataset, opts, batch = None, show_tqdm = True: self.rollout(model, dataset, opts, self.device, batch, mode = "greedy", show_tqdm = True),
                val_dataset = self.val_dataset,
        )
        

        # Initialize optimizer
        if not self.opts.use_lora:
            self.optimizer = self.baseline.get_optimizer(self.Model)
        else:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.Model.parameters()),
                lr=1e-4,
                weight_decay=1e-4
            )

        # Cosine decay
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=max(50, self.opts.n_epochs), 
            eta_min=self.opts.lr_model / 10    # lr_min
        )

        # --- setups
        setup_save_dir(self.opts.save_dir, self.opts)

    def run(self):
        if self.mode == "train" or self.mode == "train_epsodes":
            self.Train()
        elif self.mode == "resume":
            pass
        else:
            assert False, f"Modo de treinamento {self.mode} não desenvolvido!!!"
    
    def Train(self):
        initial_dropout = self.opts.dropout
        Train_epoch = self.Train_epoch if self.mode == "train" else self.Train_epoch_epsode
        n_epochs_dropout = 5
        for epoch in range(self.opts.epoch_start, self.opts.n_epochs):
            if epoch in range(n_epochs_dropout+1):
                p = max(
                    0.0,
                    initial_dropout * (1 - (epoch) / n_epochs_dropout)
                )

                set_dropout(self.Model, p)

            Train_epoch(
                epoch=epoch,
            )

    def save_model(self, epoch):

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
                    "model_state_dict": self.Model.state_dict(),
                    "baseline_state_dict": self.baseline.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                },
                os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
            )
    
    def Train_epoch(self, epoch):
        print("----------------------------------------------------------------------------------------------------")
        print(f"Start train epoch {epoch}, lr={self.optimizer.param_groups[0]['lr']} for run {self.opts.run_name}")
        
        mean_epochs = 0
        start_time = time.time()    
        
        training_dataset = self.baseline.wrap_dataset( self.Problem.make_dataset(num_samples = self.opts.epoch_size, n_cust = self.opts.graph_size) )
        
        #num_workers = min(8, self.opts.epoch_size // self.opts.batch_size)
        training_dataloader = DataLoader(training_dataset, batch_size=self.opts.batch_size, num_workers=0, shuffle=True) 
        
        self.baseline.train_epoch(
            dataloader=training_dataloader,
            model=self.Model,
            optimizer=self.optimizer,
            epoch=epoch,
            opts=self.opts,
            device=self.device,
            freeze_model_epoch = self.opts.freeze_model_epoch,
        )
            
        # Devolver Memoria pro VRAM
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        epoch_duration = time.time() - start_time
        print(f"Finished epoch {epoch}, took {time.strftime('%H:%M:%S', time.gmtime(epoch_duration))} s")

        if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
            self.baseline.save_model(self.Model, self.optimizer, epoch=epoch)

        self.baseline.epoch_callback(self.Model, epoch)

        # lr_scheduler should be called at end of epoch
        self.lr_scheduler.step()

        #torch.cuda.reset_peak_memory_stats()
        
    
    @staticmethod
    def rollout(model, dataset, opts, device, batch, mode = "greedy", show_tqdm = True):
        model.set_decode_type(mode)
        model.eval()

        def eval_model_bat(bat):
            
            #with torch.inference_mode():
            with torch.no_grad():
                depot_data, node_data = bat
                depot_data = depot_data.to(device=device, non_blocking=True)
                node_data  = node_data.to(device=device, non_blocking=True)

                edge_w = model.Problem.calc_energy(depot_data, node_data)
                start = time.time()
                
                logp, seq, costs, entropy = model(depot_data, node_data, edge_w)
                
                min_idx = costs.argmin(dim=1, keepdim=True)  # (B, 1)
                costs = costs.gather(1, min_idx)  # (B,1)
                
                costs = costs.squeeze(-1).detach().cpu()
                
                # solta refs GPU explicitamente
                del depot_data, node_data, edge_w
            
            if opts.sync:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()        
            return costs

        if batch is None:
            dl = DataLoader(dataset, batch_size=opts.eval_batch_size, num_workers=0, pin_memory=True)
            outs = []
            if show_tqdm:
                for bat in tqdm(dl):
                    outs.append(eval_model_bat(bat))
            else:
                for bat in dl:
                    outs.append(eval_model_bat(bat))
            out = torch.cat(outs, 0)
            # solta lista grande
            del outs
        else:
            out = eval_model_bat(batch)

        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return out




from nets.utils.options_train import get_options
if __name__=='__main__':
    opts=get_options()
    treinner(opts).run()




import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor")

#CUDA_LAUNCH_BLOCKING=1

import torch
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
            module.dropout = p
     
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from Problems import load_problem

from nets.AttentionModel import AttentionModel
from utils.log_utils import model_resume
from utils.tools import load_model, load_model2,  setup_save_dir

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class eval():
    def __init__(self, opts):
        """
        Docstring for __init__
        
        :param self: Description
        :param opts: Description
        """
        
        super(eval, self).__init__()
        
        self.opts = opts
        
        problem_type = opts.problem
        self.Problem = load_problem(problem_type)
        self.Problem.set_dist_type(opts.dist_type)

        # -- Set de parametros
        self.device = torch.device("cuda:0" if opts.use_cuda else "cpu")# Set the device

        self.ModelType = "attention"
        self.class_Model = {"attention": AttentionModel}.get(self.ModelType)
        
        self.Model = self.class_Model.load_pretreinandeModel(self.opts.model).to(device=self.device)

        if self.opts.models is not None:
            self.Models = [self.class_Model.load_pretreinandeModel(model).to(device=self.device) for model in self.opts.models]
        
        if self.opts.resume: model_resume(self.Model)

        if opts.use_cuda and torch.cuda.device_count() > 1:
            self.Model = torch.nn.DataParallel(self.Model)
            if self.opts.models is not None:
                self.Models = [torch.nn.DataParallel(model) for model in self.Models]

        torch.cuda.reset_peak_memory_stats()
        
    def run(self):
        self.Eval()

    # Evaluations
    def Eval(self):
        
        assert self.opts.dataset is not None or self.opts.graph_size is not None, \
            "Cannot specify result filename with more than one dataset"
        dataset_path = self.opts.dataset        
        self.eval_dataset(dataset_path)
    
    def eval_dataset(self, dataset_path, softmax_temp = 1):
        
        self.Model.pomo = self.opts.pomo
        
        self.Model.set_decode_type(
            "greedy" if self.opts.decode_strategy in ('greedy') else "sampling",
        )

        if self.opts.models is not None:
            for model in self.Models:
                model.pomo = self.opts.pomo
                model.set_decode_type(
                    "greedy" if self.opts.decode_strategy in ('greedy') else "sampling",
                )
        
        if self.opts.dataset is not None:
            dataset = self.Problem.make_dataset(num_samples = self.opts.val_size, n_cust = self.opts.graph_size, filename=dataset_path, seed = self.opts.seed)
        else:
            # Gerar dataset
            dataset = self.Model.Problem.make_dataset(num_samples = self.opts.val_size, n_cust = self.opts.graph_size, seed = self.opts.seed)
        
        run_swap = True
        models = None if self.opts.models is None else self.Models
        models_size = None if self.opts.models is None else self.opts.models_size
        results = self._eval_dataset(self.Model, models, models_size, dataset, self.device, self.opts.eval_batch_size)
        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        self.print_cost_estat(costs, durations)

        
        DATA={

                "cost_ai": list(costs),
                "tours": list([str(t) for t in tours]),
                "durations": durations,
            }

        if run_swap:
            results_swap = self._eval_dataset_swap(self.Model, dataset, self.device, self.opts.eval_batch_size, tours)
            costs_swap, tours_swap, durations_swap = zip(*results_swap)
            self.print_cost_estat(costs_swap, durations_swap)
            DATA["cost_swap"]=list(costs_swap)
            DATA["tours_swap"]=list(tours_swap)
            DATA["durations_swap"]=list(durations_swap)
            
        try:
            dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
            rand = False
        except:
            dataset_basename, ext = f"random_{self.opts.graph_size}", ".xlsx"
            rand = True
        
        model_name = "_".join(os.path.normpath(os.path.splitext(self.opts.model)[0]).split(os.sep)[-2:])
        
        results_dir = os.path.join(self.opts.results_dir, self.Model.Problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            self.opts.decode_strategy,
            self.opts.pomo if self.opts.decode_strategy != 'greedy' else '',
            softmax_temp, self.opts.offset, self.opts.offset + len(costs), ".xlsx"
        ))

        try:
            XLSX = pd.read_excel(out_file)            
            XLSX["cost_ai"] = DATA["cost_ai"]
            XLSX["tours"] = DATA["tours"]
            XLSX["durations"] = DATA["durations"]

            for tag in DATA:
                XLSX[tag] = DATA[tag]

        except:
            XLSX = pd.DataFrame(DATA)

        XLSX.to_excel(out_file, index=False)

    def _eval_dataset(self, model, models, models_size, dataset, device, eval_batch_size):
        
        model.eval()

        dataloader = DataLoader(dataset, batch_size=eval_batch_size)

        results = []
        
        for batch in tqdm(dataloader):
            start = time.time()
            costs, sequences, _ = self.rollout(model, models, models_size, dataset, self.opts, device, batch, mode = "sampling", show_tqdm = True)
            duration = time.time() - start
            for seq, cost in zip(sequences, costs):
                # seq (1, T)
                seq = [0] + np.trim_zeros(seq.squeeze(0).cpu().numpy()).tolist() + [0]  # Add depot                
                # Note VRP only
                results.append((cost.item(), seq, duration))
        return results
    
    @staticmethod
    def print_cost_estat(costs, durations):
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(
            np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations)))
        #print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations)))))

    def _eval_dataset_swap(self, model, dataset, device, eval_batch_size, tours: list):
        from torch.nn.utils.rnn import pad_sequence
        dataloader = DataLoader(dataset, batch_size=eval_batch_size)
        
        results = []

        idx = 0
        for i, batch in enumerate(tqdm(dataloader)):
            depot_data, node_data = batch
            
            depot_data = depot_data.to(device=device, non_blocking=True)
            node_data  = node_data.to(device=device, non_blocking=True)
            
            edge_W = model.Problem.calc_energy(depot_data, node_data)

            B = depot_data.shape[0]
            

            _tours = tours[idx:idx+B]

            _tours = pad_sequence(
                [torch.tensor(x) for x in _tours],
                batch_first=True,
                padding_value=0
            )
            
            _tours = _tours.unsqueeze(1)

            tours_SWAP, costs_SWAP, duration_SWAP = self.run_swap(dataset, _tours, edge_W, node_data)
            idx += B
            for seq, cost, duration in zip(tours_SWAP, costs_SWAP, duration_SWAP):
                results.append((cost[0], seq[0], duration[0]))
        return results

    def run_swap(self, dataset, tours, edge_data, node_data):        
            
            demand = node_data[:, :, 2]
            tours_SWAP, costs_SWAP, duration_SWAP = dataset.local_search_swap_mn_2optP(tours, edge_data, demand.squeeze(-1))
            
            return tours_SWAP, costs_SWAP, duration_SWAP

    @staticmethod
    def rollout(model, models, models_size, dataset, opts, device, batch, mode = "greedy", show_tqdm = True):
        model.set_decode_type(mode)
        model.eval()
        
        if models is not None:
            for m in models:
                m.set_decode_type(mode)
                #m.eval()

        def eval_model_bat(bat):
            m = model
            #with torch.inference_mode():
            with torch.no_grad():
                depot_data, node_data = bat

                depot_data = depot_data.to(device=device, non_blocking=True)
                node_data  = node_data.to(device=device, non_blocking=True)
                
                # Select model:
                if models is not None:
                    n = node_data.shape[1]
                    for idx, _n in enumerate(models_size):
                        m = models[idx]
                        if _n+16 >= n: break

                    #model = m
                    #m.set_decode_type(mode)
                    #m.eval()
                    
                m.opts.mode = "eval"
                
                edge_W = m.Problem.calc_energy(depot_data, node_data)
                
                start = time.time()
                logp, seq, costs, entropy = m(depot_data, node_data, edge_W)
                duration = time.time() - start


                min_idx = costs.argmin(dim=1, keepdim=True)  # (B, 1)
                costs = costs.gather(1, min_idx)  # (B,1)
                costs = costs.squeeze(-1).detach().cpu()

                B, _, T = seq.shape
                # Expandir índice para (B,1,T)
                min_idx_exp = min_idx.unsqueeze(-1).expand(B, 1, T)
                # Gather ao longo de dim=1 (eixo P)
                seq = seq.gather(dim=1, index=min_idx_exp)

                # solta refs GPU explicitamente
                del depot_data, node_data, edge_W
            return costs, seq, duration

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

        
        #torch.cuda.synchronize()
        #torch.cuda.empty_cache()

        return out


from nets.utils.options_eval import get_options
if __name__=='__main__':
    opts=get_options()
    eval(opts).run()
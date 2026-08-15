
from __future__ import annotations
import os
import time
import torch
from torch.utils.data import Dataset

import pickle

from .State import State
from Problems.vrp.cvrp_gurobi_paralel import solve_cvrp_dataset_gurobi_parallel as solve_cvrp_dataset_gurobi

from LS_CY.LS_fast import local_search_swap_mn_2opt_fast
#from LS_CY2.LS_fast_start_swap import local_search_swap_mn_2opt_fast
#from utils.OR_tools import solve_cvrp_ortools

from concurrent.futures import ProcessPoolExecutor, as_completed


class PROBLEM(object):

    NAME = "tsp"
    depot_dim = 2
    cust_dim = 2
    NEED_CAPACITY = False
    dist_type = "euclidian"

    @staticmethod
    def get_costs(edge_weight: torch.Tensor, sequence: torch.Tensor):
        """
        edge_weight: (B,N,N)
        sequence:    (B,T,P) com 0 como depósito e padding 0 no final
        Retorna: costs (B,P)
        """
        B, T, P = sequence.shape
        assert edge_weight.shape[0] == B
        assert edge_weight.shape[1] == edge_weight.shape[2]

        sequence = sequence.long()
        
        i = sequence[:, :-1, :]   # (B,T-1,P)
        j = sequence[:,  1:, :]   # (B,T-1,P)

        batch = torch.arange(B, device=sequence.device).view(B, 1, 1)
        seg_costs = edge_weight[batch, i, j]  # (B,T-1,P)

        costs = seg_costs.sum(dim=1)          # (B,P)
        return costs, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return ProblemDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return State.initialize(*args, **kwargs)
    
    @staticmethod
    def solver_gurobi(*args, **kwargs):
        return solve_cvrp_dataset_gurobi(*args, **kwargs)
    
    @classmethod
    def set_dist_type(cls, dist_type):
        cls.dist_type = dist_type

    @classmethod
    def calc_energy(cls, depots: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Calculado no get_item
        
        :param depots: [B, n_depots, 2]
        :type depots: torch.Tensor
        :param coords: [B, N - n_depots, 2]
        :type coords: torch.Tensor
        :return: Description
        :rtype: Tensor
        """
        #print(depots.shape)
        #print(coords.shape)

        points = torch.cat([depots, coords[:, :, :2]], dim=1)
        
        dist = (points[:, :, None, :] - points[:, None, :, :]).norm(p=2, dim=-1)
        return dist
        
        
    
    @staticmethod
    def pairwise_euclidean_fast(points):
        g = points @ points.transpose(1, 2)                # (B,N,N)
        s = (points * points).sum(dim=-1, keepdim=True)    # (B,N,1)
        dist2 = (s + s.transpose(1, 2) - 2*g).clamp_min_(0.0)
        return dist2.sqrt_()


import numpy as np
class ProblemDataset(Dataset):
    
    def __init__(self, num_samples = 10_000, n_cd = 1, n_cust = 20, filename = None, seed: int = None):
        super(ProblemDataset, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        from Problems.generate.generate import generate_coords
        
        if filename is None:
            self.__n_cust = n_cust
            # --- Generate random coords                
            self.cd_coords = generate_coords(num_samples, n_cd) # B, n_cd, 2
            self.cust_coords = generate_coords(num_samples, n_cust) # B, n_c, 2

        self.size = len(self.cust_coords)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.cd_coords[index], self.cust_coords[index]

    def get_zip_data(self):
        return list(zip(self.cd_coords, self.cust_coords))
    
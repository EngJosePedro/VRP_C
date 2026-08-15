import os

import torch
from torch.utils.data import Dataset

import pickle

from Problems.vrp.State import State
#from Problems.vrp.cvrp_gurobi_paralel import solve_cvrp_dataset_gurobi
from Problems import VRP

"""
OBJETIVO É VERIFICAR SE MODELOS ESTAO DE FATO CONSEGUINDO INTERPRETAR A GEOMETRIA DA REDE.

"""


class DIST(object):

    NAME = "DIST"
    dist_type = "euclidian"
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return ProblemDataset(*args, **kwargs)
    
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
        
        if cls.dist_type == "euclidian":
            #dist = (points[:, :, None, :] - points[:, None, :, :]).norm(p=2, dim=-1)
            dist = VRP.pairwise_euclidean_fast(points)
            return dist
        
        # Distancia entre depots = 0:
        _, n_d, _ = depots.shape
        B, n_c, _ = coords.shape
        N = n_d + n_c
        dist = torch.rand(size=(B, N, N), dtype=torch.float32, device=depots.device)
        dist[:, torch.arange(N), torch.arange(N)] = 0
        return dist
    
import math
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict

############################
#### Coordes
############################

def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_(0.0, 1.0)

def _gen_uniform(B: int, N: int) -> torch.Tensor:
    return torch.rand(B, N, 2, dtype=torch.float32)

def _gen_single_cluster(B: int, N: int, sigma: float = 0.08) -> torch.Tensor:
    # center por instância
    centers = torch.empty(B, 1, 2, dtype=torch.float32).uniform_(0.15, 0.85)
    pts = centers + sigma * torch.randn(B, N, 2, dtype=torch.float32)
    return _clamp01(pts)

def _gen_multi_cluster(B: int, N: int, k: int = 5, sigma: float = 0.05) -> torch.Tensor:
    # mistura de k clusters por instância, alocação (quase) igual
    centers = torch.empty(B, k, 2, dtype=torch.float32).uniform_(0.15, 0.85)

    base = N // k
    sizes = [base] * k
    for i in range(N - base * k):
        sizes[i % k] += 1

    out = torch.empty(B, N, 2, dtype=torch.float32)
    for b in range(B):
        chunks = []
        for j in range(k):
            c = centers[b, j].view(1, 2)
            chunks.append(c + sigma * torch.randn(sizes[j], 2, dtype=torch.float32))
        pts = torch.cat(chunks, dim=0)
        perm = torch.randperm(N)
        out[b] = pts[perm]
    return _clamp01(out)

def _gen_anisotropic_cluster(B: int, N: int, sigmas=(0.14, 0.04), angle_deg: float = 35.0) -> torch.Tensor:
    # cluster elíptico rotacionado
    centers = torch.empty(B, 1, 2, dtype=torch.float32).uniform_(0.15, 0.85)

    angle = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))
    ca, sa = torch.cos(angle), torch.sin(angle)
    R = torch.tensor([[ca, -sa], [sa, ca]], dtype=torch.float32)  # (2,2)
    S = torch.diag(torch.tensor(sigmas, dtype=torch.float32))     # (2,2)

    z = torch.randn(B, N, 2, dtype=torch.float32)
    pts = centers + (z @ S @ R.T)
    return _clamp01(pts)

def _gen_ring(B: int, N: int, r_mean: float = 0.30, r_std: float = 0.02) -> torch.Tensor:
    centers = torch.empty(B, 1, 2, dtype=torch.float32).uniform_(0.30, 0.70)  # evita clipping demais
    theta = 2.0 * torch.pi * torch.rand(B, N, 1, dtype=torch.float32)
    r = torch.tensor(r_mean, dtype=torch.float32) + torch.tensor(r_std, dtype=torch.float32) * torch.randn(B, N, 1)
    xy = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1) * r
    pts = centers + xy
    return _clamp01(pts)

def _gen_corner_hotspot(B: int, N: int, alpha: float = 6.0, beta: float = 2.0) -> torch.Tensor:
    # alpha>beta => concentra perto de 1 (canto superior direito)
    dist = torch.distributions.Beta(alpha, beta)
    u = dist.sample((B, N)).to(torch.float32)
    v = dist.sample((B, N)).to(torch.float32)
    pts = torch.stack([u, v], dim=-1)
    return _clamp01(pts)


############################
#### Demands
############################

def _triangular_percent(u: torch.Tensor, a: float, c: float, b: float) -> torch.Tensor:
    Fc = (c - a) / (b - a)
    left = u < Fc
    out = torch.empty_like(u)
    out[left] = a + torch.sqrt(u[left] * (b - a) * (c - a))
    out[~left] = b - torch.sqrt((1 - u[~left]) * (b - a) * (b - c))
    return out

def _lognormal_percent(shape, mu: float, sigma: float, lo: float, hi: float, device):
    dist = torch.distributions.LogNormal(mu, sigma)
    x = dist.sample(shape).to(device)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return lo + x * (hi - lo)

def _poisson_percent(shape, lam: float, lo: float, hi: float, device):
    dist = torch.distributions.Poisson(rate=torch.tensor(lam, device=device))
    k = dist.sample(shape).to(device)
    cap = max(lam + 4.0 * math.sqrt(max(lam, 1e-6)), 1.0)  # “cap” alto p/ escala
    x01 = torch.clamp(k / cap, 0.0, 1.0)
    return lo + x01 * (hi - lo)

def _negbin_percent(shape, total_count: float, probs: float, lo: float, hi: float, device):
    dist = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(total_count, device=device),
        probs=torch.tensor(probs, device=device),
    )
    k = dist.sample(shape).to(device)
    mean = total_count * (1 - probs) / probs
    var = total_count * (1 - probs) / (probs**2)
    cap = max(mean + 4.0 * math.sqrt(var + 1e-6), 1.0)
    x01 = torch.clamp(k / cap, 0.0, 1.0)
    return lo + x01 * (hi - lo)

@torch.no_grad()
def generate_demands_equal_mix(
    B: int,
    N: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    min_pct: float = 0.01,
    max_pct: float = 1.00,
    cfg: Dict | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Retorna:
      demand: (B, N, 1) em fração da capacidade (0.01..1.00)
      labels: (B,) indicando qual distribuição gerou cada instância
      dist_names: lista das distribuições na ordem dos labels
    """
    if cfg is None:
        cfg = {
            "uniform":   {"lo": 0.05, "hi": 0.60},
            "triangular":{"a": 0.05, "c": 0.20, "b": 0.80},
            "beta":      {"alpha": 2.0, "beta": 5.0},
            "lognormal": {"mu": -1.2, "sigma": 0.7, "lo": 0.05, "hi": 1.00},
            "poisson":   {"lam": 8.0, "lo": 0.05, "hi": 1.00},
            "negbin":    {"total_count": 5.0, "probs": 0.45, "lo": 0.05, "hi": 1.00},
        }

    dist_names = list(cfg.keys())
    K = len(dist_names)
    device_t = torch.device(device)

    demand = torch.empty((B, N, 1), device=device_t, dtype=dtype)
    labels = torch.empty((B,), device=device_t, dtype=torch.long)

    base = B // K
    rem = B % K
    counts = [base + (1 if i < rem else 0) for i in range(K)]

    start = 0
    for i, name in enumerate(dist_names):
        cnt = counts[i]
        if cnt == 0:
            continue
        end = start + cnt

        u = torch.rand((cnt, N, 1), device=device_t, dtype=dtype)

        if name == "uniform":
            lo, hi = cfg[name]["lo"], cfg[name]["hi"]
            x = lo + (hi - lo) * u

        elif name == "triangular":
            a, c, b = cfg[name]["a"], cfg[name]["c"], cfg[name]["b"]
            x = _triangular_percent(u, a=a, c=c, b=b)

        elif name == "beta":
            alpha, beta = cfg[name]["alpha"], cfg[name]["beta"]
            x01 = torch.distributions.Beta(alpha, beta).sample((cnt, N, 1)).to(device_t)
            x = min_pct + (max_pct - min_pct) * x01

        elif name == "lognormal":
            mu, sigma, lo, hi = cfg[name]["mu"], cfg[name]["sigma"], cfg[name]["lo"], cfg[name]["hi"]
            x = _lognormal_percent((cnt, N, 1), mu=mu, sigma=sigma, lo=lo, hi=hi, device=device_t)

        elif name == "poisson":
            lam, lo, hi = cfg[name]["lam"], cfg[name]["lo"], cfg[name]["hi"]
            x = _poisson_percent((cnt, N, 1), lam=lam, lo=lo, hi=hi, device=device_t)

        elif name == "negbin":
            tc, p, lo, hi = cfg[name]["total_count"], cfg[name]["probs"], cfg[name]["lo"], cfg[name]["hi"]
            x = _negbin_percent((cnt, N, 1), total_count=tc, probs=p, lo=lo, hi=hi, device=device_t)

        else:
            raise ValueError(f"Distribuição não suportada: {name}")

        demand[start:end] = torch.clamp(x, min=min_pct, max=max_pct)
        labels[start:end] = i
        start = end

    # embaralha para não ficar em blocos por distribuição
    perm = torch.randperm(B, device=device_t)
    return demand[perm], labels[perm], dist_names

class ProblemDataset(Dataset):
    """
    Gera um dataset onde num_samples é dividido igualmente entre tipos de distribuição.
    Retorna: depot (1,2), coords (N,2), label (int) indicando o tipo.
    """

    LABELS = [
        "uniform",
        "single_cluster",
        "multi_cluster",
        "anisotropic_cluster",
        "ring",
        "corner_hotspot",
    ]

    def __init__(self, num_samples: int = 10_000, size: int = 20, seed: int = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            
        assert size >= 2, "size deve ser >= 2 (1 depot + pelo menos 1 cliente)"
        N = size - 1  # clientes

        num_types = len(self.LABELS)
        base = num_samples // num_types
        rem = num_samples - base * num_types
        counts = [base] * num_types
        for i in range(rem):
            counts[i] += 1  # distribui o resto

        coords_chunks = []
        labels_chunks = []
        #demands_chunks = []
        #labelsdemand_chunks = []

        # 0) uniforme
        B = counts[0]
        coords_chunks.append(_gen_uniform(B, N))
        labels_chunks.append(torch.full((B,), 0, dtype=torch.long))
        #demands_chunks.append(_gen_uniform(B, N)[:, :, 0])
        #labelsdemand_chunks.append(torch.full((B,), 0, dtype=torch.long))

        # 1) 1 cluster
        B = counts[1]
        coords_chunks.append(_gen_single_cluster(B, N, sigma=0.08))
        #demands_chunks.append(_gen_single_cluster(B, N)[:, :, 0])
        labels_chunks.append(torch.full((B,), 1, dtype=torch.long))
        #labelsdemand_chunks.append(torch.full((B,), 1, dtype=torch.long))

        # 2) multi cluster
        B = counts[2]
        coords_chunks.append(_gen_multi_cluster(B, N, k=5, sigma=0.05))
        #demands_chunks.append(_gen_multi_cluster(B, N)[:, :, 0])
        labels_chunks.append(torch.full((B,), 2, dtype=torch.long))
        #labelsdemand_chunks.append(torch.full((B,), 2, dtype=torch.long))

        # 3) anisotrópico
        B = counts[3]
        coords_chunks.append(_gen_anisotropic_cluster(B, N, sigmas=(0.14, 0.04), angle_deg=35.0))
        #demands_chunks.append(_gen_anisotropic_cluster(B, N)[:, :, 0])
        labels_chunks.append(torch.full((B,), 3, dtype=torch.long))
        #labelsdemand_chunks.append(torch.full((B,), 3, dtype=torch.long))

        # 4) anel
        B = counts[4]
        coords_chunks.append(_gen_ring(B, N, r_mean=0.30, r_std=0.02))
        #demands_chunks.append(_gen_ring(B, N)[:, :, 0])
        labels_chunks.append(torch.full((B,), 4, dtype=torch.long))
        #labelsdemand_chunks.append(torch.full((B,), 4, dtype=torch.long))

        # 5) hotspot canto
        B = counts[5]
        coords_chunks.append(_gen_corner_hotspot(B, N, alpha=6.0, beta=2.0))
        #demands_chunks.append(_gen_corner_hotspot(B, N)[:, :, 0])
        labels_chunks.append(torch.full((B,), 5, dtype=torch.long))
        #labelsdemand_chunks.append(torch.full((B,), 5, dtype=torch.long))

        # Concatena tudo
        self.coords = torch.cat(coords_chunks, dim=0)  # (num_samples, N, 2)
        self.labels = torch.cat(labels_chunks, dim=0)  # (num_samples,)

        # índices embaralhados
        perm = torch.randperm(num_samples)

        #self.demands = torch.cat(demands_chunks, dim=0)  # (num_samples, N, 1)
        #self.labelsdemand = torch.cat(labelsdemand_chunks, dim=0)  # (num_samples,)
        # reordena pelo primeiro eixo (dim=0)
        #self.demands = self.demands[perm]           # (num_samples, N, 1)
        #self.labelsdemand = self.labelsdemand[perm] # (num_samples,)
        self.demands, self.labelsdemand, _ = generate_demands_equal_mix(
            B  = num_samples,
            N  = N,
            device = "cpu",
            dtype = torch.float32,
            min_pct = 0.01,
            max_pct = 1.00,
        )
               

        # Depot: aqui faço algo simples e consistente:
        # - para todos os casos, depot uniforme
        # (se você quiser depot correlacionado com a distribuição, eu adapto)
        self.depot = torch.rand(num_samples, 1, 2, dtype=torch.float32)

        # Embaralha para misturar tipos no dataset/
        perm = torch.randperm(num_samples)
        self.coords = self.coords[perm]
        self.depot = self.depot[perm]
        self.labels = self.labels[perm]

        self.coords = torch.cat([self.coords, self.demands], dim=2)
        
        self.size = num_samples
        self.n_nodes = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        # compatível com seu padrão: (depot, coords) + label
        return self.depot[idx], self.coords[idx], self.labels[idx], self.labelsdemand[idx]
    

    def get_zip_data(self):
        coords = self.coords[:, :, :2]   # (B, N, 2)
        demand = self.coords[:, :, 2:].squeeze(-1)   # (B, N, 1)

        return list(zip(self.depot, coords, demand))

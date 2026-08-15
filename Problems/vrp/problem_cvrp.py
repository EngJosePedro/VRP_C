
from __future__ import annotations
import os
import time
import torch
from torch.utils.data import Dataset

import pickle

from Problems.vrp.State import State
from Problems.vrp.cvrp_gurobi_paralel import solve_cvrp_dataset_gurobi_parallel as solve_cvrp_dataset_gurobi

from LS_CY.LS_fast import local_search_swap_mn_2opt_fast

from concurrent.futures import ProcessPoolExecutor, as_completed


class VRP(object):

    NAME = "VRP"
    depot_dim = 2
    cust_dim = 3

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
    def compute_customer_marginal_contrib(
        edge_w: torch.Tensor,
        seq: torch.Tensor,
        n_depots: int = 1,
    ):
        """
        Calcula contrib_i = d[prev,i] + d[i,next] - d[prev,next]
        e retorna por nó.

        Args:
            edge_w: (B, N, N)
            seq:    (B, P, T), long

        Returns:
            contrib_by_node: (B, P, N)
                depot fica zero.
                clientes não presentes ficam zero.
        """

        B, N, _ = edge_w.shape
        B2, P, T = seq.shape
        assert B == B2
        assert seq.dtype == torch.long

        device = seq.device
        dtype = edge_w.dtype

        contrib_by_node = torch.zeros(B, P, N, device=device, dtype=dtype)

        if T < 3:
            return contrib_by_node

        prev_nodes = seq[:, :, :-2]    # (B,P,T-2)
        cur_nodes  = seq[:, :, 1:-1]   # (B,P,T-2)
        next_nodes = seq[:, :, 2:]     # (B,P,T-2)

        b_idx = torch.arange(B, device=device)[:, None, None].expand(B, P, T - 2)

        d_prev_cur  = edge_w[b_idx, prev_nodes, cur_nodes]
        d_cur_next  = edge_w[b_idx, cur_nodes, next_nodes]
        d_prev_next = edge_w[b_idx, prev_nodes, next_nodes]

        contrib = d_prev_cur + d_cur_next - d_prev_next  # (B,P,T-2)

        # depot não contribui
        is_customer = cur_nodes >= n_depots
        contrib = torch.where(is_customer, contrib, torch.zeros_like(contrib))

        # scatter para (B,P,N)
        contrib_by_node.scatter_add_(
            dim=2,
            index=cur_nodes,
            src=contrib
        )

        # garante depot zero
        contrib_by_node[:, :, :n_depots] = 0.0

        return contrib_by_node

    @staticmethod
    def scatter_logp_by_customer(
        tours: torch.Tensor,
        logp_steps: torch.Tensor,
        n_nodes: int,
        n_depots: int = 1,
    ):
        """
        Converte logp por passo -> logp por cliente.

        Args:
            tours: (B,P,T)
            logp_steps:     (T,B,P)
            n_nodes: total de nós N

        Returns:
            logp_by_node: (B,P,N)
        """

        B, P, T = tours.shape

        device = logp_steps.device
        dtype = logp_steps.dtype

        logp_by_node = torch.zeros(
            B,
            P,
            n_nodes,
            device=device,
            dtype=dtype
        )

        # (B,P,T)
        logp_steps = logp_steps.permute(1, 2, 0)

        # depot não entra
        mask_customer = tours >= n_depots

        src = torch.where(
            mask_customer,
            logp_steps,
            torch.zeros_like(logp_steps)
        )

        logp_by_node.scatter_add_(
            dim=2,
            index=tours,
            src=src
        )

        logp_by_node[:, :, :n_depots] = 0.0

        return logp_by_node # B, P, N

    @staticmethod
    def get_costs___(edge_weight: torch.Tensor, sequence: torch.Tensor):
        """
        Docstring for get_costs
        
        :param edge_weight: (B, N, N) 
        :type edge_weight: torch.Tensor
        :param sequence: (B, N, pomo) Sequence of all nodes
        :type sequence: torch.Tensor
        """
        
        B, T, pomo = sequence.shape
        B, N, _ = edge_weight.shape
        #assert edge_weight.shape == (B, N, N), "Invalid edge weight matriz"

        
        # -- Check if sequence is valid (0, 1, ..., N-1):

        valid_sequence = torch.arange(start=0, end = N, dtype=sequence.dtype, device=sequence.device)[None, :, None].expand(B, N, pomo)
        
        assert (sequence.sort(dim=1)[0] == valid_sequence).all(), "Invalid Tour"
        del valid_sequence

        
        # ---- custo: somar distâncias entre pares consecutivos + fechar ciclo (FEITO COM GPT) ----
        
        i = sequence[:, :-1, :] # Origens (B, N-1, pomo)
        j = sequence[:, 1:, :] # Destinos (B, N-1, pomo)
        
        # distance[B, i, j] para todos os pares (a->b)
        batch = torch.arange(B, device=sequence.device).view(B, 1, 1)          # (B, 1, 1)
        seg_costs = edge_weight[batch, i, j]                              # (B, N-1, pomo)

        # fechamento: último -> primeiro
        last = sequence[:, -1, :]                                               # (B,pomo)
        first = sequence[:, 0, :]                                               # (B,pomo)
        close_cost = edge_weight[batch.squeeze(-1), last, first]  # (B,pomo)

        costs = seg_costs.sum(dim=1) + close_cost                      # (B,pomo)
        
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


    """def solve_cvrp_ortools(self, dataset, max_workers=None, chunksize=1):
        ""
        Resolve várias instâncias CVRP em paralelo usando OR-Tools.

        Parâmetros
        ----------
        dataset : iterable
            Cada item deve ser algo como:
                depot, coords
            onde:
                depot: tensor (1, 2) ou (nd, 2) se usar 1 depósito
                coords: tensor (N, 3), com [:, :2] = coordenadas e [:, 2] = demanda
        max_workers : int | None
            Número de processos paralelos. Se None, o Python escolhe automaticamente.
        chunksize : int
            Tamanho do chunk enviado ao ProcessPoolExecutor.map().
            Para muitos itens, chunksize > 1 pode reduzir overhead. :contentReference[oaicite:1]{index=1}

        Retorno
        -------
        routes_list : list
        costs_list : list
        durations_list : list
        ""
        

        # IMPORTANTE:
        # - No Windows, ProcessPoolExecutor exige que o ponto de entrada esteja protegido
        #   por if __name__ == "__main__": no script que chama esta função. :contentReference[oaicite:2]{index=2}
        # - A função solve_cvrp_ortools (a global importada do seu utils.OR_tools.py)
        #   precisa estar disponível em nível de módulo, o que no seu caso já está.

        # ------------------------------------------------------------------
        # 1) Pré-processa tudo no processo principal
        #    Isso evita mandar tensores complexos para os subprocessos.
        # ------------------------------------------------------------------
        jobs = []
        scale = 100000000
        cap_scale = 50

        for idx in range(len(dataset)):
            depot, coords = dataset[idx]

            demand = coords[:, 2:]
            coords_xy = coords[:, :2]

            # Junta depósito + clientes
            points = torch.cat([depot, coords_xy], dim=0)  # (N_total, 2)

            # Matriz de distâncias euclidianas inteira
            distance_matrix = (points[:, None, :] - points[None, :, :]).norm(p=2, dim=-1)
            distance_matrix = (distance_matrix * scale).round().to(torch.int64).cpu().tolist()

            # Demandas inteiras
            demands = [0] + (demand * cap_scale).round().to(torch.int64).squeeze(-1).cpu().tolist()

            # Número de veículos: sua lógica original
            total_demand = sum(demands)
            num_vehicles = max(1, int(1 + total_demand * 1.5 / cap_scale))
            vehicle_capacities = [cap_scale] * num_vehicles

            jobs.append({
                "idx": idx,
                "distance_matrix": distance_matrix,
                "demands": demands,
                "vehicle_capacities": vehicle_capacities,
                "depot": 0,
                "time_limit_seconds": 30,
                "first_solution_strategy": "PATH_CHEAPEST_ARC",
                "local_search_metaheuristic": "GUIDED_LOCAL_SEARCH",
                "solution_limit": None,
                "allow_drop_nodes": False,
                "drop_penalty": 10**6,
                "max_distance_per_vehicle": None,
                "fixed_cost_per_vehicle": None,
                "starts": None,
                "ends": None,
                "log_search": False
            })

        # ------------------------------------------------------------------
        # 2) Worker local apenas empacota chamada.
        #
        # OBS.: Em Windows, função local normalmente NÃO é picklable para
        # ProcessPoolExecutor. Então, em vez de enviar esta função ao pool,
        # vamos enviar diretamente a função global solve_cvrp_ortools com
        # argumentos simples.
        # ------------------------------------------------------------------

        routes_list = [None] * len(jobs)
        costs_list = [None] * len(jobs)
        durations_list = [None] * len(jobs)

        # ------------------------------------------------------------------
        # 3) Função auxiliar de desempacotamento usando submit
        #    Cada future chama a função global solve_cvrp_ortools.
        # ------------------------------------------------------------------
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}

            for job in jobs:
                t0 = time.time()

                future = executor.submit(
                    solve_cvrp_ortools,
                    job["distance_matrix"],
                    job["demands"],
                    job["vehicle_capacities"],
                    job["depot"],
                    job["time_limit_seconds"],
                    job["first_solution_strategy"],
                    job["local_search_metaheuristic"],
                    job["solution_limit"],
                    job["allow_drop_nodes"],
                    job["drop_penalty"],
                    job["max_distance_per_vehicle"],
                    job["fixed_cost_per_vehicle"],
                    job["starts"],
                    job["ends"],
                    job["log_search"],
                )

                future_to_idx[future] = (job["idx"], t0)

            # Coleta resultados
            for future, (idx, t0) in future_to_idx.items():
                resp = future.result()
                duration = time.time() - t0

                routes = resp["routes"]
                objective_value = resp["objective_value"]

                # Mantém apenas rotas realmente usadas:
                # ex.: [0, 0] = veículo não usado
                filtered_routes = []
                for route_info in routes:
                    route = route_info["route"]

                    # rota usada se tiver pelo menos um cliente
                    # assumindo depósito = 0
                    if any(node != 0 for node in route[1:-1]):
                        filtered_routes.append(route)

                routes_list[idx] = filtered_routes
                costs_list[idx] = None if objective_value is None else objective_value / scale
                durations_list[idx] = duration

                total_nodes = sum(len(r) for r in filtered_routes)
                print(f"Instância {idx}: {len(filtered_routes)} rotas, total de nós nas rotas = {total_nodes}")

        return routes_list, costs_list, durations_list
    """
    
import numpy as np

def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class ProblemDataset(Dataset):
    
    def __init__(self, num_samples = 10_000, n_cust = 20, filename=None, offset=0, seed: int = None, like_kool = True):
        super(ProblemDataset, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        

        self.__graph_size = n_cust

        if filename is not None:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"PKL não encontrado: {filename}")
            assert os.path.splitext(filename)[1].lower() == ".pkl"
            
            with open(filename, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, (list, tuple)):
                raise TypeError(f"Esperado list/tuple no pkl, veio: {type(data)}")
            
            slice_data = data[offset:offset + num_samples]
            
            del data
            try:
                B = len(slice_data)
                nd = len(slice_data[0][0]) // 2
                nc = len(slice_data[0][1])
                d = len(slice_data[0][2])

                self.coords = torch.zeros(B, nc, 2, dtype = torch.float32)
                self.depot = torch.zeros(B, 1, 2, dtype = torch.float32)
                self.demand = (torch.zeros(size=(B, nc))).to(torch.float32)
                
                for b in range(B):
                    depots = slice_data[b][0]
                    coords = slice_data[b][1]
                    demand = slice_data[b][2]
                    CAPACITIES = slice_data[b][3]
                    
                    self.depot[b, :nd] = torch.Tensor(depots)
                    self.coords[b] = torch.Tensor(coords)
                    self.demand[b] = torch.Tensor(demand) / CAPACITIES
                    
                    #depots, coords, demand, CAPACITIES = (*slice_data[b])
                    
                
                #self.coords = self.coords[:, : nc - nd, :]
                #self.demand = self.demand[:, : nc - nd]


            except:
                depots, coords, demand = zip(*slice_data)

                self.depot = depots
                self.coords = coords
                self.demand = demand
            
        else:
            
            if like_kool:
                
                # From VRP with RL paper https://arxiv.org/abs/1802.04240
                CAPACITIES = {
                    10: 20.,
                    20: 30.,
                    21: 30.,
                    50: 40.,
                    100: 50.,
                    200: 50.
                }
                
                self.coords = torch.zeros(num_samples, n_cust, 2)
                self.depot = torch.zeros(num_samples, 1, 2)
                self.demand = torch.zeros(num_samples, n_cust)

                for i in range(num_samples):
                    loc = torch.FloatTensor(n_cust, 2).uniform_(0, 1)
                    demand = (torch.FloatTensor(n_cust).uniform_(0, 9).int() + 1).float() / CAPACITIES[n_cust]
                    depot = torch.FloatTensor(2).uniform_(0, 1)

                    self.coords[i] = loc
                    self.depot[i] = depot
                    self.demand[i] = demand

                #print(self.coords.dtype)
                self.coords = self.coords.to(torch.float32)
                self.depot = self.depot.to(torch.float32)
                self.demand = self.demand.to(torch.float32)

            else:

                # --- Generate random coords                
                self.coords = torch.rand(num_samples, n_cust - 1, 2)
                self.depot = torch.rand(num_samples, 1, 2)
                self.demand = (torch.randint(low=1, high=9,size=(num_samples, n_cust - 1)) / self.CAPACITIES(n_cust))
                #self.demand = torch.zeros_like(self.demand)

                total_rand = True
                if total_rand:
                    from Problems.DIST.problem import ProblemDataset as randomDS
                    DS = randomDS(num_samples=num_samples, size=n_cust, seed = seed)
                    Data = DS.get_zip_data()
                    depots, coords, demand = zip(*Data)

                    depot_tensor = torch.stack(depots)      # (B, 2)
                    coords_tensor = torch.stack(coords)     # (B, N, 2)
                    demand_tensor = torch.stack(demand)    # (B, N)

                    demand_tensor = demand_tensor
                    demand_tensor = torch.round(demand_tensor * 100) / 100  # 2 casas decimais
                    demand_tensor = demand_tensor.clamp(min=0.01, max=1.00)
                    
                    self.coords = coords_tensor
                    self.depot = depot_tensor
                    self.demand = demand_tensor
        
        self.size = len(self.coords)
        #print("Leitura FInalizada")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.depot[index], torch.cat([self.coords[index], self.demand[index].unsqueeze(-1)], dim=1)

    def get_zip_data(self):
        return list(zip(self.depot, self.coords, self.demand))
    
    def get_graph_size(self):
        return self.__graph_size
    def get_Capacity(self, n = None):
        if n is None:
            n = self.get_graph_size()
        #print("----", n)
        return self.CAPACITIES(n)
    
    def CAPACITIES(self, size):
        
        if size <= 11: return 20.0
        elif size <= 21: return 30.0
        elif size <= 51: return 40.0
        elif size <= 101: return 50.0
        return 50.0

    @staticmethod
    def _ls_worker_one(args):
        """
        Worker rodando em processo separado.
        Recebe tudo serializável (listas/np arrays/float/int/bool).
        Retorna (b, p, route_out, cost_out, duration).
        """
        (b, p, route_list, dist_np, demands_np, cap,
        max_m, max_n, max_iters, first_improvement) = args

        
        t0 = time.time()

        dist_np = np.ascontiguousarray(dist_np, dtype=np.float64)
        demands_np = np.ascontiguousarray(demands_np, dtype=np.float64)
        cap = float(cap)

        cur, cost = local_search_swap_mn_2opt_fast(
            route_list,
            dist=dist_np,
            demand=demands_np,
            cap=cap,
            max_m=max_m,
            max_n=max_n,
            max_iters=max_iters,
            first_improvement=first_improvement,
            debug_check=True,
            only_run_2opt_if_promising=True,
        )
        
        dur = time.time() - t0
        
        return b, p, cur, cost, dur

    @staticmethod
    def __get_tasks(sol: torch.Tensor, dist, demand, Q_non_norm, B, P):
        demands_np_by_b = []
        for b in range(B):
            dem = torch.cat(
                [torch.zeros(1, device=sol.device),
                demand[b].to(sol.device)],
                dim=-1
            ).detach().cpu().numpy()

            if Q_non_norm is not None:
                dem = np.round(dem * Q_non_norm, decimals=0)
            demands_np_by_b.append(dem)
            
        # Converte dist para numpy uma vez por b (evita repetir em cada p)
        # Importante: .detach().cpu().numpy() antes de mandar p/ processos
        dist_np_by_b = [dist[b].detach().cpu().numpy() for b in range(B)]

        max_m = 3
        max_n = 3
        max_iters= 1_000
        first_improvement = False
        
        # Cria tasks (b,p)
        # sol[b,p] -> lista python (serializável)
        tasks = []
        sol_cpu = sol.detach()#.cpu()
        for b in range(B):
            dist_np = dist_np_by_b[b]
            dem_np = demands_np_by_b[b]
            for p in range(P):
                route_list = sol_cpu[b, p].tolist()
                tasks.append((
                    b, p, route_list, dist_np, dem_np, Q_non_norm,
                    max_m, max_n, max_iters, first_improvement
                ))
        return tasks

    def local_search_swap_mn_2optP(self, sol: torch.Tensor, dist, demand):
        
        # demand: B, nc

        B = sol.shape[0]
        P = sol.shape[1]

        # Ajuste padrão de workers
        max_workers = 4# None
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 1) - 1)

        # Converte demandas uma vez por b
        Q_non_norm = self.get_Capacity(dist.shape[1]-1)
        
        tasks = ProblemDataset.__get_tasks(sol, dist, demand, Q_non_norm, B, P)
        del sol
        
        # Pre-aloca outputs
        routes = [[None for _ in range(P)] for _ in range(B)]
        costs = [[None for _ in range(P)] for _ in range(B)]
        duration = [[None for _ in range(P)] for _ in range(B)]

        # Executa em paralelo
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(ProblemDataset._ls_worker_one, t) for t in tasks]
            for fut in as_completed(futures):
                b, p, cur, cost, dur = fut.result()
                routes[b][p] = cur
                costs[b][p] = cost
                duration[b][p] = dur

        return routes, costs, duration

    
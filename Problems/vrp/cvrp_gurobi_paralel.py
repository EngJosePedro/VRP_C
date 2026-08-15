# cvrp_gurobi_parallel.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Callable, Any, Optional
import os

import numpy as np
from tqdm import tqdm

import gurobipy as gp
from gurobipy import GRB


# ============================================================
# Helpers: parse da solução inicial e MIP start (x/y)
# ============================================================

def _split_routes_from_sequence(seq: List[int]) -> List[List[int]]:
    """
    Converte sequência no formato [0, a, b, 0, c, 0, ...] em lista de rotas
    no formato [[0,a,b,0], [0,c,0], ...].
    Lida com zeros repetidos e com ausência de 0 no começo/fim.
    """
    if seq is None:
        return []

    s = [int(v) for v in seq]
    if len(s) == 0:
        return []

    if s[0] != 0:
        s = [0] + s
    if s[-1] != 0:
        s = s + [0]

    routes: List[List[int]] = []
    cur: List[int] = [0]
    for v in s[1:]:
        if v == 0:
            # fecha rota se tiver ao menos 1 cliente
            if len(cur) > 1:
                cur.append(0)
                routes.append(cur)
            cur = [0]
        else:
            cur.append(v)

    return routes


def _build_mip_start_from_routes(
    routes: List[List[int]],
    n1: int,
    demands: np.ndarray,
    capacity: float,
    Q_non_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna matrizes start_x (n1,n1) binária e start_y (n1,n1) contínua
    consistentes com o modelo SCF implementado.

    SCF usado no seu código:
      inflow_y(j) - outflow_y(j) = demand[j]
    Um start factível para y em uma rota 0->c1->...->ck->0:
      y[0,c1] = sum(demand[c1..ck])
      y[c_t, c_{t+1}] = sum(demand[c_{t+1}..ck])
      y[ck,0] = 0
    """
    start_x = np.zeros((n1, n1), dtype=float)
    start_y = np.zeros((n1, n1), dtype=float)

    n = n1 - 1
    used = np.zeros(n1, dtype=bool)
    #print(demands)
    if Q_non_norm is not None: 
        capacity = Q_non_norm
        demands = (demands * Q_non_norm).round(decimals=0)
    
    #print(Q_non_norm)
    #print(demands.shape)
    #print(demands)
    for r in routes:
        # valida formato mínimo
        if len(r) < 3 or r[0] != 0 or r[-1] != 0:
            continue

        # valida nós
        for v in r:
            if v < 0 or v > n:
                raise ValueError(f"Nó fora do range na initial_solution: {v} (esperado 0..{n})")

        # clientes da rota
        customers = [v for v in r[1:-1] if v != 0]
        if len(customers) == 0:
            continue

        # checa repetição (opcional: em vez de erro, você pode só ignorar)
        for c in customers:
            if used[c]:
                raise ValueError(f"Cliente repetido na initial_solution: {c}")
            used[c] = True

        # checa capacidade
        load = float(np.sum(demands[customers]))
        if load > float(capacity) + 1e-9:
            #raise ValueError(f"Rota viola capacidade no MIP start: load={load} > Q={capacity}")
            print(f"Rota viola capacidade no MIP start: load={load} > Q={capacity}")

        # define arcos e fluxo SCF
        # arcos: (r[t] -> r[t+1])
        for t in range(len(r) - 1):
            i, j = r[t], r[t + 1]
            if i == j:
                continue
            start_x[i, j] = 1.0

        # y: demanda restante após sair de cada nó
        # na prática, y no arco (prev -> curr) = soma das demandas ainda não entregues a partir de curr
        suffix = 0.0
        # percorre clientes da rota de trás pra frente
        for idx in range(len(customers) - 1, -1, -1):
            suffix += float(demands[customers[idx]])

        # y[0, first] = total da rota
        first = customers[0]
        start_y[0, first] = suffix

        # para cada arco entre clientes: y[ci, c{i+1}] = soma demandas de c{i+1}..fim
        rem = suffix
        for idx in range(len(customers) - 1):
            ci = customers[idx]
            cj = customers[idx + 1]
            rem -= float(demands[ci])
            start_y[ci, cj] = rem

        # último cliente -> depot tem y = 0 (compatível com ub y[i,0]=0)
        last = customers[-1]
        start_y[last, 0] = 0.0

    # opcional: se quiser exigir que todos clientes apareçam exatamente 1x
    # missing = [i for i in range(1, n1) if not used[i]]
    # if len(missing) > 0:
    #     raise ValueError(f"Clientes faltando na initial_solution: {missing[:20]} ...")

    return start_x, start_y

# ============================================================
# Reaproveita seu reconstruidor de rotas
# ============================================================

def routes_from_xvars(x: gp.tupledict, n: int, tol: float = 0.5) -> List[List[int]]:
    succ = [-1] * (n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j and x[i, j].X > tol:
                succ[i] = j
                break

    routes = []
    used = set()
    depot_nexts = [j for j in range(1, n + 1) if x[0, j].X > tol]

    for start in depot_nexts:
        if start in used:
            continue
        route = [0, start]
        used.add(start)
        cur = start
        while True:
            nxt = succ[cur]
            if nxt == -1:
                raise RuntimeError(f"Nó {cur} sem sucessor.")
            route.append(nxt)
            if nxt == 0:
                break
            if nxt in used:
                break
            used.add(nxt)
            cur = nxt
        #routes.append(route)
        routes += route
    #print("------")
    #print(routes)
    routes = np.array(routes)
    #print(routes)
    routes = np.trim_zeros(routes)
    #print(routes)
    routes = [0] + routes.tolist() + [0]  # Add depot
    #print(routes)
    return routes


# ============================================================
# Seu solver de instância (inalterado na lógica)
# ============================================================

def __solve_cvrp_instance_gurobi(
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float = 1.0,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    threads: int = 1,
    verbose: bool = False,
) -> Tuple[float, float, float, List[List[int]]]:
    n1 = dist.shape[0]
    assert dist.shape == (n1, n1)
    assert demands.shape == (n1,)
    assert abs(demands[0]) < 1e-12
    n = n1 - 1

    model = gp.Model("cvrp_scf")
    if not verbose:
        model.Params.OutputFlag = 0
    model.Params.Threads = int(threads)
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    x = model.addVars(n1, n1, vtype=GRB.BINARY, obj=dist, name="x")
    for i in range(n1):
        x[i, i].ub = 0

    y = model.addVars(n1, n1, vtype=GRB.CONTINUOUS, lb=0.0, ub=capacity, name="y")
    for i in range(n1):
        y[i, i].ub = 0

    for i in range(1, n1):
        y[i, 0].ub = 0.0

    for i in range(1, n1):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n1) if j != i) == 1, name=f"out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(n1) if j != i) == 1, name=f"in_{i}")

    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n1)) == gp.quicksum(x[i, 0] for i in range(1, n1)),
        name="depot_balance_routes"
    )

    Q = float(capacity)
    for i in range(n1):
        for j in range(n1):
            if i != j:
                model.addConstr(y[i, j] <= Q * x[i, j], name=f"cap_{i}_{j}")

    for j in range(1, n1):
        inflow = gp.quicksum(y[i, j] for i in range(n1) if i != j)
        outflow = gp.quicksum(y[j, k] for k in range(n1) if k != j)
        model.addConstr(inflow - outflow == float(demands[j]), name=f"flow_{j}")

    total_d = float(np.sum(demands[1:]))
    out_depot = gp.quicksum(y[0, k] for k in range(1, n1))
    in_depot = gp.quicksum(y[i, 0] for i in range(1, n1))
    model.addConstr(out_depot - in_depot == total_d, name="flow_depot")

    model.ModelSense = GRB.MINIMIZE
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Gurobi terminou com status {model.Status}")

    routes = routes_from_xvars(x, n=n, tol=0.5)
    return float(model.objVal), float(model.ObjBound), float(model.Runtime), routes



def solve_cvrp_instance_gurobi(
    dist: np.ndarray,
    demands: np.ndarray,
    capacity: float = 1.0,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    threads: int = 1,
    verbose: bool = False,
    initial_solution: Optional[List[int]] = None,   # << NOVO
    Capacity: float | int = 1.0, # Capacidade sem normalizacao
) -> Tuple[float, float, float, List[List[int]]]:
    n1 = dist.shape[0]
    assert dist.shape == (n1, n1)
    assert demands.shape == (n1,)
    assert abs(demands[0]) < 1e-12
    n = n1 - 1

    model = gp.Model("cvrp_scf")
    if not verbose:
        model.Params.OutputFlag = 0
    model.Params.Threads = int(threads)
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    x = model.addVars(n1, n1, vtype=GRB.BINARY, obj=dist, name="x")
    for i in range(n1):
        x[i, i].ub = 0

    y = model.addVars(n1, n1, vtype=GRB.CONTINUOUS, lb=0.0, ub=capacity, name="y")
    for i in range(n1):
        y[i, i].ub = 0

    for i in range(1, n1):
        y[i, 0].ub = 0.0

    # ------------------------------------------------------------
    # NOVO: MIP start (se initial_solution for fornecida)
    # ------------------------------------------------------------
    if initial_solution is not None:
        #print(initial_solution)
        routes0 = _split_routes_from_sequence(initial_solution)
        #print(routes0)
        start_x, start_y = _build_mip_start_from_routes(
            routes=routes0, n1=n1, demands=demands, capacity=float(capacity), Q_non_norm = float(Capacity),
        )
        # set Start completo (n^2). Se preferir, dá para setar só arcos=1.
        for i in range(n1):
            for j in range(n1):
                if i == j:
                    continue
                x[i, j].Start = float(start_x[i, j])
                y[i, j].Start = float(start_y[i, j])

    # (resto do seu modelo fica igual)
    for i in range(1, n1):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n1) if j != i) == 1, name=f"out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(n1) if j != i) == 1, name=f"in_{i}")

    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n1)) == gp.quicksum(x[i, 0] for i in range(1, n1)),
        name="depot_balance_routes"
    )

    Q = float(capacity)
    for i in range(n1):
        for j in range(n1):
            if i != j:
                model.addConstr(y[i, j] <= Q * x[i, j], name=f"cap_{i}_{j}")

    for j in range(1, n1):
        inflow = gp.quicksum(y[i, j] for i in range(n1) if i != j)
        outflow = gp.quicksum(y[j, k] for k in range(n1) if k != j)
        model.addConstr(inflow - outflow == float(demands[j]), name=f"flow_{j}")

    total_d = float(np.sum(demands[1:]))
    out_depot = gp.quicksum(y[0, k] for k in range(1, n1))
    in_depot = gp.quicksum(y[i, 0] for i in range(1, n1))
    model.addConstr(out_depot - in_depot == total_d, name="flow_depot")

    model.ModelSense = GRB.MINIMIZE
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Gurobi terminou com status {model.Status}")

    routes = routes_from_xvars(x, n=n, tol=0.5)
    return float(model.objVal), float(model.ObjBound), float(model.Runtime), routes



# ============================================================
# Worker (processo): resolve 1 instância já em numpy
# ============================================================

def ___worker_solve_one(args):
    """
    args = (idx, dist, demands, capacity, time_limit, mip_gap, threads, verbose)
    Retorna (idx, obj, bound, rt, routes_str)
    """
    (idx, dist, demands, capacity, time_limit, mip_gap, threads, verbose) = args
    obj, bound, rt, routes = solve_cvrp_instance_gurobi(
        dist=dist,
        demands=demands,
        capacity=capacity,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        verbose=verbose,
    )
    return idx, obj, bound, rt, str(routes)

def _worker_solve_one(args):
    """
    args = (idx, dist, demands, capacity, time_limit, mip_gap, threads, verbose, initial_solution)
    Retorna (idx, obj, bound, rt, routes_str)
    """
    (idx, dist, demands, capacity, time_limit, mip_gap, threads, verbose, initial_solution, Capacity) = args
    obj, bound, rt, routes = solve_cvrp_instance_gurobi(
        dist=dist,
        demands=demands,
        capacity=capacity,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        verbose=verbose,
        initial_solution=initial_solution,  # << NOVO
        Capacity = Capacity,
    )
    return idx, obj, bound, rt, str(routes)

# ============================================================
# Função paralela: pré-calcula dist e demanda (serial),
# depois resolve em paralelo (processos)
# ============================================================

def solve_cvrp_dataset_gurobi_parallel(
    dataset,
    calc: Callable[[Any, Any], Any],
    time_limit_per_instance: float | None = None,
    mip_gap: float | None = None,
    capacity: float = 1.0,
    num_workers: int | None = None,
    threads_per_worker: int = 1,
    verbose: bool = False,
    show_progress: bool = True,
    initial_solution: Optional[List[Optional[List[int]]]] = None, 
):
    """
    dataset[idx] -> (depot, node)
      depot: (1,2) ou (2,)
      node:  (N,3) col0=x, col1=y, col2=demand em [0,1]

    Estratégia:
      1) Serial: extrai demands/dist em numpy (evita pickle de Torch/GPU).
      2) Paralelo: resolve cada instância do Gurobi em um processo.

    Retorna:
      costs, bounds, runtimes, routes_list
    """
    n_instances = len(dataset)
    if initial_solution is not None and len(initial_solution) != n_instances:
        raise ValueError(
            f"initial_solution deve ter len==n_instances ({n_instances}), "
            f"mas veio len={len(initial_solution)}"
        )
    
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    # --------------------------
    # 1) Pré-cálculo serial
    # --------------------------
    dists: List[np.ndarray] = [None] * n_instances  # type: ignore
    demands_list: List[np.ndarray] = [None] * n_instances  # type: ignore

    it = range(n_instances)
    if show_progress:
        it = tqdm(it, desc="Precomputando dist/demands (serial)")

    for idx in it:
        depot, node = dataset[idx]

        # demands (N+1)
        dem = node[:, 2].detach().cpu().numpy().astype(float, copy=False)
        demands = np.concatenate(([0.0], dem), axis=0)

        # dist (N+1,N+1)
        dist = calc(depot[None, :, :], node[None, :, :]).squeeze(0)
        dist = dist.detach().cpu().numpy().astype(float, copy=False)

        demands_list[idx] = demands
        dists[idx] = dist

    # --------------------------
    # 2) Resolver em paralelo
    # --------------------------
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args_iter = [
        (idx, dists[idx], demands_list[idx], float(capacity),
         time_limit_per_instance, mip_gap, int(threads_per_worker), bool(verbose),
         (initial_solution[idx] if initial_solution is not None else None), dataset.get_Capacity())  # << NOVO
        for idx in range(n_instances)
    ]

    costs = np.empty(n_instances, dtype=float)
    bounds = np.empty(n_instances, dtype=float)
    runtimes = np.empty(n_instances, dtype=float)
    routes_list = [None] * n_instances  # type: ignore

    with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
        futs = [ex.submit(_worker_solve_one, args) for args in args_iter]

        if show_progress:
            pbar = tqdm(total=n_instances, desc="Resolvendo Gurobi (paralelo)")
        else:
            pbar = None

        for fut in as_completed(futs):
            idx, obj, bound, rt, routes_str = fut.result()
            costs[idx] = obj
            bounds[idx] = bound
            runtimes[idx] = rt
            routes_list[idx] = routes_str
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    return costs, bounds, runtimes, routes_list




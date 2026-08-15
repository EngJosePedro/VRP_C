from typing import Any, Dict, List, Optional
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_cvrp_ortools(
    distance_matrix: List[List[float]],
    demands: List[int],
    vehicle_capacities: List[int],
    depot: int = 0,
    time_limit_seconds: int = 30,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
    solution_limit: Optional[int] = None,
    allow_drop_nodes: bool = False,
    drop_penalty: int = 10**6,
    max_distance_per_vehicle: Optional[int] = None,
    fixed_cost_per_vehicle: Optional[List[int]] = None,
    starts: Optional[List[int]] = None,
    ends: Optional[List[int]] = None,
    log_search: bool = False,
) -> Dict[str, Any]:
    """
    Resolve um CVRP usando Google OR-Tools.

    Parâmetros
    ----------
    distance_matrix : List[List[int]]
        Matriz NxN de distâncias/custos inteiros.
    demands : List[int]
        Vetor de demanda por nó. Normalmente demands[depot] = 0.
    vehicle_capacities : List[int]
        Capacidade de cada veículo.
    depot : int, default=0
        Índice do depósito, usado se starts/ends não forem fornecidos.
    time_limit_seconds : int, default=30
        Limite de tempo do solver.
    first_solution_strategy : str, default="PATH_CHEAPEST_ARC"
        Estratégia inicial do OR-Tools.
    local_search_metaheuristic : str, default="GUIDED_LOCAL_SEARCH"
        Metaheurística de melhoria.
    solution_limit : Optional[int]
        Limite opcional no número de soluções exploradas.
    allow_drop_nodes : bool, default=False
        Se True, permite não atender clientes mediante penalidade.
    drop_penalty : int, default=10**6
        Penalidade por cliente não atendido.
    max_distance_per_vehicle : Optional[int]
        Se informado, impõe limite máximo de distância por veículo.
    fixed_cost_per_vehicle : Optional[List[int]]
        Custo fixo por veículo, útil para desincentivar uso excessivo de veículos.
    starts : Optional[List[int]]
        Nó inicial de cada veículo.
    ends : Optional[List[int]]
        Nó final de cada veículo.
    log_search : bool, default=False
        Se True, ativa log do OR-Tools.

    Retorno
    -------
    Dict[str, Any]
        Dicionário com status, custo total, rotas, cargas, distâncias e nós não atendidos.
    """

    # -----------------------------
    # Validação básica
    # -----------------------------
    n = len(distance_matrix)
    if n == 0:
        raise ValueError("distance_matrix não pode ser vazia.")

    if any(len(row) != n for row in distance_matrix):
        raise ValueError("distance_matrix deve ser quadrada (NxN).")

    if len(demands) != n:
        raise ValueError("demands deve ter o mesmo tamanho da distance_matrix.")

    num_vehicles = len(vehicle_capacities)
    if num_vehicles == 0:
        raise ValueError("vehicle_capacities não pode ser vazio.")

    if starts is not None and len(starts) != num_vehicles:
        raise ValueError("starts deve ter tamanho igual ao número de veículos.")

    if ends is not None and len(ends) != num_vehicles:
        raise ValueError("ends deve ter tamanho igual ao número de veículos.")

    if fixed_cost_per_vehicle is not None and len(fixed_cost_per_vehicle) != num_vehicles:
        raise ValueError("fixed_cost_per_vehicle deve ter tamanho igual ao número de veículos.")

    if any(d < 0 for d in demands):
        raise ValueError("Esta função assume demands >= 0 para CVRP clássico de entrega/coleta.")

    # OR-Tools routing trabalha melhor com inteiros
    distance_matrix = [[int(v) for v in row] for row in distance_matrix]
    #distance_matrix = [[float(v) for v in row] for row in distance_matrix]
    demands = [int(v) for v in demands]
    #demands = [float(v) for v in demands]
    vehicle_capacities = [int(v) for v in vehicle_capacities]

    # -----------------------------
    # Manager + RoutingModel
    # -----------------------------
    if starts is not None and ends is not None:
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, starts, ends)
    else:
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)

    routing = pywrapcp.RoutingModel(manager)

    # -----------------------------
    # Callback de distância
    # -----------------------------
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # -----------------------------
    # Callback de demanda + capacidade
    # -----------------------------
    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    # Capacidade dos veículos
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # slack
        vehicle_capacities,
        True,  # start cumul at zero
        "Capacity",
    )

    capacity_dimension = routing.GetDimensionOrDie("Capacity")

    # -----------------------------
    # Limite opcional de distância
    # -----------------------------
    distance_dimension = None
    if max_distance_per_vehicle is not None:
        routing.AddDimension(
            transit_callback_index,
            0,  # slack
            int(max_distance_per_vehicle),
            True,
            "Distance",
        )
        distance_dimension = routing.GetDimensionOrDie("Distance")

    # -----------------------------
    # Custo fixo por veículo
    # -----------------------------
    if fixed_cost_per_vehicle is not None:
        for v, fixed_cost in enumerate(fixed_cost_per_vehicle):
            routing.SetFixedCostOfVehicle(int(fixed_cost), v)

    # -----------------------------
    # Permitir clientes não atendidos
    # -----------------------------
    if allow_drop_nodes:
        for node in range(n):
            if node == depot:
                continue
            routing.AddDisjunction([manager.NodeToIndex(node)], int(drop_penalty))

    # -----------------------------
    # Estratégias de busca
    # -----------------------------
    first_solution_map = {
        "AUTOMATIC": routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "PATH_MOST_CONSTRAINED_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "SWEEP": routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        "ALL_UNPERFORMED": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        "BEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
        "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        "SEQUENTIAL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION,
        "LOCAL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        "GLOBAL_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
        "LOCAL_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
        "FIRST_UNBOUND_MIN_VALUE": routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
    }

    local_search_map = {
        "AUTOMATIC": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        "GREEDY_DESCENT": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        "GENERIC_TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
    }

    if first_solution_strategy not in first_solution_map:
        raise ValueError(
            f"first_solution_strategy inválida: {first_solution_strategy}. "
            f"Opções: {list(first_solution_map.keys())}"
        )

    if local_search_metaheuristic not in local_search_map:
        raise ValueError(
            f"local_search_metaheuristic inválida: {local_search_metaheuristic}. "
            f"Opções: {list(local_search_map.keys())}"
        )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = first_solution_map[first_solution_strategy]
    search_parameters.local_search_metaheuristic = local_search_map[local_search_metaheuristic]
    search_parameters.time_limit.seconds = int(time_limit_seconds)
    search_parameters.log_search = bool(log_search)

    if solution_limit is not None:
        search_parameters.solution_limit = int(solution_limit)

    # -----------------------------
    # Resolve
    # -----------------------------
    solution = routing.SolveWithParameters(search_parameters)

    # OR-Tools mais novo: status() retorna inteiro.
    # Evite usar pywrapcp.RoutingModel.ROUTING_*
    status_code = routing.status()

    status_map = {
        0: "ROUTING_NOT_SOLVED",
        1: "ROUTING_SUCCESS",
        2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
        3: "ROUTING_FAIL",
        4: "ROUTING_FAIL_TIMEOUT",
        5: "ROUTING_INVALID",
        6: "ROUTING_INFEASIBLE",
        7: "ROUTING_OPTIMAL",
    }

    solver_status = status_map.get(status_code, f"UNKNOWN_STATUS_{status_code}")


    # Status do solver
    """status_map = {
        pywrapcp.RoutingModel.ROUTING_NOT_SOLVED: "NOT_SOLVED",
        pywrapcp.RoutingModel.ROUTING_SUCCESS: "SUCCESS",
        pywrapcp.RoutingModel.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED: "PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
        pywrapcp.RoutingModel.ROUTING_FAIL: "FAIL",
        pywrapcp.RoutingModel.ROUTING_FAIL_TIMEOUT: "FAIL_TIMEOUT",
        pywrapcp.RoutingModel.ROUTING_INVALID: "INVALID",
        pywrapcp.RoutingModel.ROUTING_INFEASIBLE: "INFEASIBLE",
        pywrapcp.RoutingModel.ROUTING_OPTIMAL: "OPTIMAL",
    }
    solver_status = status_map.get(routing.status(), f"UNKNOWN_STATUS_{routing.status()}")"""

    if solution is None:
        return {
            "status": solver_status,
            "objective_value": None,
            "routes": [],
            "dropped_nodes": [],
            "total_distance": None,
            "total_load": None,
            "message": "Nenhuma solução encontrada.",
        }

    # -----------------------------
    # Extrai solução
    # -----------------------------
    routes = []
    total_distance = 0
    total_load = 0

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)

        route_nodes = []
        route_arc_cost = 0
        route_load_progress = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)

            load_here = solution.Value(capacity_dimension.CumulVar(index))
            route_load_progress.append(load_here)

            next_index = solution.Value(routing.NextVar(index))
            route_arc_cost += routing.GetArcCostForVehicle(index, next_index, vehicle_id)
            index = next_index

        end_node = manager.IndexToNode(index)
        route_nodes.append(end_node)

        # Carga final da rota
        end_load = solution.Value(capacity_dimension.CumulVar(index))

        route_info = {
            "vehicle_id": vehicle_id,
            "route": route_nodes,
            "distance": route_arc_cost,
            "end_load": end_load,
            "load_progress_before_service": route_load_progress,
            "used": len(route_nodes) > 2 or (len(route_nodes) == 2 and route_nodes[0] != route_nodes[1]),
        }

        if distance_dimension is not None:
            route_info["distance_cumul_end"] = solution.Value(distance_dimension.CumulVar(index))

        routes.append(route_info)
        total_distance += route_arc_cost
        total_load += end_load

    # Nós descartados
    dropped_nodes = []
    for node in range(n):
        if routing.IsStart(manager.NodeToIndex(node)) or routing.IsEnd(manager.NodeToIndex(node)):
            continue

        index = manager.NodeToIndex(node)
        if solution.Value(routing.NextVar(index)) == index:
            dropped_nodes.append(node)

    return {
        "status": solver_status,
        "objective_value": solution.ObjectiveValue(),
        "routes": routes,
        "dropped_nodes": dropped_nodes,
        "total_distance": total_distance,
        "total_load": total_load,
        "num_vehicles": num_vehicles,
        "depot": depot,
    }
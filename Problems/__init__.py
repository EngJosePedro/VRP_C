from Problems.vrp.problem_cvrp import VRP
from .tsp.problem import PROBLEM as TSP

def load_problem(problem: str):
    return {
        "tsp": TSP,
        "vrp": VRP,
    }.get(problem, None)
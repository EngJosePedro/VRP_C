
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
cimport cython
cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t


# ============================================================
# Helpers básicos
# ============================================================

def _split_routes_from_sequence(seq):
    """
    Converte sequência no formato [0, a, b, 0, c, 0, ...]
    para lista de rotas [[0,a,b,0], [0,c,0], ...].
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

    routes = []
    cur = [0]
    for v in s[1:]:
        if v == 0:
            if len(cur) > 1:
                cur.append(0)
                routes.append(cur)
            cur = [0]
        else:
            cur.append(v)

    return routes


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _route_cost_c(list route, cnp.ndarray[DTYPE_t, ndim=2] dist):
    cdef Py_ssize_t i, n
    cdef double cost = 0.0
    n = len(route)
    if n < 2:
        return 0.0
    for i in range(n - 1):
        cost += dist[route[i], route[i+1]]
    return cost


def route_cost_0(route, cnp.ndarray[DTYPE_t, ndim=2] dist):
    return _route_cost_c(route, dist)


def total_cost_0(routes, cnp.ndarray[DTYPE_t, ndim=2] dist):
    cdef double s = 0.0
    cdef list r
    for r in routes:
        s += _route_cost_c(r, dist)
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _route_load_c(list route, cnp.ndarray[DTYPE_t, ndim=1] demand):
    cdef Py_ssize_t i, n
    cdef double load = 0.0
    n = len(route)
    for i in range(n):
        if route[i] != 0:
            load += demand[route[i]]
    return load


def feasible_route_capacity(route, cnp.ndarray[DTYPE_t, ndim=1] demand, double cap):
    return _route_load_c(route, demand) <= cap + 1e-9


def feasible_solution_capacity(routes, cnp.ndarray[DTYPE_t, ndim=1] demand, double cap):
    cdef list r
    for r in routes:
        if _route_load_c(r, demand) > cap + 1e-9:
            return False
    return True


def _check_routes(
    routes,
    int n1,
    cnp.ndarray[DTYPE_t, ndim=1] demands,
    double capacity,
    Q_non_norm=None,
):
    cdef int n = n1 - 1
    cdef cnp.ndarray used = np.zeros(n1, dtype=np.uint8)

    cdef list r
    cdef int v, c
    cdef list customers
    cdef double load

    for r in routes:
        if len(r) < 3 or r[0] != 0 or r[-1] != 0:
            continue

        for v in r:
            if v < 0 or v > n:
                raise ValueError(f"Nó fora do range na initial_solution: {v} (esperado 0..{n})")

        customers = [v for v in r[1:-1] if v != 0]
        if len(customers) == 0:
            continue

        for c in customers:
            if used[c]:
                raise ValueError(f"Cliente repetido na initial_solution: {c}")
            used[c] = 1

        load = 0.0
        for c in customers:
            load += demands[c]
        if load > capacity + 1e-9:
            raise ValueError(f"Rota viola capacidade no MIP start: load={load} > Q={capacity}")


# ============================================================
# 2-opt
# ============================================================

def two_opt_best_route(route, cnp.ndarray[DTYPE_t, ndim=2] dist, bint return_cost=False):
    """
    Retorna (melhor_rota, delta) ou (melhor_rota, delta, custo_final)
    """
    cdef Py_ssize_t n, i, k
    cdef double best_delta, base_cost, c_cost, delta, best_cost
    cdef bint improve
    cdef list best_route, cand

    n = len(route)
    if n < 4:
        if return_cost:
            return route, 0.0, 0.0
        return route, 0.0

    base_cost = _route_cost_c(route, dist)

    improve = True
    while improve:
        improve = False
        best_delta = 0.0
        best_route = route
        best_cost = base_cost

        for i in range(1, n - 2):
            if route[i] == 0:
                continue
            for k in range(i + 1, n - 1):
                if route[k] == 0:
                    continue

                cand = route[:i] + list(reversed(route[i:k+1])) + route[k+1:]
                c_cost = _route_cost_c(cand, dist)
                delta = c_cost - base_cost

                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_route = cand
                    best_cost = c_cost
                    improve = True

        if improve:
            route = best_route
            base_cost = best_cost
            n = len(route)

    if return_cost:
        return route, 0.0, base_cost
    return route, 0.0


# ============================================================
# swap(m,n) + 2-opt
# ============================================================

cdef bint _valid_segment(list r, int s, int L):
    cdef list seg
    if L == 0:
        return 1 <= s <= len(r) - 1
    if len(r) < L + 2:
        return False
    if not (1 <= s <= len(r) - 1 - L):
        return False
    seg = r[s:s + L]
    return len(seg) == L and all(x != 0 for x in seg)


def apply_swap_mn_with_2opt(
    routes,
    cnp.ndarray[DTYPE_t, ndim=2] dist,
    cnp.ndarray[DTYPE_t, ndim=1] demand,
    double cap,
    *,
    int a,
    int sa,
    int m,
    int b,
    int sb,
    int n,
    bint only_run_2opt_if_promising=True,
):
    """
    Mesma ideia do seu código original:
    troca segmento m da rota a com segmento n da rota b,
    roda 2-opt nas 2 rotas afetadas e devolve (new_routes, delta).
    """
    cdef list ra, rb
    cdef list seg_a, seg_b
    cdef list ra_removed, rb_removed
    cdef list new_ra, new_rb
    cdef list opt_ra, opt_rb
    cdef list new_routes
    cdef double old_local, raw_local, new_local, delta

    if a == b:
        return None
    if m < 0 or n < 0:
        return None
    if m == 0 and n == 0:
        return None
    if not (0 <= a < len(routes) and 0 <= b < len(routes)):
        return None

    ra = routes[a]
    rb = routes[b]

    if not _valid_segment(ra, sa, m):
        return None
    if not _valid_segment(rb, sb, n):
        return None

    seg_a = ra[sa:sa + m] if m > 0 else []
    seg_b = rb[sb:sb + n] if n > 0 else []

    ra_removed = ra[:sa] + ra[sa + m:] if m > 0 else ra
    rb_removed = rb[:sb] + rb[sb + n:] if n > 0 else rb

    new_ra = ra_removed[:sa] + seg_b + ra_removed[sa:]
    new_rb = rb_removed[:sb] + seg_a + rb_removed[sb:]

    if not (new_ra and new_ra[0] == 0 and new_ra[-1] == 0):
        return None
    if not (new_rb and new_rb[0] == 0 and new_rb[-1] == 0):
        return None

    if _route_load_c(new_ra, demand) > cap + 1e-9:
        return None
    if _route_load_c(new_rb, demand) > cap + 1e-9:
        return None

    old_local = _route_cost_c(ra, dist) + _route_cost_c(rb, dist)
    raw_local = _route_cost_c(new_ra, dist) + _route_cost_c(new_rb, dist)

    if only_run_2opt_if_promising and raw_local >= old_local - 1e-12:
        return None

    opt_ra, _ = two_opt_best_route(new_ra, dist)
    opt_rb, _ = two_opt_best_route(new_rb, dist)

    if _route_load_c(opt_ra, demand) > cap + 1e-9:
        return None
    if _route_load_c(opt_rb, demand) > cap + 1e-9:
        return None

    new_routes = [r.copy() for r in routes]
    new_routes[a] = opt_ra
    new_routes[b] = opt_rb

    new_local = _route_cost_c(opt_ra, dist) + _route_cost_c(opt_rb, dist)
    delta = new_local - old_local

    return new_routes, delta


# ============================================================
# Busca local principal
# ============================================================

def local_search_swap_mn_2opt_fast(
    routes,
    cnp.ndarray[DTYPE_t, ndim=2] dist,
    cnp.ndarray[DTYPE_t, ndim=1] demand,
    double cap,
    *,
    int max_m=3,
    int max_n=3,
    int max_iters=10000,
    bint first_improvement=False,
    bint debug_check=True,
    bint only_run_2opt_if_promising=True,
):
    """
    Versão Cython baseada na sua local_search_swap_mn_2opt original.
    first_improvement:
        True  -> para no primeiro movimento melhorante
        False -> best improvement
    """
    cdef list cur
    cdef int n_nodes
    cdef int it, R, m, n, a, sa, b, sb
    cdef list ra, rb
    cdef double best_delta, delta
    cdef object cand
    cdef list best_routes
    cdef bint stop_outer

    cur = _split_routes_from_sequence(routes.copy())

    if debug_check:
        if not feasible_solution_capacity(cur, demand, cap):
            raise ValueError("Solução inicial inviável por capacidade.")

    n_nodes = dist.shape[0]
    _check_routes(cur, n_nodes, demand, cap, cap)

    it = 0
    while it < max_iters:
        it += 1
        best_delta = 0.0
        best_routes = None
        stop_outer = False

        R = len(cur)

        for m in range(0, max_m + 1):
            if stop_outer:
                break

            for n in range(0, max_n + 1):
                if stop_outer:
                    break

                if m == 0 and n == 0:
                    continue

                for a in range(R):
                    if stop_outer:
                        break

                    ra = cur[a]

                    for sa in range(1, len(ra)):
                        if stop_outer:
                            break

                        for b in range(R):
                            if stop_outer:
                                break

                            if b == a:
                                continue

                            rb = cur[b]

                            for sb in range(1, len(rb)):
                                cand = apply_swap_mn_with_2opt(
                                    cur,
                                    dist,
                                    demand,
                                    cap,
                                    a=a,
                                    sa=sa,
                                    m=m,
                                    b=b,
                                    sb=sb,
                                    n=n,
                                    only_run_2opt_if_promising=only_run_2opt_if_promising,
                                )

                                if cand is None:
                                    continue

                                delta = cand[1]

                                if delta < best_delta - 1e-12:
                                    best_delta = delta
                                    best_routes = cand[0]

                                    if first_improvement:
                                        stop_outer = True
                                        break

        if best_routes is None or best_delta >= -1e-12:
            break

        cur = best_routes

    routes = []
    for ra in cur:
        routes += ra

    return routes, total_cost_0(cur, dist)
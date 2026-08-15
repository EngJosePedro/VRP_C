
import torch

# ============================================================
# Utilities
# ============================================================

@torch.no_grad()
def check_dist_matrix(dist: torch.Tensor, atol: float = 1e-6):
    if dist.dim() == 2:
        d = dist
        diag = torch.diag(d)
        max_diag = diag.abs().max().item()
        diag_ok = max_diag <= atol
        max_asym = (d - d.T).abs().max().item()
        sym_ok = max_asym <= atol
        return diag_ok, sym_ok, max_diag, max_asym

    diag = torch.diagonal(dist, dim1=-2, dim2=-1)
    max_diag = diag.abs().max().item()
    diag_ok = max_diag <= atol
    max_asym = (dist - dist.transpose(-1, -2)).abs().max().item()
    sym_ok = max_asym <= atol
    return diag_ok, sym_ok, max_diag, max_asym


@torch.no_grad()
def route_cost(distS: torch.Tensor, route: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    if route.dim() == 1:
        L = int(length.item())
        if L <= 1:
            return torch.zeros((), device=route.device, dtype=distS.dtype)
        u = route[:L - 1]
        v = route[1:L]
        if distS.dim() == 2:
            return distS[u, v].sum()
        return distS[0, u, v].sum()

    S, Lmax = route.shape
    t = torch.arange(Lmax - 1, device=route.device)[None, :].expand(S, Lmax - 1)
    mask = t < (length[:, None] - 1)
    u = route[:, :-1]
    v = route[:, 1:]
    if distS.dim() == 2:
        edge = distS[u, v]
    else:
        s_idx = torch.arange(S, device=route.device)[:, None]
        edge = distS[s_idx, u, v]
    return (edge * mask).sum(dim=1)


@torch.no_grad()
def batched_route_costs(distS: torch.Tensor, routes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    S, Lmax = routes.shape
    t = torch.arange(Lmax - 1, device=routes.device)[None, :].expand(S, Lmax - 1)
    mask = t < (lengths[:, None] - 1)
    u = routes[:, :-1]
    v = routes[:, 1:]
    s_idx = torch.arange(S, device=routes.device)[:, None]
    edge = distS[s_idx, u, v]
    return (edge * mask).sum(dim=1)


@torch.no_grad()
def double_check_small_negatives(
    distS: torch.Tensor,
    routes: torch.Tensor,
    routes_new: torch.Tensor,
    lengths: torch.Tensor,
    best_delta: torch.Tensor,
    eps: float = 1e-6
):
    sus = (best_delta < 0) & (best_delta > -eps)
    if not sus.any():
        return best_delta, None, None

    CO = batched_route_costs(distS[sus], routes[sus], lengths[sus])
    CA = batched_route_costs(distS[sus], routes_new[sus], lengths[sus])
    exact = CA - CO

    best_delta2 = best_delta.clone()
    best_delta2[sus] = exact
    return best_delta2, CO, CA


# ============================================================
# V3 Kernel: two-opt intra-route with multi-depots/satellites
# ============================================================

@torch.no_grad()
def two_opt_best_intra_routes_batched_chunked_v3(
    routes: torch.Tensor,
    distS: torch.Tensor,
    lengths: torch.Tensor,
    n_deps: int,
    i_chunk: int = 64,
    k_chunk: int = 64,
    s_chunk: int = 2048,
    avaliar=None
):
    device = routes.device
    S, Lmax = routes.shape
    dtype = distS.dtype
    N = distS.size(-1)

    can = lengths >= 5
    if avaliar is not None:
        can = can & avaliar

    if not can.any():
        return routes, torch.zeros((S,), device=device, dtype=dtype)

    best_delta = torch.full((S,), float("inf"), device=device, dtype=dtype)
    best_i = torch.zeros((S,), device=device, dtype=torch.long)
    best_k = torch.zeros((S,), device=device, dtype=torch.long)

    for s0 in range(0, S, s_chunk):
        s1 = min(S, s0 + s_chunk)
        Sb = s1 - s0

        routes_b = routes[s0:s1]
        lengths_b = lengths[s0:s1]
        can_b = can[s0:s1]

        dist_flat = distS[s0:s1].reshape(Sb, N * N)

        best_delta_b = best_delta[s0:s1]
        best_i_b = best_i[s0:s1]
        best_k_b = best_k[s0:s1]

        L = lengths_b[:, None, None]

        for i0 in range(1, Lmax - 2, i_chunk):
            i1 = min(Lmax - 2, i0 + i_chunk)
            i_blk = torch.arange(i0, i1, device=device, dtype=torch.long)
            Ci = i_blk.numel()
            if Ci == 0:
                continue

            for k0 in range(2, Lmax - 1, k_chunk):
                k1 = min(Lmax - 1, k0 + k_chunk)
                k_blk = torch.arange(k0, k1, device=device, dtype=torch.long)
                Kc = k_blk.numel()
                if Kc == 0:
                    continue

                I = i_blk[:, None].expand(Ci, Kc)
                K = k_blk[None, :].expand(Ci, Kc)

                valid = (I[None] < K[None]) & (K[None] <= (L - 2)) & (I[None] <= (L - 3))
                valid = valid & can_b[:, None, None]

                if not valid.any():
                    continue

                idx_prev = (I - 1)[None, :, :].expand(Sb, Ci, Kc)
                idx_a = I[None, :, :].expand(Sb, Ci, Kc)
                idx_c = K[None, :, :].expand(Sb, Ci, Kc)
                idx_nxt = (K + 1)[None, :, :].expand(Sb, Ci, Kc)

                prev = routes_b.gather(1, idx_prev.reshape(Sb, -1)).reshape(Sb, Ci, Kc)
                a = routes_b.gather(1, idx_a.reshape(Sb, -1)).reshape(Sb, Ci, Kc)
                c = routes_b.gather(1, idx_c.reshape(Sb, -1)).reshape(Sb, Ci, Kc)
                nxt = routes_b.gather(1, idx_nxt.reshape(Sb, -1)).reshape(Sb, Ci, Kc)

                valid = valid & (a >= n_deps) & (c >= n_deps)
                if not valid.any():
                    continue

                idx1 = (prev * N + c).reshape(Sb, -1)
                idx2 = (a * N + nxt).reshape(Sb, -1)
                idx3 = (prev * N + a).reshape(Sb, -1)
                idx4 = (c * N + nxt).reshape(Sb, -1)

                d1 = dist_flat.gather(1, idx1).reshape(Sb, Ci, Kc)
                d2 = dist_flat.gather(1, idx2).reshape(Sb, Ci, Kc)
                d3 = dist_flat.gather(1, idx3).reshape(Sb, Ci, Kc)
                d4 = dist_flat.gather(1, idx4).reshape(Sb, Ci, Kc)

                delta = d1 + d2 - d3 - d4
                delta = delta.masked_fill(~valid, float("inf"))

                flat = delta.view(Sb, -1)
                blk_best_val, blk_best_pos = torch.min(flat, dim=1)

                better = blk_best_val < best_delta_b
                if better.any():
                    best_delta_b = torch.where(better, blk_best_val, best_delta_b)

                    i_local = blk_best_pos // Kc
                    k_local = blk_best_pos % Kc

                    i_pick = i_blk[i_local]
                    k_pick = k_blk[k_local]

                    best_i_b = torch.where(better, i_pick, best_i_b)
                    best_k_b = torch.where(better, k_pick, best_k_b)

        best_delta[s0:s1] = best_delta_b
        best_i[s0:s1] = best_i_b
        best_k[s0:s1] = best_k_b

    apply = can & torch.isfinite(best_delta) & (best_delta < 0)
    if not apply.any():
        return routes, torch.zeros((S,), device=device, dtype=dtype)

    ar = torch.arange(Lmax, device=device)[None, :].expand(S, Lmax)
    i_ = best_i[:, None]
    k_ = best_k[:, None]
    in_seg = (ar >= i_) & (ar <= k_)
    ar_rev = torch.where(in_seg, (i_ + k_ - ar), ar)
    ar_rev = torch.where(apply[:, None], ar_rev, ar)

    routes_new = routes.gather(1, ar_rev)
    best_delta_out = torch.where(apply, best_delta, torch.zeros_like(best_delta))
    return routes_new, best_delta_out


# ============================================================
# V3 Apply: packed (B,P,T) with multi-depots/satellites separators
# ============================================================

@torch.no_grad()
def apply_two_opt_intra_to_solutions_fast_v3(
    sol: torch.Tensor,
    dist: torch.Tensor,
    n_deps: int,
    max_iters: int = 1,
    EPS: float = 1e-6,
    *,
    i_chunk: int = 64,
    k_chunk: int = 64,
    s_chunk: int = 2048,
    debug: bool = False,
):
    sol = sol.long()
    B, P, T = sol.shape
    device = sol.device
    BP = B * P
    seq = sol.reshape(BP, T)

    nxt = torch.zeros_like(seq)
    nxt[:, :-1] = seq[:, 1:]
    nxt[:, -1] = seq[:, -1]

    is_dep = seq < n_deps
    nxt_is_dep = nxt < n_deps

    start = is_dep & (~nxt_is_dep)
    end = (~is_dep) & nxt_is_dep

    start_pos = torch.nonzero(start, as_tuple=False)
    end_pos = torch.nonzero(end, as_tuple=False)
    if start_pos.numel() == 0 or end_pos.numel() == 0:
        return sol, torch.zeros((B, P), device=device, dtype=dist.dtype)

    start_key = start_pos[:, 0] * T + start_pos[:, 1]
    end_key = end_pos[:, 0] * T + end_pos[:, 1]
    start_pos = start_pos[start_key.argsort(stable=True)]
    end_pos = end_pos[end_key.argsort(stable=True)]

    S = min(start_pos.shape[0], end_pos.shape[0])
    start_pos = start_pos[:S]
    end_pos = end_pos[:S]

    rows = start_pos[:, 0]
    t0 = start_pos[:, 1]
    t1 = end_pos[:, 1]

    lengths = (t1 - t0 + 2).clamp(min=2)
    Lmax = int(lengths.max().item())

    offs = torch.arange(Lmax, device=device)[None, :]
    pos = offs.expand(S, Lmax)
    idx_t_raw = t0[:, None] + pos
    idx_t = torch.where(pos < lengths[:, None], idx_t_raw, t0[:, None])

    routes = seq[rows[:, None], idx_t]
    valid_pos = pos < lengths[:, None]

    distS = dist[rows // P]

    if debug:
        start_tok = routes[:, 0]
        end_tok = routes.gather(1, (lengths - 1).clamp_min(0)[:, None]).squeeze(1)
        assert (start_tok < n_deps).all(), "rota não começa em dep/sat"
        assert (end_tok < n_deps).all(), "rota não termina em dep/sat"

        inside = routes[:, 1:-1]
        inside_mask = torch.arange(routes.size(1) - 2, device=device)[None, :] < (lengths - 2)[:, None]
        bad = (inside < n_deps) & inside_mask
        assert not bad.any(), "há dep/sat interno dentro da rota extraída"

    delta_total = torch.zeros((BP,), device=device, dtype=torch.float32)
    avaliar = None

    for _ in range(max_iters):
        routes_new, best_delta = two_opt_best_intra_routes_batched_chunked_v3(
            routes, distS, lengths, n_deps=n_deps,
            i_chunk=i_chunk, k_chunk=k_chunk, s_chunk=s_chunk, avaliar=avaliar
        )

        best_delta, _, _ = double_check_small_negatives(
            distS, routes, routes_new, lengths, best_delta, eps=EPS
        )

        improved = best_delta < 0
        if not improved.any():
            break

        routes = routes_new
        delta_total.index_add_(0, rows, best_delta.float())
        avaliar = improved
        routes = torch.where(valid_pos, routes, seq[rows[:, None], idx_t])

    seq_out = seq.clone()
    old_block = seq_out[rows[:, None], idx_t]
    new_block = torch.where(valid_pos, routes, old_block)
    seq_out[rows[:, None], idx_t] = new_block

    sol_out = seq_out.view(B, P, T)
    return sol_out, delta_total.view(B, P)

import torch

# ============================================================
# Utilities (kept compatible with your current module)
# ============================================================

@torch.no_grad()
def check_dist_matrix(dist: torch.Tensor, atol: float = 1e-6):
    """
    Quick sanity checks for dist matrices:
      - diagonal ~ 0
      - symmetry
    Returns: (diag_ok, sym_ok, max_diag_abs, max_asym_abs)
    """
    if dist.dim() == 2:
        d = dist
        diag = torch.diag(d)
        max_diag = diag.abs().max().item()
        diag_ok = max_diag <= atol
        max_asym = (d - d.T).abs().max().item()
        sym_ok = max_asym <= atol
        return diag_ok, sym_ok, max_diag, max_asym

    # dist: (B,N,N)
    diag = torch.diagonal(dist, dim1=-2, dim2=-1)
    max_diag = diag.abs().max().item()
    diag_ok = max_diag <= atol
    max_asym = (dist - dist.transpose(-1, -2)).abs().max().item()
    sym_ok = max_asym <= atol
    return diag_ok, sym_ok, max_diag, max_asym


@torch.no_grad()
def route_cost(distS: torch.Tensor, route: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """
    distS: (N,N) or (S,N,N)
    route: (Lmax,) or (S,Lmax)
    length: scalar or (S,)
    Returns scalar or (S,)
    """
    if route.dim() == 1:
        L = int(length.item())
        if L <= 1:
            return torch.zeros((), device=route.device, dtype=distS.dtype)
        u = route[:L-1]
        v = route[1:L]
        if distS.dim() == 2:
            return distS[u, v].sum()
        else:
            # S==1 case
            return distS[0, u, v].sum()

    # batched
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
    """
    distS: (S,N,N)
    routes: (S,Lmax)
    lengths: (S,)
    returns: (S,)
    """
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
    """
    Recomputes exact deltas for suspicious very small negatives (numerical issues).
    Keeps behavior similar to your current code path.
    Returns: (best_delta_checked, CO, CA)
    """
    # suspicious deltas: negative but close to 0
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
# V2 Kernel: two-opt intra-route (GPU) — chunk in S, i, k
# ============================================================

@torch.no_grad()
def two_opt_best_intra_routes_batched_chunked_v2(
    routes: torch.Tensor,      # (S, Lmax) long  [0 ... 0 padding]
    distS: torch.Tensor,       # (S, N, N) float
    lengths: torch.Tensor,     # (S,) long, real length including 0 endpoints
    i_chunk: int = 64,         # chunk of i indices
    k_chunk: int = 64,         # chunk of k indices
    s_chunk: int = 2048,       # chunk of routes S
    avaliar = None             # optional mask (S,) bool: only evaluate these routes
):
    """
    Exact best-improvement 2-opt intra-route.
    Key optimizations for large BP / T:
      - Chunk over S (routes), i, and k to limit temporary tensor sizes.
      - Use distS flattened to 2D (S, N*N) and gather via linear indices (faster than 3D advanced indexing).
      - Use masked_fill instead of torch.where(full_like(inf)) inside hot loop.

    Returns:
      routes_new: (S,Lmax)
      best_delta: (S,) (negative for improvement, else 0)
    """
    device = routes.device
    S, Lmax = routes.shape
    dtype = distS.dtype
    N = distS.size(-1)

    # can: routes long enough for 2-opt and (optional) avaliar mask
    can = lengths >= 5
    if avaliar is not None:
        can = can & avaliar

    if not can.any():
        return routes, torch.zeros((S,), device=device, dtype=dtype)

    # outputs
    best_delta = torch.full((S,), float("inf"), device=device, dtype=dtype)
    best_i = torch.zeros((S,), device=device, dtype=torch.long)
    best_k = torch.zeros((S,), device=device, dtype=torch.long)

    # global k range: 2..Lmax-2 (inclusive) => torch.arange(2, Lmax-1)
    k_all = torch.arange(2, Lmax - 1, device=device, dtype=torch.long)

    # process routes in blocks to cap temp sizes
    for s0 in range(0, S, s_chunk):
        s1 = min(S, s0 + s_chunk)
        Sb = s1 - s0

        routes_b = routes[s0:s1]                # (Sb,Lmax)
        lengths_b = lengths[s0:s1]              # (Sb,)
        can_b = can[s0:s1]                      # (Sb,)

        # flatten dist for fast gather
        dist_flat = distS[s0:s1].reshape(Sb, N * N)  # (Sb, N*N)

        # local best for this block
        best_delta_b = best_delta[s0:s1]
        best_i_b = best_i[s0:s1]
        best_k_b = best_k[s0:s1]

        # Precompute L for broadcasting
        L = lengths_b[:, None, None]  # (Sb,1,1)

        # iterate i in chunks: i ∈ [1..Lmax-3] => range(1, Lmax-2)
        for i0 in range(1, Lmax - 2, i_chunk):
            i1 = min(Lmax - 2, i0 + i_chunk)
            i_blk = torch.arange(i0, i1, device=device, dtype=torch.long)  # (Ci,)
            Ci = i_blk.numel()
            if Ci == 0:
                continue

            # iterate k in chunks: k ∈ [2..Lmax-2] but must satisfy k > i
            for k0 in range(2, Lmax - 1, k_chunk):
                k1 = min(Lmax - 1, k0 + k_chunk)
                k_blk = torch.arange(k0, k1, device=device, dtype=torch.long)  # (Kc,)
                Kc = k_blk.numel()
                if Kc == 0:
                    continue

                # build I,K grids (Ci,Kc)
                I = i_blk[:, None].expand(Ci, Kc)
                K = k_blk[None, :].expand(Ci, Kc)

                # structural validity + length validity
                # i < k, k <= L-2, i <= L-3
                valid = (I[None] < K[None]) & (K[None] <= (L - 2)) & (I[None] <= (L - 3))
                valid = valid & can_b[:, None, None]

                if not valid.any():
                    continue

                # gather route nodes at prev(i-1), a(i), c(k), nxt(k+1)
                # indices: (Sb,Ci,Kc) -> gather from dim=1 with flattened indices
                idx_prev = (I - 1)[None, :, :].expand(Sb, Ci, Kc)
                idx_a    = (I     )[None, :, :].expand(Sb, Ci, Kc)
                idx_c    = (K     )[None, :, :].expand(Sb, Ci, Kc)
                idx_nxt  = (K + 1 )[None, :, :].expand(Sb, Ci, Kc)

                # gather requires (Sb, Ci*Kc)
                prev = routes_b.gather(1, idx_prev.reshape(Sb, -1)).reshape(Sb, Ci, Kc)
                a    = routes_b.gather(1, idx_a.reshape(Sb, -1)).reshape(Sb, Ci, Kc)
                c    = routes_b.gather(1, idx_c.reshape(Sb, -1)).reshape(Sb, Ci, Kc)
                nxt  = routes_b.gather(1, idx_nxt.reshape(Sb, -1)).reshape(Sb, Ci, Kc)

                # forbid depot at i/k
                valid = valid & (a != 0) & (c != 0)
                if not valid.any():
                    continue

                # dist lookups via linear indices
                # idx = u*N + v
                idx1 = (prev * N + c).reshape(Sb, -1)
                idx2 = (a    * N + nxt).reshape(Sb, -1)
                idx3 = (prev * N + a).reshape(Sb, -1)
                idx4 = (c    * N + nxt).reshape(Sb, -1)

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

                    # pos -> (i_local, k_local within this k_blk)
                    i_local = blk_best_pos // Kc
                    k_local = blk_best_pos %  Kc

                    i_pick = i_blk[i_local]
                    k_pick = k_blk[k_local]

                    best_i_b = torch.where(better, i_pick, best_i_b)
                    best_k_b = torch.where(better, k_pick, best_k_b)

        # write local block best back
        best_delta[s0:s1] = best_delta_b
        best_i[s0:s1] = best_i_b
        best_k[s0:s1] = best_k_b

    # apply reversal only where improvement exists
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
# V2 Apply: packed (B,P,T) with per-instance dist (B,N,N)
# ============================================================

@torch.no_grad()
def apply_two_opt_intra_to_solutions_fast_v2(
    sol: torch.Tensor,
    dist: torch.Tensor,
    max_iters: int = 1,
    EPS: float = 1e-6,
    *,
    i_chunk: int = 64,
    k_chunk: int = 64,
    s_chunk: int = 2048,
    debug: bool = False,
):
    """
    sol: (B,P,T) long packed with 0 separators/padding.
    dist: (B,N,N) float (GPU recommended).
    max_iters: number of 2-opt passes (each pass applies one best-improvement 2-opt per route).

    Returns:
      sol_out: (B,P,T)
      delta_total: (B,P) float
    """
    sol = sol.long()
    B, P, T = sol.shape
    device = sol.device
    BP = B * P
    seq = sol.reshape(BP, T)

    # nxt w/out wrap
    nxt = torch.zeros_like(seq)
    nxt[:, :-1] = seq[:, 1:]
    nxt[:, -1] = 0

    is0 = seq == 0
    start = is0 & (nxt != 0)          # 0 then client
    end   = (seq != 0) & (nxt == 0)   # client then 0

    start_pos = torch.nonzero(start, as_tuple=False)
    end_pos   = torch.nonzero(end,   as_tuple=False)
    if start_pos.numel() == 0 or end_pos.numel() == 0:
        return sol, torch.zeros((B, P), device=device)

    # sort to align
    start_key = start_pos[:, 0] * T + start_pos[:, 1]
    end_key   = end_pos[:, 0]   * T + end_pos[:, 1]
    start_pos = start_pos[start_key.argsort(stable=True)]
    end_pos   = end_pos[end_key.argsort(stable=True)]

    S = min(start_pos.shape[0], end_pos.shape[0])
    start_pos = start_pos[:S]
    end_pos   = end_pos[:S]

    rows = start_pos[:, 0]  # (S,)
    t0   = start_pos[:, 1]
    t1   = end_pos[:, 1]

    lengths = (t1 - t0 + 2).clamp(min=2)  # include 0 start and 0 end
    Lmax = int(lengths.max().item())

    offs = torch.arange(Lmax, device=device)[None, :]
    pos  = offs.expand(S, Lmax)
    idx_t_raw = t0[:, None] + pos
    idx_t = torch.where(pos < lengths[:, None], idx_t_raw, t0[:, None])
    routes = seq[rows[:, None], idx_t]  # (S,Lmax)

    # valid positions mask for writeback safety
    valid_pos = pos < lengths[:, None]

    # dist per extracted route
    distS = dist[rows // P]  # (S,N,N)

    if debug:
        assert (routes[:, 0] == 0).all()
        end_tok = routes.gather(1, (lengths - 1).clamp_min(0)[:, None]).squeeze(1)
        assert (end_tok == 0).all()
        inside = routes[:, 1:-1]
        bad = (inside == 0) & (torch.arange(routes.size(1) - 2, device=device)[None, :] < (lengths - 2)[:, None])
        assert not bad.any(), "há 0 interno => rotas coladas / vazias"

    delta_total = torch.zeros((BP,), device=device, dtype=torch.float32)
    avaliar = None

    for _ in range(max_iters):
        
        routes_new, best_delta = two_opt_best_intra_routes_batched_chunked_v2(
            routes, distS, lengths,
            i_chunk=i_chunk,
            k_chunk=k_chunk,
            s_chunk=s_chunk,
            avaliar=avaliar,
        )
        
        best_delta, _, _ = double_check_small_negatives(
            distS, routes, routes_new, lengths, best_delta, eps=EPS
        )

        improved = (best_delta < 0)
        if not improved.any():
            break

        routes = routes_new
        delta_total.index_add_(0, rows, best_delta.float())

        # next iter: only consider those improved
        avaliar = improved

        # ensure we never modify outside real route length
        routes = torch.where(valid_pos, routes, seq[rows[:, None], idx_t])

    # writeback to packed sequence
    seq_out = seq.clone()
    old_block = seq_out[rows[:, None], idx_t]
    new_block = torch.where(valid_pos, routes, old_block)
    seq_out[rows[:, None], idx_t] = new_block

    sol_out = seq_out.view(B, P, T)
    return sol_out, delta_total.view(B, P)

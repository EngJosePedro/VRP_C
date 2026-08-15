import torch
from typing import Optional, Tuple

# Reuse your existing GPU 2-opt kernel/utilities
try:
    from utils.two_opt_intra_v2 import (
        two_opt_best_intra_routes_batched_chunked_v2,
        batched_route_costs,
        double_check_small_negatives,
    )
except ImportError:
    from two_opt_intra_v2 import (
        two_opt_best_intra_routes_batched_chunked_v2,
        batched_route_costs,
        double_check_small_negatives,
    )


@torch.no_grad()
def _expand_demand_to_bp(demand: torch.Tensor, B: int, P: int, device) -> torch.Tensor:
    """
    Returns demand_bp: (B*P, N)
    Accepted shapes:
      - (N,)
      - (B, N)
      - (B, P, N)
    """
    demand = demand.to(device)
    if demand.dim() == 1:
        return demand[None, :].expand(B * P, -1)
    if demand.dim() == 2:
        return demand[:, None, :].expand(B, P, demand.size(-1)).reshape(B * P, -1)
    if demand.dim() == 3:
        assert demand.shape[:2] == (B, P)
        return demand.reshape(B * P, demand.size(-1))
    raise ValueError(f"Formato de demand não suportado: {tuple(demand.shape)}")


@torch.no_grad()
def _expand_cap_to_bp(cap, B: int, P: int, device, dtype) -> torch.Tensor:
    """
    Returns cap_bp: (B*P,)
    Accepted:
      - float/int scalar
      - tensor (B,)
      - tensor (B,P)
      - tensor (B*P,)
    """
    if not torch.is_tensor(cap):
        return torch.full((B * P,), float(cap), device=device, dtype=dtype)

    cap = cap.to(device=device, dtype=dtype)
    if cap.dim() == 0:
        return cap.expand(B * P)
    if cap.dim() == 1:
        if cap.numel() == B:
            return cap[:, None].expand(B, P).reshape(B * P)
        if cap.numel() == B * P:
            return cap.reshape(B * P)
    if cap.dim() == 2 and cap.shape == (B, P):
        return cap.reshape(B * P)
    raise ValueError(f"Formato de cap não suportado: {tuple(cap.shape)}")


@torch.no_grad()
def _extract_routes_from_packed(sol: torch.Tensor):
    """
    sol: (B,P,T) long packed with 0 separators/padding.

    Returns extracted routes in order of appearance:
      routes:       (S, Lmax)
      lengths:      (S,)
      row_bp:       (S,)  in [0, B*P)
      route_pos:    (S,)  local order inside each packed solution row
      start_t:      (S,)
      end_t:        (S,)
      idx_t:        (S, Lmax) indices in original packed row used to read/write this block
      valid_pos:    (S, Lmax)
    """
    B, P, T = sol.shape
    device = sol.device
    BP = B * P
    seq = sol.reshape(BP, T)

    nxt = torch.zeros_like(seq)
    nxt[:, :-1] = seq[:, 1:]
    nxt[:, -1] = 0

    is0 = seq == 0
    start = is0 & (nxt != 0)
    end = (seq != 0) & (nxt == 0)

    start_pos = torch.nonzero(start, as_tuple=False)
    end_pos = torch.nonzero(end, as_tuple=False)

    if start_pos.numel() == 0 or end_pos.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return (
            torch.empty((0, 0), device=device, dtype=torch.long),
            empty,
            empty,
            empty,
            empty,
            empty,
            torch.empty((0, 0), device=device, dtype=torch.long),
            torch.empty((0, 0), device=device, dtype=torch.bool),
        )

    start_key = start_pos[:, 0] * T + start_pos[:, 1]
    end_key = end_pos[:, 0] * T + end_pos[:, 1]
    start_pos = start_pos[start_key.argsort(stable=True)]
    end_pos = end_pos[end_key.argsort(stable=True)]

    S = min(start_pos.shape[0], end_pos.shape[0])
    start_pos = start_pos[:S]
    end_pos = end_pos[:S]

    row_bp = start_pos[:, 0]
    start_t = start_pos[:, 1]
    end_t = end_pos[:, 1]
    lengths = (end_t - start_t + 2).clamp(min=2)
    Lmax = int(lengths.max().item())

    offs = torch.arange(Lmax, device=device)[None, :]
    pos = offs.expand(S, Lmax)
    idx_t_raw = start_t[:, None] + pos
    idx_t = torch.where(pos < lengths[:, None], idx_t_raw, start_t[:, None])
    routes = seq[row_bp[:, None], idx_t]
    valid_pos = pos < lengths[:, None]

    # local route order inside each row.
    # Como start_pos foi ordenado por (row, t), as rotas ja estao agrupadas por row.
    # Evita loop Python sobre S.
    ar_s = torch.arange(S, device=device, dtype=torch.long)
    first_idx = torch.full((BP,), S, device=device, dtype=torch.long)
    first_idx.scatter_reduce_(0, row_bp, ar_s, reduce="amin", include_self=True)
    route_pos = ar_s - first_idx[row_bp]

    return routes, lengths, row_bp, route_pos, start_t, end_t, idx_t, valid_pos


@torch.no_grad()
def _route_loads(routes: torch.Tensor, lengths: torch.Tensor, demandS: torch.Tensor) -> torch.Tensor:
    """
    routes:  (S,Lmax)
    lengths: (S,)
    demandS: (S,N)
    returns: (S,)
    """
    S, Lmax = routes.shape
    pos = torch.arange(Lmax, device=routes.device)[None, :]
    mask = (pos < lengths[:, None]) & (routes != 0)
    dem = demandS.gather(1, routes.clamp_min(0))
    return (dem * mask.to(dem.dtype)).sum(dim=1)


@torch.no_grad()
def _apply_iterated_two_opt_on_routes(
    routes: torch.Tensor,
    lengths: torch.Tensor,
    distS: torch.Tensor,
    max_iters: int = 20,
    EPS: float = 1e-6,
    i_chunk: int = 64,
    k_chunk: int = 64,
    s_chunk: int = 2048,
):
    """
    Applies repeated best-improvement 2-opt until convergence or max_iters.
    routes: (S,Lmax)
    lengths:(S,)
    distS:  (S,N,N)
    """
    cur = routes
    delta_total = torch.zeros((routes.size(0),), device=routes.device, dtype=distS.dtype)
    avaliar = None

    for _ in range(max_iters):
        new_routes, best_delta = two_opt_best_intra_routes_batched_chunked_v2(
            cur,
            distS,
            lengths,
            i_chunk=i_chunk,
            k_chunk=k_chunk,
            s_chunk=s_chunk,
            avaliar=avaliar,
        )
        best_delta, _, _ = double_check_small_negatives(
            distS, cur, new_routes, lengths, best_delta, eps=EPS
        )
        improved = best_delta < 0
        if not improved.any():
            break
        cur = new_routes
        delta_total = delta_total + torch.where(improved, best_delta, torch.zeros_like(best_delta))
        avaliar = improved

    return cur, delta_total


@torch.no_grad()
def _build_candidate_routes_fixed_mn(
    ra: torch.Tensor,
    la: torch.Tensor,
    rb: torch.Tensor,
    lb: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    m: int,
    n: int,
):
    """
    Vectorized construction of swapped routes for fixed m,n.
    Inputs all batched over C candidates.
      ra, rb: (C, Lmax_in)
      la, lb: (C,)
      sa, sb: (C,)
    Returns:
      new_ra, new_la, seg_a
      new_rb, new_lb, seg_b
    """
    device = ra.device
    C = ra.size(0)
    dtype_long = torch.long

    new_la = la - m + n
    new_lb = lb - n + m
    Lout = int(torch.max(torch.max(new_la), torch.max(new_lb)).item())
    pos = torch.arange(Lout, device=device, dtype=dtype_long)[None, :].expand(C, Lout)

    # segments
    if m > 0:
        seg_a_idx = sa[:, None] + torch.arange(m, device=device, dtype=dtype_long)[None, :]
        seg_a = ra.gather(1, seg_a_idx)
    else:
        seg_a = torch.empty((C, 0), device=device, dtype=dtype_long)

    if n > 0:
        seg_b_idx = sb[:, None] + torch.arange(n, device=device, dtype=dtype_long)[None, :]
        seg_b = rb.gather(1, seg_b_idx)
    else:
        seg_b = torch.empty((C, 0), device=device, dtype=dtype_long)

    new_ra = torch.zeros((C, Lout), device=device, dtype=dtype_long)
    new_rb = torch.zeros((C, Lout), device=device, dtype=dtype_long)

    # ---- route A ----
    prefix_a = pos < sa[:, None]
    if prefix_a.any():
        src = pos.clamp_max(ra.size(1) - 1)
        new_ra[prefix_a] = ra.gather(1, src)[prefix_a]

    if n > 0:
        ins_a = (pos >= sa[:, None]) & (pos < (sa + n)[:, None])
        if ins_a.any():
            src = (pos - sa[:, None]).clamp(min=0, max=max(n - 1, 0))
            new_ra[ins_a] = seg_b.gather(1, src)[ins_a]

    suffix_a = (pos >= (sa + n)[:, None]) & (pos < new_la[:, None])
    if suffix_a.any():
        src = (pos - n + m).clamp(min=0, max=ra.size(1) - 1)
        new_ra[suffix_a] = ra.gather(1, src)[suffix_a]

    # ---- route B ----
    prefix_b = pos < sb[:, None]
    if prefix_b.any():
        src = pos.clamp_max(rb.size(1) - 1)
        new_rb[prefix_b] = rb.gather(1, src)[prefix_b]

    if m > 0:
        ins_b = (pos >= sb[:, None]) & (pos < (sb + m)[:, None])
        if ins_b.any():
            src = (pos - sb[:, None]).clamp(min=0, max=max(m - 1, 0))
            new_rb[ins_b] = seg_a.gather(1, src)[ins_b]

    suffix_b = (pos >= (sb + m)[:, None]) & (pos < new_lb[:, None])
    if suffix_b.any():
        src = (pos - m + n).clamp(min=0, max=rb.size(1) - 1)
        new_rb[suffix_b] = rb.gather(1, src)[suffix_b]

    return new_ra, new_la, seg_a, new_rb, new_lb, seg_b


@torch.no_grad()
def _repack_rows_from_route_list(
    seq: torch.Tensor,
    selected_rows: torch.Tensor,
    selected_route_a: torch.Tensor,
    selected_route_b: torch.Tensor,
    selected_new_ra: torch.Tensor,
    selected_new_la: torch.Tensor,
    selected_new_rb: torch.Tensor,
    selected_new_lb: torch.Tensor,
    routes: torch.Tensor,
    lengths: torch.Tensor,
    row_bp: torch.Tensor,
    route_pos: torch.Tensor,
):
    """
    Versão vetorizada do repack das linhas modificadas.

    Ideia central:
      - Para cada row modificada, percorremos conceitualmente as rotas antigas na
        ordem original, mas fazemos isso por tensores.
      - A rota `selected_route_a` é substituída por `selected_new_ra`.
      - A rota `selected_route_b` é substituída por `selected_new_rb`.
      - Rotas com length <= 2 são descartadas, pois representam rota vazia [0,0].
      - Para evitar duplicar zeros entre rotas, copiamos:
            primeira rota não-vazia: [0, ..., 0]
            demais rotas:           [..., 0]   # remove o 0 inicial
        Assim [0,1,2,0] + [0,3,0] vira [0,1,2,0,3,0],
        e não [0,1,2,0,0,3,0].

    Esta função elimina os loops Python sobre rows/rotas e torna desnecessário
    chamar remove_consecutive_zeros_vec no fluxo principal.
    """
    device = seq.device
    seq_out = seq.clone()
    BP, T = seq.shape

    if selected_rows.numel() == 0:
        return seq_out

    # selected_rows deve ter uma entrada por solução BP melhorada.
    # row_to_k[row] = índice local k em selected_*.
    K = selected_rows.numel()
    row_to_k = torch.full((BP,), -1, device=device, dtype=torch.long)
    row_to_k[selected_rows] = torch.arange(K, device=device, dtype=torch.long)

    # Seleciona todas as rotas antigas pertencentes às rows modificadas.
    changed_route_mask = row_to_k[row_bp] >= 0
    if not changed_route_mask.any():
        return seq_out

    gids = torch.nonzero(changed_route_mask, as_tuple=False).squeeze(-1)  # (G,)

    # Garante ordem: primeiro por row, depois pela posição local da rota.
    # Isso preserva a sequência original de rotas dentro de cada solução.
    max_rpos = int(route_pos.max().item()) + 1 if route_pos.numel() > 0 else 1
    order_key = row_bp[gids] * max_rpos + route_pos[gids]
    gids = gids[order_key.argsort(stable=True)]

    row_g = row_bp[gids]               # (G,)
    k_g = row_to_k[row_g]              # (G,)

    is_a = gids == selected_route_a[k_g]
    is_b = gids == selected_route_b[k_g]

    # Comprimento de cada bloco-fonte após substituição.
    src_len = lengths[gids].clone()
    src_len = torch.where(is_a, selected_new_la[k_g], src_len)
    src_len = torch.where(is_b, selected_new_lb[k_g], src_len)

    # Remove rotas vazias [0,0]. Na sua versão antiga isso era feito por:
    # if enter_block.shape[0] > 2: blocks.append(...)
    nonempty = src_len > 2
    if not nonempty.any():
        seq_out[selected_rows] = 0
        return seq_out

    gids = gids[nonempty]
    row_g = row_g[nonempty]
    k_g = k_g[nonempty]
    is_a = is_a[nonempty]
    is_b = is_b[nonempty]
    src_len = src_len[nonempty]
    G = gids.numel()

    # Largura máxima para armazenar blocos-fonte. Pode ser maior que a largura
    # antiga porque o swap(m,n) pode aumentar uma das duas rotas alteradas.
    Lsrc = max(routes.size(1), selected_new_ra.size(1), selected_new_rb.size(1))

    src_blocks = torch.zeros((G, Lsrc), device=device, dtype=seq.dtype)

    # Começa com as rotas antigas.
    old = routes[gids]
    src_blocks[:, :old.size(1)] = old

    # Sobrescreve as rotas A e B com as versões modificadas + 2-opt.
    if is_a.any():
        src_blocks[is_a, :selected_new_ra.size(1)] = selected_new_ra[k_g[is_a]]
    if is_b.any():
        src_blocks[is_b, :selected_new_rb.size(1)] = selected_new_rb[k_g[is_b]]

    # Primeira rota não-vazia de cada row mantém o 0 inicial.
    # Rotas seguintes omitem o primeiro 0 para não gerar 0,0 entre blocos.
    first_in_row = torch.ones((G,), device=device, dtype=torch.bool)
    first_in_row[1:] = row_g[1:] != row_g[:-1]

    copy_len = torch.where(first_in_row, src_len, src_len - 1)  # (G,)
    total_len_by_row = torch.zeros((BP,), device=device, dtype=torch.long)
    total_len_by_row.scatter_add_(0, row_g, copy_len)

    if (total_len_by_row[selected_rows] > T).any():
        max_len = int(total_len_by_row[selected_rows].max().item())
        raise RuntimeError(
            f"Reconstrução da solução excedeu T: max_len={max_len}, T={T}. "
            "Verifique o packing de entrada ou aumente T."
        )

    # Prefixo dentro de cada row, assumindo gids ordenado por row/route_pos.
    global_prefix = copy_len.cumsum(dim=0) - copy_len
    row_base = total_len_by_row.cumsum(dim=0) - total_len_by_row
    local_prefix = global_prefix - row_base[row_g]  # (G,)

    Lcopy = int(copy_len.max().item())
    pos = torch.arange(Lcopy, device=device, dtype=torch.long)[None, :].expand(G, Lcopy)
    valid = pos < copy_len[:, None]

    # Para rotas seguintes, desloca +1 para pular o depósito inicial.
    src_idx = torch.where(first_in_row[:, None], pos, pos + 1)
    tokens = src_blocks.gather(1, src_idx.clamp_max(Lsrc - 1))

    out_col = local_prefix[:, None] + pos
    out_row = row_g[:, None].expand(G, Lcopy)

    # Zera somente as rows alteradas e escreve a sequência packed compacta.
    seq_out[selected_rows] = 0
    seq_out[out_row[valid], out_col[valid]] = tokens[valid]

    return seq_out

@torch.no_grad()

@torch.no_grad()
def _build_ordered_route_pairs_by_row(row_bp: torch.Tensor, route_pos: torch.Tensor, BP: int):
    """
    Gera pares ordenados (a,b) de rotas dentro da mesma solução BP sem criar matriz SxS.

    Em vez de comparar todas as rotas contra todas as rotas:
        same_row = row_bp[:, None] == row_bp[None, :]
    usa uma tabela compacta route_table[row, route_pos] = global_route_id.

    Custo estrutural: O(BP * Rmax^2), onde Rmax é nº máximo de rotas por solução,
    normalmente muito menor que O(S^2).
    """
    device = row_bp.device
    S = row_bp.numel()
    if S == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty

    Rmax = int(route_pos.max().item()) + 1
    route_table = torch.full((BP, Rmax), -1, device=device, dtype=torch.long)
    route_table[row_bp, route_pos] = torch.arange(S, device=device, dtype=torch.long)

    a_pos = torch.arange(Rmax, device=device, dtype=torch.long)[:, None]
    b_pos = torch.arange(Rmax, device=device, dtype=torch.long)[None, :]
    pair_pos_mask = a_pos != b_pos

    A = route_table[:, :, None].expand(BP, Rmax, Rmax)
    B_ = route_table[:, None, :].expand(BP, Rmax, Rmax)
    valid = pair_pos_mask[None, :, :] & (A >= 0) & (B_ >= 0)

    pair_a = A[valid]
    pair_b = B_[valid]
    return pair_a, pair_b


@torch.no_grad()
def _pad_routes_to_len(x: torch.Tensor, L: int) -> torch.Tensor:
    """Pad right com zeros até largura L; retorna x se já tiver largura L."""
    if x.size(1) == L:
        return x
    out = x.new_zeros((x.size(0), L))
    out[:, :x.size(1)] = x
    return out


@torch.no_grad()
def _ensure_best_buffers(best_new_ra, best_new_rb, BP: int, L: int, device):
    """Cria/expande buffers (BP,L) para armazenar melhores rotas escolhidas."""
    if best_new_ra is None:
        return (
            torch.zeros((BP, L), device=device, dtype=torch.long),
            torch.zeros((BP, L), device=device, dtype=torch.long),
        )
    if best_new_ra.size(1) >= L:
        return best_new_ra, best_new_rb
    extra = L - best_new_ra.size(1)
    z = torch.zeros((BP, extra), device=device, dtype=torch.long)
    return torch.cat([best_new_ra, z], dim=1), torch.cat([best_new_rb, z], dim=1)


@torch.no_grad()
def _write_selected_best_routes(
    best_new_ra,
    best_new_rb,
    best_new_la: torch.Tensor,
    best_new_lb: torch.Tensor,
    rows_sel: torch.Tensor,
    idx_sel: torch.Tensor,
    opt_ra: torch.Tensor,
    opt_rb: torch.Tensor,
    new_la: torch.Tensor,
    new_lb: torch.Tensor,
    BP: int,
    device,
):
    """Atualiza buffers de melhores rotas por row, evitando duplicação de código."""
    L = max(opt_ra.size(1), opt_rb.size(1))
    best_new_ra, best_new_rb = _ensure_best_buffers(best_new_ra, best_new_rb, BP, L, device)

    opt_ra_w = _pad_routes_to_len(opt_ra, best_new_ra.size(1))
    opt_rb_w = _pad_routes_to_len(opt_rb, best_new_rb.size(1))

    best_new_ra[rows_sel] = opt_ra_w[idx_sel]
    best_new_rb[rows_sel] = opt_rb_w[idx_sel]
    best_new_la[rows_sel] = new_la[idx_sel]
    best_new_lb[rows_sel] = new_lb[idx_sel]
    return best_new_ra, best_new_rb

@torch.no_grad()
def apply_swap_mn_2opt_inter_routes_pytorch(
    sol: torch.Tensor,
    dist: torch.Tensor,
    demand: torch.Tensor,
    cap,
    *,
    max_m: int = 3,
    max_n: int = 3,
    max_iters: int = 100,
    first_improve: bool = False,
    two_opt_max_iters: int = 20,
    EPS: float = 1e-6,
    i_chunk: int = 64,
    k_chunk: int = 64,
    s_chunk: int = 2048,
    return_delta: bool = True,
    debug: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Busca local inter-rotas tipo swap(m,n) + 2-opt nas duas rotas alteradas.

    sol:    (B,P,T) long, sequência packed com 0 como depósito/separador/padding.
    dist:   (B,N,N) float
    demand: (N,) ou (B,N) ou (B,P,N), incluindo demanda do depósito em 0 (normalmente zero).
    cap:    escalar, (B,), (B,P) ou (B*P,)

    Retorna:
      sol_out:      (B,P,T)
      delta_total:  (B,P)  se return_delta=True

    Observação:
      - Esta primeira versão é funcional e vetorizada na geração/avaliação dos candidatos.
      - Ainda existe um laço Python na etapa de repack das linhas alteradas.
    """
    assert sol.dim() == 3, "sol deve ter shape (B,P,T)"
    assert dist.dim() == 3, "dist deve ter shape (B,N,N)"

    device = sol.device
    sol = sol.long()
    B, P, T = sol.shape
    BP = B * P
    dtype = dist.dtype

    demand_bp = _expand_demand_to_bp(demand, B, P, device)
    cap_bp = _expand_cap_to_bp(cap, B, P, device, dtype)

    seq = sol.reshape(BP, T).clone()
    delta_total = torch.zeros((BP,), device=device, dtype=dtype)

    for it in range(max_iters):
        routes, lengths, row_bp, route_pos, start_t, end_t, idx_t, valid_pos = _extract_routes_from_packed(seq.view(B, P, T))
        S = routes.size(0)
        if S == 0:
            break

        # Pares ordenados dentro de cada solução BP, sem matriz SxS.
        # Ordem preserva aproximadamente os loops por row, rota a e rota b.
        pair_a, pair_b = _build_ordered_route_pairs_by_row(row_bp, route_pos, BP)
        if pair_a.numel() == 0:
            break

        # Dados dos pares pré-computados uma vez por iteração da busca.
        pair_owner = row_bp[pair_a]
        pair_la_all = lengths[pair_a]
        pair_lb_all = lengths[pair_b]

        route_demand = demand_bp[row_bp]   # (S,N)
        route_cap = cap_bp[row_bp]         # (S,)
        route_load = _route_loads(routes, lengths, route_demand)
        route_cost = batched_route_costs(dist[row_bp // P], routes, lengths)

        # best move per packed solution row in this iteration
        best_delta_row = torch.full((BP,), float('inf'), device=device, dtype=dtype)
        best_route_a = torch.full((BP,), -1, device=device, dtype=torch.long)
        best_route_b = torch.full((BP,), -1, device=device, dtype=torch.long)
        best_new_ra = None
        best_new_rb = None
        best_new_la = torch.zeros((BP,), device=device, dtype=torch.long)
        best_new_lb = torch.zeros((BP,), device=device, dtype=torch.long)
        row_has_first = torch.zeros((BP,), device=device, dtype=torch.bool)

        for m in range(0, max_m + 1):
            for n in range(0, max_n + 1):
                if m == 0 and n == 0:
                    continue

                # unresolved rows only when first_improve=True
                if first_improve and row_has_first.all():
                    break

                pa = pair_a
                pb = pair_b
                owner = pair_owner
                la = pair_la_all
                lb = pair_lb_all
                if first_improve:
                    keep = ~row_has_first[owner]
                    if not keep.any():
                        continue
                    pa = pa[keep]
                    pb = pb[keep]
                    owner = owner[keep]
                    la = la[keep]
                    lb = lb[keep]
                    if pa.numel() == 0:
                        continue

                sa_max = (la - 1) if m == 0 else (la - 1 - m)
                sb_max = (lb - 1) if n == 0 else (lb - 1 - n)
                valid_pair = (sa_max >= 1) & (sb_max >= 1)
                if not valid_pair.any():
                    continue

                pa = pa[valid_pair]
                pb = pb[valid_pair]
                owner = owner[valid_pair]
                la = la[valid_pair]
                lb = lb[valid_pair]
                sa_max = sa_max[valid_pair]
                sb_max = sb_max[valid_pair]

                SA = int(sa_max.max().item())
                SB = int(sb_max.max().item())
                sa_grid = torch.arange(1, SA + 1, device=device)
                sb_grid = torch.arange(1, SB + 1, device=device)

                # create all candidate (pair, sa, sb)
                valid_sa = sa_grid[None, :] <= sa_max[:, None]  # (Cpair,SA)
                valid_sb = sb_grid[None, :] <= sb_max[:, None]  # (Cpair,SB)
                cand_mask = valid_sa[:, :, None] & valid_sb[:, None, :]
                if not cand_mask.any():
                    continue

                pair_idx, sa_idx, sb_idx = torch.nonzero(cand_mask, as_tuple=True)
                pa_c = pa[pair_idx]
                pb_c = pb[pair_idx]
                owner_c = owner[pair_idx]
                sa = sa_grid[sa_idx]
                sb = sb_grid[sb_idx]

                ra = routes[pa_c]
                rb = routes[pb_c]
                la_c = lengths[pa_c]
                lb_c = lengths[pb_c]

                new_ra, new_la, seg_a, new_rb, new_lb, seg_b = _build_candidate_routes_fixed_mn(
                    ra, la_c, rb, lb_c, sa, sb, m, n
                )

                # capacity check through segment demand exchange
                load_a = route_load[pa_c]
                load_b = route_load[pb_c]

                if m > 0:
                    seg_a_dem = demand_bp[owner_c].gather(1, seg_a).sum(dim=1)
                else:
                    seg_a_dem = torch.zeros_like(load_a)
                if n > 0:
                    seg_b_dem = demand_bp[owner_c].gather(1, seg_b).sum(dim=1)
                else:
                    seg_b_dem = torch.zeros_like(load_b)

                new_load_a = load_a - seg_a_dem + seg_b_dem
                new_load_b = load_b - seg_b_dem + seg_a_dem
                cap_c = cap_bp[owner_c]
                feasible = (new_load_a <= cap_c + 1e-9) & (new_load_b <= cap_c + 1e-9)
                if not feasible.any():
                    continue

                keep = feasible
                pa_c = pa_c[keep]
                pb_c = pb_c[keep]
                owner_c = owner_c[keep]
                new_ra = new_ra[keep]
                new_rb = new_rb[keep]
                new_la = new_la[keep]
                new_lb = new_lb[keep]
                la_c = la_c[keep]
                lb_c = lb_c[keep]

                if pa_c.numel() == 0:
                    continue

                dist_a = dist[row_bp[pa_c] // P]
                dist_b = dist[row_bp[pb_c] // P]
                old_cost = route_cost[pa_c] + route_cost[pb_c]

                # 2-opt on the two changed routes
                Lcand = max(new_ra.size(1), new_rb.size(1))
                if new_ra.size(1) != Lcand:
                    tmp = torch.zeros((new_ra.size(0), Lcand), device=device, dtype=new_ra.dtype)
                    tmp[:, :new_ra.size(1)] = new_ra
                    new_ra = tmp
                if new_rb.size(1) != Lcand:
                    tmp = torch.zeros((new_rb.size(0), Lcand), device=device, dtype=new_rb.dtype)
                    tmp[:, :new_rb.size(1)] = new_rb
                    new_rb = tmp

                stacked_routes = torch.cat([new_ra, new_rb], dim=0)
                stacked_lengths = torch.cat([new_la, new_lb], dim=0)
                stacked_dist = torch.cat([dist_a, dist_b], dim=0)

                stacked_opt, _ = _apply_iterated_two_opt_on_routes(
                    stacked_routes,
                    stacked_lengths,
                    stacked_dist,
                    max_iters=two_opt_max_iters,
                    EPS=EPS,
                    i_chunk=i_chunk,
                    k_chunk=k_chunk,
                    s_chunk=s_chunk,
                )

                M = pa_c.numel()
                opt_ra = stacked_opt[:M]
                opt_rb = stacked_opt[M:]
                new_cost = (
                    batched_route_costs(dist_a, opt_ra, new_la) +
                    batched_route_costs(dist_b, opt_rb, new_lb)
                )
                delta = new_cost - old_cost
                improve = delta < -1e-12
                if not improve.any():
                    continue

                cand_idx = torch.arange(delta.numel(), device=device, dtype=torch.long)

                if first_improve:
                    # first improvement per row in current scan order
                    improve_idx = cand_idx[improve]
                    improve_owner = owner_c[improve]
                    first_pos = torch.full((BP,), improve_idx.numel(), device=device, dtype=torch.long)
                    first_pos.scatter_reduce_(0, improve_owner, improve_idx, reduce='amin', include_self=True)
                    chosen_rows = (first_pos < improve_idx.numel()) & (~row_has_first)
                    if chosen_rows.any():
                        rows_sel = torch.nonzero(chosen_rows, as_tuple=False).squeeze(-1)
                        idx_sel = first_pos[rows_sel]

                        row_has_first[rows_sel] = True
                        best_delta_row[rows_sel] = delta[idx_sel]
                        best_route_a[rows_sel] = pa_c[idx_sel]
                        best_route_b[rows_sel] = pb_c[idx_sel]

                        if best_new_ra is None:
                            Ltmp = opt_ra.size(1)
                            best_new_ra = torch.zeros((BP, Ltmp), device=device, dtype=torch.long)
                            best_new_rb = torch.zeros((BP, Ltmp), device=device, dtype=torch.long)
                        elif best_new_ra.size(1) < opt_ra.size(1):
                            extra = opt_ra.size(1) - best_new_ra.size(1)
                            best_new_ra = torch.cat([best_new_ra, torch.zeros((BP, extra), device=device, dtype=torch.long)], dim=1)
                            best_new_rb = torch.cat([best_new_rb, torch.zeros((BP, extra), device=device, dtype=torch.long)], dim=1)

                        if opt_ra.size(1) < best_new_ra.size(1):
                            tmp_a = torch.zeros((opt_ra.size(0), best_new_ra.size(1)), device=device, dtype=torch.long)
                            tmp_b = torch.zeros((opt_rb.size(0), best_new_rb.size(1)), device=device, dtype=torch.long)
                            tmp_a[:, :opt_ra.size(1)] = opt_ra
                            tmp_b[:, :opt_rb.size(1)] = opt_rb
                            opt_ra_w = tmp_a
                            opt_rb_w = tmp_b
                        else:
                            opt_ra_w = opt_ra
                            opt_rb_w = opt_rb

                        best_new_ra[rows_sel] = opt_ra_w[idx_sel]
                        best_new_rb[rows_sel] = opt_rb_w[idx_sel]
                        best_new_la[rows_sel] = new_la[idx_sel]
                        best_new_lb[rows_sel] = new_lb[idx_sel]
                else:
                    # best improvement per row across all candidates
                    cand_delta = torch.full((BP,), float('inf'), device=device, dtype=dtype)
                    cand_delta.scatter_reduce_(0, owner_c[improve], delta[improve], reduce='amin', include_self=True)
                    better_rows = cand_delta < best_delta_row
                    if better_rows.any():
                        rows_sel = torch.nonzero(better_rows, as_tuple=False).squeeze(-1)
                        # choose first candidate attaining row minimum
                        row_min = cand_delta[owner_c]
                        eq_best = improve & (delta <= row_min + 1e-12)
                        idx_big = torch.full((BP,), cand_idx.numel(), device=device, dtype=torch.long)
                        idx_big.scatter_reduce_(0, owner_c[eq_best], cand_idx[eq_best], reduce='amin', include_self=True)
                        idx_sel = idx_big[rows_sel]
                        good = idx_sel < cand_idx.numel()
                        if good.any():
                            rows_sel = rows_sel[good]
                            idx_sel = idx_sel[good]

                            best_delta_row[rows_sel] = delta[idx_sel]
                            best_route_a[rows_sel] = pa_c[idx_sel]
                            best_route_b[rows_sel] = pb_c[idx_sel]

                            if best_new_ra is None:
                                Ltmp = opt_ra.size(1)
                                best_new_ra = torch.zeros((BP, Ltmp), device=device, dtype=torch.long)
                                best_new_rb = torch.zeros((BP, Ltmp), device=device, dtype=torch.long)
                            elif best_new_ra.size(1) < opt_ra.size(1):
                                extra = opt_ra.size(1) - best_new_ra.size(1)
                                best_new_ra = torch.cat([best_new_ra, torch.zeros((BP, extra), device=device, dtype=torch.long)], dim=1)
                                best_new_rb = torch.cat([best_new_rb, torch.zeros((BP, extra), device=device, dtype=torch.long)], dim=1)

                            if opt_ra.size(1) < best_new_ra.size(1):
                                tmp_a = torch.zeros((opt_ra.size(0), best_new_ra.size(1)), device=device, dtype=torch.long)
                                tmp_b = torch.zeros((opt_rb.size(0), best_new_rb.size(1)), device=device, dtype=torch.long)
                                tmp_a[:, :opt_ra.size(1)] = opt_ra
                                tmp_b[:, :opt_rb.size(1)] = opt_rb
                                opt_ra_w = tmp_a
                                opt_rb_w = tmp_b
                            else:
                                opt_ra_w = opt_ra
                                opt_rb_w = opt_rb

                            best_new_ra[rows_sel] = opt_ra_w[idx_sel]
                            best_new_rb[rows_sel] = opt_rb_w[idx_sel]
                            best_new_la[rows_sel] = new_la[idx_sel]
                            best_new_lb[rows_sel] = new_lb[idx_sel]

            if first_improve and row_has_first.all():
                break

        improve_rows = best_delta_row < -1e-12
        if not improve_rows.any():
            break

        rows_sel = torch.nonzero(improve_rows, as_tuple=False).squeeze(-1)
        seq = _repack_rows_from_route_list(
            seq,
            rows_sel,
            best_route_a[rows_sel],
            best_route_b[rows_sel],
            best_new_ra[rows_sel],
            best_new_la[rows_sel],
            best_new_rb[rows_sel],
            best_new_lb[rows_sel],
            routes,
            lengths,
            row_bp,
            route_pos,
        )
        delta_total[rows_sel] += best_delta_row[rows_sel]

        if debug:
            print(f"iter={it} improved_rows={int(improve_rows.sum().item())} total_delta={best_delta_row[improve_rows].sum().item():.6f}")

    sol_out = seq.view(B, P, T)
    if return_delta:
        return sol_out, delta_total.view(B, P)
    return sol_out, None

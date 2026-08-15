import torch
"""
@torch.no_grad()
def two_opt_best_intra_routes_batched_chunked(
    routes: torch.Tensor,      # (S, Lmax) long  [0 ... 0 padding]
    distS: torch.Tensor,       # (S, N, N) float
    lengths: torch.Tensor,     # (S,) long, comprimento real incluindo 0 nas extremidades
    i_chunk: int = 64,         # tamanho do bloco de i (ajuste p/ sua GPU)
    avaliar = None
):
    #""
    Retorna routes_new (S,Lmax) e best_delta (S,)
    - Best-improvement EXATO intra-rota.
    - Não materializa delta (S,Li,Lk) inteiro; faz argmin por blocos.
    #""
    device = routes.device
    S, Lmax = routes.shape

    best_delta = torch.full((S,), float("inf"), device=device, dtype=distS.dtype)
    #print("----", best_delta.shape)
    best_i = torch.zeros((S,), device=device, dtype=torch.long)
    best_k = torch.zeros((S,), device=device, dtype=torch.long)

    can = lengths >= 5
    if not can.any():
        return routes, torch.zeros((S,), device=device, dtype=distS.dtype)

    # Faixas máximas globais (depois mascaramos por length)
    # i ∈ [1 .. L-3], k ∈ [i+1 .. L-2]
    k_all = torch.arange(2, Lmax - 1, device=device, dtype=torch.long)  # 2..Lmax-2  (len Kmax)

    # s_idx 1D
    s_idx = torch.arange(S, device=device)  # (S,)

    routes_exp = routes[:, None, None, :]   # (S,1,1,Lmax)
    Kmax = k_all.numel()

    for i0 in range(1, Lmax - 2, i_chunk):
        i1 = min(Lmax - 2, i0 + i_chunk)
        i_blk = torch.arange(i0, i1, device=device, dtype=torch.long)
        Ci = i_blk.numel()

        I = i_blk[:, None].expand(Ci, Kmax)     # (Ci,Kmax)
        K = k_all[None, :].expand(Ci, Kmax)     # (Ci,Kmax)

        L = lengths[:, None, None]             # (S,1,1)
        valid = (I[None] < K[None]) & (K[None] <= (L - 2)) & (I[None] <= (L - 3))
        valid &= can[:, None, None]

        I3 = I[None, :, :]                      # (1,Ci,Kmax)
        K3 = K[None, :, :]

        prev = torch.take_along_dim(routes_exp, (I3 - 1)[..., None], dim=-1).squeeze(-1)  # (S,Ci,Kmax)
        a    = torch.take_along_dim(routes_exp,  I3      [..., None], dim=-1).squeeze(-1)
        c    = torch.take_along_dim(routes_exp,  K3      [..., None], dim=-1).squeeze(-1)
        nxt  = torch.take_along_dim(routes_exp, (K3 + 1)[..., None], dim=-1).squeeze(-1)

        valid &= (a != 0) & (c != 0)

        s3 = s_idx[:, None, None].expand(S, Ci, Kmax)

        delta = distS[s3, prev, c] + distS[s3, a, nxt] - distS[s3, prev, a] - distS[s3, c, nxt]
        #print("----=", delta.shape)
        delta = torch.where(valid, delta, torch.full_like(delta, float("inf")))
        #print("----=", delta.shape)

        flat = delta.view(S, -1)
        blk_best_val, blk_best_pos = torch.min(flat, dim=1)

        better = blk_best_val < best_delta
        if better.any():
            best_delta = torch.where(better, blk_best_val, best_delta)

            i_local = blk_best_pos // Kmax
            k_local = blk_best_pos %  Kmax

            i_pick = i_blk[i_local]
            k_pick = k_all[k_local]

            best_i = torch.where(better, i_pick, best_i)
            best_k = torch.where(better, k_pick, best_k)


    #""
    #s_idx = torch.arange(S, device=device)[:, None]  # (S,1)
    s_idx = torch.arange(S, device=device)  # (S,)

    routes_exp = routes[:, None, :]  # (S,1,Lmax) para take_along_dim
    Kmax = k_all.numel()

    # varre i em blocos
    for i0 in range(1, Lmax - 2, i_chunk):
        i1 = min(Lmax - 2, i0 + i_chunk)  # i vai até Lmax-3 inclusive => range(1, Lmax-2)
        i_blk = torch.arange(i0, i1, device=device, dtype=torch.long)  # (Ci,)
        Ci = i_blk.numel()

        # Monta grid (Ci, Kmax) para k e i
        I = i_blk[:, None].expand(Ci, Kmax)           # (Ci,Kmax)
        K = k_all[None, :].expand(Ci, Kmax)           # (Ci,Kmax)

        # Máscara estrutural e por comprimento
        # K <= L-2 e I <= L-3 e I < K
        L = lengths[:, None, None]                    # (S,1,1)
        valid = (I[None] < K[None]) \
              & (K[None] <= (L - 2)) \
              & (I[None] <= (L - 3))

        # pega prev,a,c,nxt: shapes (S,Ci,Kmax)
        I3 = I[None, :, :]
        K3 = K[None, :, :]

        prev = torch.take_along_dim(routes_exp, (I3 - 1), dim=-1)  # (S,Ci,Kmax)
        a    = torch.take_along_dim(routes_exp, I3,       dim=-1)
        c    = torch.take_along_dim(routes_exp, K3,       dim=-1)
        nxt  = torch.take_along_dim(routes_exp, (K3 + 1), dim=-1)

        prev = prev.squeeze(-1)
        a    = a.squeeze(-1)
        c    = c.squeeze(-1)
        nxt  = nxt.squeeze(-1)

        # não permite depósitos como i/k (segurança)
        valid &= (a != 0) & (c != 0)
        valid &= can[:, None, None]

        # delta = d(prev,c) + d(a,nxt) - d(prev,a) - d(c,nxt)
        # indexing: distS[s, u, v]
        #s3 = s_idx[:, None, None].expand(S, Ci, Kmax)
        s3 = s_idx[:, None, None].expand(S, Ci, Kmax)  # (S,Ci,Kmax)
        delta = distS[s3, prev, c] + distS[s3, a, nxt] - distS[s3, prev, a] - distS[s3, c, nxt]

        delta = torch.where(valid, delta, torch.full_like(delta, float("inf")))

        # best dentro do bloco (por rota)
        flat = delta.view(S, -1)                          # (S, Ci*Kmax)
        blk_best_val, blk_best_pos = torch.min(flat, dim=1)

        # atualiza best global
        better = blk_best_val < best_delta
        if better.any():
            best_delta = torch.where(better, blk_best_val, best_delta)

            # map pos -> (i,k)
            # pos = i_local*Kmax + k_local
            i_local = blk_best_pos // Kmax
            k_local = blk_best_pos %  Kmax

            i_pick = i_blk[i_local]       # (S,)
            k_pick = k_all[k_local]       # (S,)

            best_i = torch.where(better, i_pick, best_i)
            best_k = torch.where(better, k_pick, best_k)
    #""
    # aplica só se melhora (delta < 0)
    apply = can & torch.isfinite(best_delta) & (best_delta < 0)
    if not apply.any():
        return routes, torch.zeros((S,), device=device, dtype=distS.dtype)

    # inverter segmento [best_i..best_k] em batch
    ar = torch.arange(Lmax, device=device)[None, :].expand(S, Lmax)
    i_ = best_i[:, None]
    k_ = best_k[:, None]
    in_seg = (ar >= i_) & (ar <= k_)
    ar_rev = torch.where(in_seg, (i_ + k_ - ar), ar)
    ar_rev = torch.where(apply[:, None], ar_rev, ar)

    routes_new = routes.gather(1, ar_rev)

    best_delta = torch.where(apply, best_delta, torch.zeros_like(best_delta))
    return routes_new, best_delta

"""

@torch.no_grad()
def two_opt_best_intra_routes_batched_chunked(
    routes: torch.Tensor,      # (S, Lmax)
    distS: torch.Tensor,       # (S, N, N)
    lengths: torch.Tensor,     # (S,)
    i_chunk: int = 64,
    avaliar: torch.Tensor | None = None,  # (S,) bool ou None
):
    device = routes.device
    S, Lmax = routes.shape
    dtype = distS.dtype

    can = lengths >= 5                      # (S,)
    if avaliar is not None:
        # garante bool no device certo
        avaliar = avaliar.to(device=device, dtype=torch.bool)
        active = can & avaliar
    else:
        active = can

    # se ninguém ativo: retorna sem fazer nada
    if not active.any():
        return routes, torch.zeros((S,), device=device, dtype=dtype)

    best_delta = torch.full((S,), float("inf"), device=device, dtype=dtype)
    best_i = torch.zeros((S,), device=device, dtype=torch.long)
    best_k = torch.zeros((S,), device=device, dtype=torch.long)

    k_all = torch.arange(2, Lmax - 1, device=device, dtype=torch.long)
    s_idx = torch.arange(S, device=device, dtype=torch.long)

    routes_exp = routes[:, None, None, :]   # (S,1,1,Lmax)
    Kmax = k_all.numel()

    for i0 in range(1, Lmax - 2, i_chunk):
        i1 = min(Lmax - 2, i0 + i_chunk)
        i_blk = torch.arange(i0, i1, device=device, dtype=torch.long)
        Ci = i_blk.numel()

        I = i_blk[:, None].expand(Ci, Kmax)     # (Ci,Kmax)
        K = k_all[None, :].expand(Ci, Kmax)     # (Ci,Kmax)

        L = lengths[:, None, None]             # (S,1,1)
        valid = (I[None] < K[None]) & (K[None] <= (L - 2)) & (I[None] <= (L - 3))
        valid &= active[:, None, None]         # <<< usa active aqui

        I3 = I[None, :, :]
        K3 = K[None, :, :]

        prev = torch.take_along_dim(routes_exp, (I3 - 1)[..., None], dim=-1).squeeze(-1)
        a    = torch.take_along_dim(routes_exp,  I3      [..., None], dim=-1).squeeze(-1)
        c    = torch.take_along_dim(routes_exp,  K3      [..., None], dim=-1).squeeze(-1)
        nxt  = torch.take_along_dim(routes_exp, (K3 + 1)[..., None], dim=-1).squeeze(-1)

        valid &= (a != 0) & (c != 0)

        s3 = s_idx[:, None, None].expand(S, Ci, Kmax)

        delta = distS[s3, prev, c] + distS[s3, a, nxt] - distS[s3, prev, a] - distS[s3, c, nxt]
        delta = torch.where(valid, delta, torch.full_like(delta, float("inf")))

        flat = delta.view(S, -1)
        blk_best_val, blk_best_pos = torch.min(flat, dim=1)

        better = blk_best_val < best_delta
        if better.any():
            best_delta = torch.where(better, blk_best_val, best_delta)

            i_local = blk_best_pos // Kmax
            k_local = blk_best_pos %  Kmax

            i_pick = i_blk[i_local]
            k_pick = k_all[k_local]

            best_i = torch.where(better, i_pick, best_i)
            best_k = torch.where(better, k_pick, best_k)

    # aplica só onde ativo e melhorou
    apply = active & torch.isfinite(best_delta) & (best_delta < 0)
    if not apply.any():
        # para rotas inativas ou sem melhora -> delta 0
        return routes, torch.zeros((S,), device=device, dtype=dtype)

    ar = torch.arange(Lmax, device=device)[None, :].expand(S, Lmax)
    i_ = best_i[:, None]
    k_ = best_k[:, None]
    in_seg = (ar >= i_) & (ar <= k_)
    ar_rev = torch.where(in_seg, (i_ + k_ - ar), ar)
    ar_rev = torch.where(apply[:, None], ar_rev, ar)

    routes_new = routes.gather(1, ar_rev)

    out_delta = torch.zeros((S,), device=device, dtype=dtype)
    out_delta = torch.where(apply, best_delta, out_delta)

    return routes_new, out_delta


import torch._dynamo
torch._dynamo.config.suppress_errors = True

try:
    two_opt_best_intra_routes_batched_chunked = torch.compile(two_opt_best_intra_routes_batched_chunked)
except Exception as e:
    print("torch.compile desativado:", e)

@torch.no_grad()
def check_dist_matrix(dist: torch.Tensor, atol=1e-6, rtol=1e-6):
    if dist.dim() == 2:
        diag_ok = torch.allclose(torch.diagonal(dist), torch.zeros(dist.size(0), device=dist.device, dtype=dist.dtype), atol=atol, rtol=rtol)
        sym_ok  = torch.allclose(dist, dist.transpose(-1, -2), atol=atol, rtol=rtol)
        max_diag = torch.diagonal(dist).abs().max().item()
        max_asym = (dist - dist.transpose(-1, -2)).abs().max().item()
        return diag_ok, sym_ok, max_diag, max_asym

    elif dist.dim() == 3:
        diag = torch.diagonal(dist, dim1=-2, dim2=-1)             # (B,N)
        diag_ok = torch.allclose(diag, torch.zeros_like(diag), atol=atol, rtol=rtol)
        sym_ok  = torch.allclose(dist, dist.transpose(-1, -2), atol=atol, rtol=rtol)
        max_diag = diag.abs().max().item()
        max_asym = (dist - dist.transpose(-1, -2)).abs().max().item()
        return diag_ok, sym_ok, max_diag, max_asym

    else:
        raise ValueError("dist must be (N,N) or (B,N,N)")

@torch.no_grad()
def route_cost(distS: torch.Tensor, route: torch.Tensor, length: int):
    # route: (Lmax,)  com [0 ... 0 padding]
    # length: comprimento real incluindo os 2 depósitos
    r = route[:length].long()
    a = r[:-1]
    b = r[1:]
    return distS[a, b].sum()

@torch.no_grad()
def check_delta_one(distS_one: torch.Tensor, route: torch.Tensor, length: int, i: int, k: int):
    # aplica a reversão e compara custo
    before = route_cost(distS_one, route, length)
    new = route.clone()
    new[i:k+1] = torch.flip(new[i:k+1], dims=[0])
    after = route_cost(distS_one, new, length)
    return (after - before).item(), before.item(), after.item()


# ------- FUNCOES PARA DOUBLE CHECK DE DELTAS

@torch.no_grad()
def batched_route_costs(distS: torch.Tensor, routes: torch.Tensor, lengths: torch.Tensor):
    """
    distS:   (S,N,N) float
    routes:  (S,Lmax) long
    lengths: (S,) long  (inclui 0 inicial e 0 final)
    return:  (S,) float custo real somando dist[a_t, a_{t+1}] até length-1
    """
    S, Lmax = routes.shape
    device = routes.device

    # pares consecutivos (S, Lmax-1)
    u = routes[:, :-1].long()
    v = routes[:, 1: ].long()

    # máscara: t < length-1
    t = torch.arange(Lmax - 1, device=device)[None, :]          # (1,Lmax-1)
    mask = t < (lengths[:, None] - 1)                           # (S,Lmax-1)

    s_idx = torch.arange(S, device=device)[:, None]             # (S,1)
    edge = distS[s_idx, u, v]                                   # (S,Lmax-1)

    return (edge * mask).sum(dim=1)                             # (S,)


@torch.no_grad()
def double_check_small_negatives(distS, routes, routes_new, lengths, best_delta, eps=1e-7):
    """
    Corrige best_delta apenas onde ele está em (-eps, 0), usando custo real vetorizado.
    Retorna: best_delta_corrigido, improved_mask (best_delta < -eps), sus_mask
    """
    # suspeitos: negativos pequenos (ruído numérico)
    sus = (best_delta < 0) & (best_delta > -eps)
    if sus.any():
        # Calcula custos só uma vez (ainda é barato, e evita indexações ruins)
        c_old = batched_route_costs(distS, routes, lengths)          # (S,)
        c_new = batched_route_costs(distS, routes_new, lengths)      # (S,)
        delta_true = c_new - c_old                                   # (S,)

        # Corrige apenas suspeitos
        best_delta = best_delta.clone()
        best_delta[sus] = delta_true[sus]

    improved = best_delta < -eps
    return best_delta, improved, sus




@torch.no_grad()
def apply_two_opt_intra_to_solutions_fast(sol: torch.Tensor, dist: torch.Tensor, max_iters: int = 1, EPS = 1E-6):

    #diag_ok, sym_ok, max_diag, max_asym = check_dist_matrix(dist, atol=1e-6)
    #print(diag_ok, sym_ok, max_diag, max_asym)

    sol = sol.long()
    #print(sol)
    B, P, T = sol.shape
    device = sol.device
    BP = B * P
    seq = sol.reshape(BP, T)
   

    # nxt sem wrap
    nxt = torch.zeros_like(seq)
    nxt[:, :-1] = seq[:, 1:]
    nxt[:, -1] = 0

    is0 = seq == 0
    start = is0 & (nxt != 0)
    end   = (seq != 0) & (nxt == 0)

    start_pos = torch.nonzero(start, as_tuple=False)
    end_pos   = torch.nonzero(end,   as_tuple=False)
    if start_pos.numel() == 0 or end_pos.numel() == 0:
        return sol, torch.zeros((B, P), device=device)

    start_key = start_pos[:, 0] * T + start_pos[:, 1]
    end_key   = end_pos[:, 0]   * T + end_pos[:, 1]
    start_pos = start_pos[start_key.argsort(stable=True)]
    end_pos   = end_pos[end_key.argsort(stable=True)]

    # evita desalinhamento se tiver mismatch
    S = min(start_pos.shape[0], end_pos.shape[0])
    start_pos = start_pos[:S]
    end_pos   = end_pos[:S]

    rows = start_pos[:, 0]
    t0   = start_pos[:, 1]
    t1   = end_pos[:, 1]

    lengths = (t1 - t0 + 2).clamp(min=2)
    #Lmax = int(lengths.max().item())

    #offs = torch.arange(Lmax, device=device)[None, :]
    #idx_t = (t0[:, None] + offs).clamp(max=T-1)  # ok, mas vamos mascarar

    #routes = seq[rows[:, None], idx_t]  # (S,Lmax)


    # lengths: (S,) = (t1 - t0 + 2)  inclui o 0 inicial e o 0 final
    Lmax = int(lengths.max().item())
    offs = torch.arange(Lmax, device=device)[None, :]          # (1,Lmax)
    pos  = offs.expand(S, Lmax)                                # (S,Lmax)

    idx_t_raw = t0[:, None] + pos                              # (S,Lmax)

    # offsets fora do comprimento real viram o próprio t0 (que é 0)
    idx_t = torch.where(pos < lengths[:, None], idx_t_raw, t0[:, None])

    routes = seq[rows[:, None], idx_t]                         # (S,Lmax)

    #print("111111")
    #print(routes)

    assert (routes[:, 0] == 0).all()
    # cada linha deve ter exatamente um "0 final" na posição lengths-1
    end_tok = routes.gather(1, (lengths-1).clamp_min(0)[:,None]).squeeze(1)
    assert (end_tok == 0).all()

    # e não deve haver "0" dentro do miolo (opcional, se você quer rotas sem rotas vazias)
    inside = routes[:, 1:-1]
    bad = (inside == 0) & (torch.arange(routes.size(1)-2, device=device)[None,:] < (lengths-2)[:,None])
    assert not bad.any(), "há 0 interno => ainda está colando rotas"

    

    # máscara de posições válidas dentro do comprimento real
    pos = torch.arange(Lmax, device=device)[None, :].expand(S, Lmax)
    valid_pos = pos < lengths[:, None]

    # distS
    if dist.dim() == 2:
        distS = dist[None, :, :].expand(S, -1, -1)
    else:
        distS = dist[rows // P]

    delta_total = torch.zeros((BP,), device=device, dtype=torch.float32)
    
    avaliar = None
    for i in range(max_iters):
        
        #routes_new, best_delta = two_opt_best_intra_routes_batched_chunked(routes, distS, lengths, avaliar = avaliar)

        routes_new, best_delta = two_opt_best_intra_routes_batched_chunked(
            routes, distS, lengths, i_chunk=128, avaliar=avaliar
        )
        
        best_delta, _, _ = double_check_small_negatives(distS, routes, routes_new, lengths, best_delta, eps=1e-6)

        improved = (best_delta < 0)

        if not improved.any():
            break
        """
        for i in range(routes[improved].shape[0]):
            CO = route_cost(distS[improved][i], routes[improved][i], lengths[improved][i]).item()
            CA = route_cost(distS[improved][i], routes_new[improved][i], lengths[improved][i]).item()
            if CA - CO != 0:
                best_delta[improved][i] = CA - CO
        #        print(CO, CA, CA - CO)

        improved = (best_delta < 0)

        if not improved.any():
            break
        """

        routes = routes_new
        delta_total.index_add_(0, rows, best_delta.float())

        # Só continua avaliando quem melhorou nesta iteração
        avaliar = improved

        # GARANTE: não muda fora do comprimento real (evita corromper próximas rotas/padding)
        routes = torch.where(valid_pos, routes, seq[rows[:, None], idx_t])

        #print("--------------")
        #print(best_delta[avaliar])
        #print(routes[avaliar])
        #print(routes)
    #if i+1 == max_iters: 
    #    print("Chegou ao Limite !!!!")
    #    print(best_delta[avaliar])

    # escrever de volta só onde valid_pos
    seq_out = seq.clone()
    old_block = seq_out[rows[:, None], idx_t]
    new_block = torch.where(valid_pos, routes, old_block)
    seq_out[rows[:, None], idx_t] = new_block
    #print(seq_out.view(B, P, T))
    return seq_out.view(B, P, T), delta_total.view(B, P)






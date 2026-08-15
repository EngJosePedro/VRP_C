import torch
from typing import Tuple

# ------------------------------------------------------------
# 1) Extrai rotas (S,Lmax) a partir do sol (B,P,T)
#    (mesmo padrão do seu apply_two_opt_intra_to_solutions_fast)
# ------------------------------------------------------------
@torch.no_grad()
def extract_routes_from_solution(sol: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    sol: (B,P,T) long, packed:
         [0, a,b,0, c,d,e,0, 0,0,0]
    Retorna:
      routes:   (S, Lmax) long  (cada rota com [0 ... 0] e padding com 0)
      lengths:  (S,) long        (comprimento real da rota incluindo 0 inicial e 0 final)
      rows:     (S,) long        (índice da linha BP da qual a rota veio)
      idx_t:    (S, Lmax) long   (posições no seq original para writeback)
      valid_pos:(S, Lmax) bool   (pos < length)
      seq:      (BP, T) long     (sol flatten)
    """
    sol = sol.long()
    B, P, T = sol.shape
    device = sol.device
    BP = B * P

    seq = sol.reshape(BP, T)

    # nxt sem wrap (mesma ideia do seu código)
    nxt = torch.zeros_like(seq)
    nxt[:, :-1] = seq[:, 1:]
    nxt[:, -1] = 0

    is0 = seq == 0
    start = is0 & (nxt != 0)        # 0 seguido de cliente
    end   = (seq != 0) & (nxt == 0) # cliente seguido de 0

    start_pos = torch.nonzero(start, as_tuple=False)  # (S,2) [row, t0]
    end_pos   = torch.nonzero(end,   as_tuple=False)  # (S,2) [row, t1_last_customer]

    if start_pos.numel() == 0 or end_pos.numel() == 0:
        # sem rotas
        empty_routes = torch.zeros((0, 2), device=device, dtype=torch.long)
        empty_lengths = torch.zeros((0,), device=device, dtype=torch.long)
        empty_rows = torch.zeros((0,), device=device, dtype=torch.long)
        empty_idx = torch.zeros((0, 2), device=device, dtype=torch.long)
        empty_valid = torch.zeros((0, 2), device=device, dtype=torch.bool)
        return empty_routes, empty_lengths, empty_rows, empty_idx, empty_valid, seq

    # ordena pra alinhar start/end (igual ao seu)
    start_key = start_pos[:, 0] * T + start_pos[:, 1]
    end_key   = end_pos[:, 0]   * T + end_pos[:, 1]
    start_pos = start_pos[start_key.argsort(stable=True)]
    end_pos   = end_pos[end_key.argsort(stable=True)]

    S = min(start_pos.shape[0], end_pos.shape[0])
    start_pos = start_pos[:S]
    end_pos   = end_pos[:S]

    rows = start_pos[:, 0]            # (S,)
    t0   = start_pos[:, 1]            # posição do 0 inicial da rota
    t1   = end_pos[:, 1]              # posição do último cliente; 0 final está em t1+1

    lengths = (t1 - t0 + 2).clamp(min=2)   # inclui 0 inicial e 0 final
    Lmax = int(lengths.max().item())

    offs = torch.arange(Lmax, device=device)[None, :]      # (1,Lmax)
    pos  = offs.expand(S, Lmax)                            # (S,Lmax)
    idx_raw = t0[:, None] + pos                            # (S,Lmax)

    # fora do comprimento real -> aponta pra t0 (que é 0)
    idx_t = torch.where(pos < lengths[:, None], idx_raw, t0[:, None])
    routes = seq[rows[:, None], idx_t]                     # (S,Lmax)

    valid_pos = pos < lengths[:, None]

    return routes, lengths, rows, idx_t, valid_pos, seq


# ------------------------------------------------------------
# 2) swap_mn vetorizado em routes (S,Lmax)
#    Aplica UM movimento por elemento do batch de movimentos.
# ------------------------------------------------------------
@torch.no_grad()
def swap_mn_on_routes_batched(
    routes: torch.Tensor,          # (S,Lmax)
    lengths: torch.Tensor,         # (S,)
    distS: torch.Tensor,           # (S,N,N) (mesma convenção do 2-opt)
    sA: torch.Tensor,              # (M,) idx de rota em [0..S-1]
    sB: torch.Tensor,              # (M,) idx de rota em [0..S-1]
    a_pos: torch.Tensor,           # (M,) início do segmento em A (posição na rota, incluindo 0 inicial)
    b_pos: torch.Tensor,           # (M,) início do segmento em B
    m: int,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Troca segmento de tamanho m na rota A com segmento de tamanho n na rota B.
    - m,n ∈ {0..3} e NÃO pode (0,0).
    - aqui assumimos "no máximo um zero" (se quiser exatamente um zero, valide fora).
    Retorna:
      routes_new: (S,Lmax) com A e B atualizadas (demais iguais)
      delta_pair: (M,) delta exato (custo(A')+custo(B') - custo(A)-custo(B)), SEM 2-opt
    Observação:
      - delta aqui é sem 2-opt; você compõe com teu 2-opt depois no nível de solução.
    """
    device = routes.device
    S, Lmax = routes.shape
    dtype = distS.dtype

    if (m == 0 and n == 0):
        raise ValueError("swap_mn inválido: (m,n)=(0,0)")

    # pega rotas A e B
    A = routes[sA]         # (M,Lmax)
    B = routes[sB]
    LA = lengths[sA]       # (M,)
    LB = lengths[sB]

    # faixas válidas:
    # - se len inclui [0 ... 0], posições internas de cliente estão em [1 .. L-2]
    # - segmento de tamanho Lseg>0 precisa caber: start <= (L-2) - (Lseg-1)
    # - segmento vazio (0) permite inserção em [1..L-1] (antes do 0 final inclusive)
    a_pos = a_pos.long()
    b_pos = b_pos.long()

    if m > 0:
        okA = (a_pos >= 1) & (a_pos + m <= (LA - 1))  # a_pos+m-1 <= LA-2  => a_pos+m <= LA-1
    else:
        okA = (a_pos >= 1) & (a_pos <= (LA - 1))      # inserção em [1..LA-1]
    if n > 0:
        okB = (b_pos >= 1) & (b_pos + n <= (LB - 1))
    else:
        okB = (b_pos >= 1) & (b_pos <= (LB - 1))

    ok = okA & okB & (sA != sB)

    # se não tem nada válido, retorna sem mexer
    if not ok.any():
        return routes, torch.zeros((sA.numel(),), device=device, dtype=dtype)

    # para facilitar, vamos construir A' e B' via index mapping (gather),
    # pois comprimentos mudam quando m != n.
    #
    # A' = prefixA (0..a_pos-1) + segB + midA (a_pos+m .. LA-1) + [padding 0...]
    # B' = prefixB + segA + midB
    #
    # onde LA-1 é o índice do 0 final (posição LA-1).

    M = sA.numel()
    # comprimentos novos (inclui 0 inicial e 0 final)
    LA_new = LA - m + n
    LB_new = LB - n + m

    # Lout precisa caber no Lmax (se não couber, marcamos como inválido)
    ok = ok & (LA_new <= Lmax) & (LB_new <= Lmax)

    if not ok.any():
        return routes, torch.zeros((M,), device=device, dtype=dtype)

    # extrai segmentos (max 3) com padding seguro
    def take_segment(X: torch.Tensor, start: torch.Tensor, Lseg: int) -> torch.Tensor:
        if Lseg == 0:
            return torch.zeros((M, 0), device=device, dtype=X.dtype)
        offs = torch.arange(Lseg, device=device)[None, :]  # (1,Lseg)
        idx = start[:, None] + offs                        # (M,Lseg)
        return X.gather(1, idx.clamp(0, Lmax - 1))

    segA = take_segment(A, a_pos, m)  # (M,m)
    segB = take_segment(B, b_pos, n)  # (M,n)

    # prefix e sufixo (até 0 final inclusive no sufixo)
    # prefixA: [0 .. a_pos-1]
    # tailA  : [a_pos+m .. LA-1]  (inclui 0 final)
    def take_prefix(X, Lx, cut):
        # pega 0..cut-1 (tamanho = cut)
        # vamos criar idx (M,Lmax) e mascarar
        max_len = int(Lmax)
        ar = torch.arange(max_len, device=device)[None, :].expand(M, max_len)
        mask = ar < cut[:, None]
        out = torch.zeros((M, max_len), device=device, dtype=X.dtype)
        out[mask] = X[mask]
        return out, cut  # (M,Lmax), (M,)

    def take_tail(X, Lx, start):
        # pega start..Lx-1 (tamanho = Lx-start)
        max_len = int(Lmax)
        ar = torch.arange(max_len, device=device)[None, :].expand(M, max_len)
        # tail index in original = start + ar
        idx = start[:, None] + ar
        mask = idx < Lx[:, None]
        idx = idx.clamp(0, Lmax - 1)
        out = torch.zeros((M, max_len), device=device, dtype=X.dtype)
        out[mask] = X.gather(1, idx)[mask]
        tail_len = (Lx - start).clamp_min(0)
        return out, tail_len  # (M,Lmax), (M,)

    # prefix lengths
    preA_len = a_pos
    preB_len = b_pos

    # tail starts
    tailA_start = a_pos + m
    tailB_start = b_pos + n

    preA_blk, _ = take_prefix(A, LA, preA_len)
    preB_blk, _ = take_prefix(B, LB, preB_len)

    tailA_blk, tailA_len = take_tail(A, LA, tailA_start)
    tailB_blk, tailB_len = take_tail(B, LB, tailB_start)

    # Agora montamos A' e B' (M,Lmax) com scatter por blocos:
    # A': prefix (len preA) + segB (len n) + tailA (len tailA_len)
    # B': prefix + segA + tailB
    A_new = torch.zeros_like(A)
    B_new = torch.zeros_like(B)

    # copia prefixos
    ar = torch.arange(Lmax, device=device)[None, :].expand(M, Lmax)
    A_new = torch.where(ar < preA_len[:, None], preA_blk, A_new)
    B_new = torch.where(ar < preB_len[:, None], preB_blk, B_new)

    # coloca segB em A_new na posição preA_len
    if n > 0:
        offs = torch.arange(n, device=device)[None, :].expand(M, n)
        idxA_seg = preA_len[:, None] + offs
        A_new.scatter_(1, idxA_seg, segB)

    # coloca segA em B_new
    if m > 0:
        offs = torch.arange(m, device=device)[None, :].expand(M, m)
        idxB_seg = preB_len[:, None] + offs
        B_new.scatter_(1, idxB_seg, segA)

    # coloca tails
    # tailA vai começar em preA_len + n
    startA_tail_new = preA_len + n
    startB_tail_new = preB_len + m

    # tailA_blk tem o tail começando em ar=0; vamos pegar os primeiros tailA_len elementos
    # e escrever em A_new[startA_tail_new + t]
    def scatter_tail(Y_new, tail_blk, tail_len, start_new):
        max_len = int(Lmax)
        t = torch.arange(max_len, device=device)[None, :].expand(M, max_len)  # posição dentro do tail_blk
        mask = t < tail_len[:, None]
        vals = tail_blk  # já alinhado no começo
        idx = start_new[:, None] + t
        # só escreve onde idx < Lmax
        mask = mask & (idx < Lmax)
        idx = idx.clamp(0, Lmax - 1)
        Y_new = Y_new.clone()
        Y_new[mask] = vals[mask]
        return Y_new

    A_new = scatter_tail(A_new, tailA_blk, tailA_len, startA_tail_new)
    B_new = scatter_tail(B_new, tailB_blk, tailB_len, startB_tail_new)

    # validação final: posições >= LA_new / LB_new ficam 0
    A_new = torch.where(ar < LA_new[:, None], A_new, torch.zeros_like(A_new))
    B_new = torch.where(ar < LB_new[:, None], B_new, torch.zeros_like(B_new))

    # monta routes_out (S,Lmax) atualizando só sA/sB onde ok
    routes_out = routes.clone()
    # aplica só nos movimentos válidos
    sA_ok = sA[ok]
    sB_ok = sB[ok]
    routes_out[sA_ok] = A_new[ok]
    routes_out[sB_ok] = B_new[ok]

    # delta exato sem 2-opt (só custo de A e B)
    # custo vetorizado por arestas consecutivas usando distS[s, u, v]
    def batched_cost_one(route_block, L_block, dist_block):
        # route_block: (K,Lmax), L_block: (K,), dist_block: (K,N,N)
        K, Lm = route_block.shape
        u = route_block[:, :-1]
        v = route_block[:, 1:]
        t = torch.arange(Lm - 1, device=device)[None, :].expand(K, Lm - 1)
        mask = t < (L_block[:, None] - 1)
        k_idx = torch.arange(K, device=device)[:, None]
        edge = dist_block[k_idx, u, v]
        return (edge * mask).sum(dim=1)

    delta_pair = torch.zeros((M,), device=device, dtype=dtype)
    # calcula só onde ok (evita custo)
    if ok.any():
        # dist para as rotas selecionadas
        distA = distS[sA_ok]  # (K,N,N)
        distB = distS[sB_ok]
        oldA = batched_cost_one(A[ok],  LA[ok],  distA)
        oldB = batched_cost_one(B[ok],  LB[ok],  distB)
        newA = batched_cost_one(A_new[ok], LA_new[ok], distA)
        newB = batched_cost_one(B_new[ok], LB_new[ok], distB)
        delta_pair[ok] = (newA + newB) - (oldA + oldB)

    return routes_out, delta_pair


# ------------------------------------------------------------
# 3) End-to-end: aplica swap_mn em rotas extraídas e escreve de volta em sol
#    (com opção de compor com teu 2-opt depois)
# ------------------------------------------------------------
@torch.no_grad()
def apply_swap_mn_inter_to_solutions_fast(
    sol: torch.Tensor,          # (B,P,T)
    dist: torch.Tensor,         # (B,N,N)
    *,
    sA: torch.Tensor,           # (M,) índice no conjunto de rotas extraídas (0..S-1)
    sB: torch.Tensor,           # (M,)
    a_pos: torch.Tensor,        # (M,) posição na rota (inclui 0 inicial)
    b_pos: torch.Tensor,        # (M,)
    m: int,
    n: int,
    two_opt_fn=None,            # ex: apply_two_opt_intra_to_solutions_fast
    two_opt_iters: int = 1,
):
    """
    Aplica swap_mn (um batch de movimentos) no sol (B,P,T).
    Se two_opt_fn for fornecido, aplica 2-opt intra depois (global).
    """
    B, P, T = sol.shape
    device = sol.device

    routes, lengths, rows, idx_t, valid_pos, seq = extract_routes_from_solution(sol)

    S = routes.shape[0]
    if S == 0:
        return sol, torch.zeros((B, P), device=device, dtype=torch.float32)

    # distS por rota (mesma lógica do seu 2-opt: dist[rows//P]) :contentReference[oaicite:4]{index=4}
    distS = dist[rows // P]

    routes_new, delta_pair = swap_mn_on_routes_batched(
        routes, lengths, distS,
        sA=sA, sB=sB,
        a_pos=a_pos, b_pos=b_pos,
        m=m, n=n
    )

    # writeback: escreve routes_new em seq_out usando idx_t/valid_pos (mesmo padrão do 2-opt) 
    seq_out = seq.clone()
    old_block = seq_out[rows[:, None], idx_t]
    new_block = torch.where(valid_pos, routes_new, old_block)
    seq_out[rows[:, None], idx_t] = new_block

    sol_out = seq_out.view(B, P, T)

    # delta_total aproximado: soma dos deltas dos pares por linha BP
    # (se você passar vários movimentos na mesma linha, isso soma todos)
    BP = B * P
    delta_total = torch.zeros((BP,), device=device, dtype=torch.float32)
    delta_total.index_add_(0, rows[sA], delta_pair.float())  # usa rows do A como linha (A e B são mesma linha se você garantir)

    if two_opt_fn is not None and two_opt_iters > 0:
        sol_out, delta_2opt = two_opt_fn(sol_out, dist, max_iters=two_opt_iters)
        delta_total = delta_total + delta_2opt.reshape(-1).float()

    return sol_out, delta_total.view(B, P)


#LORA

from .get_query import GetQuery

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



@torch.no_grad()
def greedy_cluster_router_fast(
    cluster_id: torch.Tensor,   # (B,P,nc) long in [0..K-1]
    edge_w: torch.Tensor,       # (B,N,N) float, N=nd+nc
    nd: int,
    K: int,
    approx: bool = False,
    candidate_top_c: int = 32,  # usado só no approx
):
    """
    Gera rotas por cluster via greedy nearest-neighbor.
    Saída: out (B,T,P) long com índices globais, zeros separando rotas e padding 0.

    Estratégia rápida:
      - achata BP = B*P
      - pré-computa membership (BP,K,nc) via one_hot
      - loop por 'steps' (até nc) fazendo argmin batched com mask
      - pré-aloca seq_cluster: (BP,K,nc+2) [0, ..., 0]
      - depois concatena clusters e faz padding

    approx (opcional):
      - reduz candidatos por step para top-c mais próximos do depot (seed),
        ou candidate list por cluster (barato). Mantém determinismo, mas não é exatamente igual.
    """

    assert cluster_id.dtype == torch.long
    device = cluster_id.device
    B, P, nc = cluster_id.shape
    BP = B * P
    N = nd + nc
    assert edge_w.shape[1] == N and edge_w.shape[2] == N

    # --- flatten cluster_id to (BP,nc) ---
    out = cluster_id.reshape(BP, nc)

    # --- membership mask: (BP,K,nc) ---
    # one_hot: (BP,nc,K) -> permute
    mem = torch.nn.functional.one_hot(out.clamp(min=0, max=K-1), num_classes=K).to(torch.bool)  # (BP,nc,K)
    mem = mem.permute(0, 2, 1).contiguous()  # (BP,K,nc)

    # sizes per cluster (BP,K)
    sizes = mem.sum(dim=2)  # (BP,K)

    # Early exit: if everything empty => just depot
    if sizes.max().item() == 0:
        # out sequence is just [0]
        return torch.zeros((B, 1, P), device=device, dtype=torch.long)

    # --- map customers local->global ---
    cust_global = (torch.arange(nc, device=device, dtype=torch.long) + nd)  # (nc,)

    # --- expand edge_w for BP ---
    # edge_w is (B,N,N); replicate for P without copy using expand+reshape
    ew = edge_w[:, None, :, :].expand(B, P, N, N).reshape(BP, N, N)  # (BP,N,N)

    # -----------------------------
    # Optional approx: prebuild candidate list per (BP,K) from depot distances
    # -----------------------------
    cand_mask = None
    if approx:
        # depot->cust distances: (BP,nc)
        d0 = ew[:, 0, cust_global]  # (BP,nc)
        # For each (BP,K): keep only members, then take top_c nearest to depot as candidate set
        # Build a mask candidates (BP,K,nc) bool
        cand_mask = torch.zeros((BP, K, nc), device=device, dtype=torch.bool)
        top_c = min(candidate_top_c, nc)
        big = torch.tensor(1e9, device=device, dtype=d0.dtype)

        # We'll do it cluster-wise but vectorized: compute masked costs then topk along nc
        # masked_d0: (BP,K,nc)
        masked_d0 = torch.where(mem, d0[:, None, :], big)
        # take top_c smallest => use topk on negative or use kthvalue; easiest: topk on -masked with largest
        # But masked has big for non-members; we want smallest.
        vals, idx = torch.topk(-masked_d0, k=top_c, dim=2)  # picks "largest -dist" => smallest dist
        # idx: (BP,K,top_c) maybe includes non-members if cluster smaller; guard with mem check
        # scatter into cand_mask
        cand_mask.scatter_(2, idx, True)
        # ensure candidates subset of members
        cand_mask &= mem

    # -----------------------------
    # Batched greedy per cluster
    # -----------------------------
    # seq_cluster will store global node ids, padded with 0.
    # Layout: [0, v1, v2, ..., 0] with fixed length nc+2 for all.
    seq_cluster = torch.zeros((BP, K, nc + 2), device=device, dtype=torch.long)
    seq_cluster[:, :, 0] = 0  # start depot
    # last position we'll set to 0 at the end; intermediate filled progressively.

    # visited mask in customer-local index: (BP,K,nc)
    visited = torch.zeros((BP, K, nc), device=device, dtype=torch.bool)

    # current node per (BP,K): start at depot 0
    cur = torch.zeros((BP, K), device=device, dtype=torch.long)  # global ids

    # remaining count per cluster
    rem = sizes.clone()  # (BP,K)

    # We'll iterate up to max cluster size (<=nc)
    max_steps = int(rem.max().item())

    # constants
    INF = torch.tensor(1e9, device=device, dtype=ew.dtype)

    for step in range(1, max_steps + 1):
        active = rem > 0  # (BP,K)
        if not active.any():
            break

        # candidate customers allowed: members & not visited & (optional candidates)
        cand = mem & (~visited)
        if cand_mask is not None:
            cand = cand & cand_mask

        # We need costs from cur to each customer in cand.
        # Fetch row ew[bp, cur, cust_global] => (BP,K,nc) using gather:
        # ew_cur_all: for each (BP,K), gather row at index cur
        ew_rows = ew.gather(1, cur[:, :, None].expand(BP, K, N))  # (BP,K,N), gathers along dim=1 (row)
        costs_all = ew_rows[:, :, cust_global]                    # (BP,K,nc)

        # mask non-candidates to INF
        costs = torch.where(cand, costs_all, INF)

        # argmin along nc
        nxt_local = costs.argmin(dim=2)  # (BP,K) in [0..nc-1]
        # if a cluster is active but cand could be empty (approx too strict), fallback to any unvisited member
        # detect empty: min_cost == INF
        min_cost = costs.gather(2, nxt_local[:, :, None]).squeeze(2)  # (BP,K)
        need_fallback = active & (min_cost >= INF * 0.5)

        if need_fallback.any():
            # fallback candidates: mem & ~visited (ignore cand_mask)
            cand2 = mem & (~visited)
            costs2 = torch.where(cand2, costs_all, INF)
            nxt2 = costs2.argmin(dim=2)
            nxt_local = torch.where(need_fallback, nxt2, nxt_local)

        # convert to global id
        nxt_global = cust_global[nxt_local]  # (BP,K)

        # write into sequence at position=step for active clusters
        # Only write where active, else keep 0
        seq_cluster[:, :, step] = torch.where(active, nxt_global, seq_cluster[:, :, step])

        # mark visited for active
        visited.scatter_(2, nxt_local[:, :, None], active[:, :, None])

        # update cur, rem
        cur = torch.where(active, nxt_global, cur)
        rem = torch.where(active, rem - 1, rem)

    # close each cluster route with depot 0 at (size+1)
    # We already have zeros everywhere; but we want: seq_cluster[b,k, sizes+1] = 0
    # No need to write because default is 0.

    # -----------------------------
    # Concat clusters into one seq per (BP)
    # -----------------------------
    # Goal: seq = [0] + for each k with size>0: append route_k[1:] (to avoid duplicating initial 0)
    # We'll build a flat buffer with max length = 1 + sum_k (sizes_k + 1)
    route_lens = sizes + 1  # each cluster contributes (size + 1) nodes when appending r[1:] (includes final 0)
    total_len = 1 + route_lens.sum(dim=1)  # (BP,)
    maxT = int(total_len.max().item())

    seq_bp = torch.zeros((BP, maxT), device=device, dtype=torch.long)
    seq_bp[:, 0] = 0

    # pointer per instance
    ptr = torch.ones((BP,), device=device, dtype=torch.long)

    # Loop over K (K usually << nc; acceptable). Avoid b/p loops.
    for k in range(K):
        sz = sizes[:, k]  # (BP,)
        has = sz > 0
        if not has.any():
            continue

        # take route: seq_cluster[:,k,: (sz+2)] is variable; we'll copy using a masked loop over positions.
        # We'll copy r[1:sz+2] length = sz+1
        # Pre-slice full: (BP, nc+1) for r[1:]
        r = seq_cluster[:, k, 1:]  # (BP, nc+1), contains [v1..v_sz, 0, 0, ...]
        # For each position t in [0..nc] we copy if t < (sz+1)
        # This inner loop is over nc+1 which can be large. Instead, do a vectorized scatter using indices.
        # Build positions to write: pos = ptr + t
        # We'll do it with a loop over t but it's over (max cluster size +1) across all clusters; still can be big.
        # Better: write using advanced indexing with a flattened index list per BP. We'll do a batched mask.

        # Build a mask write_mask: (BP, nc+1) where t < sz+1 AND has
        t_idx = torch.arange(nc + 1, device=device)[None, :]  # (1,nc+1)
        write_mask = has[:, None] & (t_idx < (sz + 1)[:, None])  # (BP,nc+1)

        # Compute absolute positions
        pos = ptr[:, None] + t_idx  # (BP,nc+1)

        # Flatten masked writes
        rows = torch.where(write_mask)[0]
        cols = torch.where(write_mask)[1]
        seq_bp[rows, pos[rows, cols]] = r[rows, cols]

        # advance ptr by (sz+1)
        ptr = ptr + (sz + 1)

    # Ensure sequence ends with 0 (it should, because each route appends its closing 0)
    # If total_len==1 (no routes), seq is [0] already.

    # reshape back to (B,T,P)
    seq = seq_bp.reshape(B, P, maxT).permute(0, 2, 1).contiguous()  # (B,T,P)
    return seq

class GreedyClusterRouter(nn.Module):
    def __init__(self, approx: bool = False, candidate_top_c: int = 32):
        super().__init__()
        self.approx = bool(approx)
        self.candidate_top_c = int(candidate_top_c)

    @torch.no_grad()
    def forward(self, customers, cluster_id: torch.Tensor, edge_w: torch.Tensor, nd: int, K: int):
        
    
        return greedy_cluster_router_fast(
            cluster_id=cluster_id,
            edge_w=edge_w,
            nd=nd,
            K=K,
            approx=self.approx,
            candidate_top_c=self.candidate_top_c,
        )




def _reshape_by_heads(x: torch.Tensor, H: int) -> torch.Tensor:
    """
    x: (B, L, E) -> (B, H, L, Dh)
    """
    B, L, E = x.shape
    Dh = E // H
    return x.view(B, L, H, Dh).transpose(1, 2).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, L, Dh) -> (B, L, E)
    """
    B, H, L, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, L, H * Dh)


class MHA(nn.Module):
    """
    Multi-Head Attention bloco leve:
    - projeta Q,K,V
    - usa scaled_dot_product_attention
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.E = embed_dim
        self.H = num_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        q: torch.Tensor,             # (B, Lq, E)
        kv: torch.Tensor,            # (B, Lkv, E)
        attn_mask: torch.Tensor | None = None,  # bool, shape broadcastável p/ (B, H, Lq, Lkv)
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        _, Lkv, _ = kv.shape

        Q = _reshape_by_heads(self.q_proj(q), self.H)     # (B,H,Lq,Dh)
        K = _reshape_by_heads(self.k_proj(kv), self.H)    # (B,H,Lkv,Dh)
        V = _reshape_by_heads(self.v_proj(kv), self.H)    # (B,H,Lkv,Dh)

        # scaled_dot_product_attention espera mask em formato compatível com (B,H,Lq,Lkv)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B,H,Lq,Dh)

        out = self.out_proj(_merge_heads(out))            # (B,Lq,E)
        return out



class Graph_Decoder(nn.Module):
    def __init__(self, embedding_dim=256, head_num=16, qkv_dim=16,
                 logit_clipping=10.0, temperature=1.0, attn_dropout=0.0,
                 return_probs=True,
                 ):
        super().__init__()
        assert head_num * qkv_dim == embedding_dim
        self.E = embedding_dim
        self.H = head_num
        self.Dh = self.E // self.H
        self.sqrtE = math.sqrt(embedding_dim)
        self.logit_clipping = logit_clipping
        self.attn_dropout = attn_dropout
        self.return_probs = return_probs

        self.project_ctx = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.Dropout(self.attn_dropout),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )

        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)

        self.project_query = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.Dropout(self.attn_dropout),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )

        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.logit_key = None
        self.glimpse_key = None
        self.glimpse_v = None
        self.query_ctx = None

    def precompute(self, node_embedding):
        # node_embedding B, N, E
        
        self.query_ctx = self.project_ctx(node_embedding.mean(-2)[:, None,:]) # B, 1, E
        glimpse_key, glimpse_v, self.logit_key = \
            self.project_node_embeddings(node_embedding[:, :, :]).chunk(3, dim=-1) # B, N, emb_dim
        
        self.glimpse_key = _reshape_by_heads(glimpse_key, self.H)# (B,H,N,Dh)
        self.glimpse_v = _reshape_by_heads(glimpse_v, self.H)# (B,H,N,Dh)

        self._scale = 1.0 / self.sqrtE
        self._neg_large = torch.finfo(node_embedding.dtype).min

    def clear_cache(self):
        self.logit_key = None
        self.glimpse_key = None
        self.glimpse_v = None
        self.query_ctx = None
    
    def forward(self, query, const_emb, node_embedding = None, mask = None, return_score = False):
        """
        query: B, P, E
        const_emb: B, P, E
        mask: B, P, nc

        return: # B, P, N
        """
        B, P, E = query.shape
        query = self.project_query(query) + self.query_ctx + const_emb # B, P, E

        glimpse_Q = _reshape_by_heads(query, self.H)  # (B,H,P,Dh)

        #attn_visited_mask = mask.unsqueeze(1)          # (B,1,P,N)
        attn_allowed_mask = (~mask).unsqueeze(1)  # True = pode atender

        heads = F.scaled_dot_product_attention(
            glimpse_Q, self.glimpse_key, self.glimpse_v,
            attn_mask=attn_allowed_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False
        )  # (B,H,P,Dh)
                
        mh = self.project_out(heads.transpose(1, 2).reshape(B, P, self.H*self.Dh)) # (B,P,E)

        #score = torch.matmul(mh, self.logit_key.transpose(-2, -1)) * self._scale  # (B,P,N)
        score = torch.bmm(mh, self.logit_key.transpose(-2, -1)) * self._scale  # (B,P,N)

        if return_score:
            return score # B, P, N
                
        if self.logit_clipping and self.logit_clipping > 0:
            score = self.logit_clipping * torch.tanh(score)

        score = score.masked_fill(mask, self._neg_large)
        
        
        log_p = F.log_softmax(score, dim=-1)  # não divide por temperature de novo
        
        return log_p # B, P, N
        
class Decoder(nn.Module):
    """End-to-end decoder:
    """
    
    DEBUG = False
    
    def __init__(self, problem, embed_dim: int, head_num: int, dropout: float = 0.0):
        super().__init__()
        
        self.freezed = False
        self.baseline_function = None

        self.problem = problem
        self.embed_dim = int(embed_dim)
        n_heads = head_num
        
        self.get_query = GetQuery(in_dim = self.embed_dim, out_dim = self.embed_dim) #SelectedGATLayer

        
        self.decoder = Graph_Decoder(embedding_dim=self.embed_dim, head_num=n_heads, qkv_dim=self.embed_dim // n_heads,
                 logit_clipping=10.0, temperature=1.0, attn_dropout=dropout,
                 return_probs=True,
                 )

        self.emb_capacity = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        
        self.W_placeholder = nn.Parameter(torch.Tensor(self.embed_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        
        self.visited_proj = nn.Linear(3 * self.embed_dim, self.embed_dim) # Inf de nós visitados e não visitados
        self.visited_gate = nn.Linear(3 * self.embed_dim, self.embed_dim) # gate para aprender a ignorar informação desses nós

    def precompute(self, node_emb):
        self.decoder.precompute(node_emb)
    
    def clear_cache(self):
        if hasattr(self.decoder, "clear_cache"):
            self.decoder.clear_cache()

    def set_baseline_function(self, baseline_function):
        self.baseline_function = baseline_function


    def forward_decoder(self, 
                    node_emb, depots, customers, edge_w, 
                    pomo: int, 
                    decode_type: str,
                    cluster_id = None,
                    get_sequence = False,
                    forced_seq=None):
            
            B, N, E = node_emb.shape
            n_d = depots.shape[1]
            n_c = customers.shape[1]
            
            P = forced_seq.size(1) if forced_seq is not None else (1 if decode_type == "greedy" else int(pomo))
    
            logp_acc = torch.zeros(B, P, device=node_emb.device)
            entropy_acc = torch.zeros(B, P, device=node_emb.device)
    
            selected = torch.zeros(B, P, device=node_emb.device, dtype=torch.int64)
    
            b_idx = torch.arange(B, device=node_emb.device).unsqueeze(1)  # (B,1)
            p_idx = torch.arange(P, device=node_emb.device).unsqueeze(0)  # (1,P)
    
            FirstNode = torch.zeros_like(logp_acc, dtype=torch.bool)
            
            state = self.problem.make_state(depots,customers,P)
    
            #max_steps = 1 + 3 * N
            if forced_seq is not None:
                max_steps = forced_seq.size(2) - 1
            else:
                max_steps = 1 + 3 * N
            
            if get_sequence and forced_seq is None:
                seq = torch.empty((B, P, max_steps + 1), device=node_emb.device, dtype=torch.int16)
                seq[b_idx, p_idx, 0] = selected.to(seq.dtype)
            
            for step in range(1, max_steps):
                #if decode_type != "greedy": print(f" = {step} =============")
                
                mask = state.get_mask()
                
                # 2) State context
                emb_cap = self.emb_capacity(state.VEHICLE_CAPACITY.unsqueeze(-1))
                
                visited_ctx, unvisited_ctx = state.get_visited_and_unvisited_information(node_emb)
                state_ctx = torch.cat([visited_ctx, unvisited_ctx, emb_cap], dim=-1)  # (B,P,3E)
                visited_info = self.visited_proj(state_ctx)              # (B,P,E) - INFOR NODES
                visited_gate = torch.sigmoid(self.visited_gate(state_ctx))       # (B,P,E) - IGNORE INFOR NODES
                state_ctx = visited_gate * visited_info
                emb_cap = emb_cap + state_ctx
    
                # 1) Embedding do nó atual selecionado            
                query_selected = self.get_query(
                    node_emb,
                    selected,
                    mask,
                    emb_cap
                )  # (B,P,E)
    
                n_embedding = query_selected
                
                # 6) Decoder (evita clone de máscara)
                const_emb = torch.zeros(B, P, E, device=n_embedding.device)
                logp = self.decoder(n_embedding, const_emb=const_emb, node_embedding = node_emb, mask=mask)  # (B,P,N)
    
                # entropia verdadeira por passo
                probs = logp.exp()
                #print(step, probs)
                entropy_t = -(probs * logp).sum(dim=-1)  # (B,P)
                entropy_acc += entropy_t
                #if decode_type != "greedy": print(logp.exp()[0, 0])
                
                # 7) Seleção
                if forced_seq is None:
                    selected = self._select_cluster2(logp.exp(), decode_type).detach()
                else:
                    selected = forced_seq[:, :, step + 1].long()
                
                if get_sequence  and forced_seq is None:
                    seq[b_idx, p_idx, step] = selected.to(seq.dtype)
                    #if decode_type != "greedy": print(seq[0, 0, :step+1])
    
                # 9) Atualiza cluster_id e acumula logp

                logp_sel = logp.gather(2, selected.unsqueeze(-1)).squeeze(-1)
                logp_acc += logp_sel #logp.gather(2, selected.unsqueeze(-1)).squeeze(-1)  # (B,P)
                
                state = state.update(selected, edge_w, step)
    
                
                # A partir daqui, sempre 'já iniciou'
                if step == 0:
                    FirstNode.fill_(True)
    
                # 11) critério de término
                if step >= n_c:
                    all_finished = state.all_finished()
                    if all_finished and (selected < n_d).all(): break
                    
            
            state = state.get_final_cost(edge_w)
            costs = state.costs
            entropy = entropy_acc / max_steps
    
            if get_sequence:
                seq = seq[:, :, :step + 1]
            
            if forced_seq is not None:
                return logp_acc, costs, entropy
    
            if get_sequence:
                return logp_acc, seq, costs, entropy
            
            return logp_acc, costs, entropy


    def _select_cluster2(self, probs: torch.Tensor, decode_type):
        """
        
        :param probs: (B,P,Kmax)
        :param mask: Description
        """
        if self.DEBUG:
            assert (probs == probs).all(), "Probs should not contain any nans"
        
        if decode_type == "greedy":
            selected = probs.argmax(dim=-1)  # (B,1)
            
            #assert not mask.gather(2, selected.unsqueeze(
            #    -1)).any(), "Decode greedy: infeasible action has maximum probability"

        elif decode_type == "sampling" or decode_type == "sample":
            
            # Achatar (B,pomo,K) → (B*pomo, k), amostrar, e depois reformatar:
            
            B, pomo, K = probs.shape
            # (B, 3, N) -> (B*3, N)
            probs_flat = probs.reshape(-1, K)

            # Amostra 1 nó por rota
            sample_flat = probs_flat.multinomial(1)  # (B*P, 1)

            # Remove dimensão extra e volta para (S, 1)
            selected = sample_flat.squeeze(1).reshape(B, pomo)
            
            
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            
            #while mask.gather(2, selected.unsqueeze(-1)).any():
            #    print('Sampled bad values, resampling!')
                # Amostra 1 nó por rota
            #    selected_flat = probs_flat.multinomial(1)  # (B*3, 1)

                # Remove dimensão extra e volta para (B, 3)
            #    selected = selected_flat.squeeze(1).reshape(B, pomo)

        else:
            assert False, "Unknown decode type"

        
        return selected
    

    
    def save(self, filename = "node_encoder_weights.pt"):
        torch.save(self.state_dict(), filename)

    def load(self, filename = "node_encoder_weights.pt"):
        try:
            self.load_state_dict(
                torch.load(filename, map_location=self.device)
            )
        except:
            print("Erro ao ler pesos")

    def freeze(self):
        print("=========== FREEZE ===============")
        self.freezed = True
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze(self):
        self.freezed = False
        for param in self.parameters():
            param.requires_grad = True



import torch
from typing import NamedTuple


class State(NamedTuple):
    """
    Estado para problema com UM ÚNICO Depot e múltiplos clientes.

    Convenção de índices globais:
        0                 -> depot único
        1 .. n_customers  -> clientes

    Tensores principais:
        demands:       (B, Ntot), com demands[:, 0] = 0
        prev_a:        (B, P), último nó selecionado
        cur_node:      (B, P), nó atual
        used_capacity: (B, P), carga usada desde a última saída do depot
        visited_:      (B, P, Ntot), apenas clientes devem ser marcados como visitados
        costs:         (B, P), custo acumulado
    """

    # Fixed
    n_customers: int
    demands: torch.Tensor          # (B, Ntot), satélite=0, clientes>0
    ids: torch.Tensor              # (B, 1)
    p_idx: torch.Tensor            # (1, P)

    # State
    cur_node: torch.Tensor         # (B, P)
    prev_a: torch.Tensor           # (B, P)
    used_capacity: torch.Tensor    # (B, P)
    visited_: torch.Tensor         # (B, P, Ntot) uint8/bool
    costs: torch.Tensor            # (B, P)
    i: torch.Tensor                # scalar step counter

    VEHICLE_CAPACITY_: float = 1.0
    ROUTE_INIT: bool = False

    @property
    def n_depots(self) -> int:
        return 0

    @property
    def sdepot_idx(self) -> int:
        return 0

    @property
    def n_nodes(self) -> int:
        return 1 + self.n_customers

    @property
    def visited(self):
        return self.visited_.bool() if self.visited_.dtype != torch.bool else self.visited_

    @property
    def VEHICLE_CAPACITY(self):
        """Capacidade restante em cada rota parcial: (B, P)."""
        return self.VEHICLE_CAPACITY_ - self.used_capacity

    @staticmethod
    def initialize(
        depot_data: torch.Tensor,
        node_data: torch.Tensor,
        num_parallel_tours: int = 1,
        visited_dtype=torch.uint8,
        vehicle_capacity: float = 1.0,
    ):
        """
        Inicializa estado para 1 depot + clientes.

        Args:
            sat_data:
                (B, 1, F_depot) ou (B, F_depot). Usado apenas para inferir B/device/dtype.
                O depot único será sempre o nó global 0.

            node_data:
                (B, n_customers, F_node), com demanda em node_data[:, :, 2].
                Exemplo usual: [x, y, demand].

            num_parallel_tours:
                P, número de soluções paralelas/POMO.

        Returns:
            State.
        """
        if depot_data.dim() == 2:
            B = depot_data.size(0)
        else:
            B, nd, _ = depot_data.size()
            assert nd == 1, "Este State é para exatamente um depot: depot_data deve ter shape (B,1,F)."

        B2, nc, d = node_data.size()
        assert B == B2 and d >= 3, "node_data deve ser (B, n_customers, >=3), com demanda na coluna 2."

        device = node_data.device
        dtype = node_data.dtype
        P = num_parallel_tours
        Ntot = 1 + nc

        demands = torch.zeros(B, Ntot, device=device, dtype=dtype)
        demands[:, 1:] = node_data[:, :, 2]

        visited_ = torch.zeros(B, P, Ntot, device=device, dtype=visited_dtype)

        # Começa no satélite único, índice 0.
        prev_a = torch.zeros(B, P, device=device, dtype=torch.long)

        return State(
            n_customers=nc,
            demands=demands,
            ids=torch.arange(B, device=device, dtype=torch.long)[:, None],
            p_idx=torch.arange(P, device=device, dtype=torch.long).unsqueeze(0),
            prev_a=prev_a,
            cur_node=prev_a,
            used_capacity=torch.zeros(B, P, device=device, dtype=dtype),
            visited_=visited_,
            costs=torch.zeros(B, P, device=device, dtype=dtype),
            i=torch.zeros((), device=device, dtype=torch.long),
            VEHICLE_CAPACITY_=float(vehicle_capacity),
            ROUTE_INIT=False,
        )

    def all_finished(self):
        """True quando todos os clientes foram visitados. Ignora o satélite."""
        return self.visited[:, :, 1:].all()

    def get_current_node(self):
        return self.prev_a  # (B, P)

    def attrib_custumers_routes(self, cust_to_route: torch.Tensor):
        """
        Compatibilidade com versões de clusterização.

        Args:
            cust_to_route: (B, P, n_customers), 1 se cliente pertence à rota, 0 caso contrário.

        Marca como visitados/bloqueados os clientes que NÃO pertencem à rota.
        """
        visited_ = self.visited_.clone()
        visited_[:, :, 1:] = visited_[:, :, 1:] + 1 - cust_to_route
        return self._replace(visited_=visited_)

    def update(self, selected: torch.Tensor, edge_data: torch.Tensor, step):
        return self.update_F2(selected, edge_data, step=int(self.i.item()))

    def update_F2(self, selected: torch.Tensor, edge_data: torch.Tensor, step):
        """
        Atualiza o estado após escolher o próximo nó.

        Args:
            selected:  (B, P), long, índice global escolhido em [0..Ntot-1].
            edge_data: (B, Ntot, Ntot), matriz de custo/distância.
            step: mantido apenas por compatibilidade. Não é necessário.

        Regras:
            - selected == 0: voltou ao satélite, reseta capacidade.
            - selected >= 1: visitou cliente, soma demanda e marca visitado.
        """
        B, P = selected.shape
        assert self.prev_a.shape == (B, P)

        b_idx = self.ids  # (B, 1)
        prev_a = self.prev_a

        # Custo da transição prev_a -> selected.
        step_cost = edge_data[b_idx, prev_a, selected]  # (B, P)
        costs = self.costs + step_cost

        selected_is_sat = selected == 0
        selected_is_customer = selected > 0

        # demanda do nó selecionado; satélite tem demanda 0.
        demand_sel = self.demands.gather(1, selected)  # (B, P)

        # Se voltou ao satélite, reinicia capacidade; se cliente, acumula demanda.
        used_capacity = torch.where(
            selected_is_sat,
            torch.zeros_like(self.used_capacity),
            self.used_capacity + demand_sel,
        )

        # Marca apenas clientes como visitados.
        visited_ = self.visited_.clone()
        if selected_is_customer.any():
            b_arange = torch.arange(B, device=selected.device)[:, None].expand(B, P)
            p_arange = torch.arange(P, device=selected.device)[None, :].expand(B, P)
            visited_[b_arange[selected_is_customer], p_arange[selected_is_customer], selected[selected_is_customer]] = 1

        return self._replace(
            prev_a=selected,
            cur_node=selected,
            used_capacity=used_capacity,
            visited_=visited_,
            costs=costs,
            i=self.i + 1,
            ROUTE_INIT=True,
        )

    def get_mask_F2(self):
        """
        Retorna mask (B, P, Ntot), onde True = infeasible.

        Convenção:
            nó 0: depot
            nós 1..N: clientes

        Regras estilo Kool/CVRP:
            - cliente é proibido se já visitado ou se excede capacidade.
            - depot é proibido quando o nó anterior já é depot E ainda existe cliente viável,
              evitando sequência 0 -> 0 enquanto há cliente possível.
            - no primeiro passo, o depot é proibido e os clientes viáveis são liberados.
            - se não existe cliente viável, o depot fica liberado para fechar/reiniciar rota.
        """
        B, P, Ntot = self.visited.shape
        visited = self.visited

        # Clientes: índices 1..Ntot-1
        visited_cust = visited[:, :, 1:]       # (B, P, n_customers)
        demand_cust = self.demands[:, 1:]      # (B, n_customers)

        exceeds = (
            demand_cust[:, None, :] + self.used_capacity[:, :, None]
            > self.VEHICLE_CAPACITY_
        )                                      # (B, P, n_customers)

        mask_cust = visited_cust | exceeds     # (B, P, n_customers)
        any_feasible_customer = (~mask_cust).any(dim=-1)  # (B, P)

        prev_is_sat = self.prev_a == 0          # (B, P)

        # Bloqueia satélite se já estou no satélite e ainda posso ir a algum cliente.
        mask_sat = prev_is_sat & any_feasible_customer  # (B, P)

        if not self.ROUTE_INIT:
            # Primeiro movimento: não deixa escolher satélite de novo.
            mask_sat = torch.ones_like(mask_sat, dtype=torch.bool)

        mask_sat = mask_sat[:, :, None]         # (B, P, 1)

        return torch.cat([mask_sat, mask_cust], dim=-1)  # (B, P, Ntot)

    def get_mask(self):
        return self.get_mask_F2()

    def get_final_cost(self, edge_data: torch.Tensor):
        """
        Adiciona custo de retorno ao satélite único caso a rota termine em cliente.

        Args:
            edge_data: (B, Ntot, Ntot)

        Returns:
            State com costs atualizado.
        """
        b_idx = self.ids  # (B, 1)
        sat = torch.zeros_like(self.prev_a)
        step_cost = edge_data[b_idx, self.prev_a, sat]  # (B, P)
        costs = self.costs + step_cost
        return self._replace(costs=costs)


    def ___get_visited_and_unvisited_information(self, node_embed):
        # node_emb: (B, Ntot, E)
        # visited_: (B, P, Ntot) bool ou 0/1

        visited_f = self.visited.float()

        # embeddings médios dos visitados
        num_visited = visited_f.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B,P,1)

        visited_ctx = torch.einsum(
            "bpn,bne->bpe",
            visited_f,
            node_embed
        ) / num_visited  # (B,P,E)

        # embeddings médios dos não visitados
        unvisited_f = 1.0 - visited_f
        num_unvisited = unvisited_f.sum(dim=-1, keepdim=True).clamp_min(1.0)

        unvisited_ctx = torch.einsum(
            "bpn,bne->bpe",
            unvisited_f,
            node_embed
        ) / num_unvisited  # (B,P,E)

        return visited_ctx, unvisited_ctx #torch.cat([visited_ctx, unvisited_ctx], dim=-1)  # (B,P,2E)
    
    def get_visited_and_unvisited_information(self, node_embed):
        """
        node_embed: (B, Ntot, E)
        self.visited: (B, P, Ntot)

        Retorna:
            visited_ctx:   (B, P, E)
            unvisited_ctx: (B, P, E)
        """

        # considerar apenas clientes
        customer_embed = node_embed[:, self.n_depots:, :]          # (B, Nc, E)
        visited_c = self.visited[:, :, self.n_depots:]             # (B, P, Nc)

        visited_f = visited_c.float()
        unvisited_f = 1.0 - visited_f

        B, P, Nc = visited_f.shape
        E = node_embed.size(-1)

        # -------------------------
        # Clientes visitados
        # -------------------------
        num_visited = visited_f.sum(dim=-1, keepdim=True)          # (B, P, 1)

        visited_sum = torch.einsum(
            "bpn,bne->bpe",
            visited_f,
            customer_embed
        )                                                         # (B, P, E)

        visited_ctx = torch.where(
            num_visited > 0,
            visited_sum / num_visited.clamp_min(1.0),
            torch.zeros(B, P, E, device=node_embed.device, dtype=node_embed.dtype)
        )

        # -------------------------
        # Clientes não visitados
        # -------------------------
        num_unvisited = unvisited_f.sum(dim=-1, keepdim=True)      # (B, P, 1)

        unvisited_sum = torch.einsum(
            "bpn,bne->bpe",
            unvisited_f,
            customer_embed
        )                                                         # (B, P, E)

        unvisited_ctx = torch.where(
            num_unvisited > 0,
            unvisited_sum / num_unvisited.clamp_min(1.0),
            torch.zeros(B, P, E, device=node_embed.device, dtype=node_embed.dtype)
        )

        return visited_ctx, unvisited_ctx
    

    def calc_local_regret(self, selected, mask, edge_w):
        # Estimar custo marginal em adiconar cliente i 

        B, P = self.prev_a.shape
        N = edge_w.shape[1]

        # --------------------------------------------------
        # Regret local
        # --------------------------------------------------
        
        cur = self.prev_a  # (B, P), nó atual antes do update
        b_idx = self.ids  # (B, 1)

        # distâncias d[cur, j] para todos j
        # edge_w: (B, N, N)
        d_cur_all = edge_w[
            b_idx,
            cur
        ]  # (B, P, N)

        # d[j, depot], assumindo depot = 0
        d_to_depot = edge_w[:, :, 0]  # (B, N)
        d_to_depot = d_to_depot[:, None, :].expand(B, P, N)  # (B, P, N)

        # d[cur, depot]
        d_cur_depot = edge_w[
            b_idx,
            cur,
            torch.zeros_like(cur)
        ].unsqueeze(-1)  # (B, P, 1)

        # delta(j) = d[cur,j] + d[j,0] - d[cur,0]
        delta_all = d_cur_all + d_to_depot - d_cur_depot  # (B, P, N)

        # só considera candidatos viáveis
        INF = torch.finfo(edge_w.dtype).max / 10
        delta_masked = delta_all.masked_fill(mask, INF)

        # melhor delta viável
        delta_best = delta_masked.min(dim=-1).values  # (B, P)

        # delta da ação selecionada
        delta_selected = delta_all.gather(
            2,
            selected.unsqueeze(-1)
        ).squeeze(-1)  # (B, P)

        # regret >= 0 aproximadamente
        regret_t = (delta_selected - delta_best).clamp_min(0.0)  # (B, P)

        # se não houver nenhum candidato viável, zera
        has_feasible = (~mask).any(dim=-1)  # (B, P)
        regret_t = torch.where(
            has_feasible,
            regret_t,
            torch.zeros_like(regret_t)
        )

        return regret_t
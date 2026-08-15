import torch
from typing import NamedTuple, Optional

class State(NamedTuple):
    
    # Fixed
    n_depots: int
    n_sats: int
    n_customers: int
    demands: torch.Tensor          # (B, Ntot)  depot(s)=0, customers>0
    demands_on_sats: torch.Tensor # (B, P, n_sats)
    ids: torch.Tensor              # (B,1)
    p_idx: torch.Tensor             # (1,P)

    # State
    selected_sat: torch.Tensor           # (B,P)
    must_choose_customer: torch.Tensor   # (B,P) bool
    cur_node: torch.Tensor           # (B,P) last selected node index in [0..Ntot-1]
    prev_a: torch.Tensor           # (B,P) last selected node index in [0..Ntot-1]
    used_capacity: torch.Tensor    # (B,P) capacity used since last depot
    visited_: torch.Tensor         # (B,P,Ntot) uint8/bool
    costs: torch.Tensor            # (B,P) accumulated cost
    i: torch.Tensor                # scalar step counter (optional)

    VEHICLE_CAPACITY_: float = 1.0  

    ROUTE_INIT: bool = False

    @property
    def visited(self):
        return self.visited_.bool() if self.visited_.dtype != torch.bool else self.visited_

    @property
    def VEHICLE_CAPACITY(self):
        return self.VEHICLE_CAPACITY_ - self.used_capacity
    
    @staticmethod
    def initialize(depot_data: torch.Tensor,
                sats_data: torch.Tensor,
                node_data: torch.Tensor,
                edge_data: torch.Tensor,
                num_parallel_tours: int = 1,
                visited_dtype=torch.uint8,
                vehicle_capacity: float = 1.0):

        B, nd, _ = depot_data.size()
        _, n_sats, _ = sats_data.size()
        B2, nc, d = node_data.size()
        assert B == B2 and d >= 3

        Ntot = nd + n_sats + nc
        demands = torch.zeros(B, Ntot, device=node_data.device, dtype=node_data.dtype)
        demands[:, nd + n_sats:] = node_data[:, :, 2]

        visited_ = torch.zeros(B, num_parallel_tours, Ntot, device=node_data.device, dtype=visited_dtype)
        prev_a = torch.zeros(B, num_parallel_tours, device=node_data.device, dtype=torch.long)
        
        return State(
            n_depots=nd,
            n_sats=n_sats,
            n_customers=nc,
            demands=demands,
            ids=torch.arange(B, device=node_data.device, dtype=torch.long)[:, None],
            p_idx = torch.arange(num_parallel_tours, device=node_data.device).unsqueeze(0),  # (1,P)
            prev_a=prev_a,
            selected_sat=prev_a,
            cur_node=prev_a,
            used_capacity=torch.zeros(B, num_parallel_tours, device=node_data.device, dtype=node_data.dtype),
            visited_=visited_,
            costs=torch.zeros(B, num_parallel_tours, device=node_data.device, dtype=node_data.dtype),
            must_choose_customer=torch.zeros(B, num_parallel_tours, device=node_data.device, dtype=torch.bool),
            i=torch.zeros((), device=node_data.device, dtype=torch.long),
            VEHICLE_CAPACITY_=float(vehicle_capacity),
            demands_on_sats = torch.zeros(B, num_parallel_tours, nd + n_sats, device=node_data.device)
        )

    
    
    def all_finished(self):
        # finished when all customers visited (ignore depots)
        return self.visited[:, :, self.n_depots+self.n_sats:].all()

    def get_current_node(self):
        return self.prev_a  # (B,P)
    
    def attrib_custumers_routes(self, cust_to_route):
        # cust_to_route: B, P, nc -> 1 se deve entrar na rota, 0 caso contrário
        visited_ = self.visited_
        visited_[:, :, self.n_depots:] = visited_[:, :, self.n_depots:] + 1 - cust_to_route
        return self._replace(visited_ = visited_ )
    
    def update_F2(self, selected: torch.Tensor, edge_data: torch.Tensor, step: int):
        """
        selected: (B,P) long
        edge_data: (B,Ntot,Ntot)
        """
        B, P = selected.shape
        assert self.prev_a.shape == (B, P)

        nd = self.n_depots
        ns = self.n_sats
        head = nd + ns

        b_idx = self.ids  # (B,1)

        prev_a = self.prev_a
        if step == 0:
            prev_a = selected

        # custo
        step_cost = edge_data[b_idx, prev_a, selected]   # (B,P)
        costs = self.costs + step_cost

        # tipos de nó
        selected_is_sat = (selected >= nd) & (selected < head)     # (B,P)
        selected_is_customer = selected >= head                    # (B,P)
        prev_is_customer = self.prev_a >= head                     # (B,P)

        # atualiza satélite atual da rota
        selected_sat = self.selected_sat.clone()
        selected_sat[selected_is_sat] = selected[selected_is_sat]

        # demanda do nó selecionado
        demand_sel = self.demands.gather(1, selected)  # (B,P)
        demands_on_sats = self.demands_on_sats.clone()
        demands_on_sats[b_idx, self.p_idx, selected_sat] += demand_sel

        # qualquer nó de head (depósito ou satélite) reseta capacidade
        selected_is_head = selected < head
        used_capacity = torch.where(
            selected_is_head,
            torch.zeros_like(self.used_capacity),
            self.used_capacity + demand_sel
        )

        # visited: só clientes devem ser marcados
        visited_ = self.visited_.clone()
        if selected_is_customer.any():
            visited_[selected_is_customer.unsqueeze(-1).expand_as(visited_)] = visited_[selected_is_customer.unsqueeze(-1).expand_as(visited_)]
            visited_[torch.arange(B, device=selected.device)[:, None].expand(B, P)[selected_is_customer],
                    torch.arange(P, device=selected.device)[None, :].expand(B, P)[selected_is_customer],
                    selected[selected_is_customer]] = 1

        # --------------------------------------------------
        # controle de fluidez da rota
        # --------------------------------------------------
        # must_choose_customer = True  <=> acabamos de abrir rota em um satélite
        #
        # Regras:
        # - se escolheu cliente -> False
        # - se escolheu satélite:
        #       * vindo de cliente -> fechou rota -> False
        #       * vindo de satélite/depot (ou início) -> abriu rota -> True
        # --------------------------------------------------
        must_choose_customer = self.must_choose_customer.clone()

        if not self.ROUTE_INIT:
            # primeiro satélite escolhido abre rota
            must_choose_customer[selected_is_sat] = True

        else:
            # escolheu cliente => libera
            must_choose_customer[selected_is_customer] = False

            # escolheu satélite
            sat_closes_route = selected_is_sat & prev_is_customer
            sat_opens_route = selected_is_sat & (~prev_is_customer)

            must_choose_customer[sat_closes_route] = False
            must_choose_customer[sat_opens_route] = True

        return self._replace(
            prev_a=selected,
            demands_on_sats=demands_on_sats,
            cur_node=selected,
            selected_sat=selected_sat,
            used_capacity=used_capacity,
            visited_=visited_,
            costs=costs,
            must_choose_customer=must_choose_customer,
            i=self.i + 1,
            ROUTE_INIT=True
        )

    
    def get_mask_F2(self):
        """
        Returns mask (B,P,Ntot) where True = infeasible.

        Regras:
        - clientes: visitado OU excede capacidade
        - depósitos: sempre proibidos
        - satélites:
            * se ROUTE_INIT == False:
                - todos satélites liberados
                - todos clientes bloqueados
            * se must_choose_customer == True:
                - satélites bloqueados se houver cliente viável
            * se último nó foi cliente:
                - todos satélites bloqueados, exceto selected_sat
            * se último nó foi satélite e must_choose_customer == False:
                - qualquer satélite pode ser escolhido (abrir nova rota)
        """
        B, P, Ntot = self.visited.shape
        nd = self.n_depots
        ns = self.n_sats
        head = nd + ns

        visited = self.visited

        # --------------------------------------------------
        # clientes
        # --------------------------------------------------
        visited_cust = visited[:, :, head:]                     # (B,P,nc)
        demand_cust = self.demands[:, head:]                   # (B,nc)

        exceeds = (
            demand_cust[:, None, :] + self.used_capacity[:, :, None]
            > self.VEHICLE_CAPACITY_
        )                                                      # (B,P,nc)

        mask_cust = visited_cust | exceeds                     # (B,P,nc)
        any_feasible_customer = (~mask_cust).any(dim=-1)       # (B,P)

        # --------------------------------------------------
        # depósitos: sempre proibidos
        # --------------------------------------------------
        mask_dep = torch.ones((B, P, nd), device=visited.device, dtype=torch.bool)

        # --------------------------------------------------
        # satélites
        # --------------------------------------------------
        mask_sat = torch.ones((B, P, ns), device=visited.device, dtype=torch.bool)

        if not self.ROUTE_INIT:
            # primeiro nó tem de ser satélite
            mask_sat[:] = False
            mask_cust = torch.ones_like(mask_cust)

        else:
            prev_is_customer = self.prev_a >= head             # (B,P)

            # --------------------------------------------------
            # Caso 1: must_choose_customer=True
            # -> acabamos de abrir rota em satélite
            # -> próximo deve ser cliente, se houver algum viável
            # --------------------------------------------------
            force_customer = self.must_choose_customer & any_feasible_customer   # (B,P)

            # onde NÃO estamos forçando cliente, satélites ficam livres por padrão
            mask_sat = force_customer[:, :, None].expand(B, P, ns).clone()

            # --------------------------------------------------
            # Caso 2: último nó foi cliente
            # -> só pode voltar para selected_sat
            # --------------------------------------------------
            if prev_is_customer.any():
                sel_sat_local = self.selected_sat - nd   # global -> [0..ns-1]

                valid_sel = (
                    prev_is_customer
                    & (self.selected_sat >= nd)
                    & (self.selected_sat < head)
                )

                # primeiro bloqueia tudo nesses casos
                mask_sat[prev_is_customer] = True

                # libera apenas o satélite correto
                if valid_sel.any():
                    b_idx = torch.arange(B, device=visited.device)[:, None].expand(B, P)
                    p_idx = torch.arange(P, device=visited.device)[None, :].expand(B, P)

                    mask_sat[b_idx[valid_sel], p_idx[valid_sel], sel_sat_local[valid_sel]] = False

        return torch.cat([mask_dep, mask_sat, mask_cust], dim=-1)

    def _get_mask_F2(self):
        """
        Returns mask (B,P,Ntot) where True = infeasible.
        Rules (Kool-like):
        - customers infeasible if visited OR exceeds capacity
        - depot infeasible if prev_a is depot AND there exists some feasible customer (avoid depot twice)
        """
        B, P, Ntot = self.visited.shape
        nd = self.n_depots

        visited = self.visited  # (B,P,Ntot)

        # customers visited mask
        visited_cust = visited[:, :, nd+self.n_sats:]  # (B,P,nc)

        # capacity constraint: demand + used_capacity > capacity
        # demands for customers only
        demand_cust = self.demands[:, nd+self.n_sats:]  # (B,nc)
        exceeds = (demand_cust[:, None, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY_)  # (B,P,nc)

        mask_cust = visited_cust | exceeds  # (B,P,nc)

        # depot mask: forbid choosing depot twice in a row if there is at least one feasible customer
        prev_is_depot = (self.prev_a < nd + self.n_sats)  # (B,P)
        any_feasible_customer = (mask_cust == 0).any(dim=-1)  # (B,P)
        mask_depot = prev_is_depot & any_feasible_customer  # (B,P)

        # Se primeiro ponto:
        if not self.ROUTE_INIT:
            mask_depot = ~(self.prev_a < nd+self.n_sats)  # (B,P)
            mask_cust = torch.ones_like(visited_cust)#~(visited_cust == 1)

        # for nd>1: apply same logic to ALL depots (simplest Kool-like)
        mask_dep = mask_depot[:, :, None].expand(B, P, nd)  # (B,P,nd)
        
        return torch.cat([mask_dep, mask_cust], dim=-1)  # (B,P,Ntot)
    

    def get_final_cost(self, edge_data: torch.Tensor):
        """
        Add return-to-depot cost for each pomo route.
        Assumes depot index 0.
        """
        b_idx = self.ids  # (B,1)
        depot0 = torch.zeros_like(self.prev_a)

        step_cost = edge_data[b_idx, self.prev_a, self.selected_sat]   # (B,P)
        costs = self.costs + step_cost

        #C = self.costs + edge_data[b_idx, self.prev_a, depot0]
        return self._replace(costs = costs)
import torch
from typing import NamedTuple


class State(NamedTuple):
    """
    Estado para problema com UM ÚNICO Depot e múltiplos clientes.

    Convenção de índices globais:
        0                 -> depot único
        1 .. n_customers  -> clientes

    Tensores principais:
        prev_a:        (B, P), último nó selecionado
        visited_:      (B, P, Ntot), apenas clientes devem ser marcados como visitados
        costs:         (B, P), custo acumulado
    """

    # Fixed
    n_customers: int
    ids: torch.Tensor              # (B, 1)
    p_idx: torch.Tensor            # (1, P)

    # State
    first_a: torch.Tensor           # (B, P)
    prev_a: torch.Tensor           # (B, P)
    visited_: torch.Tensor         # (B, P, Ntot) uint8/bool
    costs: torch.Tensor            # (B, P)
    i: torch.Tensor                # scalar step counter

    @property
    def visited(self):
        return self.visited_.bool() if self.visited_.dtype != torch.bool else self.visited_

    @staticmethod
    def initialize(
        depot_data: torch.Tensor,
        cust_data: torch.Tensor,
        num_parallel_tours: int = 1,
        visited_dtype=torch.uint8,
    ):
        if depot_data.dim() == 2:
            B = depot_data.size(0)
        else:
            B, nd, _ = depot_data.size()
            assert nd == 1, "Este State é para exatamente um depot: depot_data deve ter shape (B,1,F)."

        B2, nc, d = cust_data.size()
        assert B == B2 and d == 2, "node_data deve ser (B, n_customers, ==2), com (x, y)."

        device = cust_data.device
        dtype = cust_data.dtype
        P = num_parallel_tours
        Ntot = nd + nc

        visited_ = torch.zeros(B, P, Ntot, device=device, dtype=visited_dtype)

        # Começa no satélite único, índice 0.
        prev_a = torch.zeros(B, P, device=device, dtype=torch.long)

        return State(
            n_customers=nc,
            ids=torch.arange(B, device=device, dtype=torch.long)[:, None],
            p_idx=torch.arange(P, device=device, dtype=torch.long).unsqueeze(0),
            first_a=prev_a,
            prev_a=prev_a,
            visited_=visited_,
            costs=torch.zeros(B, P, device=device, dtype=dtype),
            i=torch.zeros((), device=device, dtype=torch.long),
        )

    def all_finished(self):
        """True quando todos os clientes foram visitados. Ignora o satélite."""
        return self.visited.all()

    def get_current_node(self):
        return self.prev_a  # (B, P)

    def update(self, selected: torch.Tensor, edge_data: torch.Tensor, step):
        B, P = selected.shape
        assert self.prev_a.shape == (B, P)

        b_idx = self.ids  # (B, 1)
        p_idx = self.p_idx # (1, P)
        prev_a = self.prev_a
        if step == 0:
            first_a = selected
        else:
            first_a = self.first_a

        # Custo da transição prev_a -> selected.
        step_cost = edge_data[b_idx, prev_a, selected]  # (B, P)
        costs = self.costs + step_cost

        # Marca apenas clientes como visitados.
        visited_ = self.visited_.clone()
        
        visited_[b_idx, p_idx, selected] = 1.0

        #print(visited_[0])
        #print(selected[0])
        return self._replace(
            prev_a=selected,
            first_a = first_a,
            visited_=visited_,
            costs=costs,
            i=self.i + 1,
        )

    def get_mask(self):
        return self.visited


    def get_final_cost(self, edge_data: torch.Tensor):
        """
        Adiciona custo de retorno ao satélite único caso a rota termine em cliente.

        Args:
            edge_data: (B, Ntot, Ntot)

        Returns:
            State com costs atualizado.
        """
        b_idx = self.ids  # (B, 1)
        depot = self.first_a
        step_cost = edge_data[b_idx, self.prev_a, depot]  # (B, P)
        costs = self.costs + step_cost
        return self._replace(costs=costs)



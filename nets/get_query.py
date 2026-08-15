
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GetQuery(nn.Module):
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        negative_slope: float = 0.2,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        H: torch.Tensor,
        selected: torch.Tensor,
        mask: torch.Tensor | None = None,
        emb_cap: torch.Tensor | None = None,
    ):
        
        B, N, F_out = H.shape
        P = selected.size(1)

        # H_i: (B, P, F_out)
        H_i = H.gather(
            dim=1,
            index=selected.unsqueeze(-1).expand(B, P, F_out)
        )

        if emb_cap is not None:
            H_i = H_i + emb_cap

        H_i = self.W(H_i)  # (B, P, F_out)
        return H_i


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SDPAAttentionBlockEfficient(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_bias: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

        self.ff1 = nn.Linear(embed_dim, ff_dim, bias=use_bias)
        self.ff2 = nn.Linear(ff_dim, embed_dim, bias=use_bias)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"activation inválida: {activation}")

    def _attention(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        B, N, E = x.shape

        x_norm = self.norm1(x)

        qkv = self.qkv_proj(x_norm)  # (B, N, 3E)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        # q, k, v: (B, H, N, Dh)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn_mask = None
        if src_key_padding_mask is not None:
            # src_key_padding_mask: True = bloqueia
            # SDPA bool mask: True = permite atender
            attn_mask = ~src_key_padding_mask[:, None, None, :]  # (B,1,1,N)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).reshape(B, N, E)

        return self.out_proj(attn)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm2(x)
        h = self.ff1(h)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.ff2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        x = x + self._attention(x, src_key_padding_mask)
        x = x + self._ffn(x)
        return x


class NodeEncoder(nn.Module):
    def __init__(
        self,
        depot_dim: int,
        cust_dim: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = 512,
        dropout: float = 0.0,
        num_layers: int = 3,
        use_checkpoint: bool = False,
        checkpoint_embedder: bool = False,
        activation: str = "gelu",
        use_bias: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.checkpoint_embedder = checkpoint_embedder

        self.dep_proj = nn.Linear(depot_dim, embed_dim, bias=use_bias)
        self.cus_proj = nn.Linear(cust_dim, embed_dim, bias=use_bias)

        self.layers = nn.ModuleList([
            SDPAAttentionBlockEfficient(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
                use_bias=use_bias,
            )
            for _ in range(num_layers)
        ])

    def _embed(self, depot: torch.Tensor, customers: torch.Tensor) -> torch.Tensor:
        dep = self.dep_proj(depot)
        cus = self.cus_proj(customers)
        return torch.cat((dep, cus), dim=1)

    def _run_layer(self, layer, x, src_key_padding_mask):
        return layer(x, src_key_padding_mask=src_key_padding_mask)

    def forward(self, depot, customers, src_key_padding_mask=None):
        """
        depot:     (B, nd, depot_dim)
        customers: (B, nc, cust_dim)
        return:    (B, nd+nc, E)
        """

        if self.use_checkpoint and self.training and self.checkpoint_embedder:
            x = checkpoint(
                lambda d, c: self._embed(d, c),
                depot,
                customers,
                use_reentrant=False,
            )
        else:
            x = self._embed(depot, customers)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    lambda _x, layer=layer: self._run_layer(
                        layer,
                        _x,
                        src_key_padding_mask,
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return x

    def save(self, filename="node_encoder_weights.pt"):
        torch.save(self.state_dict(), filename)

    def load(self, filename="node_encoder_weights.pt", map_location="cpu"):
        try:
            self.load_state_dict(torch.load(filename, map_location=map_location))
        except Exception as e:
            print(f"Erro ao ler pesos: {e}")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
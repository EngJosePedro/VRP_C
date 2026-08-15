

import torch
import torch.nn as nn
from utils.tools import load_model, load_model2
from utils.two_opt_intra_v2 import apply_two_opt_intra_to_solutions_fast_v2 as apply_two_opt_intra_to_solutions


#######################
# Node Encoder
#######################

#from nets.NodeEncoder import NodeEncoder
from nets.NodeEncoder import NodeEncoder
from nets.decoder import Decoder


class AttentionModel(nn.Module):
    def __init__(self,
                    Problem,
                    opts,
                    dropout: float = 0.05,
                    decode_type = "sampling",
        ):

        super().__init__()

        self.opts = opts
        self.num_heads = opts.head_num
        self.Problem = Problem

        self.embedding_dim = opts.embedding_dim
        self.pomo = opts.pomo

        self.decode_type = decode_type
        self.NodeEncoder = NodeEncoder(
                        self.Problem.depot_dim, self.Problem.cust_dim,
                        self.embedding_dim, 
                        self.num_heads, 
                        ff_dim=512, 
                        dropout=dropout, 
                        num_layers=opts.n_encode_layers
        )
        
        self.decoder = Decoder(
            problem=self.Problem,
            head_num = self.num_heads,
            embed_dim=self.embedding_dim,
            dropout=dropout,
        )
    
    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    @staticmethod
    def remove_consecutive_zeros_vec(seq: torch.Tensor) -> torch.Tensor:
        """
        Remove zeros consecutivos em seq (B,P,T), mantendo apenas o primeiro zero de cada run.
        Mantém shape (B,P,T) com padding 0 no final.
        Totalmente vetorizado (sem loop Python em B,P).

        seq: Long/Int tensor (B,P,T)
        """
        B, P, T = seq.shape
        BP = B * P
        device = seq.device

        # ---- 1) máscara keep: remove (0,0) consecutivo mantendo o primeiro ----
        prev = seq[..., :-1]
        curr = seq[..., 1:]
        dup = (curr == 0) & (prev == 0)                      # (B,P,T-1)

        keep = torch.ones_like(seq, dtype=torch.bool)        # (B,P,T)
        keep[..., 1:] = ~dup                                 # mantém tudo exceto zeros consecutivos

        # ---- 2) achata (BP,T) ----
        seq_f  = seq.reshape(BP, T)
        keep_f = keep.reshape(BP, T)

        # ---- 3) posições compactadas: 0..(n_keep-1) por linha ----
        pos = keep_f.cumsum(dim=1) - 1                       # (BP,T), pos válido só onde keep=True

        # ---- 4) scatter vetorizado para compactar ----
        out = torch.zeros((BP, T), device=device, dtype=seq.dtype)

        rows, cols = torch.where(keep_f)                     # (nnz,)
        out[rows, pos[rows, cols]] = seq_f[rows, cols]       # escreve compactado

        return out.view(B, P, T)

    def apply_2opt(self, seq, edge_weights):
        # seq : B, P, T
        seq = AttentionModel.remove_consecutive_zeros_vec(seq)
            
        T = seq.size(2)

        # índice invertido
        rev_mask = (seq != 0).flip(dims=[-1])              # (B,P,T)
        last_nonzero_from_end = rev_mask.float().argmax(dim=-1)  # (B,P)
        last_nonzero = T - last_nonzero_from_end      # (B,P)
        max_len = last_nonzero.max().item() + 1

        seq = seq[:, :, :max_len] # B, P, T
        
        #print("R", time.time() - t, new_seq.shape, T)
        #t = time.time()
        seq, deltas_ = apply_two_opt_intra_to_solutions(seq, edge_weights, max_iters=1000)
        return seq, deltas_
        
    
    def forward(self, depots, # B, n_depot, depot_dim
                    customers,# B, n_cust, cust_dim 
                    edge_weights, # B, N, N
                ):
        """
        Docstring for forward
        
        :param depots: Description
        :param customers: Description
        :param edge_weights: Description

        :param get_sequence: usado para baseline e evaluetion para pegar a sequencia verdadeira gerada pela politica

        """
        node_emb = self.get_encoder(depots, customers)# B, N, E
        self.decoder.precompute(node_emb)


        logp, seq, costs, entropy = self.decoder.forward_decoder(
                    node_emb, depots, customers, edge_weights, 
                    self.pomo, 
                    self.decode_type,
                    get_sequence = True)

        if self.decode_type == "greedy" or not self.training or self.opts.mode == "eval":
            seq, delta_ = self.apply_2opt(seq, edge_weights)
        costs, _ = self.Problem.get_costs(edge_weights, seq.transpose(1, 2))
        
        return logp, seq, costs, entropy

    def get_encoder(self, depots, customers):
        return self.NodeEncoder(depots, customers)

    
    # --- UTILS
    @staticmethod
    def load_pretreinandeModel(path):
        try:
            model, _ = load_model(path, AttentionModel)
        except:
            model, _ = load_model2(path, AttentionModel)
        return model

    def set_dropout(self, p: float):
        """
        Altera dropout de todos os módulos Dropout do modelo.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = p

            # SDPA custom
            if hasattr(module, "attn_dropout"):
                module.attn_dropout = p
            if hasattr(module, "dropout"):
                module.dropout = p


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
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    
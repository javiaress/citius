import torch
import torch.nn as nn
import math
from einops import repeat, rearrange

def check_nan(tensor, name, step):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[NaN/Inf DETECTED] in {name} at step {step}")
        print(f"{name}.min: {tensor.min().item()}, max: {tensor.max().item()}")
        print(tensor)
        print ("\n")


class Modelo_ind(nn.Module):
    def __init__(self, d_acts, d_rsrc, d_embedding = 32, d_hidden = 32, device=None):
        super().__init__()
        self.d_acts = d_acts
        self.d_rsrc = d_rsrc
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        self.device = device

        self.rsrc_embedding = nn.Embedding(d_rsrc, d_embedding, padding_idx=0)
        self.act_embedding = nn.Embedding(d_acts, d_embedding, padding_idx=0)
        self.input_dim = d_embedding * 2 + 2
        self.linear_proj = nn.Linear(self.input_dim, d_hidden)
        self.attention_in = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=4, batch_first=True)

        self.lstm = nn.LSTM(input_size= d_hidden, hidden_size= d_hidden, batch_first=True)

        self.attn = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=4, batch_first=True)
        self.linear_out = nn.Linear(d_hidden, d_acts + 1) # +1 para el EOC
    
    def forward(self, x):

        act_emb = self.act_embedding(x[:, :, 0].long())       # (batch, seq, d_emb)
        rsrc_emb = self.rsrc_embedding(x[:, :, 1].long())     # (batch, seq, d_emb)
        time_prev = x[:, :, 2].unsqueeze(-1)  # [batch, seq_len, 1]
        time_case = x[:, :, 3].unsqueeze(-1)

        # Máscara de padding: True donde hay padding
        pad_mask = (x[:, :, 0] == 0)  # (B, L)

        emb_cat = torch.cat([act_emb, rsrc_emb, time_prev, time_case], dim=-1)  # (batch, seq, input_dim)
        x_proj = self.linear_proj(emb_cat)           # (batch, seq, d_hidden)

        x_attn, _ = self.attention_in(x_proj, x_proj, x_proj, key_padding_mask=pad_mask)
        
        out, _ = self.lstm(x_attn)

        out, _ = self.attn(out, out, out, key_padding_mask=pad_mask)

        out = out[:, -1, :]

        salida = self.linear_out(out)      

        return salida

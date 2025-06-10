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
    def __init__(self, d_acts, d_embedding = 32, d_hidden = 32, device=None):
        super().__init__()
        self.d_acts = d_acts
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        self.device = device
        self.act_embedding = nn.Embedding(d_acts, d_embedding, padding_idx=0)
        self.lstm = nn.LSTM(d_inner= d_hidden, hidden_size= d_hidden, batch_first=True)
        self.linear_out = nn.Linear(d_hidden, d_acts + 1) # +1 para el EOC
    
    def forward(self, x):

        x_lstm = self.act_embedding(x)       # (batch, seq, d_emb)
        
        out, _ = self.lstm(x_lstm)

        salida = self.linear_out(out)      

        return salida

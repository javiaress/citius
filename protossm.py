import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class SSM(nn.Module):
    def __init__(
        self,
        d_inner,
        d_state=16,
        dt_rank="auto",
        dt_init="random",
        dt_scale=1.0,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank

        #self.activation = "silu"
        #self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A = nn.Parameter(A)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32)) 
        self.D._no_weight_decay = True
        
    
    def forward(self, x):

        seq, _ = x.shape
        hidden_state = torch.zeros((self.d_inner, self.d_state), dtype=torch.float32)
        out = []
        hidden_previos = []

        for i in range(seq):
            x_proj = self.x_proj(x[i])
            B = x_proj[:self.d_state]
            dt = x_proj[self.d_state:self.d_state + self.dt_rank]
            C = x_proj[self.d_state + self.dt_rank:]

            print(f"dt shape: {dt.shape}\n\n")
            print(f"A shape: {self.A.shape}\n\n")
            print(f"B shape: {B.shape}\n\n")
            print(f"C shape: {C.shape}\n\n")

            dt = self.dt_proj(dt)
            print(f"dt shape: {dt.shape}\n\n")
            
            dt = dt.unsqueeze(0)
            dA = torch.einsum("ji,is->jis", dt, self.A) # 1 inner, inner state -> 1 inner state
            print(f"dA shape: {dA.shape}\n\n")

            B = B.unsqueeze(0)
            dB = torch.einsum("ji,js->jis", dt, B) # 1 inner, 1 state -> 1 inner state
            print(f"dB shape: {dB.shape}\n\n")
            
            hidden_state = hidden_state * dA + rearrange(x[i], "d -> 1 d 1") * dB
            print(f"hidden_state shape: {hidden_state.shape}\n\n")
            
            C = C.unsqueeze(0)
            y = torch.einsum("jis,js->ji", hidden_state, C)  + self.D * x[i]
            print(f"y shape: {y.shape}\n\n")

            hidden_previos.append(hidden_state)
            out.append(y)
            
            print("FIN ITER\n")
        return out, hidden_previos

model = SSM(
    d_inner=5,
    d_state=16
)

entrada = torch.tensor([[0, 1, 0, 0, 1],[0, 0, 0, 1, 0]], dtype=torch.float32)

out, hidden = model(entrada)

print(out)
print("\n\n")
print(hidden)
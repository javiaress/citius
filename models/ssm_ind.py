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

class SSM(nn.Module):
    def __init__(
        self,
        d_inner,
        d_state=32,
        dt_rank="auto",
        dt_init="random",
        dt_scale=1.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        device=None
    ):
        super().__init__()

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2
        )

        # Inicialización explícita de pesos
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)

        #nn.init.xavier_uniform_(self.dt_proj.weight)
        #nn.init.zeros_(self.dt_proj.bias)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
         # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True


        self.device = device

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32, device=device)) 
        self.D._no_weight_decay = True

    
    def forward(self, x):

        batch, seq, _ = x.shape
        hidden_state = torch.zeros((self.d_inner, self.d_state), dtype=torch.float32, device = self.device)
        out = []
        hidden_previos = []
        I = torch.eye(self.d_inner, self.d_state, device=self.device).unsqueeze(0).expand(batch, -1, -1)

        for i in range(seq):
            
            #print(f"x: {x.shape}\n\n")
            x_proj = self.x_proj(x[:,i,:])
            B = x_proj[:,:self.d_state]
            dt = x_proj[:,self.d_state:self.d_state + self.dt_rank]
            C = x_proj[:,self.d_state + self.dt_rank:]
            A = -torch.exp(self.A_log)

            #print(f"dt shape: {dt.shape}\n\n")
            #print(f"A shape: {A.shape}\n\n")
            #print(f"B shape: {B.shape}\n\n")
            #print(f"C shape: {C.shape}\n\n")

            dt = self.dt_proj(dt)
            #print(f"dt shape: {dt.shape}\n\n")

            dt = torch.clamp(dt, min=-0.1, max=0.1)
            
            deltaA = torch.einsum("ji,is->jis", dt, A) # batch inner, inner state -> batch inner state
            dA = torch.matrix_exp(deltaA)
            #print(f"dA shape: {dA.shape}\n\n")

            deltaB = torch.einsum("ji,js->jis", dt, B) # batch inner, batch state -> batch inner state
            exp_minus_I = dA - I
            deltaA = deltaA + 1e-5 * torch.eye(deltaA.size(-1), device=deltaA.device)
            A_inv_term = torch.bmm(torch.linalg.pinv(deltaA), exp_minus_I)
            dB = torch.bmm(A_inv_term, deltaB)
            #print(f"dB shape: {dB.shape}\n\n")

            check_nan(deltaA, "deltaA", i)
            check_nan(dA, "dA", i)
            check_nan(A_inv_term, "A_inv_term", i)
            check_nan(dB, "dB", i)

            mask = (x[:, i].abs().sum(dim=-1) > 1e-5).float().unsqueeze(-1).unsqueeze(-1)
            hidden_state = hidden_state * dA + rearrange(x[:,i], "b d -> b d 1") * dB
            hidden_state = hidden_state * mask + hidden_state.detach() * (1 - mask)  # evita updates con padding
            hidden_state = torch.clamp(hidden_state, min=-1e6, max=1e6)

            #print(f"hidden_state shape: {hidden_state.shape}\n\n")
            
            #C = C.unsqueeze(0)
            y = torch.einsum("jis,js->ji", hidden_state, C)  + self.D * x[:,i]
            #print(f"y shape: {y.shape}\n\n")

            hidden_previos.append(hidden_state)
            out.append(y)
            
            #print("FIN ITER\n")
        #out = torch.stack(out)  # (seq, batch, d_inner)
        hidden_previos = torch.stack(hidden_previos)  # (seq, batch, d_inner, d_state)
        
        #out = out.permute(1, 0, 2)  # (batch, seq, d_inner)
        hidden_previos = hidden_previos.permute(1, 0, 2, 3)  # (batch, seq, d_inner, d_state)
        return y, hidden_previos

class Modelo_ind(nn.Module):
    def __init__(self, d_model, d_embedding = 32, d_hidden = 32, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        self.device = device
        self.embedding = nn.Embedding(d_model, d_embedding, padding_idx=0)
        self.linear1 = nn.Linear(d_embedding, d_hidden)
        self.ssm = SSM(d_inner= d_hidden, device = device)
        self.linear2 = nn.Linear(d_hidden, d_model + 1)
    
    def forward(self, x):
        
        x_ssm = self.embedding(x) 
        out, _ = self.ssm(x_ssm)
        salida = self.linear2(out)      

        return salida

import torch
import torch.nn as nn
import math
from einops import repeat, rearrange

class SSM(nn.Module):
    def __init__(
        self,
        d_inner,
        d_state=16,
        dt_rank="auto",
        device=None
    ):
        super().__init__()

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
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

        _, seq, _ = x.shape
        hidden_state = torch.zeros((self.d_inner, self.d_state), dtype=torch.float32, device = self.device)
        out = []
        hidden_previos = []

        for i in range(seq):
            
            #print(f"x: {x.shape}\n\n")
            x_proj = self.x_proj(x[:,i,:])
            B = x_proj[:,:self.d_state]
            dt = x_proj[:,self.d_state:self.d_state + self.dt_rank]
            C = x_proj[:,self.d_state + self.dt_rank:]
            A = -torch.exp(self.A_log.float())

            #print(f"dt shape: {dt.shape}\n\n")
            #print(f"A shape: {A.shape}\n\n")
            #print(f"B shape: {B.shape}\n\n")
            #print(f"C shape: {C.shape}\n\n")

            dt = self.dt_proj(dt)
            #print(f"dt shape: {dt.shape}\n\n")
            
            #dt = dt.unsqueeze(0)
            dA = torch.einsum("ji,is->jis", dt, A) # 1 inner, inner state -> 1 inner state
            #print(f"dA shape: {dA.shape}\n\n")

            #B = B.unsqueeze(0)
            dB = torch.einsum("ji,js->jis", dt, B) # 1 inner, 1 state -> 1 inner state
            #print(f"dB shape: {dB.shape}\n\n")
            
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

class Modelo(nn.Module):
    def __init__(self, d_model, d_embedding = 32, d_hidden = 32, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.d_hidden = d_hidden
        self.device = device
        self.embedding = nn.Embedding(d_model + 1, d_embedding, padding_idx=0)
        self.linear1 = nn.Linear(d_embedding, d_hidden)
        self.ssm = SSM(d_inner= d_hidden, device = device)
        self.linear2 = nn.Linear(d_hidden, d_model + 1)
    
    def forward(self, x):
        
        #print(f"input shape: {x.shape}\n\n")
        batch_size, seq_len = x.shape
        device = self.device
        EOS_TOKEN = self.d_model

        # Inicializa secuencia generada con la entrada original
        generated = x.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        outputs = []

        for _ in range(seq_len):
            x_ssm = self.embedding(generated) 
            #print(f"embed_x shape: {embed_x.shape}\n\n")
            #x_ssm = self.linear1(embed_x)
            #print(f"x_ssm shape: {x_ssm.shape}\n\n")
            out, _ = self.ssm(x_ssm)
            #print(f"out shape: {out.shape}\n\n")
            logits = self.linear2(out)  # solo la última salida

            # Para secuencias terminadas, forzar salida 0
            logits[finished] = -1e9
            logits[finished, 0] = 1e9  # Forzamos probabilidad máxima en el padding


            next_token = torch.argmax(logits, dim=-1)  # (batch,)
            #print(f"next_token shape: {next_token.shape}\n\n")

            outputs.append(logits) # REVISAR DIMENSIONES

            # Añadir nuevo token a la secuencia
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Verifica si se ha alcanzado el token de fin de secuencia (EOS)
            finished = finished | (next_token == EOS_TOKEN)

            # Si todos los casos han terminado, salir del bucle
            #if finished.all():
                #break            

        return torch.stack(outputs, dim=1)
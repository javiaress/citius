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
        dt_rank="auto"
    ):
        super().__init__()

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank

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

        _, seq, _ = x.shape
        hidden_state = torch.zeros((self.d_inner, self.d_state), dtype=torch.float32)
        out = []
        hidden_previos = []

        for i in range(seq):
            
            #print(f"x: {x.shape}\n\n")
            x_proj = self.x_proj(x[:,i,:])
            B = x_proj[:,:self.d_state]
            dt = x_proj[:,self.d_state:self.d_state + self.dt_rank]
            C = x_proj[:,self.d_state + self.dt_rank:]

            #print(f"dt shape: {dt.shape}\n\n")
            #print(f"A shape: {self.A.shape}\n\n")
            #print(f"B shape: {B.shape}\n\n")
            #print(f"C shape: {C.shape}\n\n")

            dt = self.dt_proj(dt)
            #print(f"dt shape: {dt.shape}\n\n")
            
            #dt = dt.unsqueeze(0)
            dA = torch.einsum("ji,is->jis", dt, self.A) # 1 inner, inner state -> 1 inner state
            #print(f"dA shape: {dA.shape}\n\n")

            #B = B.unsqueeze(0)
            dB = torch.einsum("ji,js->jis", dt, B) # 1 inner, 1 state -> 1 inner state
            #print(f"dB shape: {dB.shape}\n\n")
            
            #print(f"x[:,i]: {x[:,i].shape}\n\n")
            hidden_state = hidden_state * dA + rearrange(x[:,i], "b d -> b d 1") * dB
            #print(f"hidden_state shape: {hidden_state.shape}\n\n")
            
            #C = C.unsqueeze(0)
            y = torch.einsum("jis,js->ji", hidden_state, C)  + self.D * x[:,i]
            #print(f"y shape: {y.shape}\n\n")

            hidden_previos.append(hidden_state)
            out.append(y)
            
            #print("FIN ITER\n")
        return out, hidden_previos

class Modelo(nn.Module):
    def __init__(self, d_model, d_embedding = 32, d_hidden = 64):
        super().__init__()
        self.embedding = nn.Embedding(d_model, d_embedding, padding_idx=0)
        self.linear1 = nn.Linear(d_embedding, d_hidden)
        self.ssm = SSM(d_inner= d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
    
    def forward(self, x):
        
        print(f"input shape: {x.shape}\n\n")
        embed_x = self.embedding(x) 
        print(f"embed_x shape: {embed_x.shape}\n\n")
        x_ssm = self.linear1(embed_x)
        print(f"x_ssm shape: {x_ssm.shape}\n\n")
        out, _ = self.ssm(x_ssm) 
        print(f"out shape: {out[-1].shape}\n\n")
        salida = self.linear2(out[-1])
        print(f"salida shape: {salida.shape}\n\n")
        return salida

model = Modelo(
    d_model=8
)

entrada = torch.tensor([[0, 0, 4, 1],[0, 1, 2, 5]])

out = model(entrada)

#print(out)
print("\n\n")

"""
DATOS Y ENTRENAMIENTO
"""

data_folder = './data/'
filename = 'SEPSIS'
data = pd.read_csv(data_folder + filename + '.csv')
data

# ### Getting important columns

# In[2]:


CASE_COL = 'CaseID'
ACTIVITY_COL = 'Activity'
TIMESTAMP_COL = 'time:timestamp'

data = data[[CASE_COL, ACTIVITY_COL]]



# ### Convert categorical values to labels

# In[3]:


def category_to_label(attr: pd.Series) -> (pd.Series, dict, dict):
    uniq_attr = attr.unique()
    attr_dict = {idx + 1: value for idx, value in enumerate(uniq_attr)}
    reverse_dict = {value: key for key, value in attr_dict.items()}

    attr_cat = pd.Series(map(lambda x: reverse_dict[x], attr.values))

    return attr_cat, attr_dict, reverse_dict


data.loc[:, ACTIVITY_COL], _, _ = category_to_label(data[ACTIVITY_COL])



# ### Generate temporal features

# In[5]:


data_augment = pd.DataFrame()

cases = data.groupby(CASE_COL, sort=False)
for _, case in cases:
    case = case.reset_index(drop=True)
    

    data_augment = pd.concat([data_augment, case])
    
data = data_augment


# ### Add End-of-Case special event at the end of each trace

# In[6]:


NUM_ACTIVITIES = data[ACTIVITY_COL].nunique()

data_augment = pd.DataFrame()

cases = data.groupby(CASE_COL, sort=False)
for _, case in cases:
    case = case.reset_index(drop=True)
    
    eoc_row = pd.DataFrame({CASE_COL: [case[CASE_COL][0]],
                            ACTIVITY_COL: [NUM_ACTIVITIES + 1]})
    case = pd.concat([case, eoc_row])
    case = case.reset_index(drop=True)

    data_augment = pd.concat([data_augment, case])
    
data = data_augment



# ### Split in train-validation-test sets

# In[7]:


TRAIN_SIZE = 0.64
VAL_SIZE = 0.16

# Group events by case id (traces)
df_groupby = data_augment.groupby(CASE_COL, sort=False)
cases = [case for _, case in df_groupby]

# Get splitting points
first_cut = round(len(cases) * TRAIN_SIZE)
second_cut = round(len(cases) * (TRAIN_SIZE+VAL_SIZE))

# Split in train-validation-test
train_cases = cases[:first_cut]
val_cases = cases[first_cut:second_cut]
test_cases = cases[second_cut:]

train_data = pd.concat(train_cases)
val_data = pd.concat(val_cases)
test_data = pd.concat(test_cases)

'''
print("test_data: ")
print(test_data)
print("\n\n")
'''

# ### Construct the prefixes 

# In[8]:


# Maximum trace length
MAX_LEN = max(train_data.groupby(CASE_COL, sort=False)[ACTIVITY_COL].count().max(),
              val_data.groupby(CASE_COL, sort=False)[ACTIVITY_COL].count().max())


def get_prefixes(data):
    prefixes_acts = []
    next_acts = []
    
    # Group by case
    data_group = data.groupby(CASE_COL, sort=False)
    # Iterate over cases
    for name, gr in data_group:
        gr = gr.reset_index(drop=True)
        # Iterate over events in the case
        for i in range(len(gr)):
            # This would be an empty prefix, and it doesn't make much sense to predict based on nothing
            if i == 0:
                continue

            prefixes_acts.append(gr[ACTIVITY_COL][0:i].values)

            next_acts.append(gr[ACTIVITY_COL].values)
            
    # Matrix containing the training data
    X = np.zeros((len(prefixes_acts), MAX_LEN), dtype=np.float32)
    # Target event prediction data
    Y_a = np.zeros((len(prefixes_acts), MAX_LEN), dtype=np.float32)
    
    tam_suf = np.zeros(len(prefixes_acts), dtype=np.int32)

    for i, prefix_acts in enumerate(prefixes_acts):
        left_pad = MAX_LEN - len(prefix_acts)
        left_pad_trace = MAX_LEN - len(next_acts[i])
        next_act = next_acts[i]
        for j, act in enumerate(prefix_acts):
            X[i, j + left_pad] = act
        
        for k, act in enumerate(next_act):
            Y_a[i, k + left_pad_trace] = act
            
        tam_suf[i] = len(next_acts[i]) - len(prefixes_acts[i])
    
    return X, Y_a, tam_suf

x_train, y_train, tam_suf_train = get_prefixes(train_data)
x_val, y_val, tam_suf_val = get_prefixes(val_data) 
x_test, y_test, tam_suf_test = get_prefixes(test_data)

print(x_val[1].shape)
print(x_val[1])
print("\n\n")
print(y_val[1].shape)
print(y_val[1])
print("\n\n")
print(tam_suf_val[1])
print("\n\n")


x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_val = torch.tensor(x_val)
y_val = torch.tensor(y_val)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)
tam_suf_test = torch.tensor(tam_suf_test)

model = Modelo(
    d_model=NUM_ACTIVITIES+1
)

out = model(x_train)

print(out.shape)
print("\n\n")
print(out)
print("\n\n")
print("MODELO PASADO")
"""
from torch.utils.data import DataLoader, TensorDataset

dataset_train = TensorDataset(x_train, y_train)
loader_train = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True)

dataset_val = TensorDataset(x_val, y_val)
loader_val = DataLoader(dataset=dataset_val, batch_size=16, shuffle=True)

dataset_test = TensorDataset(x_test, y_test, tam_suf_test)
loader_test = DataLoader(dataset=dataset_test, batch_size=16, shuffle=True)

import os
import pathlib
import torch
import torch.nn as nn
import numpy as np
import wandb


def acc(y_pred, y_real):

    y_pred_softmax = torch.log_softmax(y_pred, dim=-1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=-1)

    _, y_real_tags = torch.max(y_real, dim=-1)

    '''
    print(y_pred_tags)
    print("\n\n")
    print(y_real_tags)
    '''

    correct_pred = (y_pred_tags == y_real_tags).float()
    acc = correct_pred.sum() / correct_pred.numel()

    return acc

def val_test(model, val_loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to("cuda")

    val_epoch_loss = []
    val_epoch_acc = []
    
    for mini_batch in iter(val_loader):
        prefix = mini_batch[0].to("cuda")       
        y_real = mini_batch[1]

        y_pred = model(prefix)
        
        val_loss = loss_fn(y_pred, y_real)
        val_acc = acc(y_pred, y_real)
        
        val_epoch_loss.append(val_loss.item())
        val_epoch_acc.append(val_acc.item())

    return val_epoch_loss, val_epoch_acc


def fit(model, train_loader, val_loader, filename, num_fold, model_name, use_wandb):

    opt = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08)
    loss_fn = nn.CrossEntropyLoss().to("cuda")

    val_mae_best = np.inf  # Starts the best MAE value as infinite
    epochs_no_improve = 0

    for e in range(100):
        train_epoch_loss = []
        train_epoch_acc = []
        model.train()
        sum_train_loss = 0
        for mini_batch in iter(train_loader):
            prefix = mini_batch[0].to("cuda")
            y_real = mini_batch[1]

            model.zero_grad()
            y_pred = model(prefix)
            

            train_loss = loss_fn(y_pred, y_real)
            
            '''
            print(f"Tipo de y_pred: {y_pred.shape}\n")
            print(f"Tipo de targets antes de convertir: {y_real.shape}\n\n\nreal:")
            print(y_real)
            print("\n\n pred:")
            
            print(y_pred)
            print("\n\n")
            '''
            #print(prefix) #echarle un ojo a lo q significa cada dimension para hacer la funcion de acc bn

            train_acc = acc(y_pred, y_real)

            train_loss.backward()
            opt.step()

            train_epoch_loss.append(train_loss.item())
            train_epoch_acc.append(train_acc.item())

        with torch.no_grad():
            val_epoch_loss, val_epoch_acc = val_test(model, val_loader)
            
            avg_train_loss = sum(train_epoch_loss) / len(train_epoch_loss)
            avg_val_loss = sum(val_epoch_loss) / len(val_epoch_loss)
            avg_train_acc = (sum(train_epoch_acc) / len(train_epoch_acc)) * 100
            avg_val_acc = (sum(val_epoch_acc) / len(val_epoch_acc)) * 100

            print(f'Epoch {e}: | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | '
                  f'Train Acc: {avg_train_acc:.3f} | Val Acc: {avg_val_acc:.3f}')

            if use_wandb:
                wandb.log({
                    num_fold + '_train_loss': sum(train_epoch_loss) / len(train_epoch_loss),
                    num_fold + '_val_loss': sum(val_epoch_loss) / len(val_epoch_loss),
                    num_fold + '_train_acc': (sum(train_epoch_acc) / len(train_epoch_acc)) * 100,
                    num_fold + '_val_acc': (sum(val_epoch_acc) / len(val_epoch_acc)) * 100
                    })

            if sum(val_epoch_loss) / len(val_epoch_loss) < val_mae_best:
                val_mae_best = sum(val_epoch_loss) / len(val_epoch_loss)

                epochs_no_improve = 0

                if os.path.isdir('./models/' + filename + '/' + num_fold + '/'):
                    torch.save(model, "./models/" + filename + '/' + num_fold + "/" + model_name)
                else:
                    pathlib.Path('./models/' + filename + '/' + num_fold).mkdir(parents=True, exist_ok=True)
                    torch.save(model, "./models/" + filename + '/' + num_fold + "/" + model_name)
            
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 30:
                print("Early stopping")
                break

fit(model, loader_train, loader_val, "mamba", "1", "modelomamba", False)


from jellyfish._jellyfish import damerau_levenshtein_distance

def levenshtein_acc(y_pred, y_real, tam_suf):
    y_pred_softmax = torch.log_softmax(y_pred, dim=-1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=-1)

    _, y_real_tags = torch.max(y_real, dim=-1)

    y_pred_tags = y_pred_tags.cpu().numpy()
    y_real_tags = y_real_tags.cpu().numpy()
    
    acc = 0
    for i in range(len(y_pred_tags)):
        pred_seq = ''.join(map(chr, y_pred_tags[i][-tam_suf[i]:] + 161))
        real_seq = ''.join(map(chr, y_real_tags[i][-tam_suf[i]:] + 161))

        acc += 1 - damerau_levenshtein_distance(pred_seq, real_seq) / max(len(pred_seq), len(real_seq))

    return acc / len(y_pred_tags)

def test(model, val_loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to("cuda")

    val_epoch_loss = []
    val_epoch_acc = []

    for mini_batch in iter(val_loader):
        prefix = mini_batch[0].to("cuda")
        y_real = mini_batch[1]
        tam_suf = mini_batch[2]

        y_pred = model(prefix)

        val_loss = loss_fn(y_pred, y_real)
        val_acc = levenshtein_acc(y_pred, y_real, tam_suf)
        val_epoch_loss.append(val_loss.item())
        val_epoch_acc.append(val_acc)

    return val_epoch_loss, val_epoch_acc

val_epoch_loss, val_epoch_acc = test(model, loader_test)

print(f'Levenshtein Acc: {sum(val_epoch_acc) / len(val_epoch_acc)}')


print("\n\n sa cabau")
"""

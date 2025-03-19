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
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True) # MIRAR BIAS
        
        '''
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
        '''

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

        batch, _ = x.shape
        hidden_state = torch.zeros((self.d_inner, self.d_state), dtype=torch.float32)
        out = []
        hidden_previos = []

        for i in range(batch):
            x_proj = self.x_proj(x[i])
            B = x_proj[:, :self.d_state]
            dt = x_proj[:, self.d_state:self.d_state + self.dt_rank]
            C = x_proj[:, self.d_state + self.dt_rank:]

            dt = self.dt_proj(dt)
            dA = torch.einsum("ji,is->jis", dt, self.A) # 1 inner, inner state -> 1 inner state
            dB = torch.matmul(dt.T, B) # state 1, 1 inner -> state inner

            hidden_state = hidden_state * dA + x * dB

            y = torch.einsum("is,js->ji", hidden_state, C)  + self.D * x

            hidden_previos.append(hidden_state)
            out.append(y)

        return out, hidden_previos


'''
DATOS Y ENTRENAMIENTO
'''

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
    attr_dict = {idx: value for idx, value in enumerate(uniq_attr)}
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
                            ACTIVITY_COL: [NUM_ACTIVITIES]})
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
    X = np.zeros((len(prefixes_acts), MAX_LEN, NUM_ACTIVITIES+1), dtype=np.float32)
    # Target event prediction data
    Y_a = np.zeros((len(prefixes_acts), MAX_LEN, NUM_ACTIVITIES+1), dtype=np.float32)
    
    tam_suf = np.zeros(len(prefixes_acts), dtype=np.int32)

    for i, prefix_acts in enumerate(prefixes_acts):
        left_pad = MAX_LEN - len(prefix_acts)
        left_pad_trace = MAX_LEN - len(next_acts[i])
        next_act = next_acts[i]
        for j, act in enumerate(prefix_acts):
            X[i, j + left_pad, act] = 1
        
        for k, act in enumerate(next_act):
            Y_a[i, k + left_pad_trace, act] = 1
            
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

model = SSM(
    d_inner=17,
    d_state=16
)
#Haz que los prints tambien tengan texto por favor usa f strings

print(x_train.shape)
print("\n\n")

out, hidden = model(x_train)

print(out.shape)
print("\n\n")
print(out)


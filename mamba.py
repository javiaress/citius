#!/usr/bin/env python
# coding: utf-8

# # N. Tax LSTM Prediction Model

# ## Preprocessing data

# ### Reading the event log

# In[1]:


import pandas as pd
import numpy as np

data_folder = './data/'
filename = 'helpdesk'

data = pd.read_csv(data_folder + filename + '.csv')
data


# ### Getting important columns

# In[2]:


CASE_COL = 'case:concept:name'
ACTIVITY_COL = 'concept:name'
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

# como imprimo un texto y después la variable test_data
print("test_data: ")
print(test_data)
print("\n\n")
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

            next_acts.append(gr[ACTIVITY_COL].values) # LA SIGUIENTE ACTIVIDAD ES TODA LA TRAZA, PARA PONER TODAS LAS ACTIVIDADES RESTANTES SERÍA gr[ACTIVITY_COL][i:].values
            
    # Matrix containing the training data
    X = np.zeros((len(prefixes_acts), MAX_LEN, NUM_ACTIVITIES+1), dtype=np.float32)
    # Target event prediction data
    Y_a = np.zeros((len(prefixes_acts), NUM_ACTIVITIES+1), dtype=np.float32)
    
    for i, prefix_acts in enumerate(prefixes_acts):
        left_pad = MAX_LEN - len(prefix_acts)
        next_act = next_acts[i]
        for j, act in enumerate(prefix_acts):
            X[i, j + left_pad, act] = 1
        
        for k in next_act:
                Y_a[i, k] = 1
    
    return X, Y_a

x_train, y_train = get_prefixes(train_data)
x_val, y_val = get_prefixes(val_data)
x_test, y_test = get_prefixes(test_data)

print(x_train.shape)
print("\n\n")
print(y_train.shape)
print("\n\n")
print(x_val[1])
print("\n\n")
print(y_val[1])
print("\n\n")

import torch
from mamba_ssm import Mamba

x_train = torch.tensor(x_train).to("cuda")
y_train = torch.tensor(y_train).to("cuda")
x_val = torch.tensor(x_val).to("cuda")
y_val = torch.tensor(y_val).to("cuda")
x_test = torch.tensor(x_test).to("cuda")
y_test = torch.tensor(y_test).to("cuda")

model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=15, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")

y = model(x_train)

print(y.shape)
print("aplicado mamba\n\n")

from torch.utils.data import DataLoader, TensorDataset

dataset_train = TensorDataset(x_train, y_train)
loader_train = DataLoader(dataset=dataset_train, batch_size=14, shuffle=True)

dataset_val = TensorDataset(x_val, y_val)
loader_val = DataLoader(dataset=dataset_val, batch_size=14, shuffle=True)

dataset_test = TensorDataset(x_test, y_test)
loader_test = DataLoader(dataset=dataset_test, batch_size=14, shuffle=True)

import os
import pathlib
import torch
import torch.nn as nn
import numpy as np
import wandb


def acc(y_pred, y_real):

    # Obtener las predicciones para el último timestep
    y_pred_last = y_pred[:, -1, :]  # [batch_size, num_classes]

    # Aplicar log_softmax a lo largo de la dimensión de las clases (dim=1)
    y_pred_softmax = torch.log_softmax(y_pred_last, dim=1)
    
    # Obtener las etiquetas predichas (clases con la mayor probabilidad)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    
    # Obtener las etiquetas reales (clases con la mayor probabilidad)
    _, y_real_tags = torch.max(y_real, dim=1)
    
    # Comparar las etiquetas predichas con las etiquetas reales
    correct_pred = (y_pred_tags == y_real_tags).float()
    
    # Calcular la precisión: número de predicciones correctas dividido por el total de predicciones
    acc = correct_pred.sum() / correct_pred.numel()


'''
    y_pred_softmax = torch.log_softmax(y_pred, dim=2)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=2)

    correct_pred = (y_pred_tags == y_real).float()
    acc = correct_pred.sum() / correct_pred.numel()
'''
    return acc

def fit(model, train_loader, val_loader, filename, num_fold, model_name, use_wandb):

    opt = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08)
    loss_fn = nn.CrossEntropyLoss().to("cuda")

    val_mae_best = np.inf  # Starts the best MAE value as infinite

    for e in range(5):
        train_epoch_loss = []
        train_epoch_acc = []
        model.train()
        sum_train_loss = 0
        for mini_batch in iter(train_loader):
            prefix = mini_batch[0].to("cuda")
            y_real = mini_batch[1]

            model.zero_grad()
            y_pred = model(prefix)
            
            y_real = y_real.long()

            train_loss = loss_fn(y_pred, y_real)
            
            print(f"Tipo de y_pred: {y_pred.shape}\n")
            print(f"Tipo de targets antes de convertir: {y_real.shape}\n")
            print(y_real)
            print("\n\n")
            
            print(y_pred)
            
            print("\n\n")

            print(prefix) #echarle un ojo a lo q significa cada dimension para hacer la funcion de acc bn


            train_acc = acc(y_pred, y_real)
            return  #LLEGAMOS HASTA AQUI
            train_loss.backward()
            opt.step()

            train_epoch_loss.append(train_loss.item())
            train_epoch_acc.append(train_acc.item())
        
        with torch.no_grad():
            val_epoch_loss, val_epoch_acc = model.val_test(val_loader)
            
            print(f'Epoch {e}: | Train Loss: {sum(train_epoch_loss) / len(train_epoch_loss):.6f} | '
                    f'Val Loss: {sum(val_epoch_loss) / len(val_epoch_loss):.6f} | '
                    f'Train Acc: {(sum(train_epoch_acc) / len(train_epoch_acc)) * 100:.3f} | '
                    f'Val Acc: {(sum(val_epoch_acc) / len(val_epoch_acc)) * 100:.3f}')

            if use_wandb:
                wandb.log({
                    num_fold + '_train_loss': sum(train_epoch_loss) / len(train_epoch_loss),
                    num_fold + '_val_loss': sum(val_epoch_loss) / len(val_epoch_loss),
                    num_fold + '_train_acc': (sum(train_epoch_acc) / len(train_epoch_acc)) * 100,
                    num_fold + '_val_acc': (sum(val_epoch_acc) / len(val_epoch_acc)) * 100
                    })

            if sum(val_epoch_loss) / len(val_epoch_loss) < val_mae_best:
                val_mae_best = sum(val_epoch_loss) / len(val_epoch_loss)

                if os.path.isdir('../models/' + filename + '/' + num_fold + '/'):
                    torch.save(self, "../models/" + filename + '/' + num_fold + "/" + model_name)
                else:
                    pathlib.Path('../models/' + filename + '/' + num_fold).mkdir(parents=True, exist_ok=True)
                    torch.save(self, "../models/" + filename + '/' + num_fold + "/" + model_name)

fit(model, loader_train, loader_val, "mamba", "1", True, "mamba")


print("\n\n sa cabau")


'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Dense, Embedding, Dropout, \
    LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import distutils.dir_util


# ### Define the model

# In[10]:


main_input = Input(shape=(MAX_LEN, NUM_ACTIVITIES + 4), name='main_input')
l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input)  # the shared layer
b1 = BatchNormalization()(l1)
l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)  # the layer specialized in activity prediction
b2_1 = BatchNormalization()(l2_1)
l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1)  # the layer specialized in time prediction
b2_2 = BatchNormalization()(l2_2)
act_output = Dense(NUM_ACTIVITIES + 1, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)

model = Model(inputs=[main_input], outputs=[act_output, time_output])

# ### Compile the model

# In[11]:


opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt,
              metrics={"act_output": "acc", "time_output": "mae"})


# ### Train the model

# In[12]:


# Configure savings of best model
distutils.dir_util.mkpath("models/" + filename)
best_model_path = "models/" + filename + "_tax.h5"
model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=False, mode='auto')

# In[12]:


# Configure savings of best model
distutils.dir_util.mkpath("models/" + filename)
best_model_path = "models/" + filename + "_tax.h5"
model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=False, mode='auto')

# In[12]:


# Configure savings of best model
distutils.dir_util.mkpath("models/" + filename)
best_model_path = "models/" + filename + "_tax.h5"
model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=False, mode='auto')

# In[12]:


# Configure savings of best model
distutils.dir_util.mkpath("models/" + filename)
best_model_path = "models/" + filename + "_tax.h5"
model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=False, mode='auto')
# Configure early stopping when validation loss is not reducing
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
# Configure learning rate reducer
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                               min_delta=0.0001, cooldown=0, min_lr=0)

# Training
model.fit(X_train, {'act_output': Y_a_train, 'time_output': Y_t_train},
          validation_data=(X_val, {"act_output": Y_a_val, "time_output": Y_t_val}), verbose=2,
          callbacks=[early_stopping, model_checkpoint, lr_reducer],
          batch_size=MAX_LEN, epochs=200)


# ## Testing the model

# In[12]:


from jellyfish._jellyfish import damerau_levenshtein_distance
from datetime import timedelta


# ### Load the best model

# In[14]:


model.load_weights(best_model_path)
model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt,
              metrics={"act_output": "acc"})


# ### Validating on suffix prediction 

# In[15]:


def encode_tax(prefix, times, times3, divisor, divisor2, maxlen):
    X = np.zeros((1, maxlen, NUM_ACTIVITIES+4), dtype=np.float32)
    leftpad = maxlen - len(prefix)
    times2 = np.cumsum(times)
    for t, act in enumerate(prefix):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t] - midnight
        
        X[0, t + leftpad, act] = 1
        X[0, t + leftpad, NUM_ACTIVITIES] = times[t] / divisor
        X[0, t + leftpad, NUM_ACTIVITIES + 1] = times2[t] / divisor2
        X[0, t + leftpad, NUM_ACTIVITIES + 2] = timesincemidnight.seconds / 86400
        X[0, t + leftpad, NUM_ACTIVITIES + 3] = times3[t].weekday() / 7

    return X


predict_size = MAX_LEN
dl_score = []
for prefix_size in range(1, MAX_LEN):
    cases = test_data.groupby(CASE_COL)
    for _, case in cases:
        prefix_acts = case[ACTIVITY_COL][:prefix_size].values.tolist()
        prefix_t = case['times1'][:prefix_size].values.tolist()
        prefix_t3 = case[TIMESTAMP_COL][:prefix_size].tolist()
        prefix_t3 = list(map(lambda x: x.to_pydatetime(), prefix_t3))
        
        if prefix_size >= len(case):
            continue  # make no prediction for this case, since this case has ended already
            dls = 1 - (damerau_levenshtein_distance(''.join(predicted), ''.join(ground_truth)) / max(len(predicted), len(ground_truth)))
            if dls < 0:
                dls = 0
            dl_score.append(dls)

dl_score = np.mean(np.array(dl_score))

dl_score'''

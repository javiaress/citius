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

data = data[[CASE_COL, ACTIVITY_COL, TIMESTAMP_COL]]



# ### Convert categorical values to labels

# In[3]:


def category_to_label(attr: pd.Series) -> (pd.Series, dict, dict):
    uniq_attr = attr.unique()
    attr_dict = {idx: value for idx, value in enumerate(uniq_attr)}
    reverse_dict = {value: key for key, value in attr_dict.items()}

    attr_cat = pd.Series(map(lambda x: reverse_dict[x], attr.values))

    return attr_cat, attr_dict, reverse_dict


data.loc[:, ACTIVITY_COL], _, _ = category_to_label(data[ACTIVITY_COL])


# ### Convert timestamp column to correct date type

# In[4]:


data[TIMESTAMP_COL] = pd.to_datetime(data[TIMESTAMP_COL])


# ### Generate temporal features

# In[5]:


data_augment = pd.DataFrame()

cases = data.groupby(CASE_COL, sort=False)
for _, case in cases:
    case = case.reset_index(drop=True)
    
    # First temporal feature: Time since previous event
    timesincelastevent = case.loc[:, TIMESTAMP_COL].diff() / np.timedelta64(1, 's')
    timesincelastevent.iloc[0] = 0.0
    
    # Second temporal feature: Time since case start
    casestart = case.loc[0, TIMESTAMP_COL]
    timesincecasestart = (case.loc[:, TIMESTAMP_COL] - casestart) / np.timedelta64(1, 's')
    
    # Third temporal feature: Time since last midnight
    midnight = case[TIMESTAMP_COL].apply(lambda x: x.replace(hour=00, minute=00, second=00))
    timesincemidnight = (case.loc[:, TIMESTAMP_COL] - midnight) / np.timedelta64(1, 's')
    
    # Fourth temporal feature: Day of the week
    weekday = case.loc[:, TIMESTAMP_COL].dt.dayofweek
    
    case['times1'] = timesincelastevent
    case['times2'] = timesincecasestart
    case['times3'] = timesincemidnight
    case['times4'] = weekday

    # case = case.drop(columns=[TIMESTAMP_COL])

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


# ### Construct the prefixes 

# In[8]:


# Maximum trace length
MAX_LEN = max(train_data.groupby(CASE_COL, sort=False)[ACTIVITY_COL].count().max(),
              val_data.groupby(CASE_COL, sort=False)[ACTIVITY_COL].count().max())


def get_divisor(data_train, data_val):
    data_cat = pd.concat([data_train, data_val])
    divisor = np.mean(data_cat)
    
    return divisor

def get_divisor3(data_train, data_val):
    data_cat = pd.concat([data_train, data_val])
    
    list_times = []

    # Group by case
    data_group = data_cat.groupby(CASE_COL)
    # Iterate over case
    for _, gr in data_group:
        gr = gr.reset_index(drop=True)
        caseend = gr.loc[len(gr)-2, 'times2']
        timeuntilend = caseend - gr['times2'][:-1]

        list_times.append(timeuntilend.mean())

    divisor3 = np.mean(np.array(list_times))
    return divisor3

# Divisors to normalize temporal features
divisor = get_divisor(train_data['times1'], val_data['times1'])
divisor2 = get_divisor(train_data['times2'], val_data['times2']) 
divisor3 = get_divisor3(train_data, val_data)

def get_prefixes(data, divisor, divisor2):
    prefixes_acts = []
    prefixes_t = []
    prefixes_t2 = []
    prefixes_t3 = []
    prefixes_t4 = []
    next_acts = []
    next_times = []
    
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
            prefixes_t.append(gr['times1'][0:i].values)
            prefixes_t2.append(gr['times2'][0:i].values)
            prefixes_t3.append(gr['times3'][0:i].values)
            prefixes_t4.append(gr['times4'][0:i].values)

            next_acts.append(gr[ACTIVITY_COL][i])
            if i == len(gr) - 1:
                next_times.append(0)
            else:
                next_times.append(gr['times1'][i])
            
    # Matrix containing the training data
    X = np.zeros((len(prefixes_acts), MAX_LEN, NUM_ACTIVITIES+4), dtype=np.float32)
    # Target event prediction data
    Y_a = np.zeros((len(prefixes_acts), NUM_ACTIVITIES+1), dtype=np.float32)
    # Target time prediction data
    Y_t = np.zeros((len(prefixes_acts)), dtype=np.float32)
    
    for i, prefix_acts in enumerate(prefixes_acts):
        left_pad = MAX_LEN - len(prefix_acts)
        prefix_t = prefixes_t[i]
        prefix_t2 = prefixes_t2[i]
        prefix_t3 = prefixes_t3[i]
        prefix_t4 = prefixes_t4[i]
        next_act = next_acts[i]
        next_t = next_times[1]
        for j, act in enumerate(prefix_acts):
            X[i, j + left_pad, act] = 1
            X[i, j + left_pad, NUM_ACTIVITIES] = prefix_t[j] / divisor
            X[i, j + left_pad, NUM_ACTIVITIES + 1] = prefix_t2[j] / divisor2
            X[i, j + left_pad, NUM_ACTIVITIES + 2] = prefix_t3[j] / 86400
            X[i, j + left_pad, NUM_ACTIVITIES + 3] = prefix_t4[j] / 7
        
        Y_a[i, next_act] = 1
        Y_t[i] = next_t / divisor
    
    return X, Y_a, Y_t

X_train, Y_a_train, Y_t_train = get_prefixes(train_data, divisor, divisor2)
X_val, Y_a_val, Y_t_val = get_prefixes(val_data, divisor, divisor2)
# X_test, Y_a_test, Y_t_test = get_prefixes(test_data, divisor, divisor2)

print(X_train.shape)
print(type(X_train))

import torch
from mamba_ssm import Mamba

x = torch.tensor(X_train).to("cuda")

model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=18, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")

y = model(x)

print(y.shape)
print(y)
print(model)

# ## Building and training the model

# In[9]:

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
            
        ground_truth = case[ACTIVITY_COL][prefix_size:prefix_size+predict_size].values
        ground_truth_t = case['times2'][prefix_size - 1]
        case_end_time = case['times2'][len(case) - 1]
        ground_truth_t = case_end_time - ground_truth_t
        predicted = []
        total_predicted_time = 0
        for i in range(predict_size):
            enc = encode_tax(prefix_acts, prefix_t, prefix_t3, divisor, divisor2, MAX_LEN)
            y = model.predict(enc, verbose=0)  # make predictions
            # split prediction into separate activity and time predictions
            y_act = y[0][0]
            y_t = y[1][0][0]
            prediction = np.argmax(y_act)
            prefix_acts.append(prediction)
            if y_t < 0:
                y_t = 0.0
            prefix_t.append(y_t)
            if prediction == NUM_ACTIVITIES:
                break  # end of case was just predicted, therefore, stop prediction further into the future
            y_t = y_t * divisor3
            prefix_t3.append(prefix_t3[-1] + timedelta(seconds=y_t))
            total_predicted_time = total_predicted_time + y_t
            predicted.append(prediction)
        if len(ground_truth) > 0:
            predicted = list(map(lambda x: chr(x+161), predicted))
            ground_truth = list(map(lambda x: chr(x+161), ground_truth))
            dls = 1 - (damerau_levenshtein_distance(''.join(predicted), ''.join(ground_truth)) / max(len(predicted), len(ground_truth)))
            if dls < 0:
                dls = 0
            dl_score.append(dls)

dl_score = np.mean(np.array(dl_score))

dl_score'''

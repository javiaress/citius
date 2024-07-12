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
print(data)


# ### Getting important columns

# In[2]:


CASE_COL = 'case:concept:name'
ACTIVITY_COL = 'concept:name'
TIMESTAMP_COL = 'time:timestamp'

data = data[[CASE_COL, ACTIVITY_COL, TIMESTAMP_COL]]
print(data)


# ### Convert categorical values to labels

# In[3]:


def category_to_label(attr: pd.Series) -> (pd.Series, dict, dict):
    uniq_attr = attr.unique()
    attr_dict = {idx: value for idx, value in enumerate(uniq_attr)}
    reverse_dict = {value: key for key, value in attr_dict.items()}

    attr_cat = pd.Series(map(lambda x: reverse_dict[x], attr.values))

    return attr_cat, attr_dict, reverse_dict

import torch
from mamba_ssm import Mamba

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Dense, Embedding, Dropout, \
    LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import distutils.dir_util


data.loc[:, ACTIVITY_COL], _, _ = category_to_label(data[ACTIVITY_COL])
print(data)

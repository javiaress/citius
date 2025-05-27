import pandas as pd
import numpy as np
import os


def build_global_dict(*series):
    # Unir todos los valores únicos de todas las series
    all_values = pd.concat(series).unique()
    all_values_sorted = sorted(all_values)
    label_dict = {val: idx + 1 for idx, val in enumerate(all_values_sorted)}  # empezar en 1
    return label_dict

def apply_label_mapping(attr: pd.Series, label_dict: dict):
    return attr.map(label_dict).astype(int)

def group_by_case(data, case_col):
    data_augment = pd.DataFrame()
    cases = data.groupby(case_col, sort=False)
    for _, case in cases:
        case = case.reset_index(drop=True)
        data_augment = pd.concat([data_augment, case])

    return data_augment


def add_end_of_case(data, case_col, activity_col, num_activities):
    augmented = pd.DataFrame()
    cases = data.groupby(case_col, sort=False)
    for _, case in cases:
        case = case.reset_index(drop=True)
        eoc_row = pd.DataFrame({case_col: [case[case_col][0]],
                                activity_col: [num_activities + 1]})
        case = pd.concat([case, eoc_row])
        case = case.reset_index(drop=True)
        augmented = pd.concat([augmented, case])
    return augmented


def get_prefixes(data, case_col, activity_col, max_len):
    prefixes_acts, next_acts = [], []
    for _, case in data.groupby(case_col, sort=False):
        case = case.reset_index(drop=True)
        for i in range(1, len(case)):
            prefixes_acts.append(case[activity_col][0:i].values)
            next_acts.append(case[activity_col][i:].values)

    X = np.zeros((len(prefixes_acts), max_len), dtype=np.float32)
    Y = np.zeros((len(prefixes_acts), max_len), dtype=np.int32)
    lengths = np.zeros(len(prefixes_acts), dtype=np.int32)

    for i, prefix in enumerate(prefixes_acts):
        left_pad = max_len - len(prefix)
        next_act = next_acts[i]
        for j, act in enumerate(prefix):
            X[i, j + left_pad] = act
        
        for k, act in enumerate(next_act):
            Y[i, k] = act
            
        lengths[i] = len(next_acts[i])

    return X, Y, lengths

def get_prefixes_ind(data, case_col, activity_col, resource_col, max_len):
    prefixes_acts, next_acts = [], []
    prefixes_rsrc = []
    for _, case in data.groupby(case_col, sort=False):
        case = case.reset_index(drop=True)
        for i in range(1, len(case)):
            prefixes_acts.append(case[activity_col][:i].values)
            prefixes_rsrc.append(case[resource_col][:i].values)
            next_acts.append(case[activity_col][i])

    X = np.zeros((len(prefixes_acts), max_len, 2), dtype=np.float32)
    Y = np.zeros((len(prefixes_acts)), dtype=np.int32)

    for i, (a_seq, r_seq) in enumerate(zip(prefixes_acts, prefixes_rsrc)):
        left_pad = max_len - len(a_seq)
        next_act = next_acts[i]
        X[i, left_pad:, 0] = a_seq
        X[i, left_pad:, 1] = r_seq

        Y[i] = next_act

    return X, Y, 1

def load_and_preprocess_data(base_folder, case_col, activity_col, resource_col, time_col, dataset_name):
    folds_data = []

    for fold in range(5):
        train_file = f"{base_folder}/train_fold{fold}_variation0_{dataset_name}.csv"
        val_file = f"{base_folder}/val_fold{fold}_variation0_{dataset_name}.csv"
        test_file = f"{base_folder}/test_fold{fold}_variation0_{dataset_name}.csv"

        train = pd.read_csv(train_file)
        val = pd.read_csv(val_file)
        test = pd.read_csv(test_file)

        # Codificar actividades y recursos con mapeo global
        activity_dict = build_global_dict(train[activity_col], val[activity_col], test[activity_col])
        resource_dict = build_global_dict(train[resource_col], val[resource_col], test[resource_col])

        train[activity_col] = apply_label_mapping(train[activity_col], activity_dict)
        val[activity_col] = apply_label_mapping(val[activity_col], activity_dict)
        test[activity_col] = apply_label_mapping(test[activity_col], activity_dict)

        train[resource_col] = apply_label_mapping(train[resource_col], resource_dict)
        val[resource_col] = apply_label_mapping(val[resource_col], resource_dict)
        test[resource_col] = apply_label_mapping(test[resource_col], resource_dict)

        # Codificar casos
        train = group_by_case(train, case_col)
        val = group_by_case(val, case_col)
        test = group_by_case(test, case_col)

        num_activities = len(activity_dict)
        num_resources = len(resource_dict)

        # Agregar evento fin de caso
        train = add_end_of_case(train, case_col, activity_col, num_activities)
        val = add_end_of_case(val, case_col, activity_col, num_activities)
        test = add_end_of_case(test, case_col, activity_col, num_activities)

        # Longitud máxima de traza
        max_len = max(train.groupby(case_col)[activity_col].count().max(),
                     val.groupby(case_col)[activity_col].count().max(), 
                     test.groupby(case_col)[activity_col].count().max(),)

        # Prefijos
        x_train, y_train, _ = get_prefixes_ind(train, case_col, activity_col, resource_col, max_len)
        x_val, y_val, _ = get_prefixes_ind(val, case_col, activity_col, resource_col, max_len)
        x_test, y_test, tam_suf_test = get_prefixes_ind(test, case_col, activity_col, resource_col, max_len)

        folds_data.append({
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test,
            'tam_suf_test': tam_suf_test,
            'num_activities': num_activities,
            'num_resources': num_resources,
            'max_len': max_len
        })

    return folds_data

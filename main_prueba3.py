from models.ssm_ind_prueba3 import Modelo_ind as Modelo
from data.preprocess3 import load_and_preprocess_data
from training.train import fit
from training.evaluate import test
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import time

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


seed = 42

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

  torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "BPI_Challenge_2013_closed_problems" 

    model_name = "ssm_prueba3"
    
    folds_data = load_and_preprocess_data("./data", case_col="caseid", activity_col="task", resource_col="user", time_col = "end_timestamp", dataset_name= dataset_name)

    accs = []
    train_times = []
    test_times = []

    for fold_idx, fold in enumerate(folds_data):
        print(f"\n=== Fold {fold_idx} ===")
        x_train = torch.tensor(fold['x_train'], dtype=torch.float32).to(device)
        y_train = torch.tensor(fold['y_train'], dtype=torch.long).to(device)
        x_val = torch.tensor(fold['x_val'], dtype=torch.float32).to(device)
        y_val = torch.tensor(fold['y_val'], dtype=torch.long).to(device)
        x_test = torch.tensor(fold['x_test'], dtype=torch.float32).to(device)
        y_test = torch.tensor(fold['y_test'], dtype=torch.long).to(device)
        tam_suf = torch.tensor(fold['tam_suf_test'], dtype=torch.long).to(device)
        
        print(x_val[10].shape)
        print(x_val[10])
        print("\n\n")
        print(y_val[10].shape)
        print(y_val[10])
        print("\n\n")

        model = Modelo(d_acts=fold['num_activities']+1, d_rsrc= fold['num_resources']+1, device=device).to(device) #+1 para el padding

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=16, shuffle=True)

        start_time = time.time()
        fit(model, train_loader, val_loader, dataset_name, str(fold_idx), model_name, use_wandb=False)
        elapsed_time = time.time() - start_time
        print(f"Training time for fold {fold_idx}: {format_time(elapsed_time)}")
        train_times.append(elapsed_time)

        model.load_state_dict(torch.load(f"./models/{dataset_name}/{fold_idx}/{model_name}"))
        model.to(device)

        start_time = time.time()
        _, acc = test(model, test_loader, fold['num_activities'])
        elapsed_time = time.time() - start_time
        print(f"Testing time for fold {fold_idx}: {format_time(elapsed_time)}")
        test_times.append(elapsed_time)
        avg_acc = sum(acc) / len(acc)
        accs.append(avg_acc)
        print(f"Fold {fold_idx} Levenshtein Accuracy: {avg_acc:.4f}")
    
    print(f"Average Levenshtein Accuracy: {sum(accs) / len(accs):.4f}")
    print(f"Total Training Time: {format_time(sum(train_times))}")
    print(f"Total Testing Time: {format_time(sum(test_times))}")

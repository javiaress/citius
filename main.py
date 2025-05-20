from models.ssm_ind import Modelo_ind as Modelo
from data.preprocess import load_and_preprocess_data
from training.train import fit
from training.evaluate import test
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np


seed = 42

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

  torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "Helpdesk" 
    
    folds_data = load_and_preprocess_data("./data", case_col="caseid", activity_col="task", dataset_name= dataset_name)

    accs = []

    for fold_idx, fold in enumerate(folds_data):
        print(f"\n=== Fold {fold_idx} ===")
        x_train = torch.tensor(fold['x_train'], dtype=torch.long).to(device)
        y_train = torch.tensor(fold['y_train'], dtype=torch.long).to(device)
        x_val = torch.tensor(fold['x_val'], dtype=torch.long).to(device)
        y_val = torch.tensor(fold['y_val'], dtype=torch.long).to(device)
        x_test = torch.tensor(fold['x_test'], dtype=torch.long).to(device)
        y_test = torch.tensor(fold['y_test'], dtype=torch.long).to(device)
        tam_suf = torch.tensor(fold['tam_suf_test'], dtype=torch.long).to(device)
        
        print(x_val[1].shape)
        print(x_val[1])
        print("\n\n")
        print(y_val[1].shape)
        print(y_val[1])
        print("\n\n")

        model = Modelo(d_model=fold['num_activities']+1, device=device).to(device)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=16, shuffle=True)

        fit(model, train_loader, val_loader, dataset_name, str(fold_idx), "modelossm", use_wandb=False)

        model.load_state_dict(torch.load(f"./models/{dataset_name}/{fold_idx}/modelossm"))
        model.to(device)

        _, acc = test(model, test_loader, fold['num_activities'])
        avg_acc = sum(acc) / len(acc)
        accs.append(avg_acc)
        print(f"Fold {fold_idx} Levenshtein Accuracy: {avg_acc:.4f}")
    
    print(f"Average Levenshtein Accuracy: {sum(accs) / len(accs):.4f}")

import torch
import torch.nn as nn
import numpy as np
import os
import pathlib

def acc(y_pred, y_real):
    y_pred_softmax = torch.log_softmax(y_pred, dim=-1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=-1)
    correct_pred = (y_pred_tags == y_real).float()
    return correct_pred.sum() / correct_pred.numel()

def val_test(model, val_loader, device, num_activities):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0).to(device)

    val_epoch_loss = []
    val_epoch_acc = []

    for mini_batch in val_loader:
        prefix = mini_batch[0].to(device)
        y_real = mini_batch[1]

        y_pred = model(prefix)
        y_pred_loss = y_pred.view(-1, num_activities + 2)
        y_real_loss = y_real.view(-1)

        val_loss = loss_fn(y_pred_loss, y_real_loss)
        val_acc = acc(y_pred, y_real)

        val_epoch_loss.append(val_loss.item())
        val_epoch_acc.append(val_acc.item())

    return val_epoch_loss, val_epoch_acc

def fit(model, train_loader, val_loader, filename, num_fold, model_name, use_wandb=False):
    device = next(model.parameters()).device
    num_activities = model.d_model

    opt = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0).to(device)

    val_mae_best = np.inf
    epochs_no_improve = 0

    for e in range(100):
        train_epoch_loss = []
        train_epoch_acc = []
        model.train()
        for i, mini_batch in enumerate(train_loader):
            prefix = mini_batch[0].to(device)
            y_real = mini_batch[1]

            model.zero_grad()
            y_pred = model(prefix)
            
            y_pred_loss = y_pred.view(-1, num_activities + 1)
            y_real_loss = y_real.view(-1)

            train_loss = loss_fn(y_pred_loss, y_real_loss)
            train_acc = acc(y_pred, y_real)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_epoch_loss.append(train_loss.item())
            train_epoch_acc.append(train_acc.item())

        with torch.no_grad():
            val_epoch_loss, val_epoch_acc = val_test(model, val_loader, device, num_activities)

            avg_train_loss = sum(train_epoch_loss) / len(train_epoch_loss)
            avg_val_loss = sum(val_epoch_loss) / len(val_epoch_loss)
            avg_train_acc = (sum(train_epoch_acc) / len(train_epoch_acc)) * 100
            avg_val_acc = (sum(val_epoch_acc) / len(val_epoch_acc)) * 100

            print(f'Epoch {e}: | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | '
                  f'Train Acc: {avg_train_acc:.3f} | Val Acc: {avg_val_acc:.3f}')

            if use_wandb:
                import wandb
                wandb.log({
                    num_fold + '_train_loss': avg_train_loss,
                    num_fold + '_val_loss': avg_val_loss,
                    num_fold + '_train_acc': avg_train_acc,
                    num_fold + '_val_acc': avg_val_acc
                })

            if avg_val_loss < val_mae_best:
                val_mae_best = avg_val_loss
                epochs_no_improve = 0

                model_dir = f'./models/{filename}/{num_fold}'
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, model_name))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 30:
                print("Early stopping")
                break

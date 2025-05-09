import os
import pathlib
import torch
import torch.nn as nn
import numpy as np
import wandb

torch.manual_seed(42)


class NapLSTM(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()

        self.device = device

        # Tax LSTM implementation (translated from Keras)
        self.l1 = nn.LSTM(input_size=input_size,
                          hidden_size=100,
                          batch_first=True)

        self.l2 = nn.LSTM(input_size=100,
                          hidden_size=100,
                          batch_first=True)

        self.act_output = nn.Linear(in_features=100,
                                    out_features=output_size)

        self.to(device)

    def forward(self, prefix):
        prefix = prefix.type(torch.FloatTensor)
        x, _ = self.l1(prefix)
        x, _ = self.l2(x)
        x = x[:, -1]
        act_output = self.act_output(x)  # No Softmax because the use of CrossEntropyLoss

        return act_output

    def fit(self, train_loader, val_loader, filename, num_fold, model_name, use_wandb):
        opt = torch.optim.Adam(self.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        val_mae_best = np.inf  # Starts the best MAE value as infinite

        for e in range(100):
            train_epoch_loss = []
            train_epoch_acc = []
            self.train()

            sum_train_loss = 0
            for mini_batch in iter(train_loader):
                prefix = mini_batch[0].to(self.device)
                y_real = torch.max(mini_batch[1], 1)[1].to(self.device)

                self.zero_grad()
                y_pred = self(prefix)
                train_loss = loss_fn(y_pred, y_real)
                train_acc = self.multi_acc(y_pred, y_real)
                train_loss.backward()
                opt.step()

                train_epoch_loss.append(train_loss.item())
                train_epoch_acc.append(train_acc.item())

            # Validation
            with torch.no_grad():
                val_epoch_loss, val_epoch_acc = self.val_test(val_loader)

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

                    # Saves the model
                    if os.path.isdir('../models/' + filename + '/' + num_fold + '/'):
                        torch.save(self, "../models/" + filename + '/' + num_fold + "/" + model_name)
                    else:
                        pathlib.Path('../models/' + filename + '/' + num_fold).mkdir(parents=True, exist_ok=True)
                        torch.save(self, "../models/" + filename + '/' + num_fold + "/" + model_name)

    def val_test(self, data_loader):
        self.eval()
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        val_epoch_loss = []
        val_epoch_acc = []
        for mini_batch in iter(data_loader):
            prefix = mini_batch[0].to(self.device)
            y_real = torch.max(mini_batch[1], 1)[1].to(self.device)

            y_pred = self(prefix)
            val_loss = loss_fn(y_pred, y_real)
            val_acc = self.multi_acc(y_pred, y_real)

            val_epoch_loss.append(val_loss.item())
            val_epoch_acc.append(val_acc.item())

        return val_epoch_loss, val_epoch_acc

    def multi_acc(self, y_pred, y_real):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_real).float()
        acc = correct_pred.sum() / len(correct_pred)

        return acc

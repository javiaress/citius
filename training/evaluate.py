import torch
import torch.nn as nn
from jellyfish import damerau_levenshtein_distance
from training.train import acc


def levenshtein_acc(y_pred, y_real, tam_suf):
    y_pred_softmax = torch.log_softmax(y_pred, dim=-1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=-1)

    y_pred_tags = y_pred_tags.cpu().numpy()
    y_real_tags = y_real.cpu().numpy()
    tam_suf = tam_suf.cpu().numpy()

    acc = 0
    for i in range(len(y_pred_tags)):
        pred_seq = ''.join(map(chr, y_pred_tags[i][:tam_suf[i]] + 161))
        real_seq = ''.join(map(chr, y_real_tags[i][:tam_suf[i]] + 161))

        acc += 1 - damerau_levenshtein_distance(pred_seq, real_seq) / max(len(pred_seq), len(real_seq))

    return acc / len(y_pred_tags)


def test(model, test_loader, num_activities):
    device = next(model.parameters()).device
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0).to(device)

    test_epoch_loss = []
    test_epoch_acc = []

    for mini_batch in test_loader:
        prefix = mini_batch[0].to(device)
        y_real = mini_batch[1]
        tam_suf = mini_batch[2]

        y_pred = model(prefix)
        y_pred_loss = y_pred.view(-1, num_activities + 2)
        y_real_loss = y_real.view(-1)

        test_loss = loss_fn(y_pred_loss, y_real_loss)
        #test_acc = acc(y_pred, y_real)
        test_acc = levenshtein_acc(y_pred, y_real, tam_suf)

        test_epoch_loss.append(test_loss.item())
        test_epoch_acc.append(test_acc)

    return test_epoch_loss, test_epoch_acc

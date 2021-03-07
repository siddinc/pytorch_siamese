import torch
import math

from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from constants import (
  BETAS,
  WEIGHT_DECAY,
  AMSGRAD,
  N_EPOCHS,
  MARGIN,
)


class DeviceDataLoader():
  
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device
        
  def __iter__(self):
    for b in self.dl:
      yield to_device(b, self.device)

  def __len__(self):
    return len(self.dl)


def get_default_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')


def to_device(data, device):
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)


def contrastive_loss(y_pred, y_true, margin=1.0):
  loss = (y_true * torch.pow(y_pred, 2)) + ((1.0 - y_true) * torch.pow(torch.clamp(margin - y_pred, min=0.0), 2))
  return torch.mean(loss)


def accuracy(y_pred, y_true):
  pred = (y_pred < 0.5).type(y_true.dtype)
  return torch.tensor(torch.sum(pred == y_true).item() / len(y_true))


def lr_finder(model, train_loader, init_value=1e-8, final_value=1., beta = 0.98):
  optimizer = Adam(model.parameters(), lr=init_value, betas=BETAS, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)

  num = len(train_loader)-1
  mult = (final_value / init_value) ** (1/num)
  lr = init_value
  optimizer.param_groups[0]['lr'] = lr
  avg_loss = 0.
  best_loss = 0.
  batch_num = 0
  losses = []
  lrs = []

  model.train()
  for batch in tqdm(train_loader, total=len(train_loader), leave=False):
    batch_num += 1

    samples1, samples2, pair_labels = batch
    out = model(samples1, samples2)
    loss = contrastive_loss(out, pair_labels, MARGIN)

    avg_loss = beta * avg_loss + (1-beta) *loss.item()
    smoothed_loss = avg_loss / (1 - beta**batch_num)

    if batch_num > 1 and smoothed_loss > 4 * best_loss:
        lrs, losses

    if smoothed_loss < best_loss or batch_num==1:
        best_loss = smoothed_loss

    losses.append(smoothed_loss)
    lrs.append(lr)

    loss.backward()
    optimizer.step()

    lr *= mult
    optimizer.param_groups[0]['lr'] = lr

  return lrs, losses
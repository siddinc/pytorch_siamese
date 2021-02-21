import torch
import torch.nn as nn


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


class ContrastiveLoss(nn.Module):

  def __init__(self, margin=1.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, y_pred, y_true):
    loss = (y_true * torch.pow(y_pred, 2)) + ((1.0 - y_true) * torch.pow(torch.clamp(self.margin - y_pred, min=0.0), 2))
    return torch.mean(loss)


def accuracy(y_pred, y_true):
  y_pred = (y_pred > 0.5).type(y_true.dtype)
  return torch.tensor(torch.sum(y_pred == y_true).item() / len(y_true))
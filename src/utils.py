import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
  def __init__(self, margin=1.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, y_pred, y_true):
    loss = (y_true * torch.pow(y_pred, 2)) + ((1.0 - y_true) * torch.pow(torch.clamp(self.margin - y_pred, min=0.0), 2))
    return torch.mean(loss)
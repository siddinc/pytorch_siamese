import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
  get_data_loader,
)


class TrainingLoop(nn.Module):
  def training_step(self, batch):
    samples1, samples2, pair_labels = batch
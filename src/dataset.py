from torch.utils.data import DataLoader, random_split, Dataset
from itertools import combinations
import random
import torch
import numpy as np

from constants import (
  BATCH_SIZE,
  N_EPOCHS,
  N_WORKERS,
  SHUFFLE,
)


class SiameseDataset(Dataset):

  def __init__(self, samples, labels, pred=False, transform=None):
    self.samples = samples
    self.labels = labels
    self.pred = pred
    self.transform = transform

  def __len__(self):
    return len(self.samples[0])

  def __getitem__(self, index):
    pair_label = self.labels[index]
    sample1, sample2 = self.samples[index, 0, :,:], self.samples[index, 1, :,:]

    if self.transform is not None:
      sample1, sample2 = self.transform(sample1), self.transform(sample2)

    if self.pred == True:
      return sample1, sample2
    
    return sample1, sample2, pair_label


def get_default_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')
    
def to_device(data, device):
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)


class DeviceDataLoader():
  def __init__(self, dl, device):
    self.dl = dl
    self.device = device
        
  def __iter__(self):
    for b in self.dl:
      yield to_device(b, self.device)

  def __len__(self):
    return len(self.dl)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  pair_train = torch.randn((10, 2, 1, 3000))
  label_train = torch.randn((10, 1))

  train_ds = SiameseDataset(pair_train, label_train,  pred=False)
  train_dl = DeviceDataLoader(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS), device)

  for i, batch in enumerate(train_dl):
    print(batch[2])
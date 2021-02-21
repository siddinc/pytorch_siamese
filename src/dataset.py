import torch

from torch.utils.data import DataLoader, Dataset
from utils import to_device, DeviceDataLoader


class SiameseNetDataset(Dataset):

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


def get_data_loader(x, y, batch_size, pred=False, shuffle=False, num_workers=None, device=None):
  ds = SiameseNetDataset(x, y, pred=pred)
  dl = DeviceDataLoader(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), device)
  return dl
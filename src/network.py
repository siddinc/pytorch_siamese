import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
from constants import (
  BATCH_SIZE,
)


class SiameseNet(nn.Module):

  def __init__(self):
    super(SiameseNet, self).__init__()
    self.cnn = nn.Sequential(
      self.conv1d_block(1, 16, 3, stride=1, padding=1),  #3000x16
      nn.MaxPool1d(2),  #1500x16
      self.conv1d_block(16, 32, 3, 1, 1),  #1500x32
      nn.MaxPool1d(2),  #750x32
      self.conv1d_block(32, 64, 3, 1, 1),  #750x64
      nn.MaxPool1d(2),  #375x64
      self.conv1d_block(64, 128, 3, 1, 1), #375x128
      nn.AvgPool1d(375), #1, 128
      nn.Sigmoid(),
    )

    self.sigmoid = nn.Sigmoid()

  def conv1d_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
      nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
      nn.LeakyReLU(0.2, inplace=True),
      nn.BatchNorm1d(out_channels),
    )
  
  def forward_once(self, x):
    output = self.cnn(x)
    output = output.view(output.size()[0], -1)
    
    return output
  
  def forward(self, input1, input2):
    embedding1 = self.forward_once(input1)
    embedding2 = self.forward_once(input2)
    distance = nn.PairwiseDistance(p=2)(embedding1, embedding2)
    final_output = self.sigmoid(distance)
    return final_output

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = SiameseNet().to(device)
  summary(net, [(1, 3000), (1, 3000)], batch_size=16)
import torch

from utils import get_default_device
from network import get_model


if __name__ == "__main__":
  device = get_default_device()

  model = get_model(2, get_summary=False, device=device)

  x = torch.randn((5, 1, 3000), device=device)
  y = torch.randn((5, 1, 3000), device=device)

  model.eval()
  out = model(x, y)
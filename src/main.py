import torch

from utils import get_default_device
from network import get_model
from dataset import get_data_loader
from constants import (
  BATCH_SIZE,
  N_WORKERS,
  N_EPOCHS,
  MARGIN,
  NORM_DEG,
  MAX_LR,
  WEIGHT_DECAY,
)


if __name__ == "__main__":
  device = get_default_device()

  # train_dl = get_data_loader(pair_train, label_train, BATCH_SIZE, pred=False, shuffle=True, num_workers=N_WORKERS, device=device)
  # val_dl = get_data_loader(pair_val, label_val, BATCH_SIZE, pred=False, shuffle=True, num_workers=N_WORKERS, device=device)
  # test_dl = get_data_loader(pair_test, label_test, BATCH_SIZE, pred=False, shuffle=False, num_workers=N_WORKERS, device=device)

  model = model = get_model(1, NORM_DEG["manhattan"], device=device, get_summary=([(1,3000), (1,3000)], 16))

  x = torch.randn((16, 1, 3000), device=device)
  y = torch.randn((16, 1, 3000), device=device)

  model.eval()
  out = model(x, y)
  print(out)
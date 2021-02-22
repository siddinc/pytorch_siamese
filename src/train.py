import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
  get_data_loader,
)
from tqdm import tqdm
from utils import contrastive_loss, accuracy


@torch.no_grad()
def evaluate(model, val_loader):
  model.eval()
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group["lr"]


def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, opt=torch.optim.Adam):
  torch.cuda.empty_cache()
  history = []

  optimizer = opt(model.parameters(), max_lr, weight_decay=weight_decay)
  sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

  for epoch in tqdm(range(epochs)):
    model.train()
    train_losses = []
    lrs = []

    for batch in train_loader:
      loss = model.training_step(batch)
      train_losses.append(loss)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      lrs.append(get_lr(optimizer))
      sched.step()

    result = evaluate(model, val_loader)
    result["train_loss"] = torch.stack(train_losses).mean().item()
    result["lrs"] = lrs
    model.epoch_end(epoch, result)
    history.append(result)
  
  return history
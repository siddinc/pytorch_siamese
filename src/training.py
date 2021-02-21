import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (
  get_data_loader,
)
from tqdm import tqdm
from utils import contrastive_loss, accuracy


class MetricLearningBase(nn.Module):
  def __init__(self, norm_deg=1):
    super(MetricLearningBase, self).__init__()

  def training_step(self, batch):
    samples1, samples2, pair_labels = batch
    out = self(samples1, samples2)
    loss = contrastive_loss(out, pair_labels, 1.0)
    return loss

  def validation_step(self, batch):
    samples1, samples2, pair_labels = batch
    out = self(samples1, samples2)
    loss = contrastive_loss(out, pair_labels, 1.0)
    acc = accuracy(out, pair_labels)
    return {"val_loss": loss.detach(), "val_acc": acc}

  def validation_epoch_end(self, outputs):
    batch_losses = [x["val_loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x["val_acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

  def epoch_end(self, epoch=None, result=None):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
      epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


@torch.no_grad()
def evaluate(model, val_loader):
  model.eval()
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group["lr"]


def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
  torch.cuda.empty_cache()
  history = []

  optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
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
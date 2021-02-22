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


def contrastive_loss(y_pred, y_true, margin=1.0):
  loss = (y_true * torch.pow(y_pred, 2)) + ((1.0 - y_true) * torch.pow(torch.clamp(margin - y_pred, min=0.0), 2))
  return torch.mean(loss)


def accuracy(y_pred, y_true):
  y_pred = (y_pred > 0.5).type(y_true.dtype)
  return torch.tensor(torch.sum(y_pred == y_true).item() / len(y_true))


class SiameseNetBase(nn.Module):
  def __init__(self, norm_deg=1):
    super(SiameseNetBase, self).__init__()

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
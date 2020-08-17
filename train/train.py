import datetime
from time import time

import torch
from torch import nn

import utils


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [to_device(o, device) for o in obj]
    raise ValueError(f'got type {type(obj)}')


def split(self, at):
    """Split model's children to 2 groups at the child named 'at'."""
    layer_groups = []
    current_group = []
    for name, child in self.model.named_children():
        if name == at:
            layer_groups.append(nn.Sequential(*current_group))
            current_group = []
        current_group.append(child)
    layer_groups.append(nn.Sequential(*current_group))
    return layer_groups


class Trainer:

    def __init__(self, model: nn.Module, data, loss_func, opt, device=torch.device('cuda'), callbacks=None,
                 show_progress=True):
        self.device = device
        self.model = model.to(device)
        self.data = data
        self.loss_func = loss_func
        self.layer_groups = [nn.Sequential(*list(self.model.children()))]
        self.opt = opt
        callbacks = utils.listify(callbacks)
        self.callbacks = [clbk(self) for clbk in callbacks]
        self.show_progress = show_progress

    def fit(self, epochs, metrics=None):
        metrics = utils.listify(metrics)
        for epoch in range(epochs):
            start_time = time()
            for m in metrics:
                m.on_epoch_begin()
            train_avg_loss = self.fit_one_epoch()
            val_avg_loss = self.evaluate(metrics)
            epoch_time = datetime.timedelta(seconds=round(time() - start_time))

            msg = f'[{epoch}]: train loss: {train_avg_loss:.4}, val loss; {val_avg_loss:.4}, time: {epoch_time}'
            for m in metrics:
                msg += f', {m}'
            print(msg)

            for clbk in self.callbacks:
                clbk.on_epoch_end(epoch)

    def fit_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        n_items = 0

        for batch in self.train_dl():
            xb, yb = to_device(batch, self.device)
            self.opt.zero_grad()
            outputs = self.model(*utils.listify(xb))
            loss = self.loss_func(outputs, yb)
            self._backward(loss)
            running_loss += loss.item() * len(xb)
            n_items += len(xb)

        return running_loss / n_items

    def _backward(self, loss):
        loss.backward()
        self.opt.step()
        for clbk in self.callbacks:
            clbk.on_backward_end()

    @torch.no_grad()
    def evaluate(self, metrics):
        self.model.eval()
        running_loss = 0.0
        n_items = 0

        for batch in self.valid_dl():
            xb, yb = to_device(batch, self.device)
            outputs = self.model(*utils.listify(xb))
            loss = self.loss_func(outputs, yb)
            running_loss += loss.item() * len(xb)
            n_items += len(xb)
            for m in metrics:
                m(outputs, yb)

        return running_loss / n_items

    def save(self, path):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict()
        }
        torch.save(state, path)

    def load(self, path, model_only=False):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['model'])
        if not model_only:
            self.opt.load_state_dict(state_dict['optimizer'])

    def _dl(self, dl):
        if self.show_progress:
            return utils.progress_bar(dl)
        return dl

    def train_dl(self):
        return self._dl(self.data.train_dl())

    def valid_dl(self):
        return self._dl(self.data.valid_dl())

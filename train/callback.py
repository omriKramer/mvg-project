from functools import partial
from pathlib import Path

from torch import nn

from .train import Trainer


class Callback:

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    @classmethod
    def partial(cls, *args, **kwargs):
        return partial(cls, *args, **kwargs)

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_backward_end(self):
        pass


class SaveCallback(Callback):

    def __init__(self, trainer, path='.', name='model'):
        super().__init__(trainer)
        self.name = name
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch):
        self.trainer.save(self.path / f'{self.name}_{epoch}.pt')


class GradientClipping(Callback):

    def __init__(self, trainer, norm=1.):
        super().__init__(trainer)
        self.norm = norm

    def on_backward_end(self):
        nn.utils.clip_grad_norm_(self.trainer.model.parameters(), self.norm)

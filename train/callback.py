from functools import partial
from pathlib import Path

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


class SaveCallback(Callback):

    def __init__(self, trainer, path='.', name='model'):
        super().__init__(trainer)
        self.name = name
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch):
        self.trainer.save(self.path / f'{self.name}_{epoch}.pt')

from typing import Any, Protocol

from torch.utils.data import DataLoader
from ..core import LoopMixin, EngineMixin, hooks

__all__ = ['Train_EngineMixin']


class ParentProtocol(Protocol):
    mixin_loop_train: LoopMixin
    """ Mixin that handles how we loop through the dataset. """

    train_loader: DataLoader[Any]
    """
    Dataloader instance that is looped through.

    Note:
        This can be a property if you need to change the dataloader during a run.
    """


class Train_EngineMixin(EngineMixin, protocol=ParentProtocol):
    """
    EngineMixin that keeps running through a dataset forever, which is mainly used for training.
    """
    @hooks.engine_begin
    def assert_name(self) -> None:
        assert self.name == 'train', f'{self.__class__.__name__} can only be used for training'

    def __call__(self) -> None:
        if self.quit:
            return

        while True:
            if self.quit:
                return

            self.parent.mixin_loop_train.dataloader = self.parent.train_loader

            for _ in self.parent.mixin_loop_train:
                if self.quit:
                    return

            del self.parent.mixin_loop_train.dataloader

            if self.quit:
                return

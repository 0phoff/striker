from typing import Protocol

from ..core import EngineMixin, LoopMixin, hooks

__all__ = ['Train_EngineMixin']


class ParentProtocol(Protocol):
    mixin_loop_train: LoopMixin
    """ Mixin that handles how we loop through the dataset. """


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

            self.parent.mixin_loop_train.dataloader = self.parent.mixin_data.train

            for _ in self.parent.mixin_loop_train:
                if self.quit:
                    return

            del self.parent.mixin_loop_train.dataloader

            if self.quit:
                return

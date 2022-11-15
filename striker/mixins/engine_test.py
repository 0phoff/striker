from typing import Protocol, Optional, Any

from torch.utils.data import DataLoader
from ..core import LoopMixin, EngineMixin, hooks

__all__ = ['Test_EngineMixin']


class ParentProtocolTest(Protocol):
    mixin_loop_test: LoopMixin
    """ Mixin that handles how we loop through the dataset. """

    test_loader: Optional[DataLoader[Any]] = None
    """
    Dataloader instance that is looped through.

    Note:
        This can be a property if you need to change the dataloader during a run.
    """


class ParentProtocolValidation(Protocol):
    mixin_loop_validation: LoopMixin
    """ Mixin that handles how we loop through the dataset. """

    validation_loader: Optional[DataLoader[Any]] = None
    """
    Dataloader instance that is looped through.

    Note:
        This can be a property if you need to change the dataloader during a run.
    """


class Test_EngineMixin(EngineMixin):
    """
    EngineMixin that runs through a dataloader once, which is mainly used for testing and validation.
    """
    def __set_name__(self, owner: Any, name: str) -> None:
        super().__set_name__(owner, name)
        if self.name == 'validation':
            self.__protocol__ = ParentProtocolValidation
        elif self.name == 'test':
            self.__protocol__ = ParentProtocolTest

    @hooks.engine_begin
    def assert_name(self) -> None:
        assert self.name in ('test', 'validation'), f'{self.__class__.__name__} can only be used for validating or testing'

    def __call__(self) -> None:
        if self.quit:
            return

        dataloader = getattr(self.parent, f'{self.name}_loader')
        mixin = getattr(self.parent, f'mixin_loop_{self.name}')

        mixin.dataloader = dataloader

        with self.parent.eval():
            for _ in mixin:
                if self.quit:
                    return

        del mixin.dataloader

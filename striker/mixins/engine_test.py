from typing import Any, Protocol

from ..core import EngineMixin, LoopMixin

__all__ = ['Test_EngineMixin']


class ParentProtocolTest(Protocol):
    mixin_loop_test: LoopMixin
    """ Mixin that handles how we loop through the dataset. """


class ParentProtocolValidation(Protocol):
    mixin_loop_validation: LoopMixin
    """ Mixin that handles how we loop through the dataset. """


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
        else:
            raise ValueError(f'{self.__class__.__name__} can only be used for validation or testing')

    def __call__(self) -> None:
        if self.quit:
            return

        if self.name == 'validation':
            mixin = self.parent.mixin_loop_validation
            mixin.dataloader = self.parent.mixin_data.validation
        elif self.name == 'test':
            mixin = self.parent.mixin_loop_test
            mixin.dataloader = self.parent.mixin_data.test
        else:
            # Should never be reached as it is already tested in __set_name__.
            raise ValueError(f'{self.__class__.__name__} can only be used for validation or testing')

        with self.parent.eval():
            for _ in mixin:
                if self.quit:
                    return

        del mixin.dataloader

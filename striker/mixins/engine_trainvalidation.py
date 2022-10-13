from typing import Protocol, Literal, Union, cast

from ..core import EngineMixin, hooks
from .engine_train import TrainEngineMixin

__all__ = ['TrainValidationEngineMixin']


class ParentProtocol(TrainEngineMixin.__protocol__, Protocol):      # type: ignore
    validation_rate: Union[list[Union[int, slice]], int, slice, None] = slice(None, None, 1)
    """
    When to run the validation.

    Note:
        This value is used to setup a hook and thus can have a few different values:
            - None: never run the hook
            - slice: run hook periodically (according to slice specs)
            - int: run hook at specified epoch/batch
            - list[int, slice]: Combination of the above
    """

    mixin_engine_validation: EngineMixin
    """ Validation Engine that will be used. """


class TrainValidationEngineMixin(TrainEngineMixin, protocol=ParentProtocol):
    """
    EngineMixin that keeps running through a dataset forever and occasionally will run the `mixin_engine_validation`.
    This is only really useful as a training engine.
    """
    __type_check__: Literal['none', 'log', 'raise'] = 'raise'

    def __init__(self, validation_mode: Literal['batch', 'epoch'] = 'epoch') -> None:
        self.validation_mode = validation_mode

    @hooks.engine_begin
    def setup_validation_hook(self, entry: Literal['train', 'validation', 'test']) -> None:
        if entry == 'train':
            validation_rate = getattr(self.parent, 'validation_rate', slice(None, None, 1))
            if validation_rate is not None:
                if self.validation_mode == 'batch':
                    self.hooks.train_batch_end[validation_rate](self.run_validation)
                else:
                    self.hooks.train_epoch_end[validation_rate](self.run_validation)

    def run_validation(self) -> None:
        mixin = cast(EngineMixin, self.parent.mixin_engine_validation)
        with mixin.reset_quit(False):
            mixin()

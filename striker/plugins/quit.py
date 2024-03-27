import logging
from typing import Literal, Optional, Protocol

import torch

from .._engine import Engine
from ..core import Plugin, hooks

__all__ = ['QuitPlugin']
log = logging.getLogger(__name__)


class ParentProtocol(Protocol):
    max_epochs: Optional[int] = None
    """ Maximum number of epochs. """

    max_batches: Optional[int] = None
    """ Maximum number of batches. """


class QuitPlugin(Plugin, protocol=ParentProtocol):
    """
    This plugin will automatically stop training if a certain number of epochs or batches is reached.
    Additionally, it can also stop training if the loss explodes.

    Args:
        explode: Whether to stop at an exploding loss as well. Default **True**

    Note:
        We consider an exploding loss to be when the loss is infinite or NaN.
        This gets checked by running :meth:`~torch.Tensor.isinf` and :meth:`~toch.Tensor.isnan`.
    """

    parent: Engine  # Fix MyPy issues by setting a proper type of self.parent

    def __init__(self, explode: bool = True):
        self.explode = explode

    @hooks.engine_begin
    def setup_quit_hook(self, entry: Literal['train', 'validation', 'test']) -> None:
        if entry == 'train':
            max_epochs = getattr(self.parent, 'max_epochs', None)
            if max_epochs is not None:
                self.hooks.train_epoch_end[max_epochs::](self.quit_epoch)

            max_batches = getattr(self.parent, 'max_batches', None)
            if max_batches is not None:
                self.hooks.train_batch_end[max_batches::](self.quit_batch)

            if self.explode:
                self.hooks.train_batch_end(self.quit_explode)

            if max_epochs is None and max_batches is None:
                log.warning('"max_epochs" and "max_batches" are None, training might never stop.')

    def quit_epoch(self, epoch: int) -> None:
        log.info('Quitting at epoch %d', epoch)
        self.parent.quit()

    def quit_batch(self, batch: int) -> None:
        log.info('Quitting at batch %d', batch)
        self.parent.quit()

    def quit_explode(self, batch: int, loss: torch.Tensor) -> None:
        if loss.isinf() or loss.isnan():
            log.error('Infinite loss')
            self.parent.quit()

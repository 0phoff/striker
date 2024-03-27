from typing import Any, Iterator, Optional, Protocol

from torch import Tensor
from torch.utils.data import DataLoader

from ..core import LoopMixin, hooks

__all__ = ['BatchTrain_LoopMixin']


class ParentProtocol(Protocol):
    batch: int
    """
    Variable to keep track of the number of trained batches.

    Note:
        This variable gets automatically created on each :class:`~striker.Parameters` object and should thus not be manually cared for.
    """

    epoch: int
    """
    Variable to keep track of the number of trained epochs.

    Note:
        This variable gets automatically created on each :class:`~striker.Parameters` object and should thus not be manually cared for.
    """

    def forward(self, data: Any) -> Tensor:
        """
        Forward pass of the network.

        Args:
            data: data from the dataloader

        Returns:
            single loss value upon which we call ``backward()``
        """

    def optimize(self) -> None:
        """
        The optimization step of the training.
        Usually you would call ``optimizer.step()`` and ``optimizer.zero_grad()`` here.
        """

    @hooks.train_epoch_begin
    def train_epoch_begin(self, epoch: int) -> None:
        """
        Hook that gets called at the start of every epoch.

        Args:
            epoch: Number of the epoch we start.

        Note:
            The ``epoch`` argument is equal to ``self.epoch + 1``, as the ``self.epoch`` variable shows the number of completed epochs.
        """

    @hooks.train_epoch_end
    def train_epoch_end(self, epoch: int) -> None:
        """
        Hook that gets called at the end of every epoch.

        Args:
            epoch: Number of the epoch we end.
        """

    @hooks.train_batch_begin
    def train_batch_begin(self, batch: int) -> None:
        """
        Hook that gets called at the start of every batch.

        Args:
            batch: Number of the batch we start.

        Note:
            The ``batch`` argument is equal to ``self.batch + 1``, as the ``self.batch`` variable shows the number of completed batches.
        """

    @hooks.train_batch_end
    def train_batch_end(self, batch: int, loss: Tensor) -> None:
        """
        Hook that gets called at the end of every batch.

        Args:
            batch: Number of the batch we end.
            loss: Detached loss of the batch.
        """

    @hooks.data_batch
    def data_batch(self, data: Any) -> None:
        """
        Hook that gets called with for every batch of data, before using it.
        The difference with the :meth:`ParentProtocol.train_batch_begin` hook is that we pass the data as an argument
        and that most :class:`~striker.core.LoopMixin` provide this hook.

        Args:
            data: Batch of data from the dataloader.
        """


class BatchTrain_LoopMixin(LoopMixin, protocol=ParentProtocol):
    """
    LoopMixin that trains a model on batches.
    This is the default training LoopMixin.
    """

    @hooks.engine_begin
    def assert_name(self) -> None:
        assert self.name == 'train', f'{self.__class__.__name__} can only be used for training'

    def loop(self, dataloader: DataLoader[Any]) -> Optional[Iterator[None]]:
        self.parent.run_hook(type='train_epoch_begin', index=self.parent.epoch + 1, args=(self.parent.epoch + 1,))

        for data in dataloader:
            self.parent.run_hook(type='data_batch', args=(data,))
            self.parent.run_hook(type='train_batch_begin', index=self.parent.batch + 1, args=(self.parent.batch + 1))

            loss = self.parent.forward(data)
            loss.backward()
            self.parent.optimize()

            self.parent.batch += 1
            self.parent.run_hook(type='train_batch_end', index=self.parent.batch, args=(self.parent.batch, loss.detach()))

            yield None

        self.parent.epoch += 1
        self.parent.run_hook(type='train_epoch_end', index=self.parent.epoch, args=(self.parent.epoch,))

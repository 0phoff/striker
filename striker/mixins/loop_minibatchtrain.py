from typing import Any, Iterator, Optional, Protocol

from torch.utils.data import DataLoader

from .loop_batchtrain import BatchTrain_LoopMixin

__all__ = ['MiniBatchTrain_LoopMixin']


class ParentProtocol(BatchTrain_LoopMixin.__protocol__, Protocol):  # type: ignore
    batch_accumulate: int
    """ Number of batches to accumulate before calling ``optimize()``. """


class MiniBatchTrain_LoopMixin(BatchTrain_LoopMixin, protocol=ParentProtocol):
    """
    LoopMixin that trains a model on batches, but accumulates a certain number of mini-batches before optimizing.
    This is useful to emulate training on larger batches than you can fit in memory,
    but beware that it is not entirely equivalent (eg. batch normalization statistics).

    Args:
        average: Whether to take the average or the sum of the individual mini-batch losses; Default **True**
    """

    def __init__(self, average: bool = True) -> None:
        self.average = average
        self.accumulated_batches = 0

    def loop(self, dataloader: DataLoader[Any]) -> Optional[Iterator[None]]:
        self.parent.run_hook(type='train_epoch_begin', index=self.parent.epoch + 1, args=(self.parent.epoch + 1,))

        accumulated_loss = 0
        for data in dataloader:
            if self.accumulated_batches == 0:
                self.parent.run_hook(type='train_batch_begin', index=self.parent.batch + 1, args=(self.parent.batch + 1))

            # MiniBatch
            self.parent.run_hook(type='data_batch', args=(data,))
            loss = self.parent.forward(data)
            if self.average:
                loss /= self.parent.batch_accumulate
            loss.backward()
            accumulated_loss += loss.detach()
            self.accumulated_batches += 1
            if self.accumulated_batches < self.parent.batch_accumulate:
                continue

            # Batch
            self.accumulated_batches = 0
            self.parent.optimize()

            self.parent.batch += 1
            self.parent.run_hook(type='train_batch_end', index=self.parent.batch, args=(self.parent.batch, accumulated_loss))
            accumulated_loss = 0

            yield None

        self.parent.epoch += 1
        self.parent.run_hook(type='train_epoch_end', index=self.parent.epoch, args=(self.parent.epoch,))

    @property
    def num_batches(self) -> int:
        return (len(self.dataloader) if self.dataloader is not None else 0) // self.parent.batch_accumulate

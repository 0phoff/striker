from typing import Iterator, Protocol, Optional, Any

from torch.utils.data import DataLoader
from ..core import LoopMixin, hooks

__all__ = ['TestLoopMixin']


class ParentProtocol(Protocol):
    def infer(self, data: Any) -> Any:
        """
        Inference pass of the network.

        Args:
            data: data from the dataloader.

        Returns:
            output that gets aggregated in a list for :meth:`~ParentProtocol.post()``.
        """

    def post(self, output: list[Any]) -> Any:
        """
        Post-processing of the network output.

        Args:
            output: Aggregated output from :meth:`~ParentProtocol.infer()`.

        Returns:
            Post-processed output such as metrics, losses, etc.
        """

    @hooks.data_batch
    def data_batch(self, data: Any) -> None:
        """
        Hook that gets called with for every batch of data, before using it.
        The difference with the :meth:`ParentProtocol.train_batch_start` hook is that we pass the data as an argument
        and that most :class:`~striker.core.LoopMixin` provide this hook.

        Args:
            data: Batch of data from the dataloader.
        """


class ParentProtocolTest(ParentProtocol):
    @hooks.test_epoch_start
    def test_epoch_start(self) -> None:
        """
        Hook that gets called at the start of an epoch.
        """

    @hooks.test_epoch_end
    def test_epoch_end(self, output: Any) -> None:
        """
        Hook that gets called at the end of an epoch.

        Args:
            output: Final output from :meth:`ParentProtocol.post()`.
        """

    @hooks.test_batch_start
    def test_batch_start(self, batch: int) -> None:
        """
        Hook that gets called at the start of every batch.

        Args:
            batch: Number of the batch we start.
        """

    @hooks.test_batch_end
    def test_batch_end(self, batch: int, output: Any) -> None:
        """
        Hook that gets called at the end of every batch.

        Args:
            batch: Number of the batch we end.
            output: Output from the batch we processed.
        """


class ParentProtocolValidation(ParentProtocol):
    @hooks.validation_epoch_start
    def validation_epoch_start(self) -> None:
        """
        Hook that gets called at the start of an epoch.
        """

    @hooks.validation_epoch_end
    def validation_epoch_end(self, output: Any) -> None:
        """
        Hook that gets called at the end of an epoch.

        Args:
            output: Final output from :meth:`ParentProtocol.post()`.
        """

    @hooks.validation_batch_start
    def validation_batch_start(self, batch: int) -> None:
        """
        Hook that gets called at the start of every batch.

        Args:
            batch: Number of the batch we start.
        """

    @hooks.validation_batch_end
    def validation_batch_end(self, batch: int, output: Any) -> None:
        """
        Hook that gets called at the end of every batch.

        Args:
            batch: Number of the batch we end.
            output: Output from the batch we processed.
        """


class TestLoopMixin(LoopMixin, protocol=ParentProtocol):
    """
    LoopMixin that runs a model over a dataset for evaluation.
    """
    def __set_name__(self, owner: Any, name: str) -> None:
        super().__set_name__(owner, name)
        if self.name == 'validation':
            self.__protocol__ = ParentProtocolValidation
        elif self.name == 'test':
            self.__protocol__ = ParentProtocolTest

    @hooks.engine_start
    def assert_name(self) -> None:
        assert self.name in ('test', 'validation'), f'{self.__class__.__name__} can only be used for validating or testing'

    def loop(self, dataloader: DataLoader[Any]) -> Optional[Iterator[None]]:
        self.parent.run_hook(type=f'{self.name}_epoch_start')

        outputs = []
        for batch, data in enumerate(dataloader):
            self.parent.run_hook(type='data_batch', args=(data,))
            self.parent.run_hook(type=f'{self.name}_batch_start', index=batch + 1, args=(batch + 1,))
            outputs.append(self.parent.infer(data))
            self.parent.run_hook(type=f'{self.name}_batch_end', index=batch, args=(batch, outputs[-1]))
            yield

        output = self.parent.post(outputs)

        self.parent.run_hook(type=f'{self.name}_epoch_end', args=(output,))

from typing import Any, Iterator, Optional, Protocol

from torch.utils.data import DataLoader

from ..core import LoopMixin, hooks

__all__ = ['Test_LoopMixin']


def dynamic_parent_protocol(name: str, infer: str, post: str) -> Any:
    class ParentProtocol(Protocol):
        @hooks.data_batch
        def data_batch(self, data: Any) -> None:
            pass

    def _infer(self: Any, data: Any) -> Any:
        """
        Inference pass of the network.

        Args:
            data: data from the dataloader.

        Returns:
            output that gets aggregated in a list for the post-processing method.
        """
        pass

    def _post(self: Any, output: list[Any]) -> Any:
        """
        Post-processing of the network output.

        Args:
            output: Aggregated output from the inference method.

        Returns:
            Post-processed output such as metrics, losses, etc.
        """

    def epoch_begin(self: Any) -> None:
        """
        Hook that gets called at the start of an epoch.
        """

    def epoch_end(self: Any, output: Any) -> None:
        """
        Hook that gets called at the end of an epoch.

        Args:
            output: Final output from the post-processing method.
        """

    def batch_begin(self: Any, batch: int) -> None:
        """
        Hook that gets called at the start of every batch.

        Args:
            batch: Number of the batch we start.
        """

    def batch_end(self: Any, batch: int, output: Any) -> None:
        """
        Hook that gets called at the end of every batch.

        Args:
            batch: Number of the batch we end.
            output: Output from the batch we processed.
        """

    setattr(ParentProtocol, infer, _infer)
    setattr(ParentProtocol, post, _post)
    setattr(ParentProtocol, f'{name}_epoch_begin', getattr(hooks, f'{name}_epoch_begin')(epoch_begin))
    setattr(ParentProtocol, f'{name}_epoch_end', getattr(hooks, f'{name}_epoch_end')(epoch_end))
    setattr(ParentProtocol, f'{name}_batch_begin', getattr(hooks, f'{name}_batch_begin')(batch_begin))
    setattr(ParentProtocol, f'{name}_batch_end', getattr(hooks, f'{name}_batch_end')(batch_end))

    return ParentProtocol


class Test_LoopMixin(LoopMixin):
    """
    LoopMixin that runs a model over a dataset for evaluation.

    Args:
        infer_fn: Name of the inference method that will be called with each batch of data.
        post_fn: Name of the post-processing method that will be called with the aggregated results of the inference method.
    """

    def __init__(self, infer_fn: str = 'infer', post_fn: str = 'post'):
        self.infer_fn = infer_fn
        self.post_fn = post_fn

    def __set_name__(self, owner: Any, name: str) -> None:
        super().__set_name__(owner, name)
        assert self.name in {'validation', 'test'}, f'{self.__class__.__name__} can only be used for validation or testing'
        self.__protocol__ = dynamic_parent_protocol(self.name, self.infer_fn, self.post_fn)

    def loop(self, dataloader: DataLoader[Any]) -> Optional[Iterator[None]]:
        infer_fn = getattr(self.parent, self.infer_fn)
        post_fn = getattr(self.parent, self.post_fn)

        self.parent.run_hook(type=f'{self.name}_epoch_begin')

        outputs = []
        for batch, data in enumerate(dataloader):
            self.parent.run_hook(type='data_batch', args=(data,))
            self.parent.run_hook(type=f'{self.name}_batch_begin', index=batch + 1, args=(batch + 1,))
            outputs.append(infer_fn(data))
            self.parent.run_hook(type=f'{self.name}_batch_end', index=batch, args=(batch, outputs[-1]))
            yield None

        output = post_fn(outputs)

        self.parent.run_hook(type=f'{self.name}_epoch_end', args=(output,))

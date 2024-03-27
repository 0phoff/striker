from typing import Any, Literal, Optional, Protocol

from torch.utils.data import DataLoader

from ..core import DataMixin

__all__ = ['Property_DataMixin']


class ParentProtocol(Protocol):
    train_loader: Optional[DataLoader[Any]] = None
    """
    Training Dataloader instance

    Note:
        This can be a property if you need to change the dataloader during a run.
        This value can only be **None** if it is not being used in the pipeline.
    """

    validation_loader: Optional[DataLoader[Any]] = None
    """
    Testing Dataloader instance

    Note:
        This can be a property if you need to change the dataloader during a run.
        This value can only be **None** if it is not being used in the pipeline.
    """

    test_loader: Optional[DataLoader[Any]] = None
    """
    Testing Dataloader instance

    Note:
        This can be a property if you need to change the dataloader during a run.
        This value can only be **None** if it is not being used in the pipeline.
    """


class Property_DataMixin(DataMixin, protocol=ParentProtocol):
    """
    TODO
    """

    def __init__(self, cache_dataloaders: bool = False):
        self.cache_dataloaders = cache_dataloaders
        if self.cache_dataloaders:
            self.cache: dict[str, DataLoader[Any]] = {}

    def get_dataloader(self, type: Literal['train', 'validation', 'test']) -> DataLoader[Any]:
        if self.cache_dataloaders and type in self.cache:
            return self.cache[type]

        name = f'{type}_loader'
        dataloader = getattr(self.parent, name, None)
        if dataloader is None:
            raise TypeError(f'{name} is used on {self.parent.__class__.__name__} and cannot be None')

        if self.cache_dataloaders:
            self.cache[type] = dataloader

        return dataloader

    def reset_cache(self) -> None:
        if self.cache_dataloaders:
            self.cache = {}

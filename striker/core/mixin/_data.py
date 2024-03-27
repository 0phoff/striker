from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ..._engine import Engine

from abc import ABC, abstractmethod

from ._mixin import Mixin


class DataMixin(Mixin, ABC):
    """
    This mixin handles where to fetch the different DataLoader objects.

    Implementation Details:
        Simply override the ``get_dataloader`` and return the apropriate dataloader.

    User Details:
        The intended use is to call the ``train``, ``validation`` or ``test`` property each time you want to loop over the data.
        This allows the implementation to optionally change the dataloader each epoch.

        Additionally, you can modify which dataloader is used when you want to access some data.
        This can be usefull if you want to eg. perform testing on the validation data, or train on the testing data.
        The syntax for this is:

        >>> # Perform testing routine on validation data
        >>> mixin_data.test = 'validation'

        >>> # Train a model on the testing data
        >>> mixin_data.train = 'test'
    """

    parent: Engine  # Fix MyPy issues by setting a proper type of self.parent

    @abstractmethod
    def get_dataloader(self, type: Literal['train', 'validation', 'test']) -> DataLoader[Any]:
        pass

    @property
    def train(self) -> DataLoader[Any]:
        type = cast(Literal['train', 'validation', 'test'], getattr(self, 'train_type', 'train'))
        return self.get_dataloader(type)

    @train.setter
    def train(self, type: Literal['train', 'validation', 'test']) -> None:
        self.train_type = type

    @property
    def validation(self) -> DataLoader[Any]:
        type = cast(Literal['train', 'validation', 'test'], getattr(self, 'validation_type', 'validation'))
        return self.get_dataloader(type)

    @validation.setter
    def validation(self, type: Literal['train', 'validation', 'test']) -> None:
        self.validation_type = type

    @property
    def test(self) -> DataLoader[Any]:
        type = cast(Literal['train', 'validation', 'test'], getattr(self, 'test_type', 'test'))
        return self.get_dataloader(type)

    @test.setter
    def test(self, type: Literal['train', 'validation', 'test']) -> None:
        self.test_type = type

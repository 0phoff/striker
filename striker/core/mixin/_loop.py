from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ..._engine import Engine

from abc import ABC, abstractmethod

from ._mixin import Mixin


class LoopMixin(Mixin, ABC):
    """
    This mixin handles a full dataloader processing loop and is used for training, testing, validation, etc.

    Implementation Details:
        In order to create a new LoopMixin variant, simply create a new superclass
        and overwrite the :func:`~striker.core.mixin.LoopMixin.loop` method.
        This method gets a dataloader and should handle processing it.
        You can access the mixin parent throught the ``self.parent`` property.

        Optionally, you can ``yield``  in your loop function.
        This gives back control to the engine temporarily, where it can eg. check if you need to quit.

        Finally, this mixin also has a name attribute, which is set to the name of the mixin in the parent.
        However, if this name starts with 'mixin_', 'loop_' or 'mixin_loop_', it gets stripped from the name.

    User Details:
        To use a LoopMixin, first you need to set the object dataloader property to a valid DataLoader.
        Afterwards, loop over the object to start the processing.

        Note that the object handles processing all the data and simply yields
        so that the MixinParent can momentarily take back control in order to handle eg. quitting.
        You are expected to always loop through the entire mixin object.

        After the loop, you can optionally delete the dataloader property.
    """

    parent: Engine  # Fix MyPy issues by setting a proper type of self.parent
    __dataloader: Optional[DataLoader[Any]] = None
    __looping: bool = False

    @abstractmethod
    def loop(self, dataloader: DataLoader[Any]) -> Optional[Iterator[None]]:
        pass

    def __set_name__(self, owner: Any, name: str) -> None:
        for prefix in ('mixin_', 'loop_'):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        self.name = name

    def __iter__(self) -> Iterator[None]:
        assert self.__dataloader is not None, 'Should set a dataloader before looping through a LoopMixin'
        self.__looping = True
        try:
            yield from (self.loop(self.__dataloader) or ())
        finally:
            self.__looping = False

    @property
    def num_batches(self) -> int:
        return len(self.dataloader) if self.dataloader is not None else 0

    @property
    def dataloader(self) -> Optional[DataLoader[Any]]:
        return self.__dataloader

    @dataloader.setter
    def dataloader(self, value: DataLoader[Any]) -> None:
        assert not self.__looping, 'Cannot set dataloader whilst looping'
        self.__dataloader = value

    @dataloader.deleter
    def dataloader(self) -> None:
        assert not self.__looping, 'Cannot delete dataloader whilst looping'
        self.__dataloader = None

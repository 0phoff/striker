from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, Union

import torch

from .._engine import Engine
from ..core import Plugin, hooks

__all__ = ['DevicePlugin']


class ParentProtocol(Protocol):
    device: Union[torch.device, int, str] = 'cpu'
    """ The device to perform computation on. """


class DevicePlugin(Plugin, protocol=ParentProtocol):
    """
    This plugin will automatically call :meth:`~striker.Engine.to` at startup and also cast any tensors returning from the datasets.

    Note:
        The plugin only works if your datasets return either a tensor, a sequence with tensors or a mappable with tensors.
        Any other type is simply left as is.
    """

    __type_check__: Literal['none', 'log', 'raise'] = 'none'
    parent: Engine  # Fix MyPy issues by setting a proper type of self.parent

    @hooks.engine_begin
    def cast_params(self) -> None:
        self.device = torch.device(getattr(self.parent, 'device', 'cpu'))
        self.parent.to(self.device)

    @hooks.data_batch
    def cast(self, data: Any) -> None:
        if isinstance(data, torch.Tensor):
            data.data = data.data.to(self.device)
        elif isinstance(data, Sequence):
            for sub_data in data:
                self.cast(sub_data)
        elif isinstance(data, Mapping):
            for sub_data in data.values():
                self.cast(sub_data)

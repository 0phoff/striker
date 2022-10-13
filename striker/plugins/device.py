from typing import Protocol, Union, Any, Literal

from collections.abc import Sequence, Mapping
import torch
from ..core import Plugin, hooks
from .._engine import Engine

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
    parent: Engine      # Fix MyPy issues by setting a proper type of self.parent

    @hooks.engine_begin
    def cast_params(self) -> None:
        self.device = torch.device(getattr(self.parent, 'device', 'cpu'))
        self.parent.to(self.device)

    @hooks.data_batch
    def cast(self, input_data: Any) -> None:
        if isinstance(input_data, torch.Tensor):
            input_data.data = input_data.data.to(self.device)
        elif isinstance(input_data, Sequence):
            for sub_data in input_data:
                if isinstance(sub_data, torch.Tensor):
                    sub_data.data = sub_data.data.to(self.device)
        elif isinstance(input_data, Mapping):
            for sub_data in input_data.values():
                if isinstance(sub_data, torch.Tensor):
                    sub_data.data = sub_data.data.to(self.device)

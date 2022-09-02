from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
if TYPE_CHECKING:
    from ._parent import PluginParent

import copy
from .._weakref import OptionalRef
from ..hook import HookParent, HookManager, Hook


class Plugin(HookParent):
    """
    TODO
    """
    __parent: OptionalRef[PluginParent]
    __enabled: bool

    def __new__(cls, *args: Any, **kwargs: Any) -> Plugin:
        obj = cast(Plugin, super().__new__(cls))
        obj.__parent = OptionalRef()
        obj.__enabled = True
        return obj

    def bind(self, parent: PluginParent) -> Plugin:
        new = self.__class__.__new__(self.__class__)

        for name, value in self.__dict__.items():
            # HookParent.__new__ already handles the HookManager and Hooks
            if not isinstance(value, (HookManager, Hook)):
                setattr(new, name, copy.deepcopy(value))

        new.__parent.ref = parent
        return new

    @property
    def parent(self) -> PluginParent:
        parent = self.__parent.ref
        assert parent is not None, 'The parent property can only be used on bound plugins'
        return parent

    @property
    def enabled(self) -> bool:
        return self.__enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.__enabled = value

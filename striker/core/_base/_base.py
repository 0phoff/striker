from __future__ import annotations
from typing import Any, Generic, Optional, TypeVar, cast

import copy
from .._weakref import OptionalRef
from ..hook import HookParent, HookManager, Hook
from ._parent import BaseParent

T = TypeVar('T', bound=BaseParent)


class Base(HookParent, Generic[T]):
    """ Base Class for Plugins and Mixins. """
    __parent_protocol__: Optional[Any] = None
    __parent: OptionalRef[T]
    __enabled: bool

    def __init_subclass__(
        cls,
        /,
        parent_protocol: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if parent_protocol is not None:
            cls.__parent_protocol__ = parent_protocol

    def __new__(cls, *args: Any, **kwargs: Any) -> Base[T]:
        obj = cast(Base[T], super().__new__(cls))
        obj.__parent = OptionalRef()
        obj.__enabled = True
        return obj

    def bind(self, parent: T) -> Base[T]:
        new = self.__class__.__new__(self.__class__)

        for name, value in self.__dict__.items():
            # HookParent.__new__ already handles the HookManager and Hooks
            if not isinstance(value, (HookManager, Hook)):
                setattr(new, name, copy.deepcopy(value))

        new.__parent.ref = parent
        return new

    @property
    def parent(self) -> T:
        parent = self.__parent.ref
        assert parent is not None, 'The parent property can only be used on bound instances'
        return parent

    @property
    def enabled(self) -> bool:
        return self.__enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.__enabled = value
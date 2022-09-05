from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union, cast
if TYPE_CHECKING:
    from ._parent import MixinParent

import copy
from .._weakref import OptionalRef
from ..hook import Hook, HookParent, HookManager

__all__ = ['Mixin']


class Mixin(HookParent):
    """
    Base Mixin class.

    When subclassing this, you can optionally pass `parent_hasattr` and/or `parent_dir` arguments.
    Upon runtime, the the :class:`~striker.core.mixin.MixinManager` will then check that these attributes are present on the parent.

    `parent_hasattr` will use the `hasattr()` function.
    This internally uses `getattr()` and thus has the benefit of respecting any class that has a custom `__getattr__` method.
    The downside is that it actually tries to get the attribute, which might cause problems if you have custom properties that eg. change at every access.
    For mixins on the :class:`striker.Engine`, this means that it will look for attributes in your :class:`~striker.Paramters` object as well.

    `parent_dir` will simply check whether the attributes can be found in `dir(parent)`.
    This has the advantage of not trying to get the actual attribute, but will only check the object itself and not whatever might be in a custom `__getattr__` method.
    For :class:`striker.Engine`, you should use this if you specifically want a method or property on the actual engine class itself, which might take a while to compute when trying to get it.
    """
    __parent_hasattr__: set[str] = set()
    __parent_dir__: set[str] = set()
    __parent: OptionalRef[MixinParent]

    def __init_subclass__(
        cls,
        /,
        parent_hasattr: Optional[Union[set[str], str]] = None,
        parent_dir: Optional[Union[set[str], str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if isinstance(parent_hasattr, str):
            cls.__parent_hasattr__ = {*cls.__parent_hasattr__, parent_hasattr}
        elif parent_hasattr is not None:
            cls.__parent_hasattr__ = {*cls.__parent_hasattr__, *parent_hasattr}

        if isinstance(parent_dir, str):
            cls.__parent_dir__ = {*cls.__parent_dir__, parent_dir}
        elif parent_dir is not None:
            cls.__parent_dir__ = {*cls.__parent_dir__, *parent_dir}

    def __new__(cls, *args: Any, **kwargs: Any) -> Mixin:
        obj = cast(Mixin, super().__new__(cls))
        obj.__parent = OptionalRef()
        return obj

    def bind(self, parent: MixinParent) -> Mixin:
        new = self.__class__.__new__(self.__class__)

        for name, value in self.__dict__.items():
            # HookParent.__new__ already handles the HookManager and Hooks
            if not isinstance(value, (HookManager, Hook)):
                setattr(new, name, copy.deepcopy(value))

        new.__parent.ref = parent
        return new

    @property
    def parent(self) -> MixinParent:
        parent = self.__parent.ref
        assert parent is not None, 'The parent property can only be used on bound mixins'
        return parent

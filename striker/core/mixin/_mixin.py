from __future__ import annotations
from typing import Any, cast

from .._base import Base, BaseParent, BaseManager


class MixinParent(BaseParent):
    """
    TODO
    """
    mixins: MixinManager

    def __new__(cls, *args: Any, **kwargs: Any) -> MixinParent:
        obj = super().__new__(cls)
        obj.mixins = MixinManager(obj)
        return obj


class Mixin(Base[MixinParent]):
    """
    TODO
    """
    def bind(self, parent: MixinParent) -> Mixin:
        return cast(Mixin, super().bind(parent))


class MixinManager(BaseManager[Mixin]):
    def __init__(self, parent: MixinParent):
        mixins: list[Mixin] = []
        for name in dir(parent):
            try:
                value = getattr(parent, name, None)
            except BaseException:
                continue

            if isinstance(value, Mixin):
                bound_mixin = value.bind(parent)
                setattr(parent, name, bound_mixin)
                mixins.append(bound_mixin)

        self._children = tuple(mixins)

from __future__ import annotations

from typing import Any, TypeVar, cast

from .._base import Base, BaseManager, BaseParent

Self = TypeVar('Self', bound='MixinParent')


class MixinParent(BaseParent):
    mixins: MixinManager

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
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
        self._children: dict[str, Mixin] = {}
        for name in dir(parent):
            try:
                value = getattr(parent, name, None)
            except BaseException:
                continue
            if not isinstance(value, Mixin):
                continue

            bound_mixin = value.bind(parent)
            setattr(parent, name, bound_mixin)
            self._children[name] = bound_mixin

        super().__init__()

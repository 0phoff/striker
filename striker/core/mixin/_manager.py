from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Iterator
if TYPE_CHECKING:
    from collections.abc import Sequence
    from ._parent import MixinParent

from ._mixin import Mixin

__all__ = ['MixinManager']


class MixinManager:
    def __init__(self, parent: MixinParent):
        self.__mixins: tuple[Mixin, ...]

        # Bind mixins
        mixins = []
        for name in dir(parent):
            try:
                value = getattr(parent, name, None)
            except BaseException:
                continue

            if isinstance(value, Mixin):
                bound_mixin = value.bind(parent)
                setattr(parent, name, bound_mixin)
                mixins.append(bound_mixin)

        self.__mixins = tuple(mixins)

    def run(
        self,
        /,
        type: Optional[str] = None,
        index: Optional[int] = None,
        args: Sequence[Any] = [],       # NOQA: B006 - Read only argument
        kwargs: dict[str, Any] = {},    # NOQA: B006 - Read only argument
    ) -> None:
        for mixin in self.__mixins:
            mixin.hooks.run(type=type, index=index, args=args, kwargs=kwargs)

    def check(self, types: Optional[set[str]] = None) -> None:
        for mixin in self.__mixins:
            mixin.hooks.check(types)
            for attr in mixin.__parent_hasattr__:
                if not hasattr(mixin.parent, attr):
                    raise AttributeError(f'Mixin "{mixin.__class__.__name__}" requires an attribute "{attr}" on its parent "{mixin.parent.__class__.__name__}"')
            for attr in mixin.__parent_dir__:
                if attr not in dir(mixin.parent):
                    raise AttributeError(f'Mixin "{mixin.__class__.__name__}" requires an attribute "{attr}" on its parent "{mixin.parent.__class__.__name__}"')

    def __len__(self) -> int:
        return len(self.__mixins)

    def __iter__(self) -> Iterator[Mixin]:
        return iter(self.__mixins)

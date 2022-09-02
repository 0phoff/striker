from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Iterable, Any
if TYPE_CHECKING:
    from collections.abc import Sequence
    from ._parent import HookParent

from collections import defaultdict
from itertools import chain
import logging
from .._weakref import OptionalRef
from ._hook import HookDecorator, Hook

__all__ = ['HookManager']
log = logging.getLogger(__name__)


class HookManager:
    def __init__(self, parent: HookParent):
        self.__parent: OptionalRef[HookParent] = OptionalRef(parent)
        self.__types: OptionalRef[Sequence[str]] = OptionalRef(None)
        self.__check: bool = False
        self.__hooks: dict[str, set[Hook]] = defaultdict(set)

        # Bind hooks
        for name in dir(parent):
            try:
                value = getattr(parent, name, None)
            except BaseException:
                continue

            if isinstance(value, Hook):
                bound_hook = value.bind(parent)
                setattr(parent, name, bound_hook)
                self.register(bound_hook)

    def run(
        self,
        /,
        type: Optional[str] = None,
        index: Optional[int] = None,
        args: Sequence[Any] = [],       # NOQA: B006 - Read only argument
        kwargs: dict[str, Any] = {},    # NOQA: B006 - Read only argument
    ) -> None:
        # Get hooks
        hooks: Iterable[Hook]
        if type is None:
            hooks = chain(*self.__hooks.values())
        else:
            hooks = self.__hooks[type]

        # Call hooks
        for hook in hooks:
            if hook.is_active(index=index):
                hook(*args, **kwargs)

    def register(self, hook: Hook) -> None:
        self.__hooks[hook.type].add(hook)

    def check(self, types: Optional[Sequence[str]] = None) -> None:
        self.__check = True
        if types is not None:
            self.__types.ref = types

        hook_type_check = self.__parent.ref.__hook_check__
        if hook_type_check == 'none':
            return

        types = self.__types.ref or self.__parent.ref.__hook_types__
        for hook in chain(*self.__hooks.values()):
            if hook.type not in types:
                if hook_type_check == 'log':
                    log.error('Unregistered hook type "{hook.type}" in "{self.__parent().__class__.__name__}"')
                elif hook_type_check == 'raise':
                    raise TypeError('Unregistered hook type "{hook.type}" in "{self.__parent().__class__.__name__}"')

    def __getattr__(self, name: str) -> HookDecorator:
        hook_type_check = self.__parent.ref.__hook_check__
        if self.__check and hook_type_check != 'none':
            types = self.__types.ref or self.__parent.ref.__hook_types__
            if name not in types:
                if hook_type_check == 'log':
                    log.error('Unregistered hook type "{name}" in "{self.__parent().__class__.__name__}"')
                elif hook_type_check == 'raise':
                    raise TypeError('Unregistered hook type "{name}" in "{self.__parent().__class__.__name__}"')

        return HookDecorator(name, self.__parent.ref)

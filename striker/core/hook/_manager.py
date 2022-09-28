from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Iterable, Any
if TYPE_CHECKING:
    from collections.abc import Sequence
    from ._parent import HookParent

from collections import defaultdict
from contextlib import suppress
from itertools import chain
import logging

with suppress(ImportError):
    from rich import print

from .._protocol import ProtocolChecker
from .._weakref import PersistentWeakRef, OptionalRef
from ._hook import HookDecorator, Hook

log = logging.getLogger(__name__)


class HookManager:
    def __init__(self, parent: HookParent):
        self.__parent: PersistentWeakRef[HookParent] = PersistentWeakRef(parent)
        self.__protocol: OptionalRef[ProtocolChecker] = OptionalRef(None)
        self.__check: bool = False
        self.__hooks: dict[str, set[Hook]] = defaultdict(set)

        # Bind hooks
        for name in dir(parent):
            # Skip known attributes (mainly for protocol, which is a computed property)
            if name in {'__type_check__', '__protocol__', 'protocol'}:
                continue

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

    def check(self, protocol: Optional[ProtocolChecker] = None) -> None:
        self.__check = True
        if protocol is not None:
            self.__protocol.ref = protocol

        hook_type_check = self.__parent.ref.__type_check__
        if hook_type_check == 'none':
            return

        protocol_checker = self.__protocol.ref or self.__parent.ref.protocol
        for hook in chain(*self.__hooks.values()):
            if not protocol_checker.check_hook_type(hook.type):
                if hook_type_check == 'log':
                    log.error(f'Unregistered hook type "{hook.type}" in "{self.__parent.ref.__class__.__name__}"')
                elif hook_type_check == 'raise':
                    print(protocol_checker)
                    raise TypeError(f'Unregistered hook type "{hook.type}" in "{self.__parent.ref.__class__.__name__}"')

    def __getattr__(self, name: str) -> HookDecorator:
        hook_type_check = self.__parent.ref.__type_check__
        if self.__check and hook_type_check != 'none':
            protocol_checker = self.__protocol.ref or self.__parent.ref.protocol
            if not protocol_checker.check_hook_type(name):
                if hook_type_check == 'log':
                    log.error(f'Unregistered hook type "{name}" in <{self.__parent.ref.__class__.__name__}>')
                elif hook_type_check == 'raise':
                    print(protocol_checker)
                    raise TypeError(f'Unregistered hook type "{name}" in <{self.__parent.ref.__class__.__name__}>')

        return HookDecorator(name, self.__parent.ref)

    def __contains__(self, name: str) -> bool:
        protocol_checker = self.__protocol.ref or self.__parent.ref.protocol
        return protocol_checker.check_hook_type(name)

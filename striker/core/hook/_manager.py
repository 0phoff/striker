from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._parent import HookParent

import logging
from collections import defaultdict
from contextlib import suppress
from itertools import chain

with suppress(ImportError):
    from rich import print

from .._protocol import ProtocolChecker
from .._weakref import OptionalRef, PersistentWeakRef
from ._hook import Hook, HookDecorator

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
        args: Sequence[Any] = [],  # NOQA: B006 - Read only argument
        kwargs: dict[str, Any] = {},  # NOQA: B006 - Read only argument
    ) -> Callable[[], None]:
        # Get hooks
        hooks: Iterable[Hook] = chain(*self.__hooks.values()) if type is None else self.__hooks[type]

        # Split hooks
        split_hooks: tuple[list[Hook], list[Hook], list[Hook]] = ([], [], [])
        for hook in hooks:
            if hook.is_active(index=index):
                if hook.early:
                    split_hooks[0].append(hook)
                elif hook.late:
                    split_hooks[2].append(hook)
                else:
                    split_hooks[1].append(hook)

        # Run hook function
        called = 0

        def run_hooks() -> None:
            nonlocal called

            for hook in split_hooks[called]:
                hook(*args, **kwargs)

            called += 1

        return run_hooks

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
                    log.error('Unregistered hook type "%s" in "%s"', hook.type, self.__parent.ref.__class__.__name__)
                elif hook_type_check == 'raise':
                    print(protocol_checker)
                    raise TypeError(f'Unregistered hook type "{hook.type}" in "{self.__parent.ref.__class__.__name__}"')

    def __getattr__(self, name: str) -> HookDecorator:
        hook_type_check = self.__parent.ref.__type_check__
        if self.__check and hook_type_check != 'none':
            protocol_checker = self.__protocol.ref or self.__parent.ref.protocol
            if not protocol_checker.check_hook_type(name):
                if hook_type_check == 'log':
                    log.error('Unregistered hook type "%s" in <%s>', name, self.__parent.ref.__class__.__name__)
                elif hook_type_check == 'raise':
                    print(protocol_checker)
                    raise TypeError(f'Unregistered hook type "{name}" in <{self.__parent.ref.__class__.__name__}>')

        return HookDecorator(name, self.__parent.ref)

    def __contains__(self, name: str) -> bool:
        protocol_checker = self.__protocol.ref or self.__parent.ref.protocol
        return protocol_checker.check_hook_type(name)

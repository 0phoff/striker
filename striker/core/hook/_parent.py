from __future__ import annotations

from typing import Any, Literal, Optional, TypeVar, Union

from .._protocol import ProtocolChecker
from ._manager import HookManager

Self = TypeVar('Self', bound='HookParent')


class HookParent:
    """
    Any class that wants to use hooks needs to subclass :class:`~striker.core.hook.HookParent`.

    Optionally, you can give it a `hook_types` class argument and use :func:`striker.core.hook.HookManager.check <self.hooks.check()>` at runtime
    to make sure only valid hooks are registered.

    Attributes:
        hooks (HookManager): Manager to run the hooks bound on that specific class instance

    Example:
        >>> class ParentProtocol(Protocol):
        ...     @striker.hooks.a
        ...     def hook_a(self) -> None: ...
        ...
        ...     @striker.hooks.b
        ...     def hook_b(self, value: int) -> None: ...
        >>>
        >>> class Parent(striker.core.hook.HookParent, protocol=ParentProtocol):
        ...     @striker.Hooks.a
        ...     def hook_a(self):
        ...         pass
        ...
        ...     @striker.hooks.b[::10]
        ...     def hook_b(self):
        ...         pass
        ...
        ...     @striker.hooks.c[::1]
        ...     def hook_c(self, extra):
        ...         pass
        ...
        ...     def hook_a_bis(self):
        ...         pass
        ...
        ...     def __init__(self):
        ...         # Recommended: Check thath there are only valid hooks
        ...         self.hooks.check()
        ...
        ...     def run(self):
        ...         # Note that we are creating hooks on the fly on the instance and not on the class (self.hooks)
        ...         self.hooks.a[0:10](self.hook_a_bis)
        ...
        ...         # Run Hooks
        ...         self.hooks.run(type='a')
        ...         self.hooks.run(type='b', index=7)
        ...         self.hooks.run(type='c', kwargs={'extra': 123})
    """

    hooks: HookManager
    __type_check__: Literal['none', 'log', 'raise'] = 'log'
    __protocol__: Optional[Union[type, ProtocolChecker]] = None

    def __init_subclass__(cls, /, protocol: Optional[type] = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if protocol is not None:
            cls.__protocol__ = protocol

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """Attah a `hook` HookManager object to the instance."""
        obj = super().__new__(cls)
        obj.hooks = HookManager(obj)
        return obj

    @property
    def protocol(self) -> ProtocolChecker:
        if isinstance(self.__protocol__, ProtocolChecker):
            return self.__protocol__
        return ProtocolChecker().add(self.__class__.__name__, self.__protocol__)

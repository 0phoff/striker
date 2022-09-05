from __future__ import annotations
from typing import Any, Literal, Optional, Union

from ._manager import HookManager

__all__ = ['HookParent']


class HookParent():
    """
    Any class that wants to use hooks needs to subclass :class:`~striker.core.hook.HookParent`.

    Optionally, you can give it a `hook_types` class argument and use :func:`striker.core.hook.HookManager.check <self.hooks.check()>` at runtime
    to make sure only valid hooks are registered.

    Attributes:
        hooks (HookManager): Manager to run the hooks bound on that specific class instance

    Example:
        >>> class Parent(striker.core.hook.HookParent, hook_types=['a', 'b']):
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
    __hook_check__: Literal['none', 'log', 'raise'] = 'log'
    __hook_types__: set[str] = set()

    def __init_subclass__(
        cls,
        /,
        hook_types: Optional[Union[set[str], str]] = None,
    ) -> None:
        if isinstance(hook_types, str):
            cls.__hook_types__ = {*cls.__hook_types__, hook_types}
        elif hook_types is not None:
            cls.__hook_types__ = {*cls.__hook_types__, *hook_types}

    def __new__(cls, *args: Any, **kwargs: Any) -> HookParent:
        """ Attah a `hook` HookManager object to the instance. """
        obj = super().__new__(cls)
        obj.hooks = HookManager(obj)
        return obj

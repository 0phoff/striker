from __future__ import annotations
from typing import Any, cast

import inspect
from .._base import Base, BaseParent, BaseManager


class PluginParent(BaseParent):
    """
    Any class that wants to use plugins needs to subclass :class:`~striker.core.plugin.PluginParent`.

    Attributes:
        plugins (PluginManager): Manager to run the plugins bound to that specific class instance

    Example:
        >>> class Parent(striker.core.plugin.PluginParent):
        ...     plugins = [CustomPlugin1(), CustomPlugin2()]
        ...
        ...     def __init__(self):
        ...         # Recommended: Check thath there are only valid hooks
        ...         self.hooks.check()
        ...
        ...     def run(self):
        ...         # Each plugin should document what hooks they use
        ...         self.plugins.run(type='a')
        ...         self.plugins.run(type='b', index=7)
        ...         self.plugins.run(type='c', kwargs={'extra': 123})
    """
    plugins: PluginManager

    def __new__(cls, *args: Any, **kwargs: Any) -> PluginParent:
        obj = super().__new__(cls)
        obj.plugins = PluginManager(obj)
        return obj


class Plugin(Base[PluginParent]):
    """
    TODO
    """
    def bind(self, parent: PluginParent) -> Plugin:
        return cast(Plugin, super().bind(parent))


class PluginManager(BaseManager[Plugin]):
    def __init__(self, parent: PluginParent):
        self._children = tuple(
            cast(Plugin, plugin.bind(parent))
            for cls in inspect.getmro(parent.__class__)[::-1]
            for plugin in getattr(cls, 'plugins', [])
        )

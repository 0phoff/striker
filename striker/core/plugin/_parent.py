from __future__ import annotations
from typing import Any
from ._manager import PluginManager


class PluginParent:
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

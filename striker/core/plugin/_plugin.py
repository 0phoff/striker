from __future__ import annotations

import inspect
from typing import Any, Optional, TypeVar, cast

from .._base import Base, BaseManager, BaseParent

Self = TypeVar('Self', bound='PluginParent')


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
        ...         self.plugins.check()
        ...
        ...     def run(self):
        ...         # Each plugin should document what hooks they use
        ...         self.plugins.run(type='a')
        ...         self.plugins.run(type='b', index=7)
        ...         self.plugins.run(type='c', kwargs={'extra': 123})
    """

    plugins: PluginManager

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
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
        self._children = {
            cast(str, self.get_name(plugin)): cast(Plugin, plugin.bind(parent))
            for cls in inspect.getmro(parent.__class__)[::-1]
            for plugin in getattr(cls, 'plugins', [])
        }
        super().__init__()

    @staticmethod
    def get_name(item: Any) -> Optional[str]:
        name = getattr(item, '__name__', None)
        if name is None:
            name = getattr(item, '__class__', None)
            if name is not None:
                name = getattr(name, '__name__', None)

        if name is not None:
            return name.lower()
        return None

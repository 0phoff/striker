from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Optional, Any, Union, cast
if TYPE_CHECKING:
    from ._parent import PluginParent

import inspect
from ._plugin import Plugin


class PluginManager:
    __plugins: tuple[Plugin, ...]

    def __init__(self, parent: PluginParent):
        self.__plugins = tuple(
            cast(Plugin, plugin.bind(parent))
            for cls in inspect.getmro(parent.__class__)[::-1]
            for plugin in getattr(cls, 'plugins', [])
        )

    def run(
        self,
        /,
        type: Optional[str] = None,
        index: Optional[int] = None,
        args: Sequence[Any] = [],       # NOQA: B006 - Read only argument
        kwargs: dict[str, Any] = {},    # NOQA: B006 - Read only argument
    ) -> None:
        for plugin in self.__plugins:
            if plugin.enabled:
                plugin.hooks.run(type=type, index=index, args=args, kwargs=kwargs)

    def check(self, types: Optional[Sequence[str]] = None) -> None:
        for plugin in self.__plugins:
            plugin.hooks.check(types)

    def __len__(self) -> int:
        return len(self.__plugins)

    def __getitem__(self, index: Union[int, str]) -> Plugin:
        if isinstance(index, str):
            names = tuple(self.get_name(p) for p in self.__plugins if p is not None)
            index = index.lower()
            if index not in names:
                raise KeyError(f'"{index}" not found in plugins')
            return self.__plugins[names.index(index)]
        else:
            return self.__plugins[index]

    @staticmethod
    def get_name(item: Any) -> Optional[str]:
        name = getattr(item, '__name__', None)
        if name is None:
            name = getattr(item, '__class__', None)
            if name is not None:
                name = getattr(name, '__name__', None)

        if name is not None:
            return name.lower()
        else:
            return None

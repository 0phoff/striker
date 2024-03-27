from __future__ import annotations

from typing import Optional, TypeVar, cast

from ._hook import HookDecorator

Self = TypeVar('Self', bound='HookFactory')


class HookFactory:
    """
    This factory class returns proper PartialHook decorators to turn any method in a hook.

    Example:
        >>> class Engine(striker.Engine):
        ... @striker.hooks.engine_begin
        ... def startup_hook(self):
        ...     # Runs on startup
        ...     pass
        ...
        ... @striker.hooks.train_batch_end[::100]
        ... def batch_end_hook(self):
        ...     # Runs at the end of every 100th batch
        ...     pass
    """

    __instance: Optional[HookFactory] = None

    def __new__(cls: type[Self]) -> Self:
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cast(Self, cls.__instance)

    def __getattr__(self, name: str) -> HookDecorator:
        return HookDecorator(name)

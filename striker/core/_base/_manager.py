from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Optional, Sequence, TypeVar, cast

from .._protocol import ProtocolChecker
from ._base import Base

T = TypeVar('T', bound=Base)  # type: ignore[type-arg]


class BaseManager(Generic[T]):
    _children: dict[str, T]

    def __init__(self) -> None:
        self.__protocol = ProtocolChecker()
        for name, value in self._children.items():
            self.__protocol.add(name, cast(Optional[type], value.__protocol__))

    def run(
        self,
        /,
        type: Optional[str] = None,
        index: Optional[int] = None,
        args: Sequence[Any] = [],  # NOQA: B006 - Read only argument
        kwargs: dict[str, Any] = {},  # NOQA: B006 - Read only argument
    ) -> Callable[[], None]:
        # Split hooks
        run_hooks_children: list[Callable[[], None]] = []
        for child in self._children.values():
            if child.enabled:
                run_hooks_children.append(child.hooks.run(type=type, index=index, args=args, kwargs=kwargs))

        # Run hook function
        def run_hooks() -> None:
            for run_hook in run_hooks_children:
                run_hook()

        return run_hooks

    def check(self, protocol: Optional[ProtocolChecker] = None) -> None:
        for child in self._children.values():
            child.hooks.check(protocol or self.__protocol)

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, index: str) -> T:
        index = index.lower()
        if index not in self._children:
            raise KeyError(f'"{index}" not found in children')
        return self._children[index]

    def __iter__(self) -> Iterable[T]:
        return self._children.values()

    @property
    def protocol(self) -> ProtocolChecker:
        return self.__protocol

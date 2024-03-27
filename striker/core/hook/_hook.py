from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from ._parent import HookParent

import inspect
from functools import update_wrapper
from types import MethodType

HookFunction = Union[MethodType, Callable[..., Any]]


class HookDecorator:
    """
    Partially configured hook decorator.

    This is an internal class and should probably never be used by end users.
    """

    def __init__(self, type: str, parent: Optional[HookParent] = None):
        self.type = type
        self.parent = parent
        self.indices: tuple[slice, ...] = (slice(None, None, 1),)
        self.timing = 0

    def __getitem__(self, indices: Union[int, slice, tuple[Union[int, slice], ...]]) -> HookDecorator:
        if isinstance(indices, (int, slice)):
            indices = (indices,)

        self.indices = tuple(idx if isinstance(idx, slice) else slice(idx, idx + 1) for idx in indices)

        return self

    def __call__(self, fn: Callable[..., None]) -> Hook:
        assert callable(fn), 'hooks should be used as decorators on HookParent methods'
        return Hook(self.type, self.indices, fn, self.parent, self.timing)

    def set_early(self) -> HookDecorator:
        self.timing = -1
        return self

    def set_late(self) -> HookDecorator:
        self.timing = 1
        return self


class Hook:
    """
    Hooks allow you to automatically call methods at a certain point in time.

    You should not need to manually create hooks, but rather use the :class:`striker.Hooks`.
    """

    def __init__(  # noqa: C901 - This is not a complex function, it just contains a lot of checks
        self, type: str, indices: tuple[slice, ...], fn: HookFunction, parent: Optional[HookParent] = None, timing: int = 0, enabled: bool = True
    ) -> None:
        self.type = type
        self.indices = indices
        self.timing = timing
        self.enabled = enabled

        if parent is None:
            self.fn = fn
        else:
            if isinstance(fn, MethodType):
                assert fn.__self__ == parent, 'Should rebind to same parent'
                setattr(fn.__self__, fn.__name__, self)
                self.fn = MethodType(fn.__func__, parent)
            else:
                self.fn = MethodType(fn, parent)
            if hasattr(parent, 'hooks'):
                parent.hooks.register(self)

        update_wrapper(self, self.fn)

        self.__arg_count: Optional[int] = 0 if isinstance(fn, MethodType) else -1
        self.__kwarg_names: Optional[set[str]] = set()
        for param in inspect.signature(fn).parameters.values():
            if param.kind == param.POSITIONAL_ONLY and self.__arg_count is not None:
                self.__arg_count += 1
            elif param.kind == param.KEYWORD_ONLY and self.__kwarg_names is not None:
                self.__kwarg_names.add(param.name)
            elif param.kind == param.POSITIONAL_OR_KEYWORD:
                if self.__arg_count is not None:
                    self.__arg_count += 1
                if self.__kwarg_names is not None:
                    self.__kwarg_names.add(param.name)
            elif param.kind == param.VAR_POSITIONAL:
                self.__arg_count = None
            elif param.kind == param.VAR_KEYWORD:
                self.__kwarg_names = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        # Filter arguments if function takes less
        if self.__arg_count is not None:
            args = args[: self.__arg_count]
        if self.__kwarg_names is not None:
            kwargs = {name: value for name, value in kwargs.values() if name in self.__kwarg_names}
        self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f'<Hook: fn={repr(self.fn)}>'

    def bind(self, parent: HookParent) -> Hook:
        """Bind a hook to a parent instance, so it will be called as a method."""
        return self.__class__(self.type, self.indices, self.fn, parent, self.timing, self.enabled)

    def is_active(self, *, type: Optional[str] = None, index: Optional[int] = None) -> bool:
        """Check if a hook should be run under these circumstances."""
        if not self.enabled:
            return False

        if type is not None and self.type != type:
            return False

        if index is not None:
            return any(
                (slice.start is None or index >= slice.start)
                and (slice.stop is None or index < slice.stop)
                and (slice.step is None or (index - (slice.start or 0)) % slice.step == 0)
                for slice in self.indices
            )

        return True

    def set_early(self) -> None:
        self.timing = -1

    def set_late(self) -> None:
        self.timing = 1

    @property
    def early(self) -> bool:
        return self.timing == -1

    @property
    def late(self) -> bool:
        return self.timing == 1

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Union, Any, cast
if TYPE_CHECKING:
    from ._parent import HookParent

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

    def __getitem__(self, *indices: Union[int, slice]) -> HookDecorator:
        self.indices = tuple(
            idx if isinstance(idx, slice) else slice(idx, idx + 1)
            for idx in indices
        )
        return self

    def __call__(self, fn: Callable[[HookParent], None]) -> Hook:
        assert callable(fn), 'hooks should be used as decorators on HookParent methods'
        return Hook(self.type, self.indices, fn, self.parent)


class Hook:
    """
    Hooks allow you to automatically call methods at a certain point in time.

    You should not need to manually create hooks, but rather use the :class:`striker.Hooks`.
    """
    def __init__(
        self,
        type: str,
        indices: tuple[slice, ...],
        fn: HookFunction,
        parent: Optional[HookParent] = None,
        enabled: bool = True,
    ) -> None:
        self.type = type
        self.indices = indices
        self.enabled = enabled

        if parent is None:
            self.fn = fn
        else:
            self.fn = MethodType(
                cast(Callable[..., Any], fn.__func__ if isinstance(fn, MethodType) else fn),
                parent,
            )
            if hasattr(parent, 'hooks'):
                parent.hooks.register(self)

        update_wrapper(self, self.fn)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f'<Hook: fn={repr(self.fn)}>'

    def bind(self, parent: HookParent) -> Hook:
        """ Bind a hook to a parent instance, so it will be called as a method. """
        return self.__class__(
            self.type,
            self.indices,
            self.fn,
            parent,
            self.enabled,
        )

    def is_active(
        self,
        *,
        type: Optional[str] = None,
        index: Optional[int] = None,
    ) -> bool:
        """ Check if a hook should be run under these circumstances. """
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

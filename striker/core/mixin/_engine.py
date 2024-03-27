from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from ..._engine import Engine

from abc import ABC, abstractmethod
from contextlib import contextmanager

from ._mixin import Mixin


class EngineMixin(Mixin, ABC):
    """
    This mixin handles a full engine train/validation or test run and thus contains the bulk of the engine processing code.

    Implementation Details:
        Simply override the ``__call__`` method and handle processing run however you want.
        Every so often, check whether ``self.quit`` is **True** and stop the run if it is the case (by returning).

        Finally, this mixin also has a name attribute, which is set to the name of the mixin in the parent.
        However, if this name starts with 'mixin_', 'engine_' or 'mixin_engine_', it gets stripped from the name.

    User Details:
        Create a mixin object and run it by calling it.
    """

    parent: Engine  # Fix MyPy issues by setting a proper type of self.parent
    name: str
    __reset_flags: bool = True

    @abstractmethod
    def __call__(self) -> None:
        pass

    def __set_name__(self, owner: Any, name: str) -> None:
        for prefix in ('mixin_', 'engine_'):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        self.name = name

    @property
    def quit(self) -> bool:
        """Check parent __sigint__ and __quit__ flags and reset them."""
        sigint = getattr(self.parent, '__sigint__', None)
        if self.__reset_flags and sigint is not None:
            self.parent.__sigint__ = False

        quit = getattr(self.parent, '__quit__', None)
        if self.__reset_flags and quit is not None:
            self.parent.__quit__ = False

        return bool(sigint) or bool(quit)

    @staticmethod
    @contextmanager
    def reset_quit(reset: bool = True) -> Iterator[None]:
        state = EngineMixin.__reset_flags
        EngineMixin.__reset_flags = reset
        yield
        EngineMixin.__reset_flags = state

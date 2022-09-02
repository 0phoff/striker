from typing import Callable, Generic, TypeVar, Optional, Union, cast
import weakref

__all__ = ['OptionalRef', 'EnsuredWeakRef']
T = TypeVar('T', bound=object)


class OptionalRef(Generic[T]):
    __object: Union[weakref.ref[T], Callable[[], None]]

    def __init__(self, obj: Optional[T] = None):
        if obj is None:
            self.__object = lambda: None
        else:
            self.__object = weakref.ref(obj)

    @property
    def ref(self) -> Optional[T]:
        return self.__object()

    @ref.setter
    def ref(self, value: Optional[T]) -> None:
        if value is None:
            self.__object = lambda: None
        else:
            self.__object = weakref.ref(value)

    def __repr__(self) -> str:
        if isinstance(self.__object, weakref.ref):
            obj = self.__object()
            return f'<OptionalRef at {hex(id(self))}; to "{obj.__class__.__name__}" at {hex(id(obj))}>'
        else:
            return f'<OptionalRef at {hex(id(self))}; to "None">'


class EnsuredWeakRef(Generic[T]):
    def __init__(self, obj: T):
        self.__object = weakref.ref(obj)

    @property
    def ref(self) -> T:
        return cast(T, self.__object())

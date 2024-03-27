import weakref
from typing import Callable, Generic, Optional, TypeVar, Union, cast

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
        return f'<OptionalRef at {hex(id(self))}; to "None">'


class PersistentWeakRef(Generic[T]):
    """
    WeakRef an object for which you now it will always exist.

    The biggest usecase is for storing a weakref to a parent object (child cannot outlive parent).
    This is very similar to a weakref.proxy, but it doesn't raise an error (be carefull) and the actual object is returned through ``.ref``
    """

    def __init__(self, obj: T):
        self.__object = weakref.ref(obj)

    @property
    def ref(self) -> T:
        return cast(T, self.__object())

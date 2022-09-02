from typing import Generic, TypeVar, Optional
import weakref

__all__ = ['OptionalRef']
T = TypeVar('T', bound=object)


class OptionalRef(Generic[T]):
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

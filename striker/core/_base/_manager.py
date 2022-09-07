from __future__ import annotations
from typing import Generic, Iterator, Sequence, Optional, Any, TypeVar, Union

import inspect
from typing import _get_protocol_attrs  # type: ignore[attr-defined]
from ._base import Base

T = TypeVar('T', bound=Base)    # type: ignore[type-arg]


class BaseManager(Generic[T]):
    _children: tuple[T, ...]

    def run(
        self,
        /,
        type: Optional[str] = None,
        index: Optional[int] = None,
        args: Sequence[Any] = [],       # NOQA: B006 - Read only argument
        kwargs: dict[str, Any] = {},    # NOQA: B006 - Read only argument
    ) -> None:
        for child in self._children:
            if child.enabled:
                child.hooks.run(type=type, index=index, args=args, kwargs=kwargs)

    def check(self, types: Optional[set[str]] = None) -> None:
        for child in self._children:
            child.hooks.check(types)
            if child.__parent_protocol__ is not None:
                # Copied from typing._ProtocolMeta.__instancecheck__
                # https://github.com/python/cpython/blob/main/Lib/typing.py#L1889
                missing = {
                    attr: not (hasattr(child.parent, attr) and (not callable(getattr(child.__parent_protocol__, attr, None)) or getattr(child.parent, attr) is not None))
                    for attr in _get_protocol_attrs(child.__parent_protocol__)
                }

                if any(missing.values()):
                    missing_docs = []
                    annotations = getattr(child.__parent_protocol__, '__annotations__', ())
                    for name, miss in missing.items():
                        if not miss:
                            continue

                        # Type Annotation
                        if name in annotations:
                            missing_docs.append(f'  - {name}: {annotations[name]}')
                            continue

                        # Method
                        method = getattr(child.__parent_protocol__, name, None)
                        if method is not None:
                            missing_docs.append(f'  - {name}{inspect.signature(method)}')

                    raise TypeError(
                        f'"{child.parent.__class__.__name__}" is not an instance of "{child.__parent_protocol__.__name__}" ({child.__class__.__name__})\n'
                        + '\n'.join(missing_docs),
                    )

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, index: Union[int, str]) -> T:
        if isinstance(index, str):
            names = tuple(self.get_name(p) for p in self._children if p is not None)
            index = index.lower()
            if index not in names:
                raise KeyError(f'"{index}" not found in children')
            return self._children[names.index(index)]
        else:
            return self._children[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self._children)

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

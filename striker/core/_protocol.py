from __future__ import annotations

import inspect
import logging
import re
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
from typing import (  # type: ignore[attr-defined]
    EXCLUDED_ATTRIBUTES,
    Any,
    Iterable,
    Iterator,
    Optional,
)

import typeguard

with suppress(ImportError):
    from rich import print
    from rich.markup import escape
    from rich.measure import Measurement

from .hook import Hook

log = logging.getLogger(__name__)
Missing = object()
quoted_regex = re.compile(r'^[\'"].*[\'"]$')


def signature_to_string(signature: inspect.Signature) -> str:
    result = []
    render_pos_only_separator = False
    render_kw_only_separator = True
    for param in signature.parameters.values():
        formatted = param_to_string(param)

        kind = param.kind
        if kind == kind.POSITIONAL_ONLY:
            render_pos_only_separator = True
        elif render_pos_only_separator:
            result.append('/')
            render_pos_only_separator = False
        if kind == kind.VAR_POSITIONAL:
            render_kw_only_separator = False
        elif kind == kind.KEYWORD_ONLY and render_kw_only_separator:
            result.append('*')
            render_kw_only_separator = False

        result.append(formatted)

    if render_pos_only_separator:
        result.append('/')

    rendered = '({})'.format(', '.join(result))

    if signature.return_annotation is not inspect._empty:
        anno = inspect.formatannotation(signature.return_annotation)
        if quoted_regex.match(anno):
            anno = anno[1:-1]
        rendered += ' -> {}'.format(anno)

    return rendered


def param_to_string(param: inspect.Parameter) -> str:
    kind = param.kind
    formatted = param.name

    # Add annotation and default value
    if param.annotation is not inspect._empty:
        anno = anno_to_string(param.annotation)
        if quoted_regex.match(anno):
            anno = anno[1:-1]
        formatted = '{}: {}'.format(formatted, anno)

    if param.default is not inspect._empty:
        if param.annotation is not inspect._empty:
            formatted = '{} = {}'.format(formatted, repr(param.default))
        else:
            formatted = '{}={}'.format(formatted, repr(param.default))

    if kind == param.VAR_POSITIONAL:
        formatted = '*' + formatted
    elif kind == param.VAR_KEYWORD:
        formatted = '**' + formatted

    return formatted


def anno_to_string(annotation: Any, base_module: Any = None) -> str:
    # Copied from inspect.formatannotation, but adapted so dict/list show subtypes
    if getattr(annotation, '__module__', None) == 'typing':
        return repr(annotation).replace('typing.', '')

    if isinstance(annotation, type):
        if annotation.__module__ in ('builtins', base_module):
            if hasattr(annotation, '__args__'):
                return repr(annotation).replace('builtins.', '')
            return annotation.__qualname__

        return annotation.__module__ + '.' + annotation.__qualname__

    return repr(annotation)


@dataclass
class TextLine:
    """Printable line of text with rich styles or fallback marker symbols."""

    text: str
    indentation: int = 0
    style: Optional[str] = None
    marker: Optional[str] = None

    def __str__(self) -> str:
        return (' ' * self.indentation) + (self.marker or '') + self.text

    def __rich__(self) -> str:
        text = escape(self.text)
        text = text if self.style is None else f'[{self.style}]{text}[/{self.style}]'
        return (' ' * self.indentation) + text


class ProtocolCheckResult(Enum):
    PASS = auto()  # Type-checking is ok
    UNKOWN = auto()  # Could not perform type-checking or using default value
    FAIL = auto()  # Failed type-checking


@dataclass
class ProtocolAttribute:
    name: str
    doc: Optional[str]

    def get(self, instance: object) -> Any:
        return getattr(instance, self.name, Missing)

    def check(self, obj: Any) -> ProtocolCheckResult:
        raise NotImplementedError('Should be implemented in subtypes')

    @property
    def documentation(self) -> Optional[Iterable[str]]:
        if self.doc is None:
            return None
        return inspect.cleandoc(self.doc).split('\n')


@dataclass
class ProtocolAnnotation(ProtocolAttribute):
    """Wrapper around Protocol Annotations for use in :class:`ProtocolWrapper`."""

    type: type
    default: Any

    def check(self, attr: Any) -> ProtocolCheckResult:
        """Check if attribute has correct instance or if there is a default value."""
        if attr is Missing and self.default is not Missing:
            return ProtocolCheckResult.UNKOWN

        try:
            typeguard.check_type('', attr, self.type)
            return ProtocolCheckResult.PASS
        except TypeError:
            return ProtocolCheckResult.FAIL

    def __str__(self) -> str:
        return self.string()

    def string(self, obj: Any = Missing) -> str:
        if obj is Missing:
            name = self.name
            ttype = self.type
            value = self.default
        else:
            name = self.name
            ttype = type(obj)
            value = Missing

        typename = anno_to_string(ttype)
        if value is Missing:
            return f'{name}: {typename}'
        return f'{name}: {typename} = {value}'


@dataclass
class ProtocolMethod(ProtocolAttribute):
    """Wrapper around Protocol Methods for use in :class:`ProtocolWrapper`."""

    signature: inspect.Signature

    def check(self, func: Any) -> ProtocolCheckResult:
        """Only checks whether the instance has a callable attribute and not if signatures match."""
        if callable(func):
            sig = inspect.signature(getattr(func, '__func__', func))

            # Check return annotation
            if (
                (not isinstance(sig.return_annotation, str) and (sig.return_annotation != sig.empty) and (sig.return_annotation != Any))
                and (
                    not isinstance(self.signature.return_annotation, str)
                    and (self.signature.return_annotation != self.signature.empty)
                    and (self.signature.return_annotation != Any)
                )  # noqa: E501 - This would be ugly on multiple lines
                and (sig.return_annotation != self.signature.return_annotation)
            ):
                return ProtocolCheckResult.UNKOWN

            # Check argcount
            proto_argcount = len(
                [
                    param.name
                    for param in self.signature.parameters.values()
                    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and param.default is param.empty
                ]
            )
            proto_varargs = any(param for param in self.signature.parameters.values() if param.kind == param.VAR_POSITIONAL)

            func_argcount = len(
                [
                    param.name
                    for param in sig.parameters.values()
                    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and param.default is param.empty
                ]
            )
            func_varargs = any(param for param in sig.parameters.values() if param.kind == param.VAR_POSITIONAL)

            if (not proto_varargs and func_argcount > proto_argcount) or (not func_varargs and func_argcount < proto_argcount):
                return ProtocolCheckResult.FAIL

            return ProtocolCheckResult.PASS
        return ProtocolCheckResult.FAIL

    def __str__(self) -> str:
        return self.string()

    def string(self, obj: Any = Missing) -> str:
        if obj is Missing:
            return f'def {self.name}{signature_to_string(self.signature)}'

        signature = inspect.signature(obj)
        return f'def {self.name}{signature_to_string(signature)}'


@dataclass
class ProtocolHook(ProtocolMethod):
    """Wrapper around Protocol Hooks for use in :class:`ProtocolWrapper`."""

    def get(self, instance: object) -> Any:
        raise NotImplementedError()

    def check(self, hook: Hook) -> ProtocolCheckResult:
        """Only checks whether the instance has a Hook attribute and not if signatures match."""
        if isinstance(hook, Hook):
            sig = inspect.signature(getattr(hook.fn, '__func__', hook.fn))

            # Check argcount
            proto_argcount = len(
                [
                    param.name
                    for param in self.signature.parameters.values()
                    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and param.default is param.empty
                ]
            )
            proto_varargs = any(param for param in self.signature.parameters.values() if param.kind == param.VAR_POSITIONAL)

            func_argcount = len(
                [
                    param.name
                    for param in sig.parameters.values()
                    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and param.default is param.empty
                ]
            )

            # We do not care if the function takes less arguments, as this gets filtered in Hook.__call__
            if not proto_varargs and func_argcount > proto_argcount:
                return ProtocolCheckResult.FAIL

            return ProtocolCheckResult.PASS
        return ProtocolCheckResult.FAIL

    def string(self, obj: Any = Missing) -> str:
        if obj is Missing:
            return self.create_string(self.name, self.signature)
        return self.create_string(self.name, inspect.signature(obj), obj.get('__name__', None))

    @staticmethod
    def create_string(type: str, signature: inspect.Signature, name: Optional[str] = None) -> str:
        if name is None:
            return f'@hook.{type}{signature_to_string(signature)}'
        return f'@hook.{type} {name}{signature_to_string(signature)}'


@dataclass
class ProtocolAttributeDoc:
    name: str
    group: str
    docs: Optional[Iterable[str]]

    def __str__(self) -> str:
        return '\n'.join(str(t) for t in self.__str_generator())

    def __rich__(self) -> str:
        return '\n'.join(t.__rich__() for t in self.__str_generator())

    def __str_generator(self) -> Iterator[TextLine]:
        yield TextLine(f'|{self.group}|', style='bright_black i')
        yield TextLine(self.name)
        if self.docs is not None:
            yield TextLine('"""', indentation=2, style='blue')
            for line in self.docs:
                yield TextLine(line, indentation=2, style='blue')
            yield TextLine('"""', indentation=2, style='blue')


class ProtocolWrapper:
    """Transform an actual Protocol class in something more structured for runtime checking."""

    def __init__(self, protocol: type):
        self.annotations = tuple(self.get_annotations(protocol))
        self.methods = tuple(self.get_methods(protocol))
        self.hooks = tuple(self.get_hooks(protocol))

    @staticmethod
    def get_annotations(protocol: type) -> Iterator[ProtocolAnnotation]:
        for base in protocol.__mro__[:-1]:
            if base.__name__ in ('Protocol', 'Generic'):
                continue

            annotations: dict[str, type] = getattr(base, '__annotations__', {})
            for key, value in annotations.items():
                if key not in EXCLUDED_ATTRIBUTES:
                    name = getattr(key, '__name__', key)
                    yield ProtocolAnnotation(name=name, type=value, default=getattr(base, key, Missing), doc=None)

    @staticmethod
    def get_methods(protocol: type) -> Iterator[ProtocolMethod]:
        for base in protocol.__mro__[:-1]:
            if base.__name__ in ('Protocol', 'Generic'):
                continue

            for key, value in base.__dict__.items():
                if key not in EXCLUDED_ATTRIBUTES and callable(value) and not isinstance(value, Hook):
                    yield ProtocolMethod(name=key, signature=inspect.signature(value), doc=value.__doc__)

    @staticmethod
    def get_hooks(protocol: type) -> Iterator[ProtocolHook]:
        for base in protocol.__mro__[:-1]:
            if base.__name__ in ('Protocol', 'Generic'):
                continue

            for key, value in base.__dict__.items():
                if key not in EXCLUDED_ATTRIBUTES and isinstance(value, Hook):
                    yield ProtocolHook(name=value.type, signature=inspect.signature(value), doc=value.__doc__)


class ProtocolChecker:
    """Combines different Protocol classes together and adds runtime checking and pretty printing."""

    def __init__(self) -> None:
        self._protocols: dict[str, ProtocolWrapper] = {}

    def add(self, name: str, protocol: Optional[type]) -> ProtocolChecker:
        if protocol is not None:
            self._protocols[name] = ProtocolWrapper(protocol)
        return self

    def checker(self, instance: Any) -> ProtocolCheckerInstance:
        return ProtocolCheckerInstance(self, instance)

    def check(self, instance: Any) -> None:
        checker_instance = ProtocolCheckerInstance(self, instance)
        type_check = getattr(instance, '__type_check__', 'raise')

        if type_check == 'log' and not checker_instance:
            log.error('<%s> does not implement the Engine Protocol', instance.__class__.__name__)
        elif type_check == 'raise' and not checker_instance:
            print(checker_instance)
            raise TypeError(f'<{instance.__class__.__name__}> does not implement the Engine Protocol')

    def check_hook_type(self, hook_type: str) -> bool:
        hook_types = {hook.name for protocol in self._protocols.values() for hook in protocol.hooks}
        return hook_type in hook_types

    def __add__(self, other: ProtocolChecker) -> ProtocolChecker:
        new = ProtocolChecker()
        new._protocols = {**self._protocols, **other._protocols}
        return new

    def __str__(self) -> str:
        return '\n'.join(str(t) for t in self.__str_generator())

    def __rich_console__(self, console, options):  # type: ignore[no-untyped-def]
        yield from self.__str_generator()

    def __rich_measure__(self, console, options):  # type: ignore[no-untyped-def]
        return Measurement(4 + max(len(name) for name in self._protocols), options.max_width)

    def __str_generator(self) -> Iterator[TextLine]:
        yield TextLine('Protocol:', style='b')

        for name, protocol in self._protocols.items():
            yield TextLine(f'|{name}|', indentation=2, style='bright_black i')
            if not (len(protocol.annotations) or len(protocol.methods) or len(protocol.hooks)):
                yield TextLine('')
                continue

            if len(protocol.annotations):
                yield from (TextLine(str(anno), indentation=4) for anno in protocol.annotations)
                yield TextLine('')

            if len(protocol.methods):
                yield from (TextLine(str(method), indentation=4) for method in protocol.methods)
                yield TextLine('')

            if len(protocol.hooks):
                yield from (TextLine(str(hook), indentation=4) for hook in protocol.hooks)
                yield TextLine('')

    def __getitem__(self, name: str) -> ProtocolAttributeDoc:
        """Prints documentation for various protocol items."""
        for protocol_name, protocol in self._protocols.items():
            for item in chain(protocol.annotations, protocol.methods, protocol.hooks):
                if item.name == name:
                    return ProtocolAttributeDoc(str(item), protocol_name, item.documentation)
        raise AttributeError(f'"{name}" not found in Protocol')


class ProtocolCheckerInstance:
    """On the fly generated objects used for runtime checking of :class:`ProtocolChecker` objects."""

    def __init__(self, protocol: ProtocolChecker, instance: Any):
        self.protocol = protocol
        self.instance = instance

    def __str__(self) -> str:
        return '\n'.join(str(t) for t in self.__str_generator())

    def __rich__(self) -> str:
        return '\n'.join(t.__rich__() for t in self.__str_generator())

    def __rich_console__(self, console, options):  # type: ignore[no-untyped-def]
        yield from self.__str_generator()

    def __rich_measure__(self, console, options):  # type: ignore[no-untyped-def]
        return self.protocol.__rich_measure__(console, options)

    def __str_generator(self) -> Iterator[TextLine]:  # NOQA: C901 - Not sure splitting this function will make it "less" complex.
        yield TextLine('ProtocolChecker:', style='b')

        hook_types = {hook.name: hook for protocol in self.protocol._protocols.values() for hook in protocol.hooks}
        hook_to_proto_name = {hook.name: name for name, protocol in self.protocol._protocols.items() for hook in protocol.hooks}
        registered_hooks: dict[str, list[TextLine]] = defaultdict(list)
        unregistered_hooks = False

        for name in dir(self.instance):
            try:
                value = getattr(self.instance, name, None)
            except BaseException:
                continue
            if not isinstance(value, Hook):
                continue

            signature = inspect.signature(getattr(value.fn, '__func__', value.fn))
            tl = TextLine(ProtocolHook.create_string(value.type, signature, name), indentation=4)

            if value.type in hook_types:
                check = hook_types[value.type].check(value)

                if check == ProtocolCheckResult.PASS:
                    tl.marker = '+ '
                    tl.style = 'green'
                elif check == ProtocolCheckResult.UNKOWN:
                    tl.marker = '~ '
                    tl.style = 'yellow'
                else:
                    tl.marker = '- '
                    tl.style = 'red'

                registered_hooks[hook_to_proto_name[value.type]].append(tl)
            else:
                unregistered_hooks = True
                tl.indentation = 2
                tl.marker = '? '
                tl.style = 'magenta'
                yield tl

        if unregistered_hooks:
            yield TextLine('')

        for name, protocol in self.protocol._protocols.items():
            yield TextLine(f'|{name}|', indentation=2, style='bright_black i')
            if not (len(protocol.annotations) or len(protocol.methods) or len(registered_hooks[name])):
                yield TextLine('')
                continue

            if len(protocol.annotations):
                for anno in protocol.annotations:
                    obj = anno.get(self.instance)
                    check = anno.check(obj)
                    tl = TextLine(anno.string(obj), indentation=4)

                    if check == ProtocolCheckResult.PASS:
                        tl.marker = '+ '
                        tl.style = 'green'
                    elif check == ProtocolCheckResult.UNKOWN:
                        tl.marker = '~ '
                        tl.style = 'yellow'
                    else:
                        tl.marker = '- '
                        tl.style = 'red'

                    yield tl

                yield TextLine('')

            if len(protocol.methods):
                for method in protocol.methods:
                    obj = method.get(self.instance)
                    obj = getattr(obj, '__func__', obj)
                    check = method.check(obj)
                    tl = TextLine(method.string(obj), indentation=4)

                    if check == ProtocolCheckResult.PASS:
                        tl.marker = '+ '
                        tl.style = 'green'
                    elif check == ProtocolCheckResult.UNKOWN:
                        tl.marker = '~ '
                        tl.style = 'yellow'
                    else:
                        tl.marker = '- '
                        tl.style = 'red'

                    yield tl

                yield TextLine('')

            if len(registered_hooks[name]) > 0:
                yield from registered_hooks[name]
                yield TextLine('')

    def __bool__(self) -> bool:
        # Check attributes and methods
        for protocol in self.protocol._protocols.values():
            for item in chain(protocol.annotations, protocol.methods):
                if item.check(item.get(self.instance)) == ProtocolCheckResult.FAIL:
                    return False

        # Check hooks
        hook_types = {hook.name: hook for protocol in self.protocol._protocols.values() for hook in protocol.hooks}
        for name in dir(self.instance):
            try:
                value = getattr(self.instance, name, None)
            except BaseException:
                continue
            if not isinstance(value, Hook):
                continue

            if value.type not in hook_types:
                # Unregistered hook
                return False
            if hook_types[value.type].check(value) == ProtocolCheckResult.FAIL:
                # Wrong signature
                return False

        return True

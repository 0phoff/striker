from __future__ import annotations

import copy
import importlib
import importlib.abc
import importlib.util
import inspect
import logging
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Union, cast

import torch

__all__ = ['Parameters']
log = logging.getLogger(__name__)


class Parameters:
    """
    TODO
    """

    __cast: bool = False
    __init_done: bool = False
    __on_load: Optional[Callable[[Parameters], None]]
    __automatic: dict[str, Any] = {'epoch': 0, 'batch': 0}
    __skip_serialize: list[str] = [
        '_Parameters__cast',
        '_Parameters__init_done',
        '_Parameters__on_load',
        '_Parameters__automatic',
        '_Parameters__no_serialize',
    ]

    def __init__(self, **kwargs: Any):
        for key, value in self.__automatic.items():
            setattr(self, key, value)

        self.__no_serialize: list[str] = []
        for key in kwargs:
            if key.startswith('_'):
                serialize = False
                val = kwargs[key]
                key = key[1:]
            else:
                serialize = True
                val = kwargs[key]

            if not hasattr(self, key):
                setattr(self, key, val)
                if not serialize:
                    self.__no_serialize.append(key)
            else:
                log.error('%s attribute already exists as a Parameter and will not be overwritten', key)

        self.__on_load = None
        self.__init_done = True

    def save(self, filename: Union[Path, str], *keys: str) -> None:
        """
        Serialize all the hyperparameters to a pickle file.

        The network, optimizers and schedulers objects are serialized using their ``state_dict()`` functions.

        Args:
            filename (str or path): File to store the hyperparameters
            *keys (str): Which items to store (Default: store all keys that are not in ``self.__no_serialize``)

        Note:
            This function will first check if the existing attributes have a `state_dict()` function,
            in which case it will use this function to get the values needed to save.

        Warning:
            If you pass keys to save, this function will not check whether these keys are in the ``self.__no_serialize`` list.
        """
        state = {}

        if not len(keys):
            keys = tuple(k for k in vars(self) if k not in self.__no_serialize and k not in self.__skip_serialize)

        for k in keys:
            v = vars(self)[k]
            if hasattr(v, 'state_dict'):
                state[k] = v.state_dict()
            else:
                state[k] = v

        torch.save(state, filename)

    def load(self, filename: Union[Path, str], *keys: str, strict: Optional[bool] = True) -> None:
        """
        Load the hyperparameters from a serialized pickle file.

        Args:
            filename (str or path): File to load the hyperparameters from
            *keys (str): Which items to load (Default: load all keys present in the state file)
            strict (bool): Whether to perform strict loading of the state_dicts

        Note:
            This function will first check if the existing attributes have a `load_state_dict()` function,
            in which case it will use this function with the saved state to restore the values.
            The `load_state_dict()` function will first be called with both the serialized value and the `strict` argument as a keyword argument.
            If that fails because of a TypeError, it is called with only the serialized value.
            This means that you will still get an error if the strict rule is not being followed,
            but functions that have a `load_state_dict()` function without `strict` argument can be loaded as well.

        Warning:
            If you save a parameter with a `state_dict()`, but then load it into a :class:`~striker.Parameters` object without that parameter,
            the state dictionary itself will be loaded in the new object instead of using the `load_state_dict()` function.
        """
        state = torch.load(filename, 'cpu')

        if not len(keys):
            keys = tuple(k for k in state if k not in self.__skip_serialize)

        for k in keys:
            if k not in state:
                log.error('Key "%s" is not present in loaded state from "%s"', k, filename)

            v = state[k]
            current = getattr(self, k, None)

            if hasattr(current, 'load_state_dict'):
                try:
                    current.load_state_dict(v, strict=strict)  # type: ignore[union-attr]
                except TypeError:
                    current.load_state_dict(v)  # type: ignore[union-attr]
            else:
                setattr(self, k, v)

        if self.__on_load is not None:
            log.debug('Running load hook')
            self.__on_load(self)

    def reset(self) -> None:
        """Resets automatic variables epoch and batch"""
        for key, value in self.__automatic.items():
            setattr(self, key, value)

    def setup_load_hook(self, func: Callable[[Parameters], None]) -> None:
        """
        This method sets up a function that gets called after loading in weights.
        You can use this function to automatically adapt certain parameters, depending on the values that were loaded.

        Args:
            func (callable): Function that gets called with the parameter object as a single argument
        """
        assert callable(func), 'on_load should be callable: `on_load(param: Parameters) -> None`'
        self.__on_load = func

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """Recursively drill down into objects stored on Parameters and get a value.
        This item will recursively use ``getattr` to get an item from this object, and return the default value otherwise.

        If one of the intermediate objects is a dictionary we use ``obj[attr]``.
        If it is a list and the current attribute is a digit, we use ``obj[int(attr)]``.

        Args:
            name (string): Keys to get, separated by dots
            default (optional): Default value to return, if no value was found
        """
        obj = self
        for attr in name.split('.'):
            try:
                if isinstance(obj, dict):
                    obj = obj[attr]
                elif isinstance(obj, Sequence) and attr.isdigit():
                    obj = obj[int(attr)]
                else:
                    obj = getattr(obj, attr)
            except (KeyError, AttributeError, IndexError):
                return default

        return obj

    def keys(self) -> list[str]:
        """Returns the attributes of your Parameters object, similar to a python dictionary."""
        return sorted(k for k in self.__dict__ if not k.startswith('_Parameters_'))

    def values(self) -> Iterable[Any]:
        """Returns the attribute values of your Parameters object, similar to a python dictionary."""
        return (getattr(self, k) for k in self.keys())

    def items(self) -> Iterable[tuple[str, Any]]:
        """Returns the attribute keys and values of your Parameters object, similar to a python dictionary."""
        return ((k, getattr(self, k)) for k in self.keys())

    def __getattr__(self, item: str) -> Any:
        """Allow to fetch items with the underscore."""
        if item[0] == '_' and item[1:] in self.__no_serialize:  # NOQA: SIM106 - It makes no sense to handle error first
            return getattr(self, item[1:])
        raise AttributeError(f"'Parameters' object has no attribute '{item}'")

    def __setattr__(self, item: str, value: Any) -> None:
        """
        Store extra variables in this container class.

        This custom function allows to store objects after creation and mark whether are not you want to serialize them,
        by prefixing them with an underscore.
        """
        if item in self.__dict__ or not self.__init_done:
            super().__setattr__(item, value)
        elif item[0] == '_':
            if item[1:] not in self.__dict__:
                self.__no_serialize.append(item[1:])
            elif item[1:] not in self.__no_serialize:
                raise AttributeError(f'{item[1:]} already stored in this object as serializable value!')
            super().__setattr__(item[1:], value)
        else:
            super().__setattr__(item, value)

    def __repr__(self) -> str:
        """
        Print all values stored in the object as repr.

        Objects that will not be serialized are marked with an asterisk.
        """
        s = f'{self.__class__.__name__}('
        for k in sorted(self.__dict__.keys()):
            if k.startswith('_Parameters__'):
                continue

            val = self.__dict__[k]
            valrepr = repr(val)
            if '\n' in valrepr:
                valrepr = valrepr.replace('\n', '\n    ')
            if k in self.__no_serialize:
                k += '*'

            s += f'\n  {k} = {valrepr}'

        return s + '\n)'

    def __str__(self) -> str:
        """
        Print all values stored in the object as string.

        Objects that will not be serialized are marked with an asterisk.
        """
        s = f'{self.__class__.__name__}('
        for k in sorted(self.__dict__.keys()):
            if k.startswith('_Parameters__'):
                continue

            val = self.__dict__[k]
            valrepr = str(val)
            if '\n' in valrepr:
                valrepr = getattr(val, '__name__', val.__class__.__name__)
            if k in self.__no_serialize:
                k += '*'

            s += f'\n  {k} = {valrepr}'

        return s + '\n)'

    def __add__(self, other: Parameters) -> Parameters:
        """
        Add 2 Parameters together.

        This function first creates a shallow copy of the first `self` argument
        and then loops through the items in the `other` argument and adds those parameters
        if they are not already available in the new Parameters object.

        Waring:
            We only take shallow copies when adding hyperparameters together,
            so beware if you modify objects from one hyperparameter object after adding it to another.

        Note:
            When adding Parameters objects together,
            we keep the automatic variables (epoch, batch) from the first object.
            Optionally, you can reset these variables by calling the :func:`~striker.Parameters.reset()` method.
        """
        if not isinstance(other, Parameters):
            raise NotImplementedError('Can only add 2 Hyperparameters objects together')

        new = copy.copy(self)
        return new.__iadd__(other)

    def __iadd__(self, other: Parameters) -> Parameters:
        # Small performance boost by not deepcopying self.
        if not isinstance(other, Parameters):
            raise NotImplementedError('Can only add 2 Hyperparameters objects together')

        for key in other:
            if not hasattr(self, key):
                nkey = f'_{key}' if key in other.__no_serialize else key
                setattr(self, nkey, getattr(other, key))
            elif key not in Parameters.__automatic:
                log.warning('"%s" is available in both Parameters, keeping first', key)

        return self

    def __copy__(self) -> Parameters:
        new = self.__class__()
        for key, value in self.__dict__.items():
            setattr(new, key, value)
        return new

    def __contains__(self, value: str) -> bool:
        return value in self.keys()

    def __iter__(self) -> Iterator[str]:
        """Return an iterator of :func:`~striker.Parameters.keys()`, so we can loop over this object like a python dictionary."""
        return iter(self.keys())

    @classmethod
    def from_file(cls, filename: Union[Path, str], variable: str = 'params', **kwargs: Any) -> Parameters:
        """
        Create a Parameters object from a dictionary in an external configuration file.

        This function will import a file by its path and extract a variable to use as Parameters.
        The main goal of this class is to enable *"python as a config"*.
        This means that you can pass in a path to a python file, and the training code will automatically load the Parameters from this file.

        Args:
            path (str or path-like object): Path to the configuration python file
            variable (str, optional): Variable to extract from the configuration file; Default **'params'**
            **kwargs (dict, optional): Extra parameters that are passed to the extracted variable if it is a callable object

        Note:
            The extracted variable can be one of the following:

            - ``callable``: The object will be called with the optional kwargs and should return a :class:`~striker.Parameters`
            - :class:`striker.Parameters`: This object will simply be returned

        Note:
            If the ``path`` argument is a relative path,
            we first try to resolve it against the directory from where the python code is being executed.
            If we cannot find the file from there,
            we try to resolve from the directory of the python file that called this method.

        Examples:
            >>> # config.py file
            >>> params = striker.Parameters(
            ...     network = ln.engine.YoloV2(20), # Example network from Lightnet
            ...     lr = 0.001,                     # This value will be saved with the params
            ...     _batch_size = 8,                # _ means batch_size will not be serialized
            ... )

            >>> # Main training/testing file
            >>> params = ln.engine.Parameters.from_file('config.py')
            >>> print(params)
            Parameters(
               batch_size* = 8
               lr = 0.001
               network = YoloV2
            )

            By default, this function will look for a 'params' variable in your file,
            but you can change that by passing a different value to the ``variable`` argument.

            >>> # config.py file
            >>> my_custom_params = ln.engine.Parameters(...)

            >>> # Main training/testing file
            >>> params = ln.engine.Parameters.from_file('config.py', variable='my_custom_params')
            >>> print(params)
            Parameters(...)

            Finally, power users may want to be able to pass arguments to the config file!
            Just make the 'params' argument callable in your config, and you can pass in keyword arguments.

            >>> # config.py file
            >>> def params(a, b):
            ...     # Either return a dict or an HP object
            ...     return {
            ...         'a': a,
            ...         'b': b,
            ...     }

            >>> # Main training/testing file
            >>> params = ln.engine.Parameters.from_file('config.py', a=666, b='value_B')
            >>> print(params)
            Parameters(
               a = 666
               b = value_B
            )
        """
        params = load_external(Path(filename), variable)

        if callable(params):
            if cls.__cast:
                annotations = getattr(params, '__annotations__', {})
                signature = inspect.signature(params)
                for name, cast_type in annotations.items():
                    param = signature.parameters[name]
                    if param.kind == param.VAR_KEYWORD:
                        if cast_type not in (str, Any):
                            raise TypeError(f'Cannot convert **{param.name} arguments to {cast_type}')
                        continue

                    if name in kwargs and isinstance(kwargs[name], str):
                        try:
                            kwargs[name] = cast_arg(kwargs[name], cast_type)
                        except BaseException as err:
                            raise TypeError(f'Could not convert "{kwargs[name]}" to "{cast_type}"') from err

            params = params(**kwargs)
            if not isinstance(params, cls):
                raise TypeError(f'Configuration function did not return a Parameters object [{type(params)}]')

        return params

    @staticmethod
    @contextmanager
    def enable_cast(enabled: bool = True) -> Iterator[None]:
        state = Parameters.__cast
        Parameters.__cast = enabled
        yield
        Parameters.__cast = state


def load_external(filename: Path, variable: str) -> Union[Parameters, Callable[..., Parameters]]:
    tried = [str(filename.resolve())]
    if not (filename.is_file() or filename.is_absolute()):
        for stackframe in inspect.stack()[2:]:
            stackfile = Path(stackframe.filename).parent.joinpath(filename)
            tried.append(str(stackfile.resolve()))
            if stackfile.is_file():
                filename = stackfile
                break

    if not filename.is_file():
        raise FileNotFoundError(f'Could not find file, tried following paths: {tried}')

    try:
        path_import = re.sub(r'[^a-zA-Z0-9]', '_', str(filename))
        spec = cast(importlib.machinery.ModuleSpec, importlib.util.spec_from_file_location(f'striker.cfg.{path_import}', filename))
        cfg = importlib.util.module_from_spec(spec)
        cast(importlib.abc.Loader, spec.loader).exec_module(cfg)
    except AttributeError as err:
        raise ImportError(f'Failed to import the file [{filename}]. Are you sure it is a valid python file?') from err

    try:
        params = getattr(cfg, variable)
    except AttributeError as err:
        raise AttributeError(f'Configuration variable [{variable}] not found in file [{filename}]') from err

    if not callable(params) and not isinstance(params, Parameters):
        raise TypeError(f'Configuration variable "{variable}" should be a Parameters object [{type(params)}]')

    return params


def cast_arg(param: str, cast_type: type) -> Any:  # NOQA: C901 - This function is not very complex, but has lots of checks
    # String: return as is
    if cast_type == str:
        return param

    # Any: simply return string
    if cast_type == Any:
        log.warning('type is "Any", leaving input argument as string')
        return param

    # Numbers: cast number
    if cast_type in (int, float, complex):
        return cast_type(param)

    # Boolean: check for a few literal strings that mean true
    if cast_type == bool:
        if param.lower() in ('true', 't', 'yes', 'y', '1'):
            return True
        if param.lower() in ('false', 'f', 'no', 'n', '0'):
            return False
        raise ValueError(f'Could not convert "{param}" to bool')

    # None: check that string is 'none'
    if cast_type == type(None):  # NOQA: E721 - isinstance with a type causes a TypeError
        if param.lower() != 'none':
            raise ValueError(f'Expected string "none", but got "{param}"')
        return None

    # Slice: Parse x:y:z (we assume ints for the slice parts)
    if cast_type == slice:
        if ':' not in param:
            raise ValueError(f'Expected slice notation with ":", but got "{param}"')

        try:
            return slice(*(int(v) if len(v) > 0 else None for v in param.split(':')))
        except BaseException as err:
            raise ValueError(f'Could not convert "{param}" to slice') from err

    origin = getattr(cast_type, '__origin__', None)

    # Literal: check if valid and return as is
    if origin == Literal:
        values = getattr(cast_type, '__args__', [])

        if not all(isinstance(v, str) for v in values):
            log.warning('Cannot check Literal type with non-string values: %s', values)
            return param

        assert param in values, f'"{param}" is not one of the following literal strings: {values}'
        return param

    # Union/Optional: Try casting to any of the underlying types
    if origin == Union:
        # get subtypes (with str type at the end if it is in there)
        subtypes = getattr(cast_type, '__args__', [str])
        subtypes = sorted(subtypes, key=lambda t: t == str)

        for subtype in subtypes:
            try:
                return cast_arg(param, subtype)
            except BaseException:
                continue

        raise ValueError(f'Could not cast "{param}" to "{cast_type}"')

    # Tuple:
    #  - Multiple values (tuple[int, str, float]): cast to matching subtype
    #  - Ellipsis (tuple[int, ...]): cast to first subtype -> Handled by "Sequence" code below
    if inspect.isclass(origin) and issubclass(origin, tuple):
        subtypes = getattr(cast_type, '__args__', [])
        if len(subtypes) and subtypes[-1] != ...:
            params = param.split(',')
            assert len(subtypes) == len(params), f'Parameter does not contain the right amount of values for tuple type "{cast_type}".'
            return origin(cast_arg(p.strip(), subtype) for p, subtype in zip(params, subtypes))

    # Sequence/set: split on "," and cast trimmed substrings to subtype
    if inspect.isclass(origin) and (issubclass(origin, Sequence) or issubclass(origin, set)):
        subtypes = getattr(cast_type, '__args__', [])
        if len(subtypes):
            subtype = subtypes[0]
        else:
            subtype = str
            log.warning('Sequence "%s" has no subtype, returning a Sequence[str]', cast_type)

        return origin(cast_arg(s.strip(), subtype) for s in param.split(','))  # type: ignore[call-arg]

    # Mapping: split on "," and then on ":". Cast results to basetype
    if inspect.isclass(origin) and issubclass(origin, Mapping):
        subtypes = getattr(cast_type, '__args__', [])
        if len(subtypes) > 1:
            keytype = subtypes[0]
            valuetype = subtypes[1]
        elif len(subtypes):
            keytype = subtypes[0]
            valuetype = str
            log.warning('Mapping "%s" has no value subtype, returning a Mapping[%s, str]', cast_type, keytype)
        else:
            keytype = str
            valuetype = str
            log.warning('Mapping "%s" has no subtypes, returning a Mapping[str, str]', cast_type)

        values = []
        for keyvalue in param.split(','):
            key, value = keyvalue.split(':')
            values.append((cast_arg(key.strip(), keytype), cast_arg(value.strip(), valuetype)))

        return origin(values)  # type: ignore[call-arg]

    # Any other
    log.error('Unkown type "%s", cannot convert! Simply returning string', cast_type)
    return param

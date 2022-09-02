from __future__ import annotations
from typing import Any, Iterator, Union, Optional, Iterable, cast

import copy
from collections.abc import Sequence
import importlib
import inspect
import logging
from pathlib import Path
import re
import torch

__all__ = ['Parameters']
log = logging.getLogger(__name__)


class Parameters:
    """
    TODO
    """
    __init_done = False
    __automatic = {
        'epoch': 0,
        'batch': 0,
    }

    def __init__(self, **kwargs: Any):
        for key, value in self.__automatic.items():
            setattr(self, key, value)

        self.__no_serialize__: list[str] = []
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
                    self.__no_serialize__.append(key)
            else:
                log.error('%s attribute already exists as a Parameter and will not be overwritten', key)

        self.__init_done = True

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
        filename = Path(filename)
        tried = []
        if not (filename.is_file() or filename.is_absolute()):
            tried.append(str(filename))
            filename = Path(inspect.stack()[1].filename).parent.joinpath(filename)
        if not filename.is_file():
            tried.append(str(filename))
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

        if callable(params):
            params = params(**kwargs)
            if not isinstance(params, cls):
                raise TypeError(f'Configuration function did not return a Parameters object [{type(params)}]')
        elif not isinstance(params, cls):
            raise TypeError(f'Configuration variable "{variable}" should be a Parameters object [{type(params)}]')

        return params

    def save(self, filename: Union[Path, str]) -> None:
        """
        Serialize all the hyperparameters to a pickle file.

        The network, optimizers and schedulers objects are serialized using their ``state_dict()`` functions.

        Args:
            filename (str or path): File to store the hyperparameters

        Note:
            This function will first check if the existing attributes have a `state_dict()` function,
            in which case it will use this function to get the values needed to save.
        """
        state = {}

        for k, v in vars(self).items():
            if k not in self.__no_serialize__:
                if hasattr(v, 'state_dict'):
                    state[k] = v.state_dict()
                else:
                    state[k] = v

        torch.save(state, filename)

    def load(self, filename: Union[Path, str], strict: Optional[bool] = True) -> None:
        """
        Load the hyperparameters from a serialized pickle file.

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

        for k, v in state.items():
            current = getattr(self, k, None)
            if hasattr(current, 'load_state_dict'):
                try:
                    current.load_state_dict(v, strict=strict)   # type: ignore
                except TypeError:
                    current.load_state_dict(v)                  # type: ignore
            else:
                setattr(self, k, v)

    def to(self, device: Union[torch.device, str]) -> None:
        """
        Cast the parameters from the network, optimizers and schedulers to a given device.

        This function will go through all the class attributes and check if they have a `to()` function, which it will call with the device.

        Args:
            device (torch.device or string): Device to cast parameters

        Note:
            PyTorch optimizers and the ReduceLROnPlateau classes do not have a `to()` function implemented.
            For these objects, this function will go through all their necessary attributes and cast the tensors to the right device.
        """
        def manual_to(obj: dict[str, Any], device: Union[str, torch.device]) -> None:
            for param in obj.values():
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
                elif isinstance(param, dict):
                    manual_to(param, device)

        for value in self.__dict__.values():
            to = getattr(value, 'to', None)
            if callable(to):
                to(device)
            elif isinstance(value, torch.optim.Optimizer):
                manual_to(value.state, device)
            elif isinstance(value, (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                manual_to(value.__dict__, device)

    def reset(self) -> None:
        """ Resets automatic variables epoch and batch """
        for key, value in self.__automatic.items():
            setattr(self, key, value)

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """ Recursively drill down into objects stored on Parameters and get a value.
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

    def __getattr__(self, item: str) -> Any:
        """ Allow to fetch items with the underscore. """
        if item[0] == '_' and item[1:] in self.__no_serialize__:        # NOQA: SIM106 - It makes no sens to handle error first
            return getattr(self, item[1:])
        else:
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
                self.__no_serialize__.append(item[1:])
            elif item[1:] not in self.__no_serialize__:
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
            if k in self.__no_serialize__:
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
            if k in self.__no_serialize__:
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
                nkey = f'_{key}' if key in other.__no_serialize__ else key
                setattr(self, nkey, getattr(other, key))
            elif key not in Parameters.__automatic:
                log.warning('"%s" is available in both Parameters, keeping first', key)

        return self

    def __copy__(self) -> Parameters:
        new = self.__class__()
        for key, value in self.__dict__.items():
            setattr(new, key, value)
        return new

    def keys(self) -> list[str]:
        """ Returns the attributes of your Parameters object, similar to a python dictionary. """
        return sorted(k for k in self.__dict__ if not k.startswith('_Parameters_'))

    def values(self) -> Iterable[Any]:
        """ Returns the attribute values of your Parameters object, similar to a python dictionary. """
        return (getattr(self, k) for k in self.keys())

    def items(self) -> Iterable[tuple[str, Any]]:
        """ Returns the attribute keys and values of your Parameters object, similar to a python dictionary. """
        return ((k, getattr(self, k)) for k in self.keys())

    def __iter__(self) -> Iterator[str]:
        """ Return an iterator of :func:`~striker.Parameters.keys()`, so we can loop over this object like a python dictionary. """
        return iter(self.keys())

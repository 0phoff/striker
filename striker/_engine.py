import logging
import signal
from collections.abc import Sequence
from contextlib import contextmanager
from types import FrameType
from typing import Any, Iterator, Literal, Optional, Protocol, TypeVar

import torch

from ._parameter import Parameters
from .core import DataMixin, EngineMixin, HookParent, LoopMixin, MixinParent, PluginParent, hooks
from .mixins.data_property import Property_DataMixin
from .mixins.engine_train import Train_EngineMixin
from .mixins.loop_batchtrain import BatchTrain_LoopMixin

__all__ = ['Engine']
log = logging.getLogger(__name__)
T = TypeVar('T')


class ParentProtocol(Protocol):
    @hooks.engine_init
    def engine_init(self) -> None:
        """
        TODO
        """
        ...

    @hooks.engine_begin
    def engine_begin(self, entry: Literal['train', 'test', 'validation']) -> None:
        """
        TODO
        """
        ...

    @hooks.engine_end
    def engine_end(self, entry: Literal['train', 'test', 'validation']) -> None:
        """
        TODO
        """
        ...


class Engine(HookParent, PluginParent, MixinParent, protocol=ParentProtocol):
    """
    TODO
    """

    __type_check__: Literal['none', 'log', 'raise'] = 'raise'
    __entry__: Optional[Literal['train', 'test']] = None
    __init_done: bool = False

    # Mixins
    mixin_data: DataMixin = Property_DataMixin()
    mixin_engine_train: EngineMixin = Train_EngineMixin()
    mixin_loop_train: LoopMixin = BatchTrain_LoopMixin()
    mixin_engine_test: Optional[EngineMixin] = None

    def __init__(self, params: Parameters, **kwargs: Any):
        # Create 1 protocol object
        self.__protocol__ = self.protocol + self.mixins.protocol + self.plugins.protocol

        # Store parameters
        self.params = params

        # Quit handling
        self.__sigint__ = False
        self.__quit__ = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.__interrupt)

        # Set attributes
        for key in kwargs:
            if not hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                log.warning('%s attribute already exists on engine.', key)

        self.__init_done = True
        self.run_hook(type='engine_init')

    def train(self) -> None:
        self.__check()
        self.__entry__ = 'train'

        self.run_hook(type='engine_begin', args=[self.__entry__])
        try:
            self.mixin_engine_train()
        finally:
            self.run_hook(type='engine_end', args=[self.__entry__])

    def test(self, dataset: Literal['train', 'validation', 'test'] = 'test') -> None:
        self.__check()
        assert self.mixin_engine_test is not None, 'EngineMixin required for test'
        self.__entry__ = 'test'

        self.run_hook(type='engine_begin', args=[self.__entry__])
        try:
            self.mixin_data.test = dataset  # type: ignore[assignment]
            self.mixin_engine_test()
        finally:
            self.run_hook(type='engine_end', args=[self.__entry__])

    def run_hook(
        self,
        /,
        type: Optional[str] = None,
        index: Optional[int] = None,
        args: Sequence[Any] = [],  # NOQA: B006 - Read only argument
        kwargs: dict[str, Any] = {},  # NOQA: B006 - Read only argument
    ) -> None:
        """This method runs all hooks in mixins, plugins and on the engine itself."""
        # Get run functions
        run_hooks = self.hooks.run(type=type, index=index, args=args, kwargs=kwargs)
        run_plugins = self.plugins.run(type=type, index=index, args=args, kwargs=kwargs)
        run_mixins = self.mixins.run(type=type, index=index, args=args, kwargs=kwargs)

        # Run 3 times (early, normal, late)
        for _ in range(3):
            run_hooks()
            run_plugins()
            run_mixins()

    def quit(self) -> None:
        if not self.__quit__:
            log.debug('Quit function called. Waiting for gracefull exit')
            self.__quit__ = True

    def to(self, *args: Any, **kwargs: Any) -> None:
        """
        Note:
            PyTorch optimizers and the ReduceLROnPlateau classes do not have a `to()` function implemented.
            For these objects, this function will go through all their necessary attributes and cast the tensors to the right device.
        """

        def manual_to(obj: dict[Any, Any]) -> None:
            for param in obj.values():
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(*args, **kwargs)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(*args, **kwargs)
                elif isinstance(param, dict):
                    manual_to(param)

        for _name, value in self.__loop_values(
            torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            if isinstance(value, torch.nn.Module):
                value.to(*args, **kwargs)
            elif isinstance(value, torch.optim.Optimizer):
                manual_to(value.state)
            elif isinstance(value, (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                manual_to(value.__dict__)

    @contextmanager
    def eval(self) -> Iterator[None]:
        store = {}
        for name, module in self.__loop_values(torch.nn.Module):
            store[name] = module.training
            module.train(False)

        try:
            with getattr(torch, 'inference_mode', torch.no_grad)():
                yield
        finally:
            for name, module in self.__loop_values(torch.nn.Module):
                module.train(store.get(name, True))

    def __check(self) -> None:
        self.mixins.check(self.protocol)
        self.plugins.check(self.protocol)
        self.protocol.check(self)

    def __interrupt(self, signal: int, frame: Optional[FrameType]) -> None:
        if not self.__sigint__:
            log.debug('SIGINT/SIGTERM caught. Waiting for gracefull exit')
            self.__sigint__ = True

    def __loop_values(self, *types: type[T]) -> Iterator[tuple[str, T]]:
        def loop(item: dict[str, Any]) -> Iterator[tuple[str, Any]]:
            for name, value in item.items():
                if isinstance(value, Parameters):
                    for subname, subvalue in loop(value.__dict__):
                        yield (f'{name}.{subname}', subvalue)
                elif len(types) == 0 or isinstance(value, types):
                    yield (name, value)

        yield from loop(self.__dict__)

    def __getattr__(self, name: str) -> Any:
        if name == 'params':
            raise AttributeError('params not yet available on Engine')

        try:
            return getattr(self.params, name)
        except AttributeError as err:
            raise AttributeError(f'{name} attribute does not exist') from err

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__init_done and name not in dir(self) and hasattr(self.params, name):
            setattr(self.params, name, value)
        else:
            super().__setattr__(name, value)

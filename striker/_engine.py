from typing import Iterator, Literal, Optional, Any, Protocol, TypeVar

from contextlib import contextmanager
from collections.abc import Sequence
import logging
import signal
import torch
from types import FrameType

from ._parameter import Parameters
from .core import HookParent, PluginParent, MixinParent, hooks
from .core import EngineMixin, LoopMixin
from .mixins.engine_train import TrainEngineMixin
from .mixins.loop_batchtrain import BatchTrainLoopMixin

__all__ = ['Engine']
log = logging.getLogger(__name__)
T = TypeVar('T')


class ParentProtocol(Protocol):
    @hooks.engine_start
    def engine_start(self, entry: Literal['train', 'test', 'validation']) -> None:
        ...

    @hooks.engine_end
    def engine_end(self, entry: Literal['train', 'test', 'validation']) -> None:
        ...


class Engine(
    HookParent, PluginParent, MixinParent,
    protocol=ParentProtocol,
):
    """
    TODO
    """
    __type_check__: Literal['none', 'log', 'raise'] = 'raise'
    __entry__: Optional[Literal['train', 'test', 'validation']] = None
    __init_done: bool = False

    # Mixins
    mixin_engine_train: EngineMixin = TrainEngineMixin()
    mixin_loop_train: LoopMixin = BatchTrainLoopMixin()
    mixin_engine_validation: Optional[EngineMixin] = None
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

    def train(self) -> None:
        self.__check()
        self.__entry__ = 'train'

        self.run_hook(type='engine_start', args=[self.__entry__])
        self.mixin_engine_train()
        self.run_hook(type='engine_end', args=[self.__entry__])

    def validation(self) -> None:
        self.__check()
        assert self.mixin_engine_validation is not None, 'EngineMixin required for validation'
        self.__entry__ = 'validation'

        self.run_hook(type='engine_start', args=[self.__entry__])
        self.mixin_engine_validation()
        self.run_hook(type='engine_end', args=[self.__entry__])

    def test(self) -> None:
        self.__check()
        assert self.mixin_engine_test is not None, 'EngineMixin required for test'
        self.__entry__ = 'test'

        self.run_hook(type='engine_start', args=[self.__entry__])
        self.mixin_engine_test()
        self.run_hook(type='engine_end', args=[self.__entry__])

    def run_hook(
        self,
        /,
        type: Optional[str] = None,
        index: Optional[int] = None,
        args: Sequence[Any] = [],       # NOQA: B006 - Read only argument
        kwargs: dict[str, Any] = {},    # NOQA: B006 - Read only argument
    ) -> None:
        """ This method runs all hooks in mixins, plugins and on the engine itself. """
        self.hooks.run(type=type, index=index, args=args, kwargs=kwargs)
        self.mixins.run(type=type, index=index, args=args, kwargs=kwargs)
        self.plugins.run(type=type, index=index, args=args, kwargs=kwargs)

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
        def manual_to(obj: dict[str, Any]) -> None:
            for param in obj.values():
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(*args, **kwargs)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(*args, **kwargs)
                elif isinstance(param, dict):
                    manual_to(param)

        for _name, value in self.__loop_values(torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
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

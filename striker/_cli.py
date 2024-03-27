from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, cast

if TYPE_CHECKING:
    print: Any  # MyPy complains that print is not defined without this

import argparse
import inspect
import logging
import sys
from pathlib import Path

from . import Engine, Parameters
from ._parameter import load_external
from .core import Mixin
from .core._protocol import ProtocolChecker, signature_to_string

try:
    from rich import print as rprint
    from rich.markup import escape

    def print(*args: Any, **kwargs: Any) -> None:
        args = tuple(escape(a) if isinstance(a, str) else a for a in args)
        rprint(*args, **kwargs)
except ImportError:
    from builtins import print

try:
    from rich_argparse import RichHelpFormatter as HelpFormatter

    HelpFormatter.styles['argparse.groups'] = 'bold italic yellow'
except ImportError:
    from argparse import HelpFormatter

log = logging.getLogger(__name__)


class CustomFormatter(HelpFormatter):
    def __init__(self, prog: str, indent_increment: int = 2, max_help_position: int = 80, width: Optional[int] = None):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_action(self, action: argparse.Action) -> str:
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 4, self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        # no help; start on same line and add a final newline
        if not action.help:
            action_header = '%*s%s\n' % (self._current_indent, '', action_header)

        # short action name; start on the same line and pad two spaces
        elif len(action_header) <= action_width:
            action_header = '%*s%-*s  ' % (self._current_indent, '', action_width, action_header)
            indent_first = 0

        # long action name; start on the next line
        else:
            action_header = '%*s%s\n' % (self._current_indent, '', action_header)
            indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header] if action.dest != '==SUPPRESS==' or action.help else []

        # if there was help for the action, add lines of help text
        if action.help:
            help_text = self._expand_help(action)
            help_lines = self._split_lines(help_text, help_width)
            parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
            for line in help_lines[1:]:
                parts.append('%*s%s\n' % (help_position, '', line))

        # or add a newline if the description doesn't end with one
        elif len(parts) and not action_header.endswith('\n'):
            parts.append('\n')

        # if there are any sub-actions, add their help as well
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        # return a single string
        return self._join_parts(parts)


class CLI(argparse.ArgumentParser):
    __init_done: bool = False

    def __init__(self, *args: Any, **kwargs: Any):
        self.__engine: Optional[Engine] = None
        self.__param: Optional[Parameters] = None
        self.__proto: Optional[ProtocolChecker] = None

        super().__init__(*args, **kwargs)
        if self.formatter_class is argparse.HelpFormatter:
            self.formatter_class = CustomFormatter

        self.__parsers: dict[str, argparse.ArgumentParser] = {}
        subparsers = self.add_subparsers(parser_class=argparse.ArgumentParser, required=True, metavar='subcommand', title='subcommands')

        parent = argparse.ArgumentParser(add_help=False)
        parent.add_argument('-c', '--config', type=Path, required=True, help='python parameter file')
        parent.add_argument('-p', '--param', action='append', metavar='KEY=VALUE', help='keyword arguments for parameter file (multiple are allowed)')

        self.__parsers['train'] = subparsers.add_parser(
            'train', parents=[parent], formatter_class=self.formatter_class, description='Train a model', help='train a model'
        )
        self.__parsers['train'].set_defaults(subcommand='train')
        self.__parsers['train'].add_argument(
            '--dry-run', action='store_true', help='Create the engine, but do not run the training routine (useful for debugging in interactive mode)'
        )

        self.__parsers['test'] = subparsers.add_parser(
            'test',
            parents=[parent],
            formatter_class=self.formatter_class,
            description='Test a model with test data',
            help='test a model with test data',
        )
        self.__parsers['test'].set_defaults(subcommand='test')
        self.__parsers['test'].add_argument('--weights', type=Path, default=None, help='Path to stored weights')
        self.__parsers['test'].add_argument('--dataset', choices=('train', 'validation', 'test'), default='test', help='Dataset to use for testing')
        self.__parsers['test'].add_argument(
            '--dry-run', action='store_true', help='Create the engine, but do not run the testing routine (useful for debugging in interactive mode)'
        )
        self.__parsers['protocol'] = subparsers.add_parser(
            'protocol', parents=[parent], formatter_class=self.formatter_class, description='Show engine protocol', help='show engine protocol'
        )
        self.__parsers['protocol'].set_defaults(subcommand='protocol')
        self.__parsers['protocol'].add_argument('--check', action='store_true', help='Check engine against protocol in addition to printing it')

        self.__parsers['parameters'] = subparsers.add_parser(
            'parameters', parents=[parent], formatter_class=self.formatter_class, description='Show parameters', help='show parameters'
        )
        self.__parsers['parameters'].set_defaults(subcommand='parameters')
        self.__parsers['parameters'].add_argument(
            '--signature', action='store_true', help='show parameter signature with its arguments instead of creating it'
        )

        self.__init_done = True

    def __getitem__(self, name: str) -> argparse.ArgumentParser:
        return self.__parsers[name]

    def add_argument(self, *args: Any, **kwargs: Any) -> Optional[argparse.Action]:  # type: ignore[override]
        if not self.__init_done:
            return super().add_argument(*args, **kwargs)

        for parser in self.__parsers.values():
            parser.add_argument(*args, **kwargs)
        return None

    def run(
        self,
        func: Callable[[Parameters, argparse.Namespace], Engine],
        variable: str = 'params',
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> None:
        """
        Parse arguments, create the engine and run the appropriate method.

        Args:
            TODO
        """
        parsed_args = self.parse_args(args, namespace)
        if parsed_args.subcommand == 'train':
            self.__train(parsed_args, func, variable)
        elif parsed_args.subcommand == 'test':
            self.__test(parsed_args, func, variable)
        elif parsed_args.subcommand == 'protocol':
            if parsed_args.check:
                self.__protocol_check(parsed_args, func, variable)
            else:
                self.__protocol(parsed_args, func, variable)
        elif parsed_args.subcommand == 'parameters':
            if parsed_args.signature:
                self.__parameters_signature(parsed_args, func, variable)
            else:
                self.__parameters(parsed_args, func, variable)

    def __train(self, args: argparse.Namespace, func: Callable[[Parameters, argparse.Namespace], Engine], variable: str) -> None:
        params = self.__get_parameters(args, variable)

        self.__engine = func(params, args)
        assert isinstance(self.__engine, Engine), f'Function "{func.__name__}" should return an Engine instance'

        if args.dry_run:
            log.info('The --dry-run flag prevented the training routine to actually run')
            if sys.flags.interactive:
                log.info('You can access the Parameters through the <CLI.parameters> property and the Engine as <CLI.engine>')
            return

        self.__engine.train()

    def __test(self, args: argparse.Namespace, func: Callable[[Parameters, argparse.Namespace], Engine], variable: str) -> None:
        params = self.__get_parameters(args, variable)
        self.__load_weights(params, args.weights)

        self.__engine = func(params, args)
        assert isinstance(self.__engine, Engine), f'Function "{func.__name__}" should return an Engine instance'

        if args.dry_run:
            log.info('The --dry-run flag prevented the testing routine to actually run')
            if sys.flags.interactive:
                log.info('You can access the Parameters through the <CLI.parameters> property and the Engine as <CLI.engine>')
            return

        self.__engine.test(args.dataset)

    def __protocol(self, args: argparse.Namespace, func: Callable[[Parameters, argparse.Namespace], Engine], variable: str) -> None:
        EngineCls: Optional[type[Engine]] = None
        if inspect.isclass(func) and issubclass(func, Engine):
            EngineCls = func
        elif inspect.ismethod(func) and issubclass(cast(type, func.__self__), Engine):
            EngineCls = cast(type[Engine], func.__self__)

        if EngineCls is not None:
            self.__proto = ProtocolChecker().add(EngineCls.__name__, cast(Optional[type], EngineCls.__protocol__))

            # Mixins
            for name in dir(EngineCls):
                try:
                    value = getattr(EngineCls, name, None)
                except BaseException:
                    continue
                if isinstance(value, Mixin):
                    self.__proto.add(name, cast(Optional[type], value.__protocol__))

            # Plugins
            for value in getattr(EngineCls, 'plugins', []):
                name = value.__class__.__name__.lower()
                self.__proto.add(name, value.__protocol__)

            print(self.__proto)
        else:
            log.warning('Could not get custom Engine class from function, so we need to build the engine to get the protocol')
            params = self.__get_parameters(args, variable)
            self.__engine = func(params, args)
            assert isinstance(self.__engine, Engine), f'Function "{func.__name__}" should return an Engine instance'
            print(self.__engine.protocol)

    def __protocol_check(self, args: argparse.Namespace, func: Callable[[Parameters, argparse.Namespace], Engine], variable: str) -> None:
        params = self.__get_parameters(args, variable)
        self.__engine = func(params, args)
        assert isinstance(self.__engine, Engine), f'Function "{func.__name__}" should return an Engine instance'

        try:
            from rich.table import Table

            table = Table('', '', show_header=False, border_style='dim', expand=True, highlight=True, show_edge=False)
            table.add_row(self.__engine.protocol, self.__engine.protocol.checker(self.__engine))
            print(table)
        except ImportError:
            print(self.__engine.protocol)
            print(self.__engine.protocol.checker(self.__engine))

    def __parameters(self, args: argparse.Namespace, func: Callable[[Parameters, argparse.Namespace], Engine], variable: str) -> None:
        self.__param = self.__get_parameters(args, variable)
        print(self.__param)

    def __parameters_signature(self, args: argparse.Namespace, func: Callable[[Parameters, argparse.Namespace], Engine], variable: str) -> None:
        param_symbol = load_external(Path(args.config), variable)
        if not callable(param_symbol):
            raise NotImplementedError(f'"{variable}" in "{args.config}" is not a function and thus has no arguments')

        signature = inspect.signature(param_symbol)
        print(f'{variable}{signature_to_string(signature)}')

    @property
    def engine(self) -> Optional[Engine]:
        return self.__engine

    @property
    def parameters(self) -> Optional[Parameters]:
        if self.__param is not None:
            return self.__param
        if self.__engine is not None:
            return self.__engine.params
        return None

    @property
    def protocol(self) -> Optional[ProtocolChecker]:
        if self.__proto is not None:
            return self.__proto
        if self.__engine is not None:
            return self.__engine.protocol
        return None

    @staticmethod
    def __get_parameters(args: argparse.Namespace, variable: str) -> Parameters:
        param_kwargs: dict[str, str] = {}
        if args.param is not None:
            try:
                params = (p.split('=', 1) for p in args.param)
                param_kwargs = {p[0].strip(): p[1].strip() for p in params}
            except BaseException as err:
                raise ValueError(f'Could not parse parameters: {args.param}') from err

        with Parameters.enable_cast():
            return Parameters.from_file(args.config, variable, **param_kwargs)

    @staticmethod
    def __load_weights(params: Parameters, weights: Optional[Path]) -> None:
        if weights is not None:
            log.info('Loading weights from: %s', weights)
            params.load(weights)
        else:
            log.error('No weight file given to load')

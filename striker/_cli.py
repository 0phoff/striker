from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Optional, Any, Sequence
if TYPE_CHECKING:
    print: Any          # MyPy complains that print is not defined without this

import argparse
import inspect
from pathlib import Path
try:
    from rich_argparse import RichHelpFormatter as HelpFormatter
    HelpFormatter.styles['argparse.groups'] = 'bold italic yellow'
except ImportError:
    from argparse import HelpFormatter

from . import Engine, Parameters
from .core._protocol import param_to_string


class CustomFormatter(HelpFormatter):
    def __init__(
        self,
        prog: str,
        indent_increment: int = 2,
        max_help_position: int = 80,
        width: Optional[int] = None,
    ):
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
        if action.dest != '==SUPPRESS==' or action.help:
            parts = [action_header]
        else:
            parts = []

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
        super().__init__(*args, **kwargs)
        if self.formatter_class is argparse.HelpFormatter:
            self.formatter_class = CustomFormatter

        self.__parsers: dict[str, argparse.ArgumentParser] = {}
        subparsers = self.add_subparsers(
            parser_class=argparse.ArgumentParser,
            required=True,
            metavar='subcommand',
            title='subcommands',
        )

        parent = argparse.ArgumentParser(add_help=False)
        parent.add_argument(
            '-c', '--config',
            type=Path,
            required=True,
            help='python parameter file',
        )
        parent.add_argument(
            '-p', '--param',
            action='append',
            nargs=2,
            metavar=('KEY', 'VALUE'),
            help='keyword arguments for parameter file (multiple are allowed)',
        )

        self.__parsers['train'] = subparsers.add_parser(
            'train',
            parents=[parent],
            formatter_class=self.formatter_class,
            description='Train a model',
            help='train a model',
        )
        self.__parsers['train'].set_defaults(subcommand='train')

        self.__parsers['validation'] = subparsers.add_parser(
            'validation',
            parents=[parent],
            formatter_class=self.formatter_class,
            description='Test a model with validation data',
            help='test a model with validation data',
        )
        self.__parsers['validation'].set_defaults(subcommand='validation')

        self.__parsers['test'] = subparsers.add_parser(
            'test',
            parents=[parent],
            formatter_class=self.formatter_class,
            description='Test a model with test data',
            help='test a model with test data',
        )
        self.__parsers['test'].set_defaults(subcommand='test')

        self.__parsers['protocol'] = subparsers.add_parser(
            'protocol',
            parents=[parent],
            formatter_class=self.formatter_class,
            description='Show engine protocol',
            help='show engine protocol',
        )
        self.__parsers['protocol'].set_defaults(subcommand='protocol')

        self.__parsers['parameters'] = subparsers.add_parser(
            'parameters',
            parents=[parent],
            formatter_class=self.formatter_class,
            description='Show parameters',
            help='show parameters',
        )
        self.__parsers['parameters'].set_defaults(subcommand='parameters')
        self.__parsers['parameters'].add_argument(
            '-k', '--kwargs',
            action='store_true',
            help='show parameter kwargs instead of the parameters',
        )

        self.__init_done = True

    def __getitem__(self, name: str) -> argparse.ArgumentParser:
        return self.__parsers[name]

    def add_argument(self, *args: Any, **kwargs: Any) -> Optional[argparse.Action]:     # type: ignore[override]
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
    ) -> Optional[Engine]:
        parsed_args = self.parse_args(args, namespace)
        if parsed_args.subcommand == 'train':
            return self.__train(parsed_args, func, variable)
        elif parsed_args.subcommand == 'validation':
            return self.__validation(parsed_args, func, variable)
        elif parsed_args.subcommand == 'test':
            return self.__test(parsed_args, func, variable)
        elif parsed_args.subcommand == 'protocol':
            return self.__protocol(parsed_args, func, variable)
        elif parsed_args.subcommand == 'parameters':
            self.__parameters(parsed_args, func, variable)
        return None

    def __train(
        self,
        args: argparse.Namespace,
        func: Callable[[Parameters, argparse.Namespace], Engine],
        variable: str,
    ) -> Engine:
        params = self.__get_parameters(args, variable)
        engine = func(params, args)
        engine.train()

        return engine

    def __validation(
        self,
        args: argparse.Namespace,
        func: Callable[[Parameters, argparse.Namespace], Engine],
        variable: str,
    ) -> Engine:
        params = self.__get_parameters(args, variable)
        engine = func(params, args)
        engine.validation()

        return engine

    def __test(
        self,
        args: argparse.Namespace,
        func: Callable[[Parameters, argparse.Namespace], Engine],
        variable: str,
    ) -> Engine:
        params = self.__get_parameters(args, variable)
        engine = func(params, args)
        engine.test()

        return engine

    def __protocol(
        self,
        args: argparse.Namespace,
        func: Callable[[Parameters, argparse.Namespace], Engine],
        variable: str,
    ) -> Engine:
        params = self.__get_parameters(args, variable)
        engine = func(params, args)

        try:
            from rich import print
            from rich.table import Table

            table = Table('', '', show_header=False, border_style='dim', expand=True, highlight=True, show_edge=False)
            table.add_row(engine.protocol, engine.protocol.checker(engine))
            print(table)

        except ImportError:
            print(engine.protocol)
            print(engine.protocol.checker(engine))

        return engine

    def __parameters(
        self,
        args: argparse.Namespace,
        func: Callable[[Parameters, argparse.Namespace], Engine],
        variable: str,
    ) -> None:
        global print
        escape = lambda x: x                    # NOQA: E731 - lambda function is fine for this small identity func
        with suppress(ImportError):
            from rich import print
            from rich.markup import escape      # type: ignore[no-redef]

        if args.kwargs:
            param_symbol = Parameters._load_external(Path(args.config), variable)
            if not callable(param_symbol):
                raise NotImplementedError(f'"{variable}" in "{args.config}" is not a function and thus has no arguments')

            signature = inspect.signature(param_symbol)
            parameters = (param_to_string(p) for p in signature.parameters.values())

            print(escape(f'{variable}('))
            for param in parameters:
                print(escape(f'  {param}'))
            print(')')
        else:
            params = self.__get_parameters(args, variable)
            print(escape(str(params)))

    @staticmethod
    def __get_parameters(args: argparse.Namespace, variable: str) -> Parameters:
        param_kwargs = {n: v for n, v in args.param} if args.param is not None else {}
        with Parameters.enable_cast():
            return Parameters.from_file(args.config, variable, **param_kwargs)

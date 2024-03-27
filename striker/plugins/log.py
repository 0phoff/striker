import logging
import sys
from contextlib import suppress
from pathlib import Path
from typing import Literal, Optional, Protocol, Union, cast

from .._engine import Engine
from ..core import Plugin, hooks

__all__ = ['LogPlugin']
log = logging.getLogger(__name__)


class ParentProtocol(Protocol):
    log_file: Optional[Union[str, Path]] = None
    """ Path to store logging data. No logging data will be stored if this is missing. """


class LogPlugin(Plugin, protocol=ParentProtocol):
    """
    This plugin enables the :class:`~rich.logging.RichHandler` if it is installed and can also setup a logging file to store your run logs.

    Args:
        file_mode: How to open the log file. Default **'a'**
        file_level: Filter level for the logging file; Default **0**
        rich_enable: Wether to enabled the :class:`~rich.logging.RichHandler` if it is available; Default **true**
        rich_level: Filter level for the :class:`~rich.logging.RichHandler`; Default **INFO**
    """

    parent: Engine  # Fix MyPy issues by setting a proper type of self.parent

    def __init__(
        self, file_mode: Literal['a', 'w'] = 'a', file_level: int = logging.NOTSET, rich_enable: bool = True, rich_level: int = logging.INFO
    ):
        self.file_mode = file_mode
        self.file_level = file_level
        self.rich_enable = rich_enable
        self.rich_level = rich_level

    @hooks.engine_init
    def setup_handlers(self) -> None:
        # We can only setup the filehandler after the Engine is created, as we need Engine.log_file
        handlers = (self.setup_filehandler(), self.setup_streamhandler())
        filtered_handlers = tuple(h for h in handlers if h is not None)
        if len(filtered_handlers):
            logging.basicConfig(force=True, level=logging.NOTSET, handlers=filtered_handlers)

        # Optimization: We can safely disable this plugin, as there are no further hooks to run.
        self.enabled = False

    def setup_streamhandler(self) -> Optional[logging.Handler]:
        if not self.rich_enable:
            return None

        # Create Handler
        handler: Optional[logging.Handler] = None
        if sys.stdout.isatty():
            with suppress(ImportError):
                from rich.logging import RichHandler

                handler = cast(logging.Handler, RichHandler(rich_tracebacks=True, tracebacks_suppress=['striker']))
                handler.setFormatter(logging.Formatter('%(message)s', '[%X]'))
        if handler is None:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)-8s %(message)s', '[%X]'))

        # Set Level
        handler.setLevel(self.rich_level)

        return handler

    def setup_filehandler(self) -> Optional[logging.Handler]:
        log_file = getattr(self.parent, 'log_file', None)
        if log_file is None:
            log.warning('"log_file" is None, so logging data will not be saved.')
            return None

        self.log_file = Path(log_file)
        if not self.log_file.parent.exists():
            log.info('log_file folder "%s" does not exist, creating now...', self.log_file.parent)
            self.log_file.parent.mkdir(parents=True)

        # Create Handler
        handler = logging.FileHandler(filename=self.log_file, mode=self.file_mode)
        handler.setFormatter(logging.Formatter(fmt='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] | %(message)s', datefmt='%x %X'))

        # Set Level
        handler.setLevel(self.file_level)

        return handler

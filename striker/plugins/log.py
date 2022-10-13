from typing import Literal, Protocol, Union, Optional

from contextlib import suppress
import logging
import sys
from pathlib import Path
from ..core import Plugin, hooks
from .._engine import Engine

__all__ = ['LogPlugin']
log = logging.getLogger(__name__)


class ParentProtocol(Protocol):
    log_file: Optional[Union[str, Path]] = None
    """ Path to store logging data. No logging data will be stored if this is missing. """


class LogPlugin(Plugin, protocol=ParentProtocol):
    """
    This plugin enables the :class:`~rich.logging.RichHandler` if it is installed and can also setup a logging file to store your run logs.

    Args:
        logger_name: Name of the logger for which we want to store the logs in a file. Default **root logger / all loggers**
        file_mode: How to open the log file. Default **'a'**
        rich_handler: Wether to enabled the :class:`~rich.logging.RichHandler` if it is available; Default **true**
        rich_level: Filter level for the :class:`~rich.logging.RichHandler`; Default **0**
    """
    parent: Engine      # Fix MyPy issues by setting a proper type of self.parent

    def __init__(
        self,
        logger_name: str = '',
        file_mode: Literal['a', 'w'] = 'a',
        rich_handler: bool = True,
        rich_level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(logger_name)
        self.mode = file_mode
        self.rich = rich_handler
        self.level = rich_level

    @hooks.engine_begin
    def setup_logging(self) -> None:
        self.setup_streamhandler()
        self.setup_filehandler()

        # Optimization: We can safely disable this plugin, as there ar no further hooks to run.
        self.enabled = False

    def setup_streamhandler(self) -> None:
        if not self.rich:
            return

        if sys.stdout.isatty():
            with suppress(ImportError):
                from rich.logging import RichHandler

                logging.basicConfig(
                    force=True,
                    level=logging.NOTSET,
                    format='%(message)s',
                    datefmt='[%X]',
                    handlers=[RichHandler(
                        level=self.level,
                        rich_tracebacks=True,
                        tracebacks_suppress=['striker'],
                    )],
                )
        else:
            handler = logging.StreamHandler()
            handler.setLevel(self.level)

            logging.basicConfig(
                force=True,
                level=logging.NOTSET,
                format='%(levelname)-8s %(message)s',
                datefmt='[%X]',
                handlers=[handler],
            )

    def setup_filehandler(self) -> None:
        log_file = getattr(self.parent, 'log_file', None)
        if log_file is None:
            log.warning('"log_file" is None, so logging data will not be saved.')
            return

        self.log_file = Path(log_file)
        if not self.log_file.parent.exists():
            log.info('log_file folder "%s" does not exist, creating now...', self.log_file.parent)
            self.log_file.parent.mkdir(parents=True)

        handler = logging.FileHandler(filename=self.log_file, mode=self.mode)
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(logging.Formatter(
            fmt='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] | %(message)s',
            datefmt='%x %X',
        ))
        self.logger.addHandler(handler)

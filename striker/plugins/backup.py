import logging
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Union

from .._engine import Engine
from ..core import Plugin, hooks

__all__ = ['BackupPlugin']
log = logging.getLogger(__name__)


class ParentProtocol(Protocol):
    backup_folder: Optional[Union[str, Path]] = None
    """ Folder where we store backups. """

    backup_rate: Optional[Union[list[Union[int, slice]], int, slice]] = None
    """
    When to store backups.

    Note:
        This value is used to setup a hook and thus can have a few different values:
            - None: never run the hook
            - slice: run hook periodically (according to slice specs)
            - int: run hook at specified epoch/batch
            - list[int, slice]: Combination of the above
    """


class BackupPlugin(Plugin, protocol=ParentProtocol):
    """
    This plugin enables to store backups at regular intervals.

    Args:
        mode: Whether to store backups at an epoch or batch interval (specified by ``backup_rate`` in the parent). Default **epoch**
        extension: File extension to use; Default **'.param.pt'**
        final: Whether to save a final file at the end of training; Default **True**

    You can also call the save function manually, to create a backup in the correct folder (and with the correct extension):

    >>> class Engine(striker.Engine):
    ...     plugins = [BackupPlugin()]
    ...
    ...     def dummy_function(self):
    ...         self.plugins['backupplugin'].save('custom-save-file')
    """

    __type_check__: Literal['none', 'log', 'raise'] = 'none'
    parent: Engine  # Fix MyPy issues by setting a proper type of self.parent

    def __init__(self, mode: Literal['batch', 'epoch'] = 'epoch', extension: str = '.param.pt', final: bool = True) -> None:
        self.backup_mode = mode
        self.extension = extension
        self.final = final

    def save(self, name: str, *args: Any, **kwargs: Any) -> None:
        backup_path = self.backup_folder / f'{name}'
        if not ''.join(backup_path.suffixes).endswith(self.extension):
            backup_path = backup_path.with_suffix(self.extension)

        self.parent.params.save(backup_path, *args, **kwargs)
        log.info('Saved backup: %s', backup_path)

    def load(self, name: str, *args: Any, **kwargs: Any) -> None:
        backup_path = self.backup_folder / f'{name}'
        if not ''.join(backup_path.suffixes).endswith(self.extension):
            backup_path = backup_path.with_suffix(self.extension)

        if backup_path.exists():
            self.parent.params.load(backup_path, *args, **kwargs)
            log.info('Loaded backup: %s', backup_path)

    @hooks.engine_begin
    def setup_backup_hook(self, entry: Literal['train', 'validation', 'test']) -> None:
        if entry != 'train':
            self.enabled = False
            return

        backup_folder = getattr(self.parent, 'backup_folder', None)
        if backup_folder is None:
            log.warning('"backup_folder" is None, so no backups will be taken.')
            self.enabled = False
            return

        self.backup_folder = Path(backup_folder)
        if not self.backup_folder.exists():
            log.info('Backup folder "%s" does not exist, creating now...', self.backup_folder)
            self.backup_folder.mkdir(parents=True)
        elif not self.backup_folder.is_dir():
            raise ValueError(f'Backup folder "{self.backup_folder}" is not a directory')

        backup_rate = getattr(self.parent, 'backup_rate', None)
        if backup_rate is None:
            log.warning('"backup_rate" is None, so no intermediate backups will be taken.')
            return

        if self.backup_mode == 'batch':
            self.hooks.train_batch_end[backup_rate](self.run_backup)
        else:
            self.hooks.train_epoch_end[backup_rate](self.run_backup)

    @hooks.engine_end
    def final_backup(self, entry: Literal['train', 'validation', 'test']) -> None:
        if self.final and entry == 'train':
            self.save('final')

    def run_backup(self, index: int) -> None:
        self.save(f'backup-{self.backup_mode}-{index:05d}')

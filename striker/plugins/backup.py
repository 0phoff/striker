from typing import Literal, Protocol, Union, Optional

import logging
from pathlib import Path
from ..core import Plugin, hooks
from .._engine import Engine

__all__ = ['BackupPlugin']
log = logging.getLogger(__name__)


class ParentProtocol(Protocol):
    backup_folder: Optional[Union[str, Path]] = None
    """ Folder where we store backups. """

    backup_rate: Optional[Union[list[Union[int, slice]], int, slice]] = slice(None, None, 1)
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
     """
    __type_check__: Literal['none', 'log', 'raise'] = 'none'
    parent: Engine      # Fix MyPy issues by setting a proper type of self.parent

    def __init__(self, mode: Literal['batch', 'epoch'] = 'epoch') -> None:
        self.backup_mode = mode

    @hooks.engine_start
    def setup_backup_hook(self, entry: Literal['train', 'validation', 'test']) -> None:
        if entry != 'train':
            self.enabled = False
            return

        backup_folder = getattr(self.parent, 'backup_folder', None)
        if backup_folder is None:
            log.warn('"backup_folder" is None, so no backups will be taken.')
            self.enabled = False
            return

        self.backup_folder = Path(backup_folder)
        if not self.backup_folder.exists():
            log.info('Backup folder "%s" does not exist, creating now...', self.backup_folder)
            self.backup_folder.mkdir(parents=True)
        elif not self.backup_folder.is_dir():
            raise ValueError(f'Backup folder "{self.backup_folder}" is not a directory')

        backup_rate = getattr(self.parent, 'backup_rate', slice(None, None, 1))
        if backup_rate is None:
            log.warn('"backup_rate" is None, so no backups will be taken.')
            self.enabled = False
            return

        if self.backup_mode == 'batch':
            self.hooks.train_batch_end[backup_rate](self.run_backup)
        else:
            self.hooks.train_epoch_end[backup_rate](self.run_backup)

    def run_backup(self, index: int) -> None:
        backup = self.backup_folder / f'backup-{self.backup_mode}-{index:05d}.state.pt'
        self.parent.params.save(backup)
        log.info('Saved backup: %s', backup)
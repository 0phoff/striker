from typing import TYPE_CHECKING, Literal, Optional, Protocol, cast

if TYPE_CHECKING:
    from rich.progress import Task, TaskID

import datetime
import logging
import sys
import time

from rich.progress import BarColumn, MofNCompleteColumn, Progress, ProgressColumn, TextColumn, filesize  # type: ignore[attr-defined]
from rich.table import Column
from rich.text import Text

from .._engine import Engine
from ..core import Plugin, hooks

__all__ = ['ProgressBarPlugin']
log = logging.getLogger(__name__)


class ParentProtocol(Protocol):
    max_epochs: Optional[int] = None
    """ Maximum number of epochs. """

    max_batches: Optional[int] = None
    """ Maximum number of batches. """


class ProgressBarPlugin(Plugin, protocol=ParentProtocol):
    """
    This plugin shows a Rich progress bar during runs.

    Note:
        You can append extra information to the different progress bars by setting the appropriate strings on this plugin:
        (syntax is shown as it should be used in the engine)
            - ``self.plugins['progressbarplugin'].train_text``
            - ``self.plugins['progressbarplugin'].validation_text``
            - ``self.plugins['progressbarplugin'].test_text``

    Note:
        The ``max_epochs`` and/or ``max_batches`` values are purely used for the progress bar and will not actually stop the training.
        See :class:`~striker.plugins.quit.QuitPlugin` for a plugin that actually stops training.
    """

    __type_check__: Literal['none', 'log', 'raise'] = 'none'
    parent: Engine  # Fix MyPy issues by setting a proper type of self.parent

    PRINT_DELAY_MIN: int = 1 * 60
    PRINT_DELAY_MAX: int = 10 * 60
    PRINT_TOTAL_MIN: int = 30 * 60
    PRINT_TOTAL_MAX: int = 10 * 60 * 60

    def __init__(self) -> None:
        self.tty = sys.stdout.isatty()

    @hooks.engine_begin
    def start_progress(self, entry: Literal['train', 'test', 'validation']) -> None:
        self.__time_start = time.perf_counter()
        max_epochs = getattr(self.parent, 'max_epochs', None)
        max_batches = getattr(self.parent, 'max_batches', None)
        self.max_batches = max_batches is not None

        self.progress = Progress(
            TextColumn('{task.description}'),
            BarColumn(pulse_style='bar.back'),
            CompletedColumn(table_column=Column(justify='right')),
            TimeColumn(),
            SpeedColumn(),
            TextColumn('{task.fields[post_text]}'),
            auto_refresh=False,
            speed_estimate_period=24 * 60 * 60,
        )

        if entry == 'train':
            self.p_epoch = self.progress.add_task(
                'Train', completed=self.parent.epoch, total=max_epochs, marker=' E', post_text='', last_print=self.progress.get_time(), first=True
            )

            self.p_batch = self.progress.add_task(
                '',
                start=self.max_batches,
                visible=False,
                completed=self.parent.batch if self.max_batches else 0,
                total=max_batches,
                marker=' B',
                post_text=self.train_text,
                last_print=self.progress.get_time(),
            )

        if entry in ('train', 'validation'):
            self.p_validation = self.progress.add_task(
                'Validation', start=False, visible=False, marker=' B', post_text=self.validation_text, last_print=self.progress.get_time()
            )

        if entry == 'test':
            self.p_test = self.progress.add_task(
                'Test', start=False, visible=False, marker=' B', post_text=self.test_text, last_print=self.progress.get_time()
            )

        if self.tty:
            self.progress.start()

    @hooks.engine_end
    def stop_progress(self, entry: Literal['train', 'test', 'validation']) -> None:
        self.__time_stop = time.perf_counter()

        if self.tty:
            self.progress.stop()
            self.progress.console.clear_live()

        delta = round(self.__time_stop - self.__time_start)
        log.info('Engine %s run took %s', entry, datetime.timedelta(seconds=delta))

    @hooks.train_epoch_begin
    def train_epoch_begin(self) -> None:
        if self.max_batches:
            self.progress.update(self.p_batch, visible=True, start=True)
        else:
            self.progress.reset(self.p_batch, visible=True, total=self.parent.mixin_loop_train.num_batches)
        self.refresh()
        if not self.tty:
            self.print_train()

    @hooks.train_batch_end
    def train_batch_end(self) -> None:
        completed = self.parent.batch if self.max_batches else None
        self.progress.update(self.p_batch, advance=1, completed=completed)
        self.refresh()
        if not self.tty:
            self.print_train()

    @hooks.train_epoch_end
    def train_epoch_end(self, epoch: int) -> None:
        self.progress.update(self.p_epoch, completed=epoch)
        self.refresh()
        if not self.tty:
            self.print_train()

    @hooks.validation_epoch_begin
    def validation_epoch_begin(self) -> None:
        self.progress.reset(self.p_validation, visible=True, total=self.parent.mixin_loop_validation.num_batches)
        self.refresh()
        if not self.tty:
            self.print_test(self.p_validation)

    @hooks.validation_batch_end
    def validation_batch_end(self) -> None:
        self.progress.update(self.p_validation, advance=1)
        self.refresh()
        if not self.tty:
            self.print_test(self.p_validation)

    @hooks.validation_epoch_end
    def validation_epoch_end(self) -> None:
        self.progress.update(self.p_validation, visible=False)
        self.refresh()

    @hooks.test_epoch_begin
    def test_epoch_begin(self) -> None:
        self.progress.reset(self.p_test, visible=True, total=self.parent.mixin_loop_test.num_batches)
        self.refresh()
        if not self.tty:
            self.print_test(self.p_test)

    @hooks.test_batch_end
    def test_batch_end(self) -> None:
        self.progress.update(self.p_test, advance=1)
        self.refresh()
        if not self.tty:
            self.print_test(self.p_test)

    @hooks.test_epoch_end
    def test_epoch_end(self) -> None:
        self.progress.update(self.p_test, visible=False)
        self.refresh()

    def refresh(self) -> None:
        """Update all progress bars post_text and refresh progress."""
        p_train = getattr(self, 'p_batch', None)
        if p_train is not None:
            self.progress.update(p_train, post_text=self.train_text)

        p_validation = getattr(self, 'p_validation', None)
        if p_validation is not None:
            self.progress.update(p_validation, post_text=self.validation_text)

        p_test = getattr(self, 'p_test', None)
        if p_test is not None:
            self.progress.update(p_test, post_text=self.test_text)

        self.progress.refresh()

    def print_train(self) -> None:
        """Print training progress bar (in case we do not have TTY connected)."""
        with self.progress._lock:
            current_time = self.progress.get_time()
            task_epoch = self.progress._tasks[self.p_epoch]
            task_batch = self.progress._tasks[self.p_batch]

            if self.should_print(task_batch if self.max_batches else task_epoch, current_time):
                texts = [task_epoch.description]

                # Completed
                completed_epoch = str(cast(ProgressColumn, self.progress.columns[2]).render(task_epoch)).strip()
                completed_batch = str(cast(ProgressColumn, self.progress.columns[2]).render(task_batch)).strip()
                texts.append(f'{completed_epoch} {completed_batch}')

                # Time
                texts.append(str(cast(ProgressColumn, self.progress.columns[3]).render(task_batch if self.max_batches else task_epoch)).strip())

                # Post Text
                texts.append(str(cast(ProgressColumn, self.progress.columns[5]).render(task_batch)).strip())

                print(' | '.join(t for t in texts if len(t)), flush=True)

    def print_test(self, task_id: 'TaskID') -> None:
        """Print test/validation progress bar (in case we do not have TTY connected)."""
        with self.progress._lock:
            current_time = self.progress.get_time()
            task = self.progress._tasks[task_id]

            if self.should_print(task, current_time):
                texts = [task.description]

                # Completed
                texts.append(str(cast(ProgressColumn, self.progress.columns[2]).render(task)).strip())

                # Time
                texts.append(str(cast(ProgressColumn, self.progress.columns[3]).render(task)).strip())

                # Post Text
                texts.append(str(cast(ProgressColumn, self.progress.columns[5]).render(task)).strip())

                print(' | '.join(t for t in texts if len(t)), flush=True)

    def should_print(self, task: 'Task', current_time: float) -> bool:
        first = task.fields.get('first', None)
        if (first is not None and first) or (first is None and task.completed == 0) or (task.completed == task.total):
            # Always print first/last
            task.fields['last_print'] = current_time
            if first is not None:
                task.fields['first'] = False
            return True

        remaining = task.time_remaining
        elapsed = task.elapsed
        delay: float = 0
        if elapsed is not None and elapsed >= self.PRINT_TOTAL_MAX:
            # Elapsed is already bigger than max, so delay should be max
            delay = self.PRINT_DELAY_MAX
        elif remaining is None or elapsed is None:
            # We cannot compute total estimated time, so delay is minimal
            delay = self.PRINT_DELAY_MIN
        else:
            # Map total time of TOTAL_MIN-TOTAL_MAX -> delay of DELAY_MIN-DELAY_MAX
            time_total = elapsed + remaining
            time_clamped = min(max(0, time_total - self.PRINT_TOTAL_MIN), self.PRINT_TOTAL_MAX)
            slope = (self.PRINT_DELAY_MAX - self.PRINT_DELAY_MIN) / 36000
            delay = self.PRINT_DELAY_MIN + (time_clamped * slope)

        if current_time - task.fields['last_print'] >= delay:
            task.fields['last_print'] = current_time
            return True

        return False

    @property
    def train_text(self) -> str:
        text = getattr(self, '_train_text', None)
        return f'\[{text}]' if text is not None else ''  # NOQA: W605 - rich requires escape of [

    @train_text.setter
    def train_text(self, value: str) -> None:
        self._train_text = value

    @property
    def validation_text(self) -> str:
        text = getattr(self, '_validation_text', None)
        return f'\[{text}]' if text is not None else ''  # NOQA: W605 - rich requires escape of [

    @validation_text.setter
    def validation_text(self, value: str) -> None:
        self._validation_text = value

    @property
    def test_text(self) -> str:
        text = getattr(self, '_test_text', None)
        return f'\[{text}]' if text is not None else ''  # NOQA: W605 - rich requires escape of [

    @test_text.setter
    def test_text(self, value: str) -> None:
        self._test_text = value


class SpeedColumn(ProgressColumn):
    """Taken from rich.progress.TaskProgressColumn"""

    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        if speed is None:
            return Text('', style='progress.remaining')

        if speed < 1:
            speed = 1 / speed
            unit, suffix = filesize.pick_unit_and_suffix(int(speed), ['s', 'm', 'h'], 60)
            speed /= unit

            # Over 1.5 days, so we show in days instead of hours
            if suffix == 'h' and speed >= 36:
                speed /= 24
                suffix = 'd'

            return Text(f'({speed:.1f} {suffix}/it)', style='progress.remaining')

        unit, suffix = filesize.pick_unit_and_suffix(int(speed), ['', '×10³', '×10⁶', '×10⁹', '×10¹²'], 1000)
        speed /= unit

        return Text(f'({speed:.1f}{suffix} it/s)', style='progress.remaining')

    def render(self, task: 'Task') -> Text:
        return self.render_speed(task.finished_speed or task.speed)


class TimeColumn(ProgressColumn):
    """Taken from rich.progress.TimeRemainingColumn and rich.progress.TimeElapsedColumn"""

    def render(self, task: 'Task') -> Text:
        if task.finished or task.total is None:
            return self.render_elapsed(task, 'progress.elapsed')

        return Text.assemble(
            self.render_elapsed(task, 'progress.elapsed'), Text('/', style='progress.elapsed'), self.render_remaining(task, 'progress.elapsed')
        )

    def render_elapsed(self, task: 'Task', style: str) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text('--:--:--', style=style)

        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        formatted = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

        return Text(formatted, style=style)

    def render_remaining(self, task: 'Task', style: str) -> Text:
        if task.time_remaining is None:
            return Text('--:--:--', style=style)

        minutes, seconds = divmod(int(task.time_remaining), 60)
        hours, minutes = divmod(minutes, 60)
        formatted = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

        return Text(formatted, style=style)


class CompletedColumn(MofNCompleteColumn):
    def render(self, task: 'Task') -> Text:
        completed = int(task.completed)
        if task.total is not None:
            total = int(task.total)
            total_width = len(str(total))
            text = f'{completed:{total_width}d}{self.separator}{total}'
        else:
            text = f'{completed}'

        marker = task.fields.get('marker', '')
        return Text(f'{text}{marker}', style='progress.download')

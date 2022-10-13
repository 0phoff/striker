from typing import TYPE_CHECKING, Literal, Protocol, Optional
if TYPE_CHECKING:
    from rich.progress import Task

from datetime import timedelta
from rich.text import Text
from rich.progress import (     # type: ignore[attr-defined]
    Progress,
    ProgressColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    filesize,
)
from ..core import Plugin, hooks
from .._engine import Engine

__all__ = ['ProgressBarPlugin']


class ParentProtocol(Protocol):
    max_epochs: Optional[int] = None
    """ Maximum number of epochs. """


class ProgressBarPlugin(Plugin, protocol=ParentProtocol):
    """
    This plugin shows a Rich progress bar during runs.

    Note:
        There are 4 different progress bars, that are shown depending on the run:
            - train_epochs: This progress bar shows the number of completed training epochs
            - train_batches: This progress bar show the number of completed training batches
            - validation_batches: This progress bar show the number of completed validation batches
            - test_batches: This progress bar show the number of completed test batches

        You can append extra information to each of these progress bars by setting the appropriate strings on this plugin:
        (syntax is shown as it should be used in the engine)
            - ``self.plugins['progressbarplugin'].train_epoch_text``
            - ``self.plugins['progressbarplugin'].train_batch_text``
            - ``self.plugins['progressbarplugin'].validation_text``
            - ``self.plugins['progressbarplugin'].test_text``

    Note:
        The ``max_epoch`` value is purely used for the progress bar and will not actually stop the training.
        See :class:`~striker.plugins.quit.QuitPlugin` for a plugin that actually stops training.
    """
    __type_check__: Literal['none', 'log', 'raise'] = 'none'
    parent: Engine      # Fix MyPy issues by setting a proper type of self.parent

    @hooks.engine_begin
    def start_progress(self, entry: Literal['train', 'test', 'validation']) -> None:
        self.progress = Progress(
            TextColumn('{task.description}'),
            BarColumn(),
            MofNCompleteColumn(separator=' / '),
            TimeColumn(),
            SpeedColumn(),
            TextColumn('{task.fields[post_text]}'),
            auto_refresh=False,
        )
        self.progress.start()

        if entry == 'train':
            self.p_epoch = self.progress.add_task(
                'Epoch',
                completed=self.parent.epoch,
                total=getattr(self.parent, 'max_epochs', None),
                post_text=self.train_epoch_text,
            )
            self.p_batch = self.progress.add_task(
                'Batch',
                start=False,
                visible=False,
                post_text=self.train_batch_text,
            )

        if entry in ('train', 'validation'):
            self.p_validation = self.progress.add_task(
                'Validation',
                start=False,
                visible=False,
                post_text=self.validation_text,
            )

        if entry == 'test':
            self.p_test = self.progress.add_task(
                'Validation',
                start=False,
                visible=False,
                post_text=self.test_text,
            )

    @hooks.engine_end
    def stop_progress(self) -> None:
        self.progress.stop()
        self.progress.console.clear_live()

    @hooks.train_epoch_begin
    def train_epoch_start(self) -> None:
        self.progress.reset(
            self.p_batch,
            visible=True,
            total=self.parent.mixin_loop_train.num_batches,
            post_text=self.train_batch_text,
        )
        self.progress.refresh()

    @hooks.train_batch_end
    def train_batch_end(self) -> None:
        self.progress.update(
            self.p_batch,
            advance=1,
            post_text=self.train_batch_text,
            refresh=True,
        )

    @hooks.train_epoch_end
    def train_epoch_end(self, epoch: int) -> None:
        self.progress.update(
            self.p_epoch,
            completed=epoch,
            post_text=self.train_epoch_text,
            refresh=True,
        )

    @hooks.validation_epoch_begin
    def validation_epoch_start(self) -> None:
        self.progress.reset(
            self.p_validation,
            visible=True,
            total=self.parent.mixin_loop_validation.num_batches,
            post_text=self.validation_text,
        )
        self.progress.refresh()

    @hooks.validation_batch_end
    def validation_batch_end(self) -> None:
        self.progress.update(
            self.p_validation,
            advance=1,
            post_text=self.validation_text,
            refresh=True,
        )

    @hooks.validation_epoch_end
    def validation_epoch_end(self) -> None:
        self.progress.update(self.p_validation, visible=False, refresh=True)

    @hooks.test_epoch_begin
    def test_epoch_start(self) -> None:
        self.progress.reset(
            self.p_test,
            visible=True,
            total=self.parent.mixin_loop_test.num_batches,
            post_text=self.test_text,
        )
        self.progress.refresh()

    @hooks.test_batch_end
    def test_batch_end(self) -> None:
        self.progress.update(
            self.p_test,
            advance=1,
            post_text=self.test_text,
            refresh=True,
        )

    @hooks.test_epoch_end
    def test_epoch_end(self) -> None:
        self.progress.update(self.p_test, visible=False, refresh=True)

    @property
    def train_epoch_text(self) -> str:
        text = getattr(self, '_train_epoch_text', None)
        return f'\[{text}]' if text is not None else ''     # NOQA: W605 - rich requires escape of [

    @train_epoch_text.setter
    def train_epoch_text(self, value: str) -> None:
        self._train_epoch_text = value

    @property
    def train_batch_text(self) -> str:
        text = getattr(self, '_train_batch_text', None)
        return f'\[{text}]' if text is not None else ''     # NOQA: W605 - rich requires escape of [

    @train_batch_text.setter
    def train_batch_text(self, value: str) -> None:
        self._train_batch_text = value

    @property
    def validation_text(self) -> str:
        text = getattr(self, '_validation_text', None)
        return f'\[{text}]' if text is not None else ''     # NOQA: W605 - rich requires escape of [

    @validation_text.setter
    def validation_text(self, value: str) -> None:
        self._validation_text = value

    @property
    def test_text(self) -> str:
        text = getattr(self, '_test_text', None)
        return f'\[{text}]' if text is not None else ''     # NOQA: W605 - rich requires escape of [

    @test_text.setter
    def test_text(self, value: str) -> None:
        self._test_text = value


class SpeedColumn(ProgressColumn):
    """ Taken from rich.progress.TaskProgressColumn """
    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        if speed is None:
            return Text('', style='progress.remaining')

        if speed < 1:
            unit, suffix = filesize.pick_unit_and_suffix(
                int(1 / speed),
                ['', '×10³', '×10⁶', '×10⁹', '×10¹²'],
                1000,
            )
            data_speed = 1 / (speed * unit)
            return Text(f'({data_speed:.1f}{suffix} s/it)', style='progress.remaining')
        else:
            unit, suffix = filesize.pick_unit_and_suffix(
                int(speed),
                ['', '×10³', '×10⁶', '×10⁹', '×10¹²'],
                1000,
            )
            data_speed = speed / unit
            return Text(f'({data_speed:.1f}{suffix} it/s)', style='progress.remaining')

    def render(self, task: 'Task') -> Text:
        return self.render_speed(task.finished_speed or task.speed)


class TimeColumn(ProgressColumn):
    """ Taken from rich.progress.TimeRemainingColumn and rich.progress.TimeElapsedColumn """
    def render(self, task: 'Task') -> Text:
        if task.finished:
            return self.render_elapsed(task, 'progress.elapsed')
        else:
            return Text.assemble(
                self.render_elapsed(task, 'progress.elapsed'),
                Text(' / ', style='progress.elapsed'),
                self.render_remaining(task, 'progress.elapsed'),
            )

    def render_elapsed(self, task: 'Task', style: str) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text('--:--:--', style=style)

        delta = timedelta(seconds=int(elapsed))
        return Text(str(delta), style=style)

    def render_remaining(self, task: 'Task', style: str) -> Text:
        if task.total is None:
            return Text('', style=style)
        if task.time_remaining is None:
            return Text('--:--:--', style=style)

        minutes, seconds = divmod(int(task.time_remaining), 60)
        hours, minutes = divmod(minutes, 60)
        formatted = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

        return Text(formatted, style=style)

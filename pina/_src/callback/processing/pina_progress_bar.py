"""Module for the Processing Callbacks."""

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.progress_bar import (
    get_standard_metrics,
)
from pina._src.core.utils import check_consistency


class PINAProgressBar(TQDMProgressBar):
    """
    Custom progress bar callback for PINA training workflows.

    This callback extends the default Lightning progress bar by filtering the
    displayed metrics.

    Metrics can refer either to condition-specific losses, identified by the
    names assigned to the problem conditions, or to global losses. Global losses
    are selected using ``"train"``, ``"val"``, or ``"test"``, and are internally
    expanded to the corresponding logged loss metrics.

    :Example:

        >>> progress_bar = PINAProgressBar(metrics="val")
        >>> progress_bar._sorted_metrics
        ['val']
    """

    GLOBAL_LOSS_KEYS = ("train", "val", "test")

    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
        "{rate_noinv_fmt}{postfix}]"
    )

    def __init__(self, metrics="val", **kwargs):
        """
        Initialization of the :class:`PINAProgressBar`.

        :param metrics: The names of the metrics to be shown in the progress
            bar. Each entry can be either a key of a condition defined in the
            problem or one of the global loss keys: ``"train"``, ``"val"``, or
            ``"test"``. These global keys are internally expanded to the
            corresponding logged loss names. Default is ``"val"``.
        :type metrics: str | list(str) | tuple(str)
        :param dict kwargs: Additional keyword arguments passed to
            :class:`lightning.pytorch.callbacks.TQDMProgressBar`.
        :raises TypeError: If ``metrics`` contains non-string elements.
        """
        super().__init__(**kwargs)

        # Check consistency
        check_consistency(metrics, str)

        # Convert to list if a single string is provided
        if isinstance(metrics, str):
            metrics = [metrics]

        # Store the sorted metrics for later use in get_metrics
        self._sorted_metrics = sorted(metrics)

    def get_metrics(self, trainer, __):
        """
        Retrieve and filter metrics to be displayed in the progress bar.

        This method combines standard Lightning metrics with user-selected
        progress bar metrics, retaining only the metrics specified at
        initialization.

        :param Trainer trainer: The trainer managing the training loop.
        :param __: Placeholder argument, not used.
        :return: Dictionary containing the metrics to display.
        :rtype: dict

        .. note::
            This method overrides the default Lightning behaviour. It can be
            further customized by subclassing.
        """
        # Retrieve standard metrics and user-selected progress bar metrics
        standard_metrics = get_standard_metrics(trainer)
        progress_bar_metrics = trainer.progress_bar_metrics

        # Filter progress bar metrics to include only specified keys
        if progress_bar_metrics:
            progress_bar_metrics = {
                key: progress_bar_metrics[key]
                for key in progress_bar_metrics
                if key in self._sorted_metrics
            }

        return {**standard_metrics, **progress_bar_metrics}

    def setup(self, trainer, pl_module, stage):
        """
        Configure the metrics to track before execution starts.

        The requested metrics must be either names assigned to problem
        conditions or global loss keys. The accepted global loss keys are
        ``"train"``, ``"val"``, and ``"test"``.

        :param Trainer trainer: The trainer instance managing the execution.
        :param BaseSolver pl_module: The solver module being executed.
        :param str stage: Current execution stage.
        :raises KeyError: If a metric key is neither a condition key nor one of
            ``"train"``, ``"val"``, or ``"test"``.
        """
        # Get the condition keys from the problem
        condition_keys = trainer.solver.problem.conditions.keys()
        for key in self._sorted_metrics:
            if key not in condition_keys and key not in self.GLOBAL_LOSS_KEYS:
                raise KeyError(
                    f"Key '{key}' is not a valid metric. It must be either a "
                    f"problem condition key or one of {self.GLOBAL_LOSS_KEYS}."
                )

        # Add the appropriate suffix to the metric names based on batch size
        suffix = "_loss_epoch" if trainer.batch_size is not None else "_loss"
        self._sorted_metrics = [
            metric + suffix for metric in self._sorted_metrics
        ]

        return super().setup(trainer, pl_module, stage)

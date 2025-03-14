"""Module for the LinearWeightUpdate callback."""

import warnings
from lightning.pytorch.callbacks import Callback
from ..utils import check_consistency
from ..loss import ScalarWeighting


class LinearWeightUpdate(Callback):
    """
    Callback to linearly adjust the weight of a condition from an
    initial value to a target value over a specified number of epochs.
    """

    def __init__(
        self, target_epoch, condition_name, initial_value, target_value
    ):
        """
        Callback initialization.

        :param int target_epoch: The epoch at which the weight of the condition
            should reach the target value.
        :param str condition_name: The name of the condition whose weight
            should be adjusted.
        :param float initial_value: The initial value of the weight.
        :param float target_value: The target value of the weight.
        """
        super().__init__()
        self.target_epoch = target_epoch
        self.condition_name = condition_name
        self.initial_value = initial_value
        self.target_value = target_value

        # Check consistency
        check_consistency(self.target_epoch, int, subclass=False)
        check_consistency(self.condition_name, str, subclass=False)
        check_consistency(self.initial_value, (float, int), subclass=False)
        check_consistency(self.target_value, (float, int), subclass=False)

    def on_train_start(self, trainer, pl_module):
        """
        Initialize the weight of the condition to the specified `initial_value`.

        :param Trainer trainer: A :class:`~pina.trainer.Trainer` instance.
        :param SolverInterface pl_module: A
            :class:`~pina.solver.solver.SolverInterface` instance.
        """
        # Check that the target epoch is valid
        if not 0 < self.target_epoch <= trainer.max_epochs:
            raise ValueError(
                "`target_epoch` must be greater than 0"
                " and less than or equal to `max_epochs`."
            )

        # Check that the condition is a problem condition
        if self.condition_name not in pl_module.problem.conditions:
            raise ValueError(
                f"`{self.condition_name}` must be a problem condition."
            )

        # Check that the initial value is not equal to the target value
        if self.initial_value == self.target_value:
            warnings.warn(
                "`initial_value` is equal to `target_value`. "
                "No effective adjustment will be performed.",
                UserWarning,
            )

        # Check that the weighting schema is ScalarWeighting
        if not isinstance(pl_module.weighting, ScalarWeighting):
            raise ValueError("The weighting schema must be ScalarWeighting.")

        # Initialize the weight of the condition
        pl_module.weighting.weights[self.condition_name] = self.initial_value

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Adjust at each epoch the weight of the condition.

        :param Trainer trainer: A :class:`~pina.trainer.Trainer` instance.
        :param SolverInterface pl_module: A
            :class:`~pina.solver.solver.SolverInterface` instance.
        """
        if 0 < trainer.current_epoch <= self.target_epoch:
            pl_module.weighting.weights[self.condition_name] += (
                self.target_value - self.initial_value
            ) / (self.target_epoch - 1)

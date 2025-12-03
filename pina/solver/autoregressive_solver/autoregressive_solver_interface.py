"""Module for the Autoregressive solver interface."""

from abc import abstractmethod
import torch
from torch.nn.modules.loss import _Loss

from ..solver import SolverInterface
from ...utils import check_consistency
from ...loss.loss_interface import LossInterface
from ...loss import TimeWeightingInterface, ConstantTimeWeighting
from ...condition import AutoregressiveCondition


class AutoregressiveSolverInterface(SolverInterface):

    accepted_conditions_types = AutoregressiveCondition

    def __init__(self, loss=None, **kwargs):

        if loss is None:
            loss = torch.nn.MSELoss()

        super().__init__(**kwargs)

        check_consistency(loss, (LossInterface, _Loss), subclass=False)
        self._loss_fn = loss

    def optimization_cycle(self, batch):
        """
        Optimization cycle for this family of solvers.
        Iterates over each conditions and each time applies the specialized loss_data function.
        """

        condition_loss = {}
        for condition_name, points in batch:
            condition = self.problem.conditions[condition_name]

            unroll_length = getattr(condition, "unroll_length", None)
            time_weighting = getattr(condition, "time_weighting", None)

            if "unroll" in points:
                loss = self.loss_data(
                    points["input"],
                    points["unroll"],
                    unroll_length,
                    time_weighting,
                )
            condition_loss[condition_name] = loss
        return condition_loss

    @abstractmethod
    def loss_data(self, input, target, unroll_length, time_weighting):
        """
        Computes the data loss for each condition.
        N.B.: unroll_length and time_weighting are attributes of the condition.

        :param torch.Tensor input: Initial states.
        :param torch.Tensor target: Target sequences.
        :param int unroll_length: The number of steps to unroll (attribute of the condition).
        :param TimeWeightingInterface time_weighting: The time weighting strategy (attribute of the condition).
        :return: The average loss over all unroll steps.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def predict(self, initial_state, num_steps):
        """
        Make recursive predictions starting from an initial state.

        :param torch.Tensor initial_state: Initial state tensor.
        :param int num_steps: Number of steps to predict ahead.
        :return: Tensor of predictions.
        :rtype: torch.Tensor
        """
        pass

    @property
    def loss(self):
        """
        The loss function to be minimized.

        :return: The loss function to be minimized.
        :rtype: torch.nn.Module
        """
        return self._loss_fn

    @property
    def time_weighting(self):
        """
        The time weighting strategy.
        """
        return self._time_weighting

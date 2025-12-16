"""Module for the Autoregressive solver interface."""

from abc import abstractmethod
import torch
from torch.nn.modules.loss import _Loss

from ..solver import SolverInterface
from ...utils import check_consistency
from ...loss.loss_interface import LossInterface
from ...condition import DataCondition


class AutoregressiveSolverInterface(SolverInterface):

    def __init__(self, unroll_instructions, loss=None, **kwargs):
        """
        Initialization of the :class:`AutoregressiveSolverInterface` class.
        :param dict unroll_instructions: A dictionary specifying how to unroll each condition.
        this is supposed to map condition names to dict objects with unroll instructions.
        :param loss: The loss function to be minimized. If None, defaults to MSELoss.
        :type loss: torch.nn.Module or LossInterface, optional
        """

        super().__init__(**kwargs)

        if loss is None:
            loss = torch.nn.MSELoss()

        check_consistency(loss, (LossInterface, _Loss), subclass=False)
        self._loss_fn = loss
        self._unroll_instructions = unroll_instructions

    def optimization_cycle(self, batch):
        """
        Optimization cycle for this family of solvers.
        Iterates over each conditions and each time applies the specialized loss_data function.
        :param dict batch: A dictionary mapping condition names to data batches.
        :return: A dictionary mapping condition names to computed loss values.
        :rtype: dict
        """

        condition_loss = {}
        for condition_name, points in batch:
            condition_unroll_instructions = self._unroll_instructions[condition_name]
            loss = self.loss_data(
                    points["input"],
                    condition_unroll_instructions,
                )
            condition_loss[condition_name] = loss
        return condition_loss

    @abstractmethod
    def loss_data(self, input, condition_unroll_instructions):
        """
        Computes the data loss for each condition.
        N.B.: This loss_data function must make use of unroll_instructions to know how to unroll the model.

        :param torch.Tensor input: all training data.
        :param dict condition_unroll_instructions: instructions on how to unroll the model for this condition.
        :return: Computed loss value.
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
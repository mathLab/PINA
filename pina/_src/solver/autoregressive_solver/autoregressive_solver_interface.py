"""Module for the Autoregressive Solver Interface."""

from abc import abstractmethod
from pina._src.condition.data_condition import DataCondition
from pina._src.solver.solver import SolverInterface


class AutoregressiveSolverInterface(SolverInterface):
    # TODO: fix once the AutoregressiveCondition is implemented.
    """
    Abstract interface for all autoregressive solvers.

    Any solver implementing this interface is expected to be designed to learn
    dynamical systems in an autoregressive manner. The solver should handle
    conditions of type :class:`~pina.condition.data_condition.DataCondition`.
    """

    accepted_conditions_types = (DataCondition,)

    @abstractmethod
    def preprocess_step(self, current_state, **kwargs):
        """
        Pre-process the current state before passing it to the model's forward.

        :param current_state: The current state to be preprocessed.
        :type current_state: torch.Tensor | LabelTensor
        :param dict kwargs: Additional keyword arguments for pre-processing.
        :return: The preprocessed state for the given step.
        :rtype: torch.Tensor | LabelTensor
        """

    @abstractmethod
    def postprocess_step(self, predicted_state, **kwargs):
        """
        Post-process the state predicted by the model.

        :param predicted_state: The predicted state tensor from the model.
        :type predicted_state: torch.Tensor | LabelTensor
        :param dict kwargs: Additional keyword arguments for post-processing.
        :return: The post-processed predicted state tensor.
        :rtype: torch.Tensor | LabelTensor
        """

    # TODO: remove once the AutoregressiveCondition is implemented.
    @abstractmethod
    def loss_autoregressive(self, input, **kwargs):
        """
        Compute the loss for each autoregressive condition.

        :param input: The input tensor containing unroll windows.
        :type input: torch.Tensor | LabelTensor
        :param dict kwargs: Additional keyword arguments for loss computation.
        :return: The scalar loss value for the given batch.
        :rtype: torch.Tensor | LabelTensor
        """

    @abstractmethod
    def predict(self, starting_value, num_steps, **kwargs):
        """
        Generate predictions by recursively applying the model.

        :param starting_value: The initial state from which to start prediction.
            The initial state must be of shape ``[trajectories, 1, features]``,
            where the trajectory dimension can be used for batching.
        :type starting_value: torch.Tensor | LabelTensor
        :param int num_steps: The number of autoregressive steps to predict.
        :param dict kwargs: Additional keyword arguments.
        :return: The predicted trajectory, including the initial state. It has
            shape ``[trajectories, num_steps + 1, features]``, where the first
            step corresponds to the initial state.
        :rtype: torch.Tensor | LabelTensor
        """

    @property
    @abstractmethod
    def loss(self):
        """
        The loss function to be minimized.

        :return: The loss function to be minimized.
        :rtype: torch.nn.Module
        """

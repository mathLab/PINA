import torch
from torch.nn.modules.loss import _Loss

from pina.utils import check_consistency
from pina.solver.solver import SingleSolverInterface
from pina.condition import AutoregressiveCondition
from pina.loss import (
    LossInterface,
    TimeWeightingInterface,
    ConstantTimeWeighting,
)
from .autoregressive_solver_interface import AutoregressiveSolverInterface


class AutoregressiveSolver(
    AutoregressiveSolverInterface, SingleSolverInterface
):
    """
    Autoregressive Solver class.
    """

    accepted_conditions_types = AutoregressiveCondition

    def __init__(
        self,
        problem,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=False,
    ):
        """
        Initialization of the :class:`AutoregressiveSolver` class.
        """
        super().__init__(
            problem=problem,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )

    def loss_data(self, input, target, unroll_length, time_weighting):
        """
        Compute the data loss for the recursive autoregressive solver.
        This will be applied to each condition individually.
        """
        steps_to_predict = unroll_length - 1
        # weights are passed from the condition
        weights = time_weighting(steps_to_predict, device=input.device)

        total_loss = 0.0
        current_state = input

        for step in range(steps_to_predict):

            predicted_next_state = self.forward(
                current_state
            )  # [batch_size, features]
            actual_next_state = target[:, step, :]  # [batch_size, features]

            step_loss = self.loss(predicted_next_state, actual_next_state)

            total_loss += step_loss * weights[step]

            current_state = predicted_next_state.detach()

        return total_loss

    def predict(self, initial_state, num_steps):
        """
        Make recursive predictions starting from an initial state.
        """
        self.eval()  # Set model to evaluation mode

        current_state = initial_state
        predictions = [current_state]  # Store initial state without batch dim
        with torch.no_grad():
            for step in range(num_steps):
                next_state = self.forward(current_state)
                predictions.append(next_state)  # Keep batch dim for storage
                current_state = next_state

        return torch.stack(predictions)

import torch
from pina.utils import check_consistency
from pina.solver.solver import SingleSolverInterface
from pina.condition import DataCondition
from .autoregressive_solver_interface import AutoregressiveSolverInterface


class AutoregressiveSolver(
    AutoregressiveSolverInterface, SingleSolverInterface
):
    """
    Autoregressive Solver class.
    """

    accepted_conditions_types = DataCondition

    def __init__(
        self,
        unroll_instructions,
        problem,
        model,
        eps=None,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=False,
    ):
        """
        Initialization of the :class:`AutoregressiveSolver` class.
        :param dict unroll_instructions: A dictionary specifying how to unroll each condition.
        this is supposed to map condition names to dict objects with unroll instructions.
        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The model to be trained.
        :param torch.nn.Module or LossInterface or None loss: The loss function to be minimized. If None, defaults to MSELoss.
        :param TorchOptimizer or None optimizer: The optimizer to be used. If None, no optimization is performed.
        :param TorchScheduler or None scheduler: The learning rate scheduler to be used. If None, no scheduling is performed.
        :param Weighting or None weighting: The weighting scheme for combining losses from different conditions. If None, equal weighting is applied.
        :param bool use_lt: Whether to use learning rate tuning.
        """

        super().__init__(
            unroll_instructions=unroll_instructions,
            problem=problem,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )

    def loss_data(self, data, condition_unroll_instructions):
        """
        Compute the data loss for the recursive autoregressive solver.
        This will be applied to each condition individually.
        :param torch.Tensor data: all training data.
        :param dict condition_unroll_instructions: instructions on how to unroll the model for this condition.
        :return: Computed loss value.
        :rtype: torch.Tensor
        """

        initial_data, unroll_data = self.create_unroll_windows(
            data, condition_unroll_instructions
        )

        unroll_length = condition_unroll_instructions["unroll_length"]
        current_state = initial_data # [num_unrolls, features]

        losses = []
        for step in range(unroll_length):

            predicted_state = self.forward(current_state)  # [num_unrolls, features]
            target_state = unroll_data[:, step, :]  # [num_unrolls, features]
            step_loss = self._loss_fn(predicted_state, target_state)
            losses.append(step_loss)
            current_state = predicted_state
        
        step_losses = torch.stack(losses)  # [unroll_length]

        with torch.no_grad():
            weights = self.compute_adaptive_weights(step_losses.detach(), condition_unroll_instructions)
        
        weighted_loss = (step_losses * weights).sum()
        return weighted_loss

    def create_unroll_windows(self, data, condition_unroll_instructions):
        """
        Create unroll windows for each condition from the data based on the provided instructions.
        :param torch.Tensor data: The full data tensor.
        :param dict condition_unroll_instructions: Instructions on how to unroll the model for this condition.
        :return: Tuple of initial data and unroll data tensors.
        :rtype: (torch.Tensor, torch.Tensor)
        """

        unroll_length = condition_unroll_instructions["unroll_length"]

        start_list = []
        unroll_list = []
        for starting_index in self.decide_starting_indices(
            data, condition_unroll_instructions
        ):
            idx = starting_index.item()
            start = data[idx]
            target_start = idx + 1
            unroll = data[target_start : target_start + unroll_length, :]
            start_list.append(start)
            unroll_list.append(unroll)
        initial_data = torch.stack(start_list) # [num_unrolls, features]
        unroll_data = torch.stack(unroll_list) # [num_unrolls, unroll_length, features]
        return initial_data, unroll_data

    def decide_starting_indices(self, data, condition_unroll_instructions):
        """
        Decide the starting indices for unrolling based on the provided instructions.
        :param torch.Tensor data: The full data tensor.
        :param dict condition_unroll_instructions: Instructions on how to unroll the model for this condition.
        :return: Tensor of starting indices.
        :rtype: torch.Tensor
        """
        n_step, n_features = data.shape
        num_unrolls = condition_unroll_instructions.get("num_unrolls", None)
        unroll_length = condition_unroll_instructions["unroll_length"]
        randomize = condition_unroll_instructions.get("randomize", True)

        max_start = n_step - unroll_length
        indices = torch.arange(max_start, device=data.device)

        if num_unrolls is not None and num_unrolls < len(indices):
            indices = indices[:num_unrolls]

        if randomize:
            indices = indices[torch.randperm(len(indices), device=data.device)]

        return indices
    
    def compute_adaptive_weights(self, step_losses, condition_unroll_instructions):
        """
        Compute adaptive weights for each time step based on cumulative losses.
        :param torch.Tensor step_losses: Tensor of shape [unroll_length] containing losses at each time step.
        :return: Tensor of shape [unroll_length] containing normalized weights.
        :rtype: torch.Tensor
        """
        num_steps = len(step_losses)
        eps = condition_unroll_instructions.get("eps", None)
        if eps is None:
            weights =  torch.ones_like(step_losses)
        else:
            weights = torch.exp(-eps * torch.cumsum(step_losses, dim=0))
        
        return weights / weights.sum()

    def predict(self, initial_state, num_steps):
        """
        Make recursive predictions starting from an initial state.
        :param torch.Tensor initial_state: Initial state tensor.
        :param int num_steps: Number of steps to predict ahead.
        :return: Tensor of predictions.
        :rtype: torch.Tensor
        """
        self.eval()  # Set model to evaluation mode
        
        current_state = initial_state
        predictions = [current_state]
        
        with torch.no_grad():
            for step in range(num_steps):
                next_state = self.forward(current_state)
                predictions.append(next_state)
                current_state = next_state
        
        return torch.stack(predictions)
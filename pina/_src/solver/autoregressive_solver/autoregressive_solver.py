import torch
from pina._src.solver.autoregressive_solver.autoregressive_solver_interface import (
    AutoregressiveSolverInterface,
)
from pina._src.solver.solver import SingleSolverInterface
from pina._src.loss.loss_interface import LossInterface
from pina._src.core.utils import check_consistency


class AutoregressiveSolver(
    AutoregressiveSolverInterface, SingleSolverInterface
):
    r"""
    The autoregressive Solver for learning dynamical systems.

    This solver learns a one-step transition function
    :math:`\mathcal{M}: \mathbb{R}^n \rightarrow \mathbb{R}^n` that maps
    a state :math:`\mathbf{y}_t` to the next state :math:`\mathbf{y}_{t+1}`.

    During training, the model is unrolled over multiple time steps to
    learn long-term dynamics. Given an initial state :math:`\mathbf{y}_0`,
    the model generates predictions recursively:

    .. math::
        \hat{\mathbf{y}}_{t+1} = \mathcal{M}(\hat{\mathbf{y}}_t),
        \quad \hat{\mathbf{y}}_0 = \mathbf{y}_0

    The loss is computed over the entire unroll window:

    .. math::
        \mathcal{L} = \sum_{t=1}^{T} w_t \|\hat{\mathbf{y}}_t - \mathbf{y}_t\|^2

    where :math:`w_t` are exponential weights that down-weight later predictions
    to stabilize training.
    """

    def __init__(
        self,
        problem,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=False,
        reset_weights_at_epoch_start=True,
    ):
        """
        Initialization of the :class:`AutoregressiveSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is ``None``.
        :param Optimizer optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Scheduler scheduler: Learning rate scheduler.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: Whether to use LabelTensors. Default is ``False``.
        :param bool reset_weights_at_epoch_start: If ``True``, the running
            averages used for adaptive weighting are reset at the start of each
            epoch. Setting this parameter to ``False`` can improve training
            stability, especially when data are scarce. Default is ``True``.
        :raise ValueError: If the provided loss function is not compatible.
        :raise ValueError: If ``reset_weights_at_epoch_start`` is not a boolean.
        """
        super().__init__(
            problem=problem,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )

        # Check consistency
        loss = loss or torch.nn.MSELoss()
        check_consistency(
            loss, (LossInterface, torch.nn.modules.loss._Loss), subclass=False
        )
        check_consistency(reset_weights_at_epoch_start, bool)

        # Initialization
        self._loss_fn = loss
        self.reset_weights_at_epoch_start = reset_weights_at_epoch_start
        self._running_avg = {}
        self._step_count = {}

    def on_train_epoch_start(self):
        """
        Clean up running averages at the start of each epoch if
        ``reset_weights_at_epoch_start`` is True.
        """
        if self.reset_weights_at_epoch_start:
            self._running_avg.clear()
            self._step_count.clear()

    def optimization_cycle(self, batch):
        """
        The optimization cycle for autoregressive solvers.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The losses computed for all conditions in the batch.
        :rtype: dict
        """
        # Store losses for each condition in the batch
        condition_loss = {}

        # Loop through each condition and compute the autoregressive loss
        for condition_name, points in batch:
            # TODO: remove setting once AutoregressiveCondition is implemented
            # TODO: pass a temporal weighting schema in the __init__
            if hasattr(self.problem.conditions[condition_name], "settings"):
                settings = self.problem.conditions[condition_name].settings
                eps = settings.get("eps", None)
                kwargs = settings.get("kwargs", {})
            else:
                eps = None
                kwargs = {}

            loss = self.loss_autoregressive(
                points["input"],
                condition_name=condition_name,
                eps=eps,
                **kwargs,
            )
            condition_loss[condition_name] = loss
        return condition_loss

    def loss_autoregressive(
        self,
        input,
        condition_name,
        eps=None,
        aggregation_strategy=None,
        **kwargs,
    ):
        """
        Compute the loss for each autoregressive condition.

        :param input: The input tensor containing unroll windows.
        :type input: torch.Tensor | LabelTensor
        :param dict kwargs: Additional keyword arguments for loss computation.
        :raise ValueError: If ``input`` has less than 4 dimensions.
        :return: The scalar loss value for the given batch.
        :rtype: torch.Tensor | LabelTensor
        """
        # Check input dimensionality
        if input.dim() < 4:
            raise ValueError(
                "The provided input tensor must have at least 4 dimensions:"
                " [trajectories, windows, time_steps, *features]."
                f" Got shape {input.shape}."
            )

        # Initialize current state and loss list
        current_state = input[:, :, 0]
        losses = []

        # Iterate through the unroll window and compute the loss for each step
        for step in range(1, input.shape[2]):

            # Predict
            processed_input = self.preprocess_step(current_state, **kwargs)
            output = self.forward(processed_input)
            predicted_state = self.postprocess_step(output, **kwargs)

            # Compute step loss
            target_state = input[:, :, step]
            step_loss = self._loss_fn(predicted_state, target_state, **kwargs)
            losses.append(step_loss)

            # Update current state for the next step
            current_state = predicted_state

        # Stack step losses into a tensor of shape [time_steps - 1]
        step_losses = torch.stack(losses).as_subclass(torch.Tensor)

        # Compute adaptive weights based on running averages of step losses
        with torch.no_grad():
            condition_name = condition_name or "default"
            weights = self._get_weights(condition_name, step_losses, eps)

        # Aggregate the weighted step losses into a single scalar loss value
        if aggregation_strategy is None:
            aggregation_strategy = torch.mean

        return aggregation_strategy(step_losses * weights)

    def preprocess_step(self, current_state, **kwargs):
        """
        Pre-process the current state before passing it to the model's forward.

        :param current_state: The current state to be preprocessed.
        :type current_state: torch.Tensor | LabelTensor
        :param dict kwargs: Additional keyword arguments for pre-processing.
        :return: The preprocessed state for the given step.
        :rtype: torch.Tensor | LabelTensor
        """
        return current_state

    def postprocess_step(self, predicted_state, **kwargs):
        """
        Post-process the state predicted by the model.

        :param predicted_state: The predicted state tensor from the model.
        :type predicted_state: torch.Tensor | LabelTensor
        :param dict kwargs: Additional keyword arguments for post-processing.
        :return: The post-processed predicted state tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        return predicted_state

    def _get_weights(self, condition_name, step_losses, eps):
        """
        Return cached weights or compute new ones.

        :param str condition_name: The name of the current condition.
        :param torch.Tensor step_losses: The tensor of per-step losses.
        :param float eps: The weighting parameter.
        :return: The weights tensor.
        :rtype: torch.Tensor
        """
        # Determine the key for caching based on the condition name
        key = condition_name or "default"

        # Initialize the key if not in the running averages.
        if key not in self._running_avg:
            self._running_avg[key] = step_losses.detach().clone()
            self._step_count[key] = 1

        # Update running averages and counts
        else:
            self._step_count[key] += 1
            value = step_losses.detach() - self._running_avg[key]
            self._running_avg[key] += value / self._step_count[key]

        return self._compute_adaptive_weights(self._running_avg[key], eps)

    def _compute_adaptive_weights(self, step_losses, eps):
        """
        Compute temporal adaptive weights.

        :param torch.Tensor step_losses: The tensor of per-step losses.
        :param float eps: The weighting parameter.
        :return: The weights tensor.
        :rtype: torch.Tensor
        """
        # If eps is None, return uniform weights
        if eps is None:
            return torch.ones_like(step_losses)

        # Compute cumulative loss and apply exponential weighting
        cumulative_loss = -eps * torch.cumsum(step_losses, dim=0)

        return torch.exp(cumulative_loss)

    def predict(self, initial_state, n_steps, **kwargs):
        """
        Generate predictions by recursively calling the model's forward.

        :param initial_state: The initial state from which to start prediction.
            The initial state must be of shape ``[trajectories, 1, *features]``.
        :type initial_state: torch.Tensor | LabelTensor
        :param int n_steps: The number of autoregressive steps to predict.
        :param dict kwargs: Additional keyword arguments.
        :raise ValueError: If the provided initial_state tensor has less than 3
            dimensions.
        :return: The predicted trajectory, including the initial state. It has
            shape ``[trajectories, n_steps + 1, *features]``, where the first
            step corresponds to the initial state.
        :rtype: torch.Tensor | LabelTensor
        """
        # Set model to evaluation mode for prediction
        self.eval()

        # Check intial state dimensionality
        if initial_state.dim() < 3:
            raise ValueError(
                "The provided initial_state tensor must have at least 3"
                "dimensions: [trajectories, time_steps, *features]."
                f" Got shape {initial_state.shape}."
            )

        # Initialize the list of predictions with the initial state
        predictions = [initial_state]

        # Generate predictions recursively for n_steps
        with torch.no_grad():
            for _ in range(n_steps):
                input = self.preprocess_step(predictions[-1], **kwargs)
                output = self.forward(input)
                next_state = self.postprocess_step(output, **kwargs)
                predictions.append(next_state)

        return torch.stack(predictions, dim=2)

    # TODO: integrate in the Autoregressive Condition once implemented
    @staticmethod
    def unroll(data, unroll_length, n_unrolls=None, randomize=True):
        """
        Create unrolling time windows from temporal data.

        This function takes as input a tensor of shape
        ``[trajectories, time_steps, *features]`` and produces a tensor of shape
        ``[trajectories, windows, unroll_length, *features]``.
        Each window contains a sequence of subsequent states used for computing
        the multi-step loss during training.

        :param data: The temporal data tensor to be unrolled.
        :type data: torch.Tensor | LabelTensor
        :param int unroll_length: The number of time steps in each window.
        :param int n_unrolls: The maximum number of windows to return.
            If ``None``, all valid windows are returned. Default is ``None``.
        :param bool randomize: If ``True``, starting indices are randomly
            permuted before applying ``n_unrolls``. Default is ``True``.
        :raise ValueError: If the input ``data`` has less than 3 dimensions.
        :raise ValueError: If ``unroll_length`` is greater or equal to the
            number of time steps in ``data``.
        :return: A tensor of unrolled windows.
        :rtype: torch.Tensor | LabelTensor
        """
        # Check input dimensionality
        if data.dim() < 3:
            raise ValueError(
                "The provided data tensor must have at least 3 dimensions:"
                " [trajectories, time_steps, *features]."
                f" Got shape {data.shape}."
            )

        # Determine valid starting indices for unroll windows
        start_idx = AutoregressiveSolver._get_start_idx(
            n_steps=data.shape[1],
            unroll_length=unroll_length,
            n_unrolls=n_unrolls,
            randomize=randomize,
        )

        # Create unroll windows by slicing the data tensor at starting indices
        windows = [data[:, s : s + unroll_length] for s in start_idx]

        return torch.stack(windows, dim=1)

    @staticmethod
    def _get_start_idx(n_steps, unroll_length, n_unrolls=None, randomize=True):
        """
        Determine starting indices for unroll windows.

        :param int n_steps: The total number of time steps in the data.
        :param int unroll_length: The number of time steps in each window.
        :param int n_unrolls: The maximum number of windows to return.
            If ``None``, all valid windows are returned. Default is ``None``.
        :param bool randomize: If ``True``, starting indices are randomly
            permuted before applying ``n_unrolls``. Default is ``True``.
        :raise ValueError: If ``unroll_length`` is greater or equal to the
            number of time steps in ``data``.
        :return: A tensor of starting indices for unroll windows.
        :rtype: torch.Tensor
        """
        # Calculate the last valid starting index for unroll windows
        last_idx = n_steps - unroll_length

        # Raise error if no valid windows can be created
        if last_idx < 0:
            raise ValueError(
                f"Cannot create unroll windows: unroll_length ({unroll_length})"
                " cannot be greater or equal to the number of time_steps"
                f" ({n_steps})."
            )

        # Generate ordered starting indices for unroll windows
        indices = torch.arange(last_idx + 1)

        # Permute indices if randomization is enabled
        if randomize:
            indices = indices[torch.randperm(len(indices))]

        # Limit the number of windows if n_unrolls is specified
        if n_unrolls is not None and n_unrolls < len(indices):
            indices = indices[:n_unrolls]

        return indices

    @property
    def loss(self):
        """
        The loss function to be minimized.

        :return: The loss function to be minimized.
        :rtype: torch.nn.Module
        """
        return self._loss_fn

from typing import Callable
import torch
from pina._src.solver.single_model_simple_solver import SingleModelSimpleSolver
from pina._src.condition.time_series_condition import TimeSeriesCondition
from pina._src.core.utils import check_consistency


class AutoregressiveSolver(SingleModelSimpleSolver):
    r"""
    The autoregressive solver for learning dynamical systems.

    This solver learns a one-step transition function
    :math:`\mathcal{M}: \mathbb{R}^n \rightarrow \mathbb{R}^n` that maps a state
    :math:`\mathbf{y}_t` to the next state :math:`\mathbf{y}_{t+1}`.

    During training, the model is recursively unrolled over multiple time steps
    to learn long-term dynamics. Given an initial state :math:`\mathbf{y}_0`,
    predictions are generated autoregressively:

    .. math::

        \hat{\mathbf{y}}_{t+1} = \mathcal{M}(\hat{\mathbf{y}}_t),
        \qquad \hat{\mathbf{y}}_0 = \mathbf{y}_0

    At each step, the predicted state is fed back as input for the next
    prediction. The solver computes a per-step loss between the predicted and
    target states over the entire unroll window.

    The final training loss is obtained by applying ``aggregation_strategy`` to
    the weighted per-step losses:

    .. math::

        \mathcal{L} =A \left( \left \{w_t \, \ell_t \right\}_{t=1}^{T} \right),

    where :math:`\ell_t` denotes the loss between :math:`\hat{\mathbf{y}}_t` and
    :math:`\mathbf{y}_t`, and :math:`A` is the aggregation function.

    The weights :math:`w_t` are computed adaptively from the cumulative
    per-step losses using an exponential decay:

    .. math::

        w_t = \exp \left( -\varepsilon \sum_{i=1}^{t} \ell_i \right)

    For non-negative losses, the cumulative loss is non-decreasing, so later
    time steps receive smaller or equal weights. This can stabilize training
    during long autoregressive rollouts by progressively reducing the
    contribution of later predictions.
    """

    # The conditions accepted by this solver
    accepted_conditions_types = (TimeSeriesCondition,)

    def __init__(
        self,
        problem,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=False,
        eps=0.0,
        aggregation_strategy=torch.mean,
        reset_weights_at_epoch_start=True,
        kwargs=None,
    ):
        """
        Initialization of the :class:`AutoregressiveSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param torch.nn.Module loss: The loss function to be minimized.
            If ``None``, the :class:`torch.nn.MSELoss` loss is used.
            Default is ``None``.
        :param OptimizerInterface optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param SchedulerInterface scheduler: Learning rate scheduler.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: Whether to use LabelTensors. Default is ``False``.
        :param eps: The weighting parameter for the exponential weighting
            scheme. If set to ``0.0``, no weighting is applied.
            Default is ``0.0``.
        :type eps: float | int
        :param Callable aggregation_strategy: The function used to aggregate
            the time step losses. Default is ``torch.mean``.
        :param bool reset_weights_at_epoch_start: If ``True``, the running
            averages used for adaptive weighting are reset at the start of each
            epoch. Setting this parameter to ``False`` can improve training
            stability, especially when data are scarce. Default is ``True``.
        :param dict kwargs: Additional keyword arguments for the solver.
        :raises ValueError: If eps is not a float or int.
        :raises ValueError: If aggregation_strategy is not a callable function.
        :raises ValueError: If reset_weights_at_epoch_start is not a boolean.
        """
        super().__init__(
            problem=problem,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss,
            use_lt=use_lt,
        )

        # Check consistency
        check_consistency(eps, (float, int))
        check_consistency(aggregation_strategy, Callable)
        check_consistency(reset_weights_at_epoch_start, bool)

        # Initialization
        self.reset_weights_at_epoch_start = reset_weights_at_epoch_start
        self.aggregation_strategy = aggregation_strategy
        self.eps = eps
        self._kwargs = kwargs or {}
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

    def _get_weights(self, condition_name, step_losses):
        """
        Return temporal adaptive weights for the current condition.

        This method maintains an online running average of the per-step losses
        over batches for each condition. The actual computation of the adaptive
        weights is done by the :meth:`_compute_adaptive_weights` method, which
        applies an exponential decay to the cumulative loss.

        :param str condition_name: The name of the current condition.
            Used as the key for the running average cache.
        :param torch.Tensor step_losses: The tensor of per-step losses.
        :return: The temporal adaptive weights for the current condition.
        :rtype: torch.Tensor
        """
        # Determine the key for caching based on the condition name
        key = condition_name or "default"

        # Reduce over all non-time dimensions to ensure batch size consistency
        reduce_dims = tuple(range(1, step_losses.dim()))
        step_losses = step_losses.detach().mean(dim=reduce_dims, keepdim=True)

        # Initialize the key if not in the running averages.
        if key not in self._running_avg:
            self._running_avg[key] = step_losses.detach().clone()
            self._step_count[key] = 1

        # Update running averages and counts
        else:
            self._step_count[key] += 1
            value = step_losses.detach() - self._running_avg[key]
            self._running_avg[key] += value / self._step_count[key]

        return self._compute_adaptive_weights(self._running_avg[key])

    def _compute_adaptive_weights(self, step_losses):
        r"""
        Compute temporal adaptive weights from a tensor of per-step losses.

        Given a tensor of running average of per-step losses, it computes
        cumulative losses along the time dimension and applies an exponential
        decay:

        .. math::

            w_t = \exp \left( -\varepsilon \sum_{i=1}^{t} \ell_i \right)

        Therefore, later time steps receive smaller weights when the cumulative
        loss increases. This helps to stabilize training by reducing the
        influence of later predictions when the model is still learning previous
        steps.

        :param torch.Tensor step_losses: The running average of per-step losses,
            used to compute the temporal weights.
        :return: The exponential temporal adaptive weights.
        :rtype: torch.Tensor
        """
        # Compute cumulative loss and apply exponential weighting
        cumulative_loss = -self.eps * torch.cumsum(step_losses, dim=0)

        return torch.exp(cumulative_loss)

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

    def optimization_cycle(self, batch):
        """
        Compute one reduced, aggregated loss per condition in the batch.

        For TimeSeriesCondition, this method evaluates the condition to obtain
        per-step residuals, applies the pointwise loss function to each step,
        computes adaptive weights based on the step-wise losses, and returns
        the aggregated weighted loss.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The reduced, aggregated losses for all conditions.
        :rtype: dict[str, torch.Tensor]
        """
        condition_losses = {}

        for condition_name, data in batch:
            condition = self.problem.conditions[condition_name]
            condition_data = dict(data)

            # Evaluate condition to get per-step residuals
            self.step_residuals = condition.evaluate(condition_data, self)

            # Apply the loss function to each step-wise residual
            step_losses = self._loss_fn(
                self.step_residuals, torch.zeros_like(self.step_residuals)
            )

            # Compute adaptive weights and aggregate the step-wise losses
            with torch.no_grad():
                name = condition_name or "default"
                weights = self._get_weights(name, step_losses)

            # Aggregate using the configured strategy
            aggregated_loss = self.aggregation_strategy(step_losses * weights)
            condition_losses[condition_name] = aggregated_loss

        return condition_losses

    def predict(self, initial_state, n_steps, **kwargs):
        """
        Generate predictions by recursively calling the model's forward.

        :param initial_state: The initial state from which to start prediction.
            The initial state must be of shape ``[trajectories, 1, *features]``.
        :type initial_state: torch.Tensor | LabelTensor
        :param int n_steps: The number of autoregressive steps to predict.
        :param dict kwargs: Additional keyword arguments.
        :raises ValueError: If the provided initial_state tensor has less than 3
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
                "dimensions: [trajectories, 1, *features]."
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

        return torch.cat(predictions, dim=1)

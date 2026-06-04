"""Module for the causal physics-informed single-model solver class."""

import torch
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.solver.mixin.physics_informed_mixin import _PhysicsInformedMixin
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.core.utils import check_consistency, check_positive_integer
from pina._src.problem.time_dependent_problem import TimeDependentProblem
from pina._src.solver.single_model_solver import SingleModelSolver
from pina._src.core.label_tensor import LabelTensor
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class CausalPhysicsInformedSingleModelSolver(
    _PhysicsInformedMixin, SingleModelSolver
):
    r"""
    Single-model solver for causal physics-informed learning problems.

    This solver approximates the solution of a time-dependent differential
    problem using a single model and a causality-aware training objective. It is
    intended for problems whose conditions include equation residuals and
    boundary residuals evaluated across ordered time snapshots. It can be used
    only for forward problems, due to the causal nature of the training
    objective.

    Given a model :math:`\mathcal{M}`, the predicted solution is

    .. math::

        \hat{\mathbf{u}}(\mathbf{x}, t) = \mathcal{M}(\mathbf{x}, t).

    The solver minimizes a causal residual loss that weights each time snapshot
    according to the residuals accumulated at previous times. For a time
    dependent problem with governing equation operator :math:`\mathcal{A}` in
    the domain :math:`\Omega` and boundary operator :math:`\mathcal{B}` on the
    boundary :math:`\partial\Omega`, the objective can be written as

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{N_t} \sum_{i=1}^{N_t}
        \omega_i \mathcal{L}_r(t_i),

    where the residual loss at time :math:`t` is

    .. math::

        \mathcal{L}_r(t) = \frac{1}{N_{\Omega}} \sum_{j=1}^{N_{\Omega}}
        \mathcal{L}\left( \mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_j, t) \right)
        + \frac{1}{N_{\partial\Omega}} \sum_{j=1}^{N_{\partial\Omega}}
        \mathcal{L} \left( \mathcal{B}[\hat{\mathbf{u}}](\mathbf{x}_j, t)
        \right).

    The causal weights are defined as

    .. math::

        \omega_i = \exp \left( -\epsilon \sum_{k=1}^{i-1} \mathcal{L}_r(t_k)
        \right),

    where :math:`\epsilon` is a hyperparameter controlling the strength of the
    causal weighting, and :math:`\mathcal{L}` is the selected loss function,
    typically the mean squared error.

    .. seealso::

        **Original reference**: Wang, S., Sankaran, S., & Perdikaris, P. (2024).
        *Respecting causality for training physics-informed neural networks.*
        Computer Methods in Applied Mechanics and Engineering, 421, 116813.
        DOI: `10.1016/j.cma.2024.116813
        <https://doi.org/10.1016/j.cma.2024.116813>`_.

    .. note::

        This solver is compatible only with problems inheriting from
        :class:`~pina.problem.time_dependent_problem.TimeDependentProblem`.
    """

    # Accepted conditions types for this solver
    accepted_conditions_types = (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
        eps=100,
        n_steps=10,
        regularized_conditions=None,
    ):
        """
        Initialization of the :class:`CausalPhysicsInformedSingleModelSolver`
        class.

        :param BaseProblem problem: The problem to be solved.
        :param torch.nn.Module model: The model used by the solver.
        :param TorchOptimizer optimizer: The optimizer used by the solver.
            If ``None``, the ``torch.optim.Adam`` optimizer with a learning rate
            of ``0.001`` is used. Default is ``None``.
        :param TorchScheduler scheduler: The scheduler used by the solver.
            If ``None``, the ``torch.optim.lr_scheduler.ConstantLR`` scheduler
            with a factor of ``1.0`` is used. Default is ``None``.
        :param BaseWeighting weighting: The weighting strategy used to combine
            condition losses. If ``None``, no weighting is applied. Default is
            ``None``.
        :param loss: The loss function used to compute residual losses.
            If ``None``, :class:`torch.nn.MSELoss` is used. Default is ``None``.
        :param eps: The exponential decay parameter. Default is ``100``.
        :type eps: float | int
        :param int n_steps: The number of equispaced temporal steps used to
            compute the causal loss. Default is ``10``.
        :param regularized_conditions: The names of the conditions that should
            receive causal regularization. Default is ``None``.
        :raises ValueError: If the problem is not time-dependent.
        :raises ValueError: If the user does not specify any regularized
            conditions.
        :raises ValueError: If any of the specified ``regularized_conditions``
            are not present in the ``problem``'s conditions.
        :raises ValueError: If ``eps`` is not a float or int.
        :raises ValueError: If ``n_steps`` is not a positive integer.
        """
        # Ensure the problem is time-dependent
        if not isinstance(problem, TimeDependentProblem):
            raise ValueError(
                "Causal physics-informed solvers require the problem to be an "
                f"instance of TimeDependentProblem. Got {type(problem)}."
            )

        # Ensure the user specified valid regularized conditions
        if regularized_conditions is None:
            raise ValueError(
                "Causal physics-informed solvers require the user to specify "
                "the conditions that should receive causal regularization. "
                "Apply causal regularization only to time-dependent conditions."
            )

        # Check consistency
        check_consistency(eps, (int, float))
        check_consistency(regularized_conditions, str)
        check_positive_integer(n_steps, strict=True)

        # Map conditions to list if a single condition is provided
        if not isinstance(regularized_conditions, (list, tuple)):
            regularized_conditions = [regularized_conditions]

        # Ensure that all regularized conditions are present in the problem
        problem_conditions = set(problem.conditions.keys())
        for condition in regularized_conditions:
            if condition not in problem_conditions:
                raise ValueError(
                    f"Condition '{condition}' is not present in the problem."
                )

        # Initialize the parent class
        SingleModelSolver.__init__(
            self,
            problem=problem,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            loss=loss,
            use_lt=True,
        )

        # Initialize the causal regularization parameters
        self.eps = eps
        self.n_steps = n_steps
        self.regularized_conditions = regularized_conditions

    def _compute_condition_loss(self, condition, data, batch_idx):
        """
        Compute the scalar loss for a given condition and its data.

        :param BaseCondition condition: The condition for which to compute the
            loss.
        :param dict data: The data corresponding to the condition.
        :param int batch_idx: The index of the current batch.
        :return: The scalar loss for the condition.
        :rtype: torch.Tensor
        """
        # If the condition is not regularized, or is a supervised (target)
        # condition, use the standard loss computation
        if condition.name not in self.regularized_conditions or isinstance(
            condition, InputTargetCondition
        ):
            return super()._compute_condition_loss(condition, data, batch_idx)

        # Clone the input tensor if it exists to avoid in-place modifications
        if "input" in data and hasattr(data["input"], "clone"):
            data = dict(data)
            data["input"] = data["input"].clone()

        # Extract the temporal domain
        time_domain = self.problem.temporal_domain

        # Define the time steps for the causal loss computation
        if time_domain.range:
            time_steps = torch.linspace(
                time_domain.range[self.temporal_variable][0],
                time_domain.range[self.temporal_variable][1],
                self.n_steps,
                device=data["input"].device,
                dtype=data["input"].dtype,
            )

        # If no range is defined, use the unique temporal value
        else:
            time_steps = torch.tensor(
                [time_domain.fixed[self.temporal_variable]],
                device=data["input"].device,
                dtype=data["input"].dtype,
            )

        # Initialize the list to store the loss for each time step
        time_loss = []

        # Iterate over the time steps
        for step in time_steps:

            # Append the temporal variable to the spatial input tensor
            spatial_pts = data["input"].extract(self.spatial_variables)
            time_pts = LabelTensor(
                torch.ones(spatial_pts.shape[0], 1, device=spatial_pts.device)
                * step,
                labels=[self.temporal_variable],
            )
            pts = {
                "input": LabelTensor.cat(
                    [spatial_pts, time_pts], dim=1
                ).requires_grad_(True)
            }

            # Compute and store the residual tensor for the condition
            self.residual_tensor = condition.evaluate(pts, self)

            # Retrieve condition name for more complex weighting schemes
            condition_name = (
                condition.name if hasattr(condition, "name") else None
            )

            # Compute the tensor loss from the residual tensor
            condition_tensor_loss = self._loss_from_residual(condition_name)

            # Append the loss for the current time step to the list
            time_loss.append(condition_tensor_loss)

        # Compute the time-adaptive weights for the causal loss
        time_loss = torch.stack(time_loss)
        with torch.no_grad():
            weights = self._compute_weights(time_loss)

        # Compute the scalar loss from the tensor loss and return it
        condition_scalar_loss = self._apply_reduction(weights * time_loss)

        return condition_scalar_loss

    def _compute_weights(self, loss):
        """
        Compute the temporal adaptive weights for the causal loss based on the
        cumulative loss.

        :param LabelTensor loss: The physics loss values.
        :return: The computed weights for the physics loss.
        :rtype: LabelTensor
        """
        # Compute the cumulative loss and apply exponential decay
        cumulative_loss = torch.cumsum(loss, dim=0)
        return torch.exp(-self.eps * cumulative_loss)

    @property
    def temporal_variable(self):
        """
        The temporal variable of the problem.

        :return: The temporal variable name.
        :rtype: str
        :raises ValueError: If the problem does not have exactly one temporal
            variable.
        """
        # Extract the temporal variable from the problem
        temporal_variables = self.problem.temporal_variables

        # Raise error if there is not exactly one temporal variable
        if len(temporal_variables) != 1:
            raise ValueError(
                "Causal physics-informed solvers require exactly one temporal "
                f"variable. Got {temporal_variables}."
            )

        return temporal_variables[0]

    @property
    def spatial_variables(self):
        """
        The spatial variables of the problem.

        :return: The spatial variable names.
        :rtype: list[str]
        :raises ValueError: If the problem does not have any spatial variables.
        """
        # Determine the spatial variables by excluding the temporal variable
        spatial_variables = [
            v
            for v in self.problem.input_variables
            if v != self.temporal_variable
        ]

        # Raise error if there are no spatial variables left
        if not spatial_variables:
            raise ValueError(
                "Causal physics-informed solvers require at least one spatial "
                "variable in addition to time."
            )

        return spatial_variables

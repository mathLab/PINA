"""Module for the gradient physics-informed single-model solver class."""

from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.solver.single_model_solver import SingleModelSolver
from pina._src.solver.mixin.gradient_enhanced_mixin import (
    GradientEnhancedMixin,
)
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class GradientPhysicsInformedSingleModelSolver(
    PhysicsInformedMixin, GradientEnhancedMixin, SingleModelSolver
):
    r"""
    Single-model solver for gradient-enhanced physics-informed learning
    problems.

    This solver approximates the solution of a differential problem using a
    single model and augments the standard physics-informed objective with
    gradient-enhanced residual terms. It can be used for both forward and
    inverse problems.

    Given a model :math:`\mathcal{M}`, the predicted solution is

    .. math::

        \hat{\mathbf{u}}(\mathbf{x}) = \mathcal{M}(\mathbf{x}).

    The solver minimizes both the residuals of the differential operators
    defining the problem and the gradients of those residuals with respect to
    the input variables. For a problem with governing equation operator
    :math:`\mathcal{A}` in the domain :math:`\Omega` and boundary operator
    :math:`\mathcal{B}` on the boundary :math:`\partial\Omega`, the objective
    can be written as

    .. math::

        \mathcal{L}_{\mathrm{problem}} = \frac{1}{N_{\Omega}}
        \sum_{i=1}^{N_{\Omega}} \mathcal{L}
        \left( \mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_i) \right)
        + \frac{1}{N_{\partial\Omega}} \sum_{i=1}^{N_{\partial\Omega}}
        \mathcal{L} \left( \mathcal{B}[\hat{\mathbf{u}}](\mathbf{x}_i) \right)
        + \frac{1}{N_{\Omega}} \sum_{i=1}^{N_{\Omega}} \mathcal{L}
        \left( \nabla_{\mathbf{x}} \mathcal{A}[\hat{\mathbf{u}}](\mathbf{x}_i)
        \right) + \frac{1}{N_{\partial\Omega}} \sum_{i=1}^{N_{\partial\Omega}}
        \mathcal{L} \left( \nabla_{\mathbf{x}} \mathcal{B}[\hat{\mathbf{u}}]
        (\mathbf{x}_i) \right),

    where :math:`\mathcal{L}` is the selected loss function, typically the mean
    squared error.

    .. seealso::

        **Original reference**: Yu, J., Lu, L., Meng, X., & Karniadakis, G. E.
        (2022). *Gradient-enhanced physics-informed neural networks for forward
        and inverse PDE problems.* Computer Methods in Applied Mechanics and
        Engineering, 393, 114823.
        DOI: `10.1016/j.cma.2022.114823
        <https://doi.org/10.1016/j.cma.2022.114823>`_.
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
        regularization_weight=1.0,
        regularized_conditions=None,
    ):
        """
        Initialization of the :class:`GradientPhysicsInformedSingleModelSolver`
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
        :param regularization_weight: The weight of the gradient regularization
            term. Default is ``1.0``.
        :type regularization_weight: float | int
        :param regularized_conditions: The names of the conditions that should
            receive gradient regularization. If ``None``, all conditions are
            regularized. Default is ``None``.
        """
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

        # Initialize the gradient-enhanced components
        self._init_gradient_enhanced_components(
            regularization_weight=regularization_weight,
            regularized_conditions=regularized_conditions,
        )

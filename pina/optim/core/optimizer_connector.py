"""Module for the PINA Optimizer."""

from .optim_connector_interface import OptimizerConnectorInterface, _HooksOptim


class OptimizerConnector(OptimizerConnectorInterface, _HooksOptim):
    """
    Abstract base class for defining an optimizer connector. All specific
    optimizers connectors should inherit form this class and implement the
    required methods.
    """

    def __init__(self, optimizer_class, **optimizer_class_kwargs):
        """
        Initialize connector parameters

        :param torch.optim.Optimizer optimizer_class: The torch optimizer class.
        :param dict optimizer_class_kwargs: The optimizer kwargs.
        """
        super().__init__()
        self._optimizer_class = optimizer_class
        self._optimizer_instance = None
        self._optim_kwargs = optimizer_class_kwargs
        self._solver = None

    def parameter_hook(self, parameters):
        """
        Abstract method to define the hook logic for the optimizer. This hook
        is used to initialize the optimizer instance with the given parameters.

        :param dict parameters: The parameters of the model to be optimized.
        """
        self._optimizer_instance = self._optimizer_class(
            parameters, **self._optim_kwargs
        )

    def solver_hook(self, solver):
        """
        Method to define the hook logic for the optimizer. This hook
        is used to hook the optimizer instance with the given parameters.

        :param SolverInterface solver: The solver to hook.
        """
        if not self.hooks_done["parameter_hook"]:
            raise RuntimeError(
                "Cannot run 'solver_hook' before 'parameter_hook'. "
                "Please call 'parameter_hook' first to initialize "
                "the solver parameters."
            )
        # hook to both instance and connector the solver
        self._solver = solver
        self._optimizer_instance.solver = solver

    def _register_hooks(self, **kwargs):
        """
        Register the optimizers hooks. This method inspects keyword arguments
        for known keys (`parameters`, `solver`, ...) and applies the
        corresponding hooks.

        It allows flexible integration with
        different workflows without enforcing a strict method signature.

        This method is used inside the
        :class:`~pina.solver.solver.SolverInterface` class.

        :param kwargs: Expected keys may include:
            - ``parameters``: Parameters to be registered for optimization.
            - ``solver``: Solver instance.
        """
        # parameter hook
        parameters = kwargs.get("parameters", None)
        if parameters is not None:
            self.parameter_hook(parameters)
        # solver hook
        solver = kwargs.get("solver", None)
        if solver is not None:
            self.solver_hook(solver)

    @property
    def solver(self):
        """
        Get the solver hooked to the optimizer.
        """
        if not self.hooks_done["solver_hook"]:
            raise RuntimeError(
                "Solver has not been hooked."
                "Override the method solver_hook to hook the solver to "
                "the optimizer."
            )
        return self._solver

    @property
    def instance(self):
        """
        Get the optimizer instance.

        :return: The optimizer instance
        :rtype: torch.optim.Optimizer
        """
        return self._optimizer_instance

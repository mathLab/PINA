"""Module for the SingleSolverInterface base class."""

from abc import ABCMeta
import torch

from pina._src.problem.inverse_problem import InverseProblem
from pina._src.optim.optimizer_interface import Optimizer
from pina._src.optim.scheduler_interface import Scheduler
from pina._src.core.utils import check_consistency
from pina._src.solver.solver_interface import SolverInterface


class SingleSolverInterface(SolverInterface, metaclass=ABCMeta):
    """
    Base class for PINA solvers using a single :class:`torch.nn.Module`.
    """

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`SingleSolverInterface` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param Optimizer optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is
            used. Default is ``None``.
        :param Scheduler scheduler: The scheduler to be used.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
        """
        if optimizer is None:
            optimizer = self.default_torch_optimizer()

        if scheduler is None:
            scheduler = self.default_torch_scheduler()

        super().__init__(problem=problem, use_lt=use_lt, weighting=weighting)

        # check consistency of models argument and encapsulate in list
        check_consistency(model, torch.nn.Module)
        # check scheduler consistency and encapsulate in list
        check_consistency(scheduler, Scheduler)
        # check optimizer consistency and encapsulate in list
        check_consistency(optimizer, Optimizer)

        # initialize the model (needed by Lightining to go to different devices)
        self._pina_models = torch.nn.ModuleList([model])
        self._pina_optimizers = [optimizer]
        self._pina_schedulers = [scheduler]

    def forward(self, x):
        """
        Forward pass implementation.

        :param x: Input tensor.
        :type x: torch.Tensor | LabelTensor | Graph | Data
        :return: Solver solution.
        :rtype: torch.Tensor | LabelTensor | Graph | Data
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Optimizer configuration for the solver.

        :return: The optimizer and the scheduler
        :rtype: tuple[list[Optimizer], list[Scheduler]]
        """
        self.optimizer.hook(self.model.parameters())
        if isinstance(self.problem, InverseProblem):
            self.optimizer.instance.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.instance], [self.scheduler.instance])

    @property
    def model(self):
        """
        The model used for training.

        :return: The model used for training.
        :rtype: torch.nn.Module
        """
        return self._pina_models[0]

    @property
    def scheduler(self):
        """
        The scheduler used for training.

        :return: The scheduler used for training.
        :rtype: Scheduler
        """
        return self._pina_schedulers[0]

    @property
    def optimizer(self):
        """
        The optimizer used for training.

        :return: The optimizer used for training.
        :rtype: Optimizer
        """
        return self._pina_optimizers[0]

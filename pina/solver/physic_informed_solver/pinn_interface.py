"""Module for Physics Informed Neural Network Interface."""

from abc import ABCMeta, abstractmethod
import torch
from torch.nn.modules.loss import _Loss

from ..solver import SolverInterface
from ...utils import check_consistency
from ...loss.loss_interface import LossInterface
from ...problem import InverseProblem
from ...condition import (
    InputTargetCondition,
    InputEquationCondition,
    DomainEquationCondition,
)


class PINNInterface(SolverInterface, metaclass=ABCMeta):
    """
    Base PINN solver class. This class implements the Solver Interface
    for Physics Informed Neural Network solver.

    This class can be used to define PINNs with multiple ``optimizers``,
    and/or ``models``.
    By default it takes :class:`~pina.problem.abstract_problem.AbstractProblem`,
    so the user can choose what type of problem the implemented solver,
    inheriting from this class, is designed to solve.
    """

    accepted_conditions_types = (
        InputTargetCondition,
        InputEquationCondition,
        DomainEquationCondition,
    )

    def __init__(self, problem, loss=None, **kwargs):
        """
        :param AbstractProblem problem: A problem definition instance.
        :param torch.nn.Module loss: The loss function to be minimized,
            default `None`.
        """

        if loss is None:
            loss = torch.nn.MSELoss()

        super().__init__(problem=problem, use_lt=True, **kwargs)

        # check consistency
        check_consistency(loss, (LossInterface, _Loss), subclass=False)

        # assign variables
        self._loss = loss

        # inverse problem handling
        if isinstance(self.problem, InverseProblem):
            self._params = self.problem.unknown_parameters
            self._clamp_params = self._clamp_inverse_problem_params
        else:
            self._params = None
            self._clamp_params = lambda: None

        self.__metric = None

    def optimization_cycle(self, batch):
        return self._run_optimization_cycle(batch, self.loss_phys)

    @torch.set_grad_enabled(True)
    def validation_step(self, batch):
        losses = self._run_optimization_cycle(batch, self._residual_loss)
        loss = self.weighting.aggregate(losses).as_subclass(torch.Tensor)
        self.store_log("val_loss", loss, self.get_batch_size(batch))
        return loss

    @torch.set_grad_enabled(True)
    def test_step(self, batch):
        losses = self._run_optimization_cycle(batch, self._residual_loss)
        loss = self.weighting.aggregate(losses).as_subclass(torch.Tensor)
        self.store_log("test_loss", loss, self.get_batch_size(batch))
        return loss

    def loss_data(self, input_pts, output_pts):
        """
        The data loss for the PINN solver. It computes the loss between
        the network output against the true solution. This function
        should not be override if not intentionally.

        :param LabelTensor input_pts: The input to the neural networks.
        :param LabelTensor output_pts: The true solution to compare the
            network solution.
        :return: The residual loss averaged on the input coordinates
        :rtype: torch.Tensor
        """
        return self._loss(self.forward(input_pts), output_pts)

    @abstractmethod
    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the physics informed solver based on given
        samples and equation. This method must be override by all inherited
        classes and it is the core to define a new physics informed solver.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor
        """

    def compute_residual(self, samples, equation):
        """
        Compute the residual for Physics Informed learning. This function
        returns the :obj:`~pina.equation.equation.Equation` specified in the
        :obj:`~pina.condition.Condition` evaluated at the ``samples`` points.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The residual of the neural network solution.
        :rtype: LabelTensor
        """
        try:
            residual = equation.residual(samples, self.forward(samples))
        except TypeError:
            # this occurs when the function has three inputs (inverse problem)
            residual = equation.residual(
                samples, self.forward(samples), self._params
            )
        return residual

    def _residual_loss(self, samples, equation):
        residuals = self.compute_residual(samples, equation)
        return self.loss(residuals, torch.zeros_like(residuals))

    def _run_optimization_cycle(self, batch, loss_residuals):
        condition_loss = {}
        for condition_name, points in batch:
            self.__metric = condition_name
            # if equations are passed
            if "target" not in points:
                input_pts = points["input"]
                condition = self.problem.conditions[condition_name]
                loss = loss_residuals(
                    input_pts.requires_grad_(), condition.equation
                )
            # if data are passed
            else:
                input_pts = points["input"]
                output_pts = points["target"]
                loss = self.loss_data(
                    input_pts=input_pts.requires_grad_(), output_pts=output_pts
                )
            # append loss
            condition_loss[condition_name] = loss
        # clamp unknown parameters in InverseProblem (if needed)
        self._clamp_params()
        return condition_loss

    def _clamp_inverse_problem_params(self):
        """
        Clamps the parameters of the inverse problem
        solver to the specified ranges.
        """
        for v in self._params:
            self._params[v].data.clamp_(
                self.problem.unknown_parameter_domain.range_[v][0],
                self.problem.unknown_parameter_domain.range_[v][1],
            )

    @property
    def loss(self):
        """
        Loss used for training.
        """
        return self._loss

    @property
    def current_condition_name(self):
        """
        The current condition name.
        """
        return self.__metric

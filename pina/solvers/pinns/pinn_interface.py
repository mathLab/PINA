""" Module for Physics Informed Neural Network Interface."""

import torch
from abc import ABCMeta, abstractmethod
from torch.nn.modules.loss import _Loss

from ..solver import SolverInterface
from ...utils import check_consistency
from ...loss.loss_interface import LossInterface
from ...problem import InverseProblem
from ...condition import (
    InputOutputPointsCondition,
    InputPointsEquationCondition,
    DomainEquationCondition
)


class PINNInterface(SolverInterface, metaclass=ABCMeta):
    """
    Base PINN solver class. This class implements the Solver Interface
    for Physics Informed Neural Network solvers.

    This class can be used to define PINNs with multiple ``optimizers``, 
    and/or ``models``.
    By default it takes :class:`~pina.problem.abstract_problem.AbstractProblem`,
    so the user can choose which type of problem the implemented solver,
    inheriting from this class, is designed to solve.
    """
    accepted_conditions_types = (
        InputOutputPointsCondition,
        InputPointsEquationCondition,
        DomainEquationCondition
    )

    def __init__(self,
                 problem,
                 loss=None,
                 **kwargs):
        """
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        """

        if loss is None:
            loss = torch.nn.MSELoss()

        super().__init__(problem=problem,
                         use_lt=True,
                         **kwargs)

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
        condition_loss = {}
        for condition_name, points in batch:
            self.__metric = condition_name
            # if equations are passed
            if 'output_points' not in points:
                input_pts = points['input_points']
                condition = self.problem.conditions[condition_name]
                loss = self.loss_phys(
                    input_pts.requires_grad_(),
                    condition.equation
                )
            # if data are passed
            else:
                input_pts = points['input_points']
                output_pts = points['output_points']
                loss = self.loss_data(
                    input_pts=input_pts,
                    output_pts=output_pts
                )
            # append loss
            condition_loss[condition_name] = loss
        # clamp unknown parameters in InverseProblem (if needed)
        self._clamp_params()
        return condition_loss

    @torch.set_grad_enabled(True)
    def validation_step(self, batch):
        super().validation_step(batch)

    @torch.set_grad_enabled(True)
    def test_step(self, batch):
        super().test_step(batch)
    
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
        pass

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
        except (TypeError):
            # this occurs when the function has three inputs (inverse problem)
            residual = equation.residual(
                samples,
                self.forward(samples),
                self._params
            )
        return residual

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

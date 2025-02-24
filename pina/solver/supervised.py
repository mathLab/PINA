"""Module for SupervisedSolver"""

import torch
from torch.nn.modules.loss import _Loss
from .solver import SingleSolverInterface
from ..utils import check_consistency
from ..loss.loss_interface import LossInterface
from ..condition import InputOutputPointsCondition


class SupervisedSolver(SingleSolverInterface):
    r"""
    SupervisedSolver solver class. This class implements a SupervisedSolver,
    using a user specified ``model`` to solve a specific ``problem``.

    The  Supervised Solver class aims to find
    a map between the input :math:`\mathbf{s}:\Omega\rightarrow\mathbb{R}^m`
    and the output :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`. The input
    can be discretised in space (as in :obj:`~pina.solver.rom.ROMe2eSolver`),
    or not (e.g. when training Neural Operators).

    Given a model :math:`\mathcal{M}`, the following loss function is
    minimized during training:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathbf{u}_i - \mathcal{M}(\mathbf{v}_i))

    where :math:`\mathcal{L}` is a specific loss function,
    default Mean Square Error:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    In this context :math:`\mathbf{u}_i` and :math:`\mathbf{v}_i` means that
    we are seeking to approximate multiple (discretised) functions given
    multiple (discretised) input functions.
    """

    accepted_conditions_types = InputOutputPointsCondition

    def __init__(
        self,
        problem,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        :param WeightingInterface weighting: The loss weighting to use.
        :param bool use_lt: Using LabelTensors as input during training.
        """
        if loss is None:
            loss = torch.nn.MSELoss()

        super().__init__(
            model=model,
            problem=problem,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )

        # check consistency
        check_consistency(
            loss, (LossInterface, _Loss, torch.nn.Module), subclass=False
        )
        self._loss = loss

    def optimization_cycle(self, batch):
        """
        Perform an optimization cycle by computing the loss for each condition
        in the given batch.

        :param batch: A batch of data, where each element is a tuple containing
                    a condition name and a dictionary of points.
        :type batch: list of tuples (str, dict)
        :return: The computed loss for the all conditions in the batch,
            cast to a subclass of `torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict(torch.Tensor)
        """
        condition_loss = {}
        for condition_name, points in batch:
            input_pts, output_pts = (
                points["input_points"],
                points["output_points"],
            )
            condition_loss[condition_name] = self.loss_data(
                input_pts=input_pts, output_pts=output_pts
            )
        return condition_loss

    def loss_data(self, input_pts, output_pts):
        """
        The data loss for the Supervised solver. It computes the loss between
        the network output against the true solution. This function
        should not be override if not intentionally.

        :param input_pts: The input to the neural networks.
        :type input_pts: LabelTensor | torch.Tensor
        :param output_pts: The true solution to compare the
            network solution.
        :type output_pts: LabelTensor | torch.Tensor
        :return: The residual loss.
        :rtype: torch.Tensor
        """
        return self._loss(self.forward(input_pts), output_pts)

    @property
    def loss(self):
        """
        Loss for training.
        """
        return self._loss

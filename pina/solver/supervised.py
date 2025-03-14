"""Module for the Supervised solver."""

import torch
from torch.nn.modules.loss import _Loss
from .solver import SingleSolverInterface
from ..utils import check_consistency
from ..loss.loss_interface import LossInterface
from ..condition import InputTargetCondition


class SupervisedSolver(SingleSolverInterface):
    r"""
    Supervised Solver solver class. This class implements a Supervised Solver,
    using a user specified ``model`` to solve a specific ``problem``.

    The  Supervised Solver class aims to find a map between the input
    :math:`\mathbf{s}:\Omega\rightarrow\mathbb{R}^m` and the output
    :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`.

    Given a model :math:`\mathcal{M}`, the following loss function is
    minimized during training:

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathbf{u}_i - \mathcal{M}(\mathbf{v}_i)),

    where :math:`\mathcal{L}` is a specific loss function, typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    In this context, :math:`\mathbf{u}_i` and :math:`\mathbf{v}_i` indicates
    the will to approximate multiple (discretised) functions given multiple
    (discretised) input functions.
    """

    accepted_conditions_types = InputTargetCondition

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
        Initialization of the :class:`SupervisedSolver` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param torch.nn.Module loss: The loss function to be minimized.
            If `None`, the :class:`torch.nn.MSELoss` loss is used.
            Default is `None`.
        :param Optimizer optimizer: The optimizer to be used.
            If `None`, the :class:`torch.optim.Adam` optimizer is used.
            Default is ``None``.
        :param Scheduler scheduler: Learning rate scheduler.
            If `None`, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If `None`, no weighting schema is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
            Default is ``True``.
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
        The optimization cycle for the solvers.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """
        condition_loss = {}
        for condition_name, points in batch:
            input_pts, output_pts = (
                points["input"],
                points["target"],
            )
            condition_loss[condition_name] = self.loss_data(
                input_pts=input_pts, output_pts=output_pts
            )
        return condition_loss

    def loss_data(self, input_pts, output_pts):
        """
        Compute the data loss for the Supervised solver by evaluating the loss
        between the network's output and the true solution. This method should
        not be overridden, if not intentionally.

        :param input_pts: The input points to the neural network.
        :type input_pts: LabelTensor | torch.Tensor
        :param output_pts: The true solution to compare with the network's
            output.
        :type output_pts: LabelTensor | torch.Tensor
        :return: The supervised loss, averaged over the number of observations.
        :rtype: torch.Tensor
        """
        return self._loss(self.forward(input_pts), output_pts)

    @property
    def loss(self):
        """
        The loss function to be minimized.

        :return: The loss function to be minimized.
        :rtype: torch.nn.Module
        """
        return self._loss

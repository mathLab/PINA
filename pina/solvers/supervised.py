""" Module for SupervisedSolver """

import torch
from torch.nn.modules.loss import _Loss
from ..optim import TorchOptimizer, TorchScheduler
from .solver import SolverInterface
from ..label_tensor import LabelTensor
from ..utils import check_consistency
from ..loss.loss_interface import LossInterface


class SupervisedSolver(SolverInterface):
    r"""
    SupervisedSolver solver class. This class implements a SupervisedSolver,
    using a user specified ``model`` to solve a specific ``problem``.

    The  Supervised Solver class aims to find
    a map between the input :math:`\mathbf{s}:\Omega\rightarrow\mathbb{R}^m`
    and the output :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`. The input
    can be discretised in space (as in :obj:`~pina.solvers.rom.ROMe2eSolver`),
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
    accepted_condition_types = ['supervised']
    __name__ = 'SupervisedSolver'

    def __init__(self,
                 problem,
                 model,
                 loss=None,
                 optimizer=None,
                 scheduler=None,
                 extra_features=None):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        """
        if loss is None:
            loss = torch.nn.MSELoss()

        if optimizer is None:
            optimizer = TorchOptimizer(torch.optim.Adam, lr=0.001)

        if scheduler is None:
            scheduler = TorchScheduler(torch.optim.lr_scheduler.ConstantLR)

        super().__init__(models=model,
                         problem=problem,
                         optimizers=optimizer,
                         schedulers=scheduler,
                         extra_features=extra_features)

        # check consistency
        check_consistency(loss, (LossInterface, _Loss), subclass=False)
        self._loss = loss
        self._model = self._pina_models[0]
        self._optimizer = self._pina_optimizers[0]
        self._scheduler = self._pina_schedulers[0]

    def forward(self, x):
        """Forward pass implementation for the solver.

        :param torch.Tensor x: Input tensor.
        :return: Solver solution.
        :rtype: torch.Tensor
        """

        output = self._model(x)

        output.labels = self.problem.output_variables
        return output

    def configure_optimizers(self):
        """Optimizer configuration for the solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        self._optimizer.hook(self._model.parameters())
        self._scheduler.hook(self._optimizer)
        return ([self._optimizer.optimizer_instance],
                [self._scheduler.scheduler_instance])

    def training_step(self, batch, batch_idx):
        """Solver training step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :param batch_idx: The batch index.
        :type batch_idx: int
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """
        condition_idx = batch.supervised.condition_indices

        for condition_id in range(condition_idx.min(), condition_idx.max() + 1):

            condition_name = self._dataloader.condition_names[condition_id]
            condition = self.problem.conditions[condition_name]
            pts = batch.supervised.input_points
            out = batch.supervised.output_points
            if condition_name not in self.problem.conditions:
                raise RuntimeError("Something wrong happened.")

            # for data driven mode
            if not hasattr(condition, "output_points"):
                raise NotImplementedError(
                    f"{type(self).__name__} works only in data-driven mode.")
            output_pts = out[condition_idx == condition_id]
            input_pts = pts[condition_idx == condition_id]

            input_pts.labels = pts.labels
            output_pts.labels = out.labels

            loss = self.loss_data(input_pts=input_pts, output_pts=output_pts)
            loss = loss.as_subclass(torch.Tensor)

        self.log("mean_loss", float(loss), prog_bar=True, logger=True)
        return loss

    def loss_data(self, input_pts, output_pts):
        """
        The data loss for the Supervised solver. It computes the loss between
        the network output against the true solution. This function
        should not be override if not intentionally.

        :param LabelTensor input_pts: The input to the neural networks.
        :param LabelTensor output_pts: The true solution to compare the
            network solution.
        :return: The residual loss averaged on the input coordinates
        :rtype: torch.Tensor
        """
        return self._loss(self.forward(input_pts), output_pts)

    @property
    def scheduler(self):
        """
        Scheduler for training.
        """
        return self._scheduler

    @property
    def optimizer(self):
        """
        Optimizer for training.
        """
        return self._optimizer

    @property
    def model(self):
        """
        Neural network for training.
        """
        return self._model

    @property
    def loss(self):
        """
        Loss for training.
        """
        return self._loss

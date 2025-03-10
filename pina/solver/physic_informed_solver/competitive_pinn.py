"""Module for Competitive PINN."""

import copy
import torch

from ...problem import InverseProblem
from .pinn_interface import PINNInterface
from ..solver import MultiSolverInterface


class CompetitivePINN(PINNInterface, MultiSolverInterface):
    r"""
    Competitive Physics Informed Neural Network (PINN) solver class.
    This class implements Competitive Physics Informed Neural
    Network solver, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    The Competitive Physics Informed Network aims to find
    the solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`
    of the differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    with a minimization (on ``model`` parameters) maximation (
    on ``discriminator`` parameters) of the loss function

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(D(\mathbf{x}_i)\mathcal{A}[\mathbf{u}](\mathbf{x}_i))+
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(D(\mathbf{x}_i)\mathcal{B}[\mathbf{u}](\mathbf{x}_i))

    where :math:`D` is the discriminator network, which tries to find the points
    where the network performs worst, and :math:`\mathcal{L}` is a specific loss
    function, default Mean Square Error:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Zeng, Qi, et al.
        "Competitive physics informed networks." International Conference on
        Learning Representations, ICLR 2022
        `OpenReview Preprint <https://openreview.net/forum?id=z9SIj-IM7tn>`_.

    .. warning::
        This solver does not currently support the possibility to pass
        ``extra_feature``.
    """

    def __init__(
        self,
        problem,
        model,
        discriminator=None,
        optimizer_model=None,
        optimizer_discriminator=None,
        scheduler_model=None,
        scheduler_discriminator=None,
        weighting=None,
        loss=None,
    ):
        """
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module model: The neural network model to use
            for the model.
        :param torch.nn.Module discriminator: The neural network model to use
            for the discriminator. If ``None``, the discriminator network will
            have the same architecture as the model network.
        :param torch.optim.Optimizer optimizer_model: The neural network
            optimizer to use for the model network; default `None`.
        :param torch.optim.Optimizer optimizer_discriminator: The neural network
            optimizer to use for the discriminator network; default `None`.
        :param torch.optim.LRScheduler scheduler_model: Learning rate scheduler
            for the model; default `None`.
        :param torch.optim.LRScheduler scheduler_discriminator: Learning rate
            scheduler for the discriminator; default `None`.
        :param WeightingInterface weighting: The weighting schema to use;
            default `None`.
        :param torch.nn.Module loss: The loss function to be minimized;
            default `None`.
        """
        if discriminator is None:
            discriminator = copy.deepcopy(model)

        super().__init__(
            models=[model, discriminator],
            problem=problem,
            optimizers=[optimizer_model, optimizer_discriminator],
            schedulers=[scheduler_model, scheduler_discriminator],
            weighting=weighting,
            loss=loss,
        )

        # Set automatic optimization to False
        self.automatic_optimization = False

    def forward(self, x):
        r"""
        Forward pass implementation for the PINN solver. It returns the function
        evaluation :math:`\mathbf{u}(\mathbf{x})` at the control points
        :math:`\mathbf{x}`.

        :param LabelTensor x: Input tensor for the PINN solver. It expects
            a tensor :math:`N \times D`, where :math:`N` the number of points
            in the mesh, :math:`D` the dimension of the problem,
        :return: PINN solution evaluated at contro points.
        :rtype: LabelTensor
        """
        return self.neural_net(x)

    def training_step(self, batch):
        """
        Solver training step, overridden to perform manual optimization.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """
        # train model
        self.optimizer_model.instance.zero_grad()
        loss = super().training_step(batch)
        self.manual_backward(loss)
        self.optimizer_model.instance.step()
        # train discriminator
        self.optimizer_discriminator.instance.zero_grad()
        loss = super().training_step(batch)
        self.manual_backward(-loss)
        self.optimizer_discriminator.instance.step()
        return loss

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the Competitive PINN solver based on given
        samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor
        """
        # Compute discriminator bets
        discriminator_bets = self.discriminator(samples)

        # Compute residual and multiply discriminator_bets
        residual = self.compute_residual(samples=samples, equation=equation)
        residual = residual * discriminator_bets

        # Compute competitive residual.
        loss_val = self.loss(
            torch.zeros_like(residual, requires_grad=True),
            residual,
        )
        return loss_val

    def configure_optimizers(self):
        """
        Optimizer configuration for the Competitive PINN solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        # If the problem is an InverseProblem, add the unknown parameters
        # to the parameters to be optimized
        self.optimizer_model.hook(self.neural_net.parameters())
        self.optimizer_discriminator.hook(self.discriminator.parameters())
        if isinstance(self.problem, InverseProblem):
            self.optimizer_model.instance.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )
        self.scheduler_model.hook(self.optimizer_model)
        self.scheduler_discriminator.hook(self.optimizer_discriminator)
        return (
            [
                self.optimizer_model.instance,
                self.optimizer_discriminator.instance,
            ],
            [
                self.scheduler_model.instance,
                self.scheduler_discriminator.instance,
            ],
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This method is called at the end of each training batch, and ovverides
        the PytorchLightining implementation for logging the checkpoints.

        :param torch.Tensor outputs: The output from the model for the
            current batch.
        :param tuple batch: The current batch of data.
        :param int batch_idx: The index of the current batch.
        :return: Whatever is returned by the parent
            method ``on_train_batch_end``.
        :rtype: Any
        """
        # increase by one the counter of optimization to save loggers
        (
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed
        ) += 1

        return super().on_train_batch_end(outputs, batch, batch_idx)

    @property
    def neural_net(self):
        """
        Returns the neural network model.

        :return: The neural network model.
        :rtype: torch.nn.Module
        """
        return self.models[0]

    @property
    def discriminator(self):
        """
        Returns the discriminator model (if applicable).

        :return: The discriminator model.
        :rtype: torch.nn.Module
        """
        return self.models[1]

    @property
    def optimizer_model(self):
        """
        Returns the optimizer associated with the neural network model.

        :return: The optimizer for the neural network model.
        :rtype: torch.optim.Optimizer
        """
        return self.optimizers[0]

    @property
    def optimizer_discriminator(self):
        """
        Returns the optimizer associated with the discriminator (if applicable).

        :return: The optimizer for the discriminator.
        :rtype: torch.optim.Optimizer
        """
        return self.optimizers[1]

    @property
    def scheduler_model(self):
        """
        Returns the scheduler associated with the neural network model.

        :return: The scheduler for the neural network model.
        :rtype: torch.optim.lr_scheduler._LRScheduler
        """
        return self.schedulers[0]

    @property
    def scheduler_discriminator(self):
        """
        Returns the scheduler associated with the discriminator (if applicable).

        :return: The scheduler for the discriminator.
        :rtype: torch.optim.lr_scheduler._LRScheduler
        """
        return self.schedulers[1]

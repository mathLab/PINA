""" Module for CompetitivePINN """

import torch
import copy

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from torch.optim.lr_scheduler import ConstantLR

from .basepinn import PINNInterface
from pina.utils import check_consistency
from pina.problem import InverseProblem


class CompetitivePINN(PINNInterface):
    r"""
    Competitive Physics Informed Neural Network (PINN) solver class.
    This class implements Competitive Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
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
        loss=torch.nn.MSELoss(),
        optimizer_model=torch.optim.Adam,
        optimizer_model_kwargs={"lr": 0.001},
        optimizer_discriminator=torch.optim.Adam,
        optimizer_discriminator_kwargs={"lr": 0.001},
        scheduler_model=ConstantLR,
        scheduler_model_kwargs={"factor": 1, "total_iters": 0},
        scheduler_discriminator=ConstantLR,
        scheduler_discriminator_kwargs={"factor": 1, "total_iters": 0},
    ):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module model: The neural network model to use
            for the model.
        :param torch.nn.Module discriminator: The neural network model to use
            for the discriminator. If ``None``, the discriminator network will
            have the same architecture as the model network.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.optim.Optimizer optimizer_model: The neural
            network optimizer to use for the model network
            , default is `torch.optim.Adam`.
        :param dict optimizer_model_kwargs: Optimizer constructor keyword
            args. for the model.
        :param torch.optim.Optimizer optimizer_discriminator: The neural
            network optimizer to use for the discriminator network
            , default is `torch.optim.Adam`.
        :param dict optimizer_discriminator_kwargs: Optimizer constructor
            keyword args. for the discriminator.
        :param torch.optim.LRScheduler scheduler_model: Learning
            rate scheduler for the model.
        :param dict scheduler_model_kwargs: LR scheduler constructor
            keyword args.
        :param torch.optim.LRScheduler scheduler_discriminator: Learning
            rate scheduler for the discriminator.
        """
        if discriminator is None:
            discriminator = copy.deepcopy(model)

        super().__init__(
            models=[model, discriminator],
            problem=problem,
            optimizers=[optimizer_model, optimizer_discriminator],
            optimizers_kwargs=[
                optimizer_model_kwargs,
                optimizer_discriminator_kwargs,
            ],
            extra_features=None,  # CompetitivePINN doesn't take extra features
            loss=loss,
        )

        # set automatic optimization for GANs
        self.automatic_optimization = False

        # check consistency
        check_consistency(scheduler_model, LRScheduler, subclass=True)
        check_consistency(scheduler_model_kwargs, dict)
        check_consistency(scheduler_discriminator, LRScheduler, subclass=True)
        check_consistency(scheduler_discriminator_kwargs, dict)

        # assign schedulers
        self._schedulers = [
            scheduler_model(self.optimizers[0], **scheduler_model_kwargs),
            scheduler_discriminator(
                self.optimizers[1], **scheduler_discriminator_kwargs
            ),
        ]

        self._model = self.models[0]
        self._discriminator = self.models[1]

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
        # train one step of the model
        with torch.no_grad():
            discriminator_bets = self.discriminator(samples)
        loss_val = self._train_model(samples, equation, discriminator_bets)
        self.store_log(loss_value=float(loss_val))
        # detaching samples from the computational graph to erase it and setting
        # the gradient to true to create a new computational graph.
        # In alternative set `retain_graph=True`.
        samples = samples.detach()
        samples.requires_grad = True
        # train one step of discriminator
        discriminator_bets = self.discriminator(samples)
        self._train_discriminator(samples, equation, discriminator_bets)
        return loss_val

    def loss_data(self, input_tensor, output_tensor):
        """
        The data loss for the PINN solver. It computes the loss between the
        network output against the true solution.

        :param LabelTensor input_tensor: The input to the neural networks.
        :param LabelTensor output_tensor: The true solution to compare the
            network solution.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        self.optimizer_model.zero_grad()
        loss_val = (
            super()
            .loss_data(input_tensor, output_tensor)
            .as_subclass(torch.Tensor)
        )
        loss_val.backward()
        self.optimizer_model.step()
        return loss_val

    def configure_optimizers(self):
        """
        Optimizer configuration for the Competitive PINN solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        # if the problem is an InverseProblem, add the unknown parameters
        # to the parameters that the optimizer needs to optimize
        if isinstance(self.problem, InverseProblem):
            self.optimizer_model.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )
        return self.optimizers, self._schedulers

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
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += (
            1
        )
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def _train_discriminator(self, samples, equation, discriminator_bets):
        """
        Trains the discriminator network of the Competitive PINN.

        :param LabelTensor samples: Input samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation representing
            the physics.
        :param Tensor discriminator_bets: Predictions made by the discriminator
            network.
        """
        # manual optimization
        self.optimizer_discriminator.zero_grad()
        # compute residual, we detach because the weights of the generator
        # model are fixed
        residual = self.compute_residual(
            samples=samples, equation=equation
        ).detach()
        # compute competitive residual, the minus is because we maximise
        competitive_residual = residual * discriminator_bets
        loss_val = -self.loss(
            torch.zeros_like(competitive_residual, requires_grad=True),
            competitive_residual,
        ).as_subclass(torch.Tensor)
        # backprop
        self.manual_backward(loss_val)
        self.optimizer_discriminator.step()
        return

    def _train_model(self, samples, equation, discriminator_bets):
        """
        Trains the model network of the Competitive PINN.

        :param LabelTensor samples: Input samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation representing
            the physics.
        :param Tensor discriminator_bets: Predictions made by the discriminator.
            network.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        # manual optimization
        self.optimizer_model.zero_grad()
        # compute residual (detached for discriminator) and log
        residual = self.compute_residual(samples=samples, equation=equation)
        # store logging
        with torch.no_grad():
            loss_residual = self.loss(torch.zeros_like(residual), residual)
        # compute competitive residual, discriminator_bets are detached becase
        # we optimize only the generator model
        competitive_residual = residual * discriminator_bets.detach()
        loss_val = self.loss(
            torch.zeros_like(competitive_residual, requires_grad=True),
            competitive_residual,
        ).as_subclass(torch.Tensor)
        # backprop
        self.manual_backward(loss_val)
        self.optimizer_model.step()
        return loss_residual

    @property
    def neural_net(self):
        """
        Returns the neural network model.

        :return: The neural network model.
        :rtype: torch.nn.Module
        """
        return self._model

    @property
    def discriminator(self):
        """
        Returns the discriminator model (if applicable).

        :return: The discriminator model.
        :rtype: torch.nn.Module
        """
        return self._discriminator

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
        return self._schedulers[0]

    @property
    def scheduler_discriminator(self):
        """
        Returns the scheduler associated with the discriminator (if applicable).

        :return: The scheduler for the discriminator.
        :rtype: torch.optim.lr_scheduler._LRScheduler
        """
        return self._schedulers[1]

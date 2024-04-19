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
    """
    TODO

    .. warning Does Not Support Extra Features
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
            loss=loss
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
            scheduler_model(
                self.optimizers[0], **scheduler_model_kwargs
            ),
            scheduler_discriminator(
                self.optimizers[1], **scheduler_discriminator_kwargs
            ),
        ]

        self._model = self.models[0]
        self._discriminator = self.models[1]


    def forward(self, x):
        """
        Forward pass implementation for the PINN solver.
        :param LabelTensor x: Input tensor for the PINN solver. It expects
            a tensor :math:`N \times D`, where :math:`N` the number of points
            in the mesh, :math:`D` the dimension of the problem,
        :return: PINN solution evaluated at the input points.
        :rtype: LabelTensor
        """
        return self.neural_net(x)


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


    def loss_phys(self, samples, equation):
        # train one step of discriminator
        discriminator_bets = self.discriminator(samples.clone())
        self._train_discriminator(samples, equation, discriminator_bets)
        # detaching samples from the computational graph to erase it and setting
        # the gradient to true to create a new computational graph.
        # In alternative set `retain_graph=True`.
        samples = samples.detach()
        samples.requires_grad = True
        # train one step of the model
        loss_val = self._train_model(samples, equation, discriminator_bets)
        self.store_log(loss_value=float(loss_val))
        return loss_val
    
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
        residual = self.compute_residual(samples=samples,
                                         equation=equation).detach()
        # compute competitive residual, the minus is because we maximise
        competitive_residual = residual * discriminator_bets
        loss_val = - self.loss(
            torch.zeros_like(competitive_residual, requires_grad=True),
            competitive_residual
        ).as_subclass(torch.Tensor)
        # backprop
        loss_val.backward()
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
            loss_residual = self.loss(
                                torch.zeros_like(residual),
                                residual
                                )
        # compute competitive residual, discriminator_bets are detached becase
        # we optimize only the generator model
        competitive_residual = residual * discriminator_bets.detach()
        loss_val = self.loss(
            torch.zeros_like(competitive_residual, requires_grad=True),
            competitive_residual
        ).as_subclass(torch.Tensor)
        # backprop
        loss_val.backward()
        self.optimizer_model.step()
        return loss_residual

    def loss_data(self, input_tensor, output_tensor):
        """
        Computes the data loss for the PINN solver based on input,
        output, and condition name. This function is a wrapper of the function
        :meth:`loss_data` used internally in PINA to handle the logging step.

        :param LabelTensor input_tensor: The input to the neural networks.
        :param LabelTensor output_tensor: The true solution to compare the
            network solution.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        self.optimizer_model.zero_grad()
        loss_val = super().loss_data(
            input_tensor, output_tensor).as_subclass(torch.Tensor)
        loss_val.backward()
        self.optimizer_model.step()
        return loss_val

    
    @property
    def neural_net(self):
        return self._model

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def optimizer_model(self):
        return self.optimizers[0]

    @property
    def optimizer_discriminator(self):
        return self.optimizers[1]

    @property
    def scheduler_model(self):
        return self._schedulers[0]

    @property
    def scheduler_discriminator(self):
        return self._schedulers[1]
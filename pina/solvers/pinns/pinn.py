""" Module for PINN """

import torch

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from torch.optim.lr_scheduler import ConstantLR

from .basepinn import PINNInterface
from ...utils import check_consistency
from ...loss import LossInterface
from ...problem import InverseProblem
from torch.nn.modules.loss import _Loss

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class PINN(PINNInterface):
    """
    PINN solver class. This class implements Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    .. seealso::

        **Original reference**: Karniadakis, G. E., Kevrekidis, I. G., Lu, L.,
        Perdikaris, P., Wang, S., & Yang, L. (2021).
        Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.
        <https://doi.org/10.1038/s42254-021-00314-5>`_.
    """

    def __init__(
        self,
        problem,
        model,
        extra_features=None,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        scheduler=ConstantLR,
        scheduler_kwargs={"factor": 1, "total_iters": 0},
    ):
        """
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param torch.optim.Optimizer optimizer: The neural network optimizer to
            use; default is :class:`torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param torch.optim.LRScheduler scheduler: Learning
            rate scheduler.
        :param dict scheduler_kwargs: LR scheduler constructor keyword args.
        """
        super().__init__(
            models=[model],
            problem=problem,
            optimizers=[optimizer],
            optimizers_kwargs=[optimizer_kwargs],
            extra_features=extra_features,
        )

        # check consistency
        check_consistency(scheduler, LRScheduler, subclass=True)
        check_consistency(scheduler_kwargs, dict)

        # assign variables
        self._scheduler = scheduler(self.optimizers[0], **scheduler_kwargs)
        self._neural_net = self.models[0]


    def forward(self, x):
        """
        Forward pass implementation for the PINN
        solver.

        :param LabelTensor x: Input tensor for the PINN solver. It expects
            a tensor :math:`N \times D`, where :math:`N` the number of points
            in the mesh, :math:`D` the dimension of the problem,
        :return: PINN solution.
        :rtype: LabelTensor
        """
        return self.neural_net(x)


    def loss_phys(self, samples, equation):
        residual = equation.residual(
                samples, self.forward(samples), self._params
            )
        loss_phys =  self.loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        return loss_phys


    def configure_optimizers(self):
        """
        Optimizer configuration for the PINN
        solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        # if the problem is an InverseProblem, add the unknown parameters
        # to the parameters that the optimizer needs to optimize
        if isinstance(self.problem, InverseProblem):
            self.optimizers[0].add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )
        return self.optimizers, [self.scheduler]


    @property
    def scheduler(self):
        """
        Scheduler for the PINN training.
        """
        return self._scheduler


    @property
    def neural_net(self):
        """
        Neural network for the PINN training.
        """
        return self._neural_net
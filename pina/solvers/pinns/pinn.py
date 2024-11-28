""" Module for Physics Informed Neural Network. """

import torch

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0


from .basepinn import PINNInterface
from ...problem import InverseProblem


class PINN(PINNInterface):
    r"""
    Physics Informed Neural Network (PINN) solver class.
    This class implements Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    The Physics Informed Network aims to find
    the solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`
    of the differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}

    minimizing the loss function

    .. math::
        \mathcal{L}_{\rm{problem}} = \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{A}[\mathbf{u}](\mathbf{x}_i)) +
        \frac{1}{N}\sum_{i=1}^N
        \mathcal{L}(\mathcal{B}[\mathbf{u}](\mathbf{x}_i))

    where :math:`\mathcal{L}` is a specific loss function, default Mean Square Error:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::

        **Original reference**: Karniadakis, G. E., Kevrekidis, I. G., Lu, L.,
        Perdikaris, P., Wang, S., & Yang, L. (2021).
        Physics-informed machine learning. Nature Reviews Physics, 3, 422-440.
        DOI: `10.1038 <https://doi.org/10.1038/s42254-021-00314-5>`_.
    """

    __name__ = 'PINN'

    def __init__(
        self,
        problem,
        model,
        extra_features=None,
        loss=None,
        optimizer=None,
        scheduler=None,
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
            models=model,
            problem=problem,
            optimizers=optimizer,
            schedulers=scheduler,
            extra_features=extra_features,
            loss=loss,
        )

        # assign variables
        self._neural_net = self.models[0]

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
        Computes the physics loss for the PINN solver based on given
        samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor
        """
        residual = self.compute_residual(samples=samples, equation=equation)
        loss_value = self.loss(
            torch.zeros_like(residual), residual
        )
        return loss_value

    def configure_optimizers(self):
        """
        Optimizer configuration for the PINN
        solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        # if the problem is an InverseProblem, add the unknown parameters
        # to the parameters that the optimizer needs to optimize


        self._optimizer.hook(self._model.parameters())
        if isinstance(self.problem, InverseProblem):
            self._optimizer.optimizer_instance.add_param_group(
                    {
                        "params": [
                            self._params[var]
                            for var in self.problem.unknown_variables
                        ]
                    }
                )
        self._scheduler.hook(self._optimizer)
        return ([self._optimizer.optimizer_instance],
                [self._scheduler.scheduler_instance])

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

""" Module for Physics Informed Neural Network. """

import torch

from .pinn_interface import PINNInterface
from ..solver import SingleSolverInterface
from ...problem import InverseProblem


class PINN(PINNInterface, SingleSolverInterface):
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
        optimizer=None,
        scheduler=None,
        weighting=None,
        loss=None,
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
            model=model,
            problem=problem,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            weighting=weighting
        )

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
        Optimizer configuration for the PINN solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        # If the problem is an InverseProblem, add the unknown parameters
        # to the parameters to be optimized.
        self.optimizer.hook(self.model.parameters())
        if isinstance(self.problem, InverseProblem):
            self.optimizer.optimizer_instance.add_param_group(
                    {
                        "params": [
                            self._params[var]
                            for var in self.problem.unknown_variables
                        ]
                    }
                )
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.optimizer_instance],
                [self.scheduler.scheduler_instance])
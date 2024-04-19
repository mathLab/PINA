""" Module for GPINN """

import torch


from torch.optim.lr_scheduler import ConstantLR

from .pinn import PINN
from pina.operators import grad
from pina.problem import SpatialProblem


class GPINN(PINN):
    """
    Gradient enhanced PINN solver class. This class implements
    Gradient enhanced Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    .. seealso::

        **Original reference**: Yu, Jeremy, et al. "Gradient-enhanced
        physics-informed neural networks for forward and inverse
        PDE problems." Computer Methods in Applied Mechanics
        and Engineering 393 (2022): 114823.
        <https://doi.org/10.1016/j.cma.2022.114823>`_.

    .. note::
        This class can only work for problems inheriting
        from at least :class:`~pina.problem.spatial_problem.SpatialProblem`
        class.
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
        :param AbstractProblem problem: The formulation of the problem. It must
            inherit from at least
            :class:`~pina.problem.spatial_problem.SpatialProblem` in order to
            compute the gradient of the loss.
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
                        problem=problem,
                        model=model,
                        extra_features=extra_features,
                        loss=loss,
                        optimizer=optimizer,
                        optimizer_kwargs=optimizer_kwargs,
                        scheduler=scheduler,
                        scheduler_kwargs=scheduler_kwargs,
        )
        if not isinstance(self.problem, SpatialProblem):
            raise ValueError('Gradient PINN computes the gradient of the '
                             'PINN loss with respect to the spatial '
                             'coordinates, thus the PINA problem must be '
                             'a SpatialProblem.')

    
    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the GPINN solver based on given
        samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor
        """
        # classical PINN loss
        residual = self.compute_residual(samples=samples, equation=equation)
        loss_value = self.loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        self.store_log(loss_value=float(loss_value))
        # gradient PINN loss
        loss_value = loss_value.reshape(-1, 1)
        loss_value.labels = ['__LOSS']
        loss_grad = grad(loss_value, samples, d=self.problem.spatial_variables)
        g_loss_phys = self.loss(
            torch.zeros_like(loss_grad, requires_grad=True), loss_grad
        )
        return loss_value + g_loss_phys
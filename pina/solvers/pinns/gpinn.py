""" Module for PINN """

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

    
    def _loss_phys(self, samples, equation, condition_name):
        """
        Computes the physics loss for the PINN solver based on input,
        output, and condition name. This function is a wrapper of the function
        :meth:`loss_phys` used internally in PINA to handle the logging step.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :param str condition_name: The condition name for tracking purposes.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        # classical PINN loss
        loss_val = self.loss_phys(samples, equation)
        self.store_log(name=condition_name+'_loss', loss_val=float(loss_val))
        # gradient PINN loss
        loss_val = loss_val.reshape(-1, 1)
        loss_val.labels = ['__LOSS']
        loss_grad = grad(loss_val, samples, d=self.problem.spatial_variables)
        g_loss_phys = self.loss(
            torch.zeros_like(loss_grad, requires_grad=True), loss_grad
        )
        return (loss_val + g_loss_phys).as_subclass(torch.Tensor)
import torch

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from pina.solvers import PINN
from torch.optim.lr_scheduler import ConstantLR


class SAPINN(PINN):
    """
    This class aims to implements the Self-Adaptive PINN solver,
    using a user specified "model" to solve a specific "problem".

    .. seealso::
    **Original reference**: McClenny, Levi D., and Ulisses M. Braga-Neto.
    "Self-adaptive physics-informed neural networks."
    Journal of Computational Physics 474 (2023): 111722.
    <https://doi.org/10.1016/j.jcp.2022.111722>`_.
    """
    
    def __init__(
            self,
            problem,
            model,
            extra_features=None,
            mask="polinomial",
            loss=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam,
            optimizer_kwargs={"lr" : 0.001},
            scheduler=ConstantLR,
            scheduler_kwargs={"factor" : 1, "total_iters" : 0}
    ):
        """
        :param AbstractProblem problem: The formulation of the problem.
        :param torch.nn.Module model: The neural network model to use.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.nn.Module extra_features: The additional input
            features to use as augmented input.
        :param str mask: type of mask applied to weights for the
            self adaptive strategy
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
            loss=loss,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs
        )
    
    def _generate_weigths(self):
        pass

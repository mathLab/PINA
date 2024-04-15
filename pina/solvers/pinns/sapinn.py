import torch

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from pina import LabelTensor
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
            mask_type={"type": "polynomial", "coefficient": [2]},
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
        :param dict mask: type of mask applied to weights for the
            self adaptive strategy
            mask_type["type"] -> polynomial, sigmoid
            mask_type["coefficient"] -> list of coefficient
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
            optimizer_kwargs=optimizer_kwargs,
            extra_features=extra_features,
            loss=loss,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs
        )

        # Attributes for inputs
        self.mask_type = mask_type

        # Attributes
        self.mask = self._initialize_mask
        self.weights = self._generate_weigths()

    def _initialize_mask(self, x):
        if self.mask_type["type"] == "polynomial":
            if not isinstance(self.mask_type["coefficient"], list):
                self.mask_type["coefficient"] = [self.mask_type["coefficient"]]
            if len(self.mask_type["coefficient"]) != 1:
                raise ValueError('Polynomial mask type requires only one coefficient')
            return x ** self.mask_type["coefficient"][0]
        if self.mask_type["type"] == "sigmoid":
            sig = torch.nn.Sigmoid()
            if not isinstance(self.mask_type["coefficient"], list):
                raise ValueError('Sigmoid mask type has to be a list of coefficients')
            if len(self.mask_type["coefficient"]) != 3:
                raise ValueError('Sigmoid mask type requires three elements in the list')
            coeffs = self.mask_type["coefficient"]
            return coeffs[0]*sig(torch.tensor(coeffs[1]*x.numpy() + coeffs[2]))
        raise ValueError("key type of mask_type Error")

    
    def _generate_weigths(self):
        weigths_initilization = dict()
        for key in self.problem.input_pts.keys():
            weigths_initilization[key] = LabelTensor(
                self.mask(torch.rand(size = self.problem.input_pts[key].tensor.shape)),
                labels=self.problem.input_variables
            )
        return weigths_initilization

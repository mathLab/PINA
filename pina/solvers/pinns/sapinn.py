import torch

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from pina import LabelTensor
from .basepinn import PINNInterface
from pina.utils import check_consistency
from pina.problem import InverseProblem

from torch.optim.lr_scheduler import ConstantLR

class SAPINNWeightsModel(torch.nn.Module):
    """
    This class aims to implements the weights of the Self-Adaptive
    PINN solver.
    """

    def __init__(self, dict_mask : dict, size : tuple) -> None:
        super().__init__()
        self.type_mask = dict_mask["type"]
        self.weigth_of_mask = dict_mask["coefficient"]
        self.sa_weights = torch.nn.Parameter(torch.randn(size=size))

        if self.type_mask == "polynomial" and self._polynomial_consistency():
            self.func = self._polynomial_func
        elif self.type_mask == "sigmoid" and self._sigmoidal_consistency():
            self.func = self._sigmoid_func
            pass
        else:
            raise ValueError("type key of dict_mask not allowed")
    
    def _polynomial_consistency(self):
        if not isinstance(self.weigth_of_mask, list):
            self.weigth_of_mask = [self.weigth_of_mask]
        if len(self.weigth_of_mask) != 1:
            raise ValueError("coefficient key of dict_mask not coherent with type key. Polynomial mask type requires only one coefficient")
        return True
    
    def _polynomial_func(self, x):
        return x ** self.weigth_of_mask[0]
    
    def _sigmoidal_consistency(self):
        if not isinstance(self.weigth_of_mask, list):
            raise ValueError('Sigmoid mask type has to be a list of coefficients')
        if len(self.weigth_of_mask) != 3:
            raise ValueError('Sigmoid mask type requires three elements in the list')
        return True
    
    def _sigmoid_func(self, x):
        return self.weigth_of_mask[0]*torch.nn.Sigmoid(self.weigth_of_mask[1]*x+ self.weigth_of_mask[2])
    
    def forward(self, x):
        return self.func(self.sa_weights * x)

class SAPINN(PINNInterface):
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
            optimizer_weights=torch.optim.Adam,
            optimizer_kwargs={"lr" : 0.001},
            optimizer_weights_kwargs={"lr" : 0.001},
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
            models=self._interface_models(problem, model, mask_type),
            problem=problem,
            optimizers=self._interface_optimizers(problem, optimizer, optimizer_weights),
            optimizers_kwargs=self._interface_optimizers_kwargs(problem, optimizer_kwargs, optimizer_weights_kwargs),
            extra_features=extra_features,
            loss=loss
        )

        # Controllo massimizzazione
        try:
            for idx in range(1, len(self.optimizers)):
                self.optimizers[idx].maximize = True
        except:
            raise ValueError("Select an optimizer with the maximize attribute")

        # check consistency
        check_consistency(scheduler, LRScheduler, subclass=True)
        check_consistency(scheduler_kwargs, dict)

        # assign variables
        self._scheduler = scheduler(self.optimizers[0], **scheduler_kwargs)
        self._neural_net = self.models[0]

        # dict - condition_name : index in self.models
        self.dict_condition_idx = dict()
        i = 0
        for key in self.problem.input_pts.keys():
            self.dict_condition_idx[key] = 1+i
            i += 1
    
    def _interface_models(self, problem, model, mask_type):
        weights_models = [
            SAPINNWeightsModel(
                dict_mask=mask_type,
                size=value.tensor.shape
            )
            for _, value in problem.input_pts.items()
        ]
        interface_models = [model]
        interface_models.extend(weights_models)
        return interface_models
    
    def _interface_optimizers(self, problem, optimizer, optimizer_weights):
        interface_optimizers = [optimizer]
        interface_optimizers.extend([optimizer_weights for _, _ in problem.input_pts.items()])
        return interface_optimizers
    
    def _interface_optimizers_kwargs(self, problem, optimizer_kwargs, optimizer_weights_kwargs):
        interface_optimizers_kwargs = [optimizer_kwargs]
        interface_optimizers_kwargs.extend([optimizer_weights_kwargs for _, _ in problem.input_pts.items()])
        return interface_optimizers_kwargs
    
    ###########################################################################################################
    # DA pinn.py
    
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
    
    ###########################################################################################################

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
        loss_val = self.loss_phys(samples, equation, self.models[self.dict_condition_idx[condition_name]])
        self.store_log(name=condition_name+'_loss', loss_val=float(loss_val))
        return loss_val.as_subclass(torch.Tensor)
    
    def loss_phys(self, samples, equation, weight_model):
        try:
            residual = weight_model(equation.residual(samples, self.forward(samples)))
        except (
            TypeError
        ):  # this occurs when the function has three inputs, i.e. inverse problem
            residual = weight_model(equation.residual(
                samples, self.forward(samples), self._params
            ))
        return self.loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
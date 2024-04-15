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

        # Attributes for inputs
        self.mask_type = mask_type

        # Attributes
        self.mask_model = self._initialize_mask()

        super().__init__(
            models=[model, model],
            problem=problem,
            optimizers=[optimizer, optimizer_weights],
            optimizers_kwargs=[optimizer_kwargs, optimizer_weights_kwargs],
            extra_features=extra_features,
            loss=loss
        )
        
        ####################################################################
        # Da PINN
        # check consistency
        check_consistency(scheduler, LRScheduler, subclass=True)
        check_consistency(scheduler_kwargs, dict)

        # assign variables
        self._scheduler = scheduler(self.optimizers[0], **scheduler_kwargs)
        self._neural_net = self.models[0]
        ####################################################################

        self.weights = self._generate_weigths()

        # Controllo massimizzazione
        try:
            optimizer_weights.maximize = True
        except:
            raise ValueError("Select an optimizer with the maximize attribute")
        
        self.configure_optimizers_weights()
    
    def _initialize_mask(self):
        if self.mask_type["type"] == "polynomial":
            if not isinstance(self.mask_type["coefficient"], list):
                self.mask_type["coefficient"] = [self.mask_type["coefficient"]]
            if len(self.mask_type["coefficient"]) != 1:
                raise ValueError('Polynomial mask type requires only one coefficient')
            return lambda x: x ** self.mask_type["coefficient"][0]
        if self.mask_type["type"] == "sigmoid":
            if not isinstance(self.mask_type["coefficient"], list):
                raise ValueError('Sigmoid mask type has to be a list of coefficients')
            if len(self.mask_type["coefficient"]) != 3:
                raise ValueError('Sigmoid mask type requires three elements in the list')
            coeffs = self.mask_type["coefficient"]
            return lambda x: coeffs[0]*torch.nn.Sigmoid(coeffs[1]*x+ coeffs[2])
        raise ValueError("key type of mask_type Error")
    
    def _generate_weigths(self):
        weigths_initilization = dict()
        for key in self.problem.input_pts.keys():
            weigths_initilization[key] = torch.nn.Parameter(
                torch.rand(size = self.problem.input_pts[key].tensor.shape)
            )
        return weigths_initilization
    
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
        loss_val = self.loss_phys(samples, equation, self.weights[condition_name])
        self.store_log(name=condition_name+'_loss', loss_val=float(loss_val))
        return loss_val.as_subclass(torch.Tensor)
    
    def loss_phys(self, samples, equation, weights):
        try:
            residual = weights * equation.residual(samples, self.forward(samples))
        except (
            TypeError
        ):  # this occurs when the function has three inputs, i.e. inverse problem
            residual = weights * equation.residual(
                samples, self.forward(samples), self._params
            )
        return self.loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
    
    def configure_optimizers_weights(self):
        """
        Optimizer configuration for the SA-PINN
        solver related to adaptive weights.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        self.optimizers[1].add_param_group(
            {
                "params": [
                    self.weights[key]
                    for key in self.weights.keys()
                ]
            }
        )
        return self.optimizers, [self.scheduler]
    
    #######################################################################
    # DA PINN
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
    #######################################################################
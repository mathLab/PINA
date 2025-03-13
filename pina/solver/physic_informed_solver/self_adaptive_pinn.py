"""Module for Self-Adaptive PINN."""

from copy import deepcopy
import torch

from ...utils import check_consistency
from ...problem import InverseProblem
from ..solver import MultiSolverInterface
from .pinn_interface import PINNInterface


class Weights(torch.nn.Module):
    """
    Implementation of the mask model for the self-adaptive weights of the
    :class:`SelfAdaptivePINN` solver.
    """

    def __init__(self, func):
        """
        Initialization of the :class:`Weights` class.

        :param torch.nn.Module func: the mask model.
        """
        super().__init__()
        check_consistency(func, torch.nn.Module)
        self.sa_weights = torch.nn.Parameter(torch.Tensor())
        self.func = func

    def forward(self):
        """
        Forward pass implementation for the mask module.

        :return: evaluation of self adaptive weights through the mask.
        :rtype: torch.Tensor
        """
        return self.func(self.sa_weights)


class SelfAdaptivePINN(PINNInterface, MultiSolverInterface):
    r"""
    Self-Adaptive Physics-Informed Neural Network (SelfAdaptivePINN) solver
    class. This class implements the Self-Adaptive Physics-Informed Neural
    Network solver, using a user specified ``model`` to solve a specific
    ``problem``. It can be used to solve both forward and inverse problems.

    The Self-Adapive Physics-Informed Neural Network solver aims to find the
    solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m` of a differential
    problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}
    
    integrating pointwise loss evaluation using a mask :math:m and self-adaptive
    weights, which allow the model to focus on regions of the domain where the
    residual is higher.

    The loss function to solve the problem is

    .. math::

        \mathcal{L}_{\rm{problem}} = \frac{1}{N} \sum_{i=1}^{N_\Omega} m
        \left( \lambda_{\Omega}^{i} \right) \mathcal{L} \left( \mathcal{A}
        [\mathbf{u}](\mathbf{x}) \right) + \frac{1}{N} 
        \sum_{i=1}^{N_{\partial\Omega}}
        m \left( \lambda_{\partial\Omega}^{i} \right) \mathcal{L} 
        \left( \mathcal{B}[\mathbf{u}](\mathbf{x})
        \right),
    
    denoting the self adaptive weights as
    :math:`\lambda_{\Omega}^1, \dots, \lambda_{\Omega}^{N_\Omega}` and
    :math:`\lambda_{\partial \Omega}^1, \dots, 
    \lambda_{\Omega}^{N_\partial \Omega}`
    for :math:`\Omega` and :math:`\partial \Omega`, respectively.

    The Self-Adaptive Physics-Informed Neural Network solver identifies the
    solution and appropriate self adaptive weights by solving the following
    optimization problem:

    .. math::

        \min_{w} \max_{\lambda_{\Omega}^k, \lambda_{\partial \Omega}^s}
        \mathcal{L} ,
    
    where :math:`w` denotes the network parameters, and :math:`\mathcal{L}` is a
    specific loss function, , typically the MSE:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::
        **Original reference**: McClenny, Levi D., and Ulisses M. Braga-Neto.
        "Self-adaptive physics-informed neural networks."
        Journal of Computational Physics 474 (2023): 111722.
        DOI: `10.1016/
        j.jcp.2022.111722 <https://doi.org/10.1016/j.jcp.2022.111722>`_.
    """

    def __init__(
        self,
        problem,
        model,
        weight_function=torch.nn.Sigmoid(),
        optimizer_model=None,
        optimizer_weights=None,
        scheduler_model=None,
        scheduler_weights=None,
        weighting=None,
        loss=None,
    ):
        """
        Initialization of the :class:`SelfAdaptivePINN` class.

        :param AbstractProblem problem: The problem to be solved.
        :param torch.nn.Module model: The model to be used.
        :param torch.nn.Module weight_function: The Self-Adaptive mask model.
            Default is ``torch.nn.Sigmoid()``.
        :param torch.optim.Optimizer optimizer_model: The optimizer of the
            ``model``. If `None`, the Adam optimizer is used.
            Default is ``None``.
        :param torch.optim.Optimizer optimizer_weights: The optimizer of the
            ``weight_function``. If `None`, the Adam optimizer is used.
            Default is ``None``.
        :param torch.optim.LRScheduler scheduler_model: Learning rate scheduler
            for the ``model``. If `None`, the constant learning rate scheduler
            is used. Default is ``None``.
        :param torch.optim.LRScheduler scheduler_weights: Learning rate
            scheduler for the ``weight_function``. If `None`, the constant
            learning rate scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If `None`, no weighting schema is used. Default is ``None``.
        :param torch.nn.Module loss: The loss function to be minimized.
            If `None`, the Mean Squared Error (MSE) loss is used.
            Default is `None`.
        """
        # check consistency weitghs_function
        check_consistency(weight_function, torch.nn.Module)

        # create models for weights
        weights_dict = {}
        for condition_name in problem.conditions:
            weights_dict[condition_name] = Weights(weight_function)
        weights_dict = torch.nn.ModuleDict(weights_dict)

        super().__init__(
            models=[model, weights_dict],
            problem=problem,
            optimizers=[optimizer_model, optimizer_weights],
            schedulers=[scheduler_model, scheduler_weights],
            weighting=weighting,
            loss=loss,
        )

        # Set automatic optimization to False
        self.automatic_optimization = False

        self._vectorial_loss = deepcopy(self.loss)
        self._vectorial_loss.reduction = "none"

    def forward(self, x):
        """
        Forward pass.

        :param LabelTensor x: Input tensor.
        :return: The output of the neural network.
        :rtype: LabelTensor
        """
        return self.model(x)

    def training_step(self, batch):
        """
        Solver training step, overridden to perform manual optimization.

        :param dict batch: The batch element in the dataloader.
        :return: The aggregated loss.
        :rtype: LabelTensor
        """
        # Weights optimization
        self.optimizer_weights.instance.zero_grad()
        loss = super().training_step(batch)
        self.manual_backward(-loss)
        self.optimizer_weights.instance.step()

        # Model optimization
        self.optimizer_model.instance.zero_grad()
        loss = super().training_step(batch)
        self.manual_backward(loss)
        self.optimizer_model.instance.step()

        return loss

    def configure_optimizers(self):
        """
        Optimizer configuration.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        # If the problem is an InverseProblem, add the unknown parameters
        # to the parameters to be optimized
        self.optimizer_model.hook(self.model.parameters())
        self.optimizer_weights.hook(self.weights_dict.parameters())
        if isinstance(self.problem, InverseProblem):
            self.optimizer_model.instance.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )
        self.scheduler_model.hook(self.optimizer_model)
        self.scheduler_weights.hook(self.optimizer_weights)
        return (
            [self.optimizer_model.instance, self.optimizer_weights.instance],
            [self.scheduler_model.instance, self.scheduler_weights.instance],
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This method is called at the end of each training batch and overrides
        the PyTorch Lightning implementation to log checkpoints.

        :param torch.Tensor outputs: The ``model``'s output for the current
            batch.
        :param dict batch: The current batch of data.
        :param int batch_idx: The index of the current batch.
        """
        # increase by one the counter of optimization to save loggers
        (
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed
        ) += 1

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_start(self):
        """
        This method is called at the start of the training process to set the
        self-adaptive weights as parameters of the mask model.

        :raises NotImplementedError: If the batch size is not ``None``.
        """
        if self.trainer.batch_size is not None:
            raise NotImplementedError(
                "SelfAdaptivePINN only works with full "
                "batch size, set batch_size=None inside "
                "the Trainer to use the solver."
            )
        device = torch.device(
            self.trainer._accelerator_connector._accelerator_flag
        )

        # Initialize the self adaptive weights only for training points
        for (
            condition_name,
            tensor,
        ) in self.trainer.data_module.train_dataset.input.items():
            self.weights_dict[condition_name].sa_weights.data = torch.rand(
                (tensor.shape[0], 1), device=device
            )
        return super().on_train_start()

    def on_load_checkpoint(self, checkpoint):
        """
        Override of the Pytorch Lightning ``on_load_checkpoint`` method to
        handle checkpoints for Self-Adaptive Weights. This method should not be
        overridden, if not intentionally.

        :param dict checkpoint: Pytorch Lightning checkpoint dict.
        """
        # First initialize self-adaptive weights with correct shape,
        # then load the values from the checkpoint.
        for condition_name, _ in self.problem.input_pts.items():
            shape = checkpoint["state_dict"][
                f"_pina_models.1.{condition_name}.sa_weights"
            ].shape
            self.weights_dict[condition_name].sa_weights.data = torch.rand(
                shape
            )
        return super().on_load_checkpoint(checkpoint)

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the physics-informed solver based on the
        provided samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation.
        :return: The computed physics loss.
        :rtype: LabelTensor
        """
        residual = self.compute_residual(samples, equation)
        weights = self.weights_dict[self.current_condition_name].forward()
        loss_value = self._vectorial_loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        return self._vect_to_scalar(weights * loss_value)

    def _vect_to_scalar(self, loss_value):
        """
        Computation of the scalar loss.

        :param LabelTensor loss_value: the tensor of pointwise losses.
        :raises RuntimeError: If the loss reduction is not ``mean`` or ``sum``.
        :return: The computed scalar loss.
        :rtype LabelTensor
        """
        if self.loss.reduction == "mean":
            ret = torch.mean(loss_value)
        elif self.loss.reduction == "sum":
            ret = torch.sum(loss_value)
        else:
            raise RuntimeError(
                f"Invalid reduction, got {self.loss.reduction} "
                "but expected mean or sum."
            )
        return ret

    @property
    def model(self):
        """
        The model.

        :return: The model.
        :rtype: torch.nn.Module
        """
        return self.models[0]

    @property
    def weights_dict(self):
        """
        The self-adaptive weights.

        :return: The self-adaptive weights.
        :rtype: torch.nn.Module
        """
        return self.models[1]

    @property
    def scheduler_model(self):
        """
        The scheduler associated to the model.

        :return: The scheduler for the model.
        :rtype: torch.optim.lr_scheduler._LRScheduler
        """
        return self.schedulers[0]

    @property
    def scheduler_weights(self):
        """
        The scheduler associated to the mask model.

        :return: The scheduler for the mask model.
        :rtype: torch.optim.lr_scheduler._LRScheduler
        """
        return self.schedulers[1]

    @property
    def optimizer_model(self):
        """
        Returns the optimizer associated to the model.

        :return: The optimizer for the model.
        :rtype: torch.optim.Optimizer
        """
        return self.optimizers[0]

    @property
    def optimizer_weights(self):
        """
        The optimizer associated to the mask model.

        :return: The optimizer for the mask model.
        :rtype: torch.optim.Optimizer
        """
        return self.optimizers[1]

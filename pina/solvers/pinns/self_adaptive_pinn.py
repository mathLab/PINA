""" Module for Self-Adaptive PINN. """

import torch
from copy import deepcopy

from pina.utils import check_consistency
from pina.problem import InverseProblem
from ..solver import MultiSolverInterface
from .pinn_interface import PINNInterface


class Weights(torch.nn.Module):
    """
    This class aims to implements the mask model for the
    self-adaptive weights of the Self-Adaptive PINN solver.
    """

    def __init__(self, func):
        """
        :param torch.nn.Module func: the mask module of SAPINN.
        """
        super().__init__()
        check_consistency(func, torch.nn.Module)
        self.sa_weights = torch.nn.Parameter(torch.Tensor())
        self.func = func

    def forward(self):
        """
        Forward pass implementation for the mask module.
        It returns the function on the weights evaluation.

        :return: evaluation of self adaptive weights through the mask.
        :rtype: torch.Tensor
        """
        return self.func(self.sa_weights)


class SelfAdaptivePINN(PINNInterface, MultiSolverInterface):
    r"""
    Self Adaptive Physics Informed Neural Network (SelfAdaptivePINN)
    solver class. This class implements Self-Adaptive Physics Informed Neural
    Network solvers, using a user specified ``model`` to solve a specific
    ``problem``. It can be used for solving both forward and inverse problems.

    The Self Adapive Physics Informed Neural Network aims to find
    the solution :math:`\mathbf{u}:\Omega\rightarrow\mathbb{R}^m`
    of the differential problem:

    .. math::

        \begin{cases}
        \mathcal{A}[\mathbf{u}](\mathbf{x})=0\quad,\mathbf{x}\in\Omega\\
        \mathcal{B}[\mathbf{u}](\mathbf{x})=0\quad,
        \mathbf{x}\in\partial\Omega
        \end{cases}
    
    integrating the pointwise loss evaluation through a mask :math:`m` and
    self adaptive weights that permit to focus the loss function on 
    specific training samples.
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

    Self Adaptive Physics Informed Neural Network identifies the solution
    and appropriate self adaptive weights by solving the following problem

    .. math::

        \min_{w} \max_{\lambda_{\Omega}^k, \lambda_{\partial \Omega}^s}
        \mathcal{L} ,
    
    where :math:`w` denotes the network parameters, and
    :math:`\mathcal{L}` is a specific loss
    function, default Mean Square Error:

    .. math::
        \mathcal{L}(v) = \| v \|^2_2.

    .. seealso::
        **Original reference**: McClenny, Levi D., and Ulisses M. Braga-Neto.
        "Self-adaptive physics-informed neural networks."
        Journal of Computational Physics 474 (2023): 111722.
        DOI: `10.1016/
        j.jcp.2022.111722 <https://doi.org/10.1016/j.jcp.2022.111722>`_.
    """

    def __init__(self,
                 problem,
                 model,
                 weight_function=torch.nn.Sigmoid(),
                 optimizer_model=None,
                 optimizer_weights=None,
                 scheduler_model=None,
                 scheduler_weights=None,
                 weighting=None,
                 loss=None):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module model: The neural network model to use
            for the model.
        :param torch.nn.Module weight_function: The neural network model
            related to the mask of SAPINN.
            default :obj:`~torch.nn.Sigmoid`.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.optim.Optimizer optimizer_model: The neural
            network optimizer to use for the model network
            , default is `torch.optim.Adam`.
        :param torch.optim.Optimizer optimizer_weights: The neural
            network optimizer to use for mask model model,
            default is `torch.optim.Adam`.
        :param torch.optim.LRScheduler scheduler_model: Learning
            rate scheduler for the model.
        :param torch.optim.LRScheduler scheduler_weights: Learning
            rate scheduler for the mask model.
        """
        # check consistency weitghs_function
        check_consistency(weight_function, torch.nn.Module)

        # create models for weights
        weights_dict = {}
        for condition_name in problem.conditions:
            weights_dict[condition_name] = Weights(weight_function)
        weights_dict = torch.nn.ModuleDict(weights_dict)

        super().__init__(models=[model, weights_dict],
                         problem=problem,
                         optimizers=[optimizer_model, optimizer_weights],
                         schedulers=[scheduler_model, scheduler_weights],
                         weighting=weighting,
                         loss=loss)

        # Set automatic optimization to False
        self.automatic_optimization = False

        self._vectorial_loss = deepcopy(self.loss)
        self._vectorial_loss.reduction = "none"

    def forward(self, x):
        """
        Forward pass implementation for the PINN
        solver. It returns the function
        evaluation :math:`\mathbf{u}(\mathbf{x})` at the control points
        :math:`\mathbf{x}`.

        :param LabelTensor x: Input tensor for the SAPINN solver. It expects
            a tensor :math:`N \\times D`, where :math:`N` the number of points
            in the mesh, :math:`D` the dimension of the problem,
        :return: PINN solution.
        :rtype: LabelTensor
        """
        return self.model(x)

    def training_step(self, batch):
        """
        Solver training step, overridden to perform manual optimization.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """
        self.optimizer_model.instance.zero_grad()
        self.optimizer_weights.instance.zero_grad()
        loss = super().training_step(batch)
        self.optimizer_model.instance.step()
        self.optimizer_weights.instance.step()
        return loss

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the SAPINN solver based on given
        samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: torch.Tensor
        """
        # Train the weights
        weighted_loss = self._loss_phys(samples, equation)
        loss_value = -weighted_loss.as_subclass(torch.Tensor)
        self.manual_backward(loss_value)

        # Detach samples from the existing computational graph and
        # create a new one by setting requires_grad to True.
        # In alternative set `retain_graph=True`.
        samples = samples.detach()
        samples.requires_grad_()# = True

        # Train the model
        weighted_loss = self._loss_phys(samples, equation)
        loss_value = weighted_loss.as_subclass(torch.Tensor)
        self.manual_backward(loss_value)

        return loss_value

    def loss_data(self, input_pts, output_pts):
        """
        Computes the data loss for the SAPINN solver based on input and
        output. It computes the loss between the
        network output against the true solution.

        :param LabelTensor input_pts: The input to the neural networks.
        :param LabelTensor output_pts: The true solution to compare the
            network solution.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        residual = self.forward(input_pts) - output_pts
        loss = self._vectorial_loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        loss_value = self._vect_to_scalar(loss).as_subclass(torch.Tensor)
        self.manual_backward(loss_value)
        return loss_value

    def configure_optimizers(self):
        """
        Optimizer configuration for the SelfAdaptive PINN solver.

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
            [self.optimizer_model.instance,
             self.optimizer_weights.instance],
            [self.scheduler_model.instance,
             self.scheduler_weights.instance]
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This method is called at the end of each training batch, and ovverides
        the PytorchLightining implementation for logging the checkpoints.

        :param torch.Tensor outputs: The output from the model for the
            current batch.
        :param tuple batch: The current batch of data.
        :param int batch_idx: The index of the current batch.
        :return: Whatever is returned by the parent
            method ``on_train_batch_end``.
        :rtype: Any
        """
        # increase by one the counter of optimization to save loggers
        (
            self.trainer.fit_loop.epoch_loop.manual_optimization
            .optim_step_progress.total.completed
        ) += 1

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_start(self):
        """
        This method is called at the start of the training for setting
        the self adaptive weights as parameters of the mask model.

        :return: Whatever is returned by the parent
            method ``on_train_start``.
        :rtype: Any
        """
        if self.trainer.batch_size is not None:
            raise NotImplementedError("SelfAdaptivePINN only works with full "
                                      "batch size, set batch_size=None inside "
                                      "the Trainer to use the solver.")
        device = torch.device(
            self.trainer._accelerator_connector._accelerator_flag
        )
        self.trainer.datamodule.train_dataset

        # Initialize the self adaptive weights only for training points
        for condition_name, tensor in (
            self.trainer.data_module.train_dataset.input_points.items()
        ):
            self.weights_dict[condition_name].sa_weights.data = (
                torch.rand((tensor.shape[0], 1), device=device)
            )
        return super().on_train_start()

    def on_load_checkpoint(self, checkpoint):
        """
        Override the Pytorch Lightning ``on_load_checkpoint`` to handle
        checkpoints for Self-Adaptive Weights. This method should not be
        overridden if not intentionally.

        :param dict checkpoint: Pytorch Lightning checkpoint dict.
        """
        # First initialize self-adaptive weights with correct shape, 
        # then load the values from the checkpoint.
        for condition_name, _ in self.problem.input_pts.items():
            shape = checkpoint['state_dict'][
                f"_pina_models.1.{condition_name}.sa_weights"
            ].shape
            self.weights_dict[condition_name].sa_weights.data = (
                torch.rand(shape)
            )
        return super().on_load_checkpoint(checkpoint)

    def _loss_phys(self, samples, equation):
        """
        Computation of the physical loss for SelfAdaptive PINN solver.

        :param LabelTensor samples: Input samples to evaluate the physics loss.
        :param EquationInterface equation: the governing equation representing
            the physics.

        :return: tuple with weighted and not weighted scalar loss
        :rtype: List[LabelTensor, LabelTensor]
        """
        residual = self.compute_residual(samples, equation)
        weights = self.weights_dict[self.current_condition_name].forward()
        loss_value = self._vectorial_loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        return self._vect_to_scalar(weights * loss_value)

    def _vect_to_scalar(self, loss_value):
        """
        Elaboration of the pointwise loss through the mask model and the
        self adaptive weights

        :param LabelTensor loss_value: the matrix of pointwise loss

        :return: the scalar loss
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
        Return the mask models associate to the application of
        the mask to the self adaptive weights for each loss that
        compones the global loss of the problem.

        :return: The ModuleDict for mask models.
        :rtype: torch.nn.ModuleDict
        """
        return self.models[0]
    
    @property
    def weights_dict(self):
        """
        Return the mask models associate to the application of
        the mask to the self adaptive weights for each loss that
        compones the global loss of the problem.

        :return: The ModuleDict for mask models.
        :rtype: torch.nn.ModuleDict
        """
        return self.models[1]

    @property
    def scheduler_model(self):
        """
        Returns the scheduler associated with the neural network model.

        :return: The scheduler for the neural network model.
        :rtype: torch.optim.lr_scheduler._LRScheduler
        """
        return self.schedulers[0]

    @property
    def scheduler_weights(self):
        """
        Returns the scheduler associated with the mask model (if applicable).

        :return: The scheduler for the mask model.
        :rtype: torch.optim.lr_scheduler._LRScheduler
        """
        return self.schedulers[1]

    @property
    def optimizer_model(self):
        """
        Returns the optimizer associated with the neural network model.

        :return: The optimizer for the neural network model.
        :rtype: torch.optim.Optimizer
        """
        return self.optimizers[0]

    @property
    def optimizer_weights(self):
        """
        Returns the optimizer associated with the mask model (if applicable).

        :return: The optimizer for the mask model.
        :rtype: torch.optim.Optimizer
        """
        return self.optimizers[1]

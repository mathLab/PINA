import torch
from copy import deepcopy

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

from .basepinn import PINNInterface
from pina.utils import check_consistency
from pina.problem import InverseProblem

from torch.optim.lr_scheduler import ConstantLR


class Weights(torch.nn.Module):
    """
    This class aims to implements the mask model for
    self adaptive weights of the Self-Adaptive
    PINN solver.
    """

    def __init__(self, func):
        """
        :param torch.nn.Module func: the mask module of SAPINN
        """
        super().__init__()
        check_consistency(func, torch.nn.Module)
        self.sa_weights = torch.nn.Parameter(torch.Tensor())
        self.func = func

    def forward(self):
        """
        Forward pass implementation for the mask module.
        It returns the function on the weights
        evaluation.

        :return: evaluation of self adaptive weights through the mask.
        :rtype: torch.Tensor
        """
        return self.func(self.sa_weights)


class SAPINN(PINNInterface):
    r"""
    Self Adaptive Physics Informed Neural Network (SAPINN) solver class.
    This class implements Self-Adaptive Physics Informed Neural
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

    def __init__(
        self,
        problem,
        model,
        weights_function=torch.nn.Sigmoid(),
        extra_features=None,
        loss=torch.nn.MSELoss(),
        optimizer_model=torch.optim.Adam,
        optimizer_model_kwargs={"lr": 0.001},
        optimizer_weights=torch.optim.Adam,
        optimizer_weights_kwargs={"lr": 0.001},
        scheduler_model=ConstantLR,
        scheduler_model_kwargs={"factor": 1, "total_iters": 0},
        scheduler_weights=ConstantLR,
        scheduler_weights_kwargs={"factor": 1, "total_iters": 0},
    ):
        """
        :param AbstractProblem problem: The formualation of the problem.
        :param torch.nn.Module model: The neural network model to use
            for the model.
        :param torch.nn.Module weights_function: The neural network model
            related to the mask of SAPINN.
            default :obj:`~torch.nn.Sigmoid`.
        :param list(torch.nn.Module) extra_features: The additional input
            features to use as augmented input. If ``None`` no extra features
            are passed. If it is a list of :class:`torch.nn.Module`,
            the extra feature list is passed to all models. If it is a list
            of extra features' lists, each single list of extra feature
            is passed to a model.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        :param torch.optim.Optimizer optimizer_model: The neural
            network optimizer to use for the model network
            , default is `torch.optim.Adam`.
        :param dict optimizer_model_kwargs: Optimizer constructor keyword
            args. for the model.
        :param torch.optim.Optimizer optimizer_weights: The neural
            network optimizer to use for mask model model,
            default is `torch.optim.Adam`.
        :param dict optimizer_weights_kwargs: Optimizer constructor
            keyword args. for the mask module.
        :param torch.optim.LRScheduler scheduler_model: Learning
            rate scheduler for the model.
        :param dict scheduler_model_kwargs: LR scheduler constructor
            keyword args.
        :param torch.optim.LRScheduler scheduler_weights: Learning
            rate scheduler for the mask model.
        :param dict scheduler_model_kwargs: LR scheduler constructor
            keyword args.
        """

        # check consistency weitghs_function
        check_consistency(weights_function, torch.nn.Module)

        # create models for weights
        weights_dict = {}
        for condition_name in problem.conditions:
            weights_dict[condition_name] = Weights(weights_function)
        weights_dict = torch.nn.ModuleDict(weights_dict)

        super().__init__(
            models=[model, weights_dict],
            problem=problem,
            optimizers=[optimizer_model, optimizer_weights],
            optimizers_kwargs=[
                optimizer_model_kwargs,
                optimizer_weights_kwargs,
            ],
            extra_features=extra_features,
            loss=loss,
        )

        # set automatic optimization
        self.automatic_optimization = False

        # check consistency
        check_consistency(scheduler_model, LRScheduler, subclass=True)
        check_consistency(scheduler_model_kwargs, dict)
        check_consistency(scheduler_weights, LRScheduler, subclass=True)
        check_consistency(scheduler_weights_kwargs, dict)

        # assign schedulers
        self._schedulers = [
            scheduler_model(self.optimizers[0], **scheduler_model_kwargs),
            scheduler_weights(self.optimizers[1], **scheduler_weights_kwargs),
        ]

        self._model = self.models[0]
        self._weights = self.models[1]

        self._vectorial_loss = deepcopy(loss)
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
        return self.neural_net(x)

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
        # train weights
        self.optimizer_weights.zero_grad()
        weighted_loss, _ = self._loss_phys(samples, equation)
        loss_value = -weighted_loss.as_subclass(torch.Tensor)
        self.manual_backward(loss_value)
        self.optimizer_weights.step()

        # detaching samples from the computational graph to erase it and setting
        # the gradient to true to create a new computational graph.
        # In alternative set `retain_graph=True`.
        samples = samples.detach()
        samples.requires_grad = True

        # train model
        self.optimizer_model.zero_grad()
        weighted_loss, loss = self._loss_phys(samples, equation)
        loss_value = weighted_loss.as_subclass(torch.Tensor)
        self.manual_backward(loss_value)
        self.optimizer_model.step()

        # store loss without weights
        self.store_log(loss_value=float(loss))
        return loss_value

    def loss_data(self, input_tensor, output_tensor):
        """
        Computes the data loss for the SAPINN solver based on input and
        output. It computes the loss between the
        network output against the true solution.

        :param LabelTensor input_tensor: The input to the neural networks.
        :param LabelTensor output_tensor: The true solution to compare the
            network solution.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        # train weights
        self.optimizer_weights.zero_grad()
        weighted_loss, _ = self._loss_data(input_tensor, output_tensor)
        loss_value = -weighted_loss.as_subclass(torch.Tensor)
        self.manual_backward(loss_value)
        self.optimizer_weights.step()

        # detaching samples from the computational graph to erase it and setting
        # the gradient to true to create a new computational graph.
        # In alternative set `retain_graph=True`.
        input_tensor = input_tensor.detach()
        input_tensor.requires_grad = True

        # train model
        self.optimizer_model.zero_grad()
        weighted_loss, loss = self._loss_data(input_tensor, output_tensor)
        loss_value = weighted_loss.as_subclass(torch.Tensor)
        self.manual_backward(loss_value)
        self.optimizer_model.step()

        # store loss without weights
        self.store_log(loss_value=float(loss))
        return loss_value

    def configure_optimizers(self):
        """
        Optimizer configuration for the SAPINN
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
        return self.optimizers, self._schedulers

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
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += (
            1
        )
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_start(self):
        """
        This method is called at the start of the training for setting
        the self adaptive weights as parameters of the mask model.

        :return: Whatever is returned by the parent
            method ``on_train_start``.
        :rtype: Any
        """
        device = torch.device(
            self.trainer._accelerator_connector._accelerator_flag
        )
        for condition_name, tensor in self.problem.input_pts.items():
            self.weights_dict.torchmodel[condition_name].sa_weights.data = (
                torch.rand((tensor.shape[0], 1), device=device)
            )
        return super().on_train_start()

    def on_load_checkpoint(self, checkpoint):
        """
        Overriding the Pytorch Lightning ``on_load_checkpoint`` to handle
        checkpoints for Self Adaptive Weights. This method should not be
        overridden if not intentionally.

        :param dict checkpoint: Pytorch Lightning checkpoint dict.
        """
        for condition_name, tensor in self.problem.input_pts.items():
            self.weights_dict.torchmodel[condition_name].sa_weights.data = (
                torch.rand((tensor.shape[0], 1))
            )
        return super().on_load_checkpoint(checkpoint)

    def _loss_phys(self, samples, equation):
        """
        Elaboration of the physical loss for the SAPINN solver.

        :param LabelTensor samples: Input samples to evaluate the physics loss.
        :param EquationInterface equation: the governing equation representing
            the physics.

        :return: tuple with weighted and not weighted scalar loss
        :rtype: List[LabelTensor, LabelTensor]
        """
        residual = self.compute_residual(samples, equation)
        return self._compute_loss(residual)

    def _loss_data(self, input_tensor, output_tensor):
        """
        Elaboration of the loss related to data for the SAPINN solver.

        :param LabelTensor input_tensor: The input to the neural networks.
        :param LabelTensor output_tensor: The true solution to compare the
            network solution.

        :return: tuple with weighted and not weighted scalar loss
        :rtype: List[LabelTensor, LabelTensor]
        """
        residual = self.forward(input_tensor) - output_tensor
        return self._compute_loss(residual)

    def _compute_loss(self, residual):
        """
        Elaboration of the pointwise loss through the mask model and the
        self adaptive weights

        :param LabelTensor residual: the matrix of residuals that have to
            be weighted

        :return: tuple with weighted and not weighted loss
        :rtype List[LabelTensor, LabelTensor]
        """
        weights = self.weights_dict.torchmodel[
            self.current_condition_name
        ].forward()
        loss_value = self._vectorial_loss(
            torch.zeros_like(residual, requires_grad=True), residual
        )
        return (
            self._vect_to_scalar(weights * loss_value),
            self._vect_to_scalar(loss_value),
        )

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
    def neural_net(self):
        """
        Returns the neural network model.

        :return: The neural network model.
        :rtype: torch.nn.Module
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
        return self._scheduler[0]

    @property
    def scheduler_weights(self):
        """
        Returns the scheduler associated with the mask model (if applicable).

        :return: The scheduler for the mask model.
        :rtype: torch.optim.lr_scheduler._LRScheduler
        """
        return self._scheduler[1]

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

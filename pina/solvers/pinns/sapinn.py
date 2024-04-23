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
    This class aims to implements the weights of the Self-Adaptive
    PINN solver.
    """

    def __init__(self, func):
        """
        TODO
        """
        super().__init__()
        check_consistency(func, torch.nn.Module)
        self.sa_weights = torch.nn.Parameter(
            torch.Tensor()
        )
        self.func = func
    
    def forward(self):
        return self.func(self.sa_weights)

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
            weights_function=torch.nn.Sigmoid(),
            extra_features=None,
            loss=torch.nn.MSELoss(),
            optimizer_model=torch.optim.Adam,
            optimizer_model_kwargs={"lr" : 0.001},
            optimizer_weights=torch.optim.Adam,
            optimizer_weights_kwargs={"lr" : 0.001},
            scheduler_model=ConstantLR,
            scheduler_model_kwargs={"factor" : 1, "total_iters" : 0},
            scheduler_weights=ConstantLR,
            scheduler_weights_kwargs={"factor" : 1, "total_iters" : 0}
    ):
        """
        weights_function - torch.nn.___ funzione di attivazione per il modello sui pesi per ogni peso
        # number of points fixed in the training
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
            optimizers_kwargs=[optimizer_model_kwargs, optimizer_weights_kwargs],
            extra_features=extra_features,
            loss=loss
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
            scheduler_model(
                self.optimizers[0], **scheduler_model_kwargs
            ),
            scheduler_weights(
                self.optimizers[1], **scheduler_weights_kwargs
            ),
        ]

        self._model = self.models[0]
        self._weights = self.models[1]

        self._vectorial_loss = deepcopy(loss)
        self._vectorial_loss.reduction = "none"
    
    def on_train_start(self):
        for condition_name, tensor in self.problem.input_pts.items():
            self.weights_dict.torchmodel[condition_name].sa_weights.data = torch.rand(
                (tensor.shape[0], 1),
                dtype = tensor.dtype,
                device = tensor.device
            )
        return super().on_train_start()
    
    def on_train_batch_end(self,outputs, batch, batch_idx):
        """
        This method is called at the end of each training batch, and ovverides
        the PytorchLightining implementation for logging the checkpoints.

        :param outputs: The output from the model for the current batch.
        :type outputs: Any
        :param batch: The current batch of data.
        :type batch: Any
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :return: Whatever is returned by the parent
            method ``on_train_batch_end``.
        :rtype: Any
        """
        # increase by one the counter of optimization to save loggers
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += 1
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
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
        return self.optimizers, self._schedulers
    
    def _loss_data(self, input_tensor, output_tensor):
        """
        TODO
        """
        residual = self.forward(input_tensor) - output_tensor
        return self._compute_loss(residual)

    def _compute_loss(self, residual):
        weights = self.weights_dict.torchmodel[self.current_condition_name].forward()
        loss_value = self._vectorial_loss(torch.zeros_like(residual, requires_grad=True), residual)
        return self._vect_to_scalar(weights * loss_value), self._vect_to_scalar(loss_value)

    def loss_data(self, input_tensor, output_tensor):
        """
        Computes the data loss for the PINN solver based on input,
        output, and condition name. This function is a wrapper of the function
        :meth:`loss_data` used internally in PINA to handle the logging step.

        :param LabelTensor input_tensor: The input to the neural networks.
        :param LabelTensor output_tensor: The true solution to compare the
            network solution.
        :return: The computed data loss.
        :rtype: torch.Tensor
        """
        # train weights
        self.optimizer_weights.zero_grad()
        weighted_loss, _ = self._loss_data(input_tensor, output_tensor)
        loss_value = - weighted_loss.as_subclass(torch.Tensor)
        self.manual_backward(loss_value)
        self.optimizer_weights.step()

        # detaching samples from the computational graph to erase it and setting
        # the gradient to true to create a new computational graph.
        # In alternative set `retain_graph=True`.
        samples = samples.detach()
        samples.requires_grad = True

        # train model
        self.optimizer_model.zero_grad()
        weighted_loss, loss = self._loss_data(input_tensor, output_tensor)
        loss_value = weighted_loss.as_subclass(torch.Tensor)
        self.manual_backward(loss_value)
        self.optimizer_model.step()

        # store loss without weights
        self.store_log(loss_value=float(loss))
        return loss_value

    def _vect_to_scalar(self, loss_value):
        if self.loss.reduction == "mean":
            ret = torch.mean(loss_value)
        elif self.loss.reduction == "sum":
            ret = torch.sum(loss_value)
        else:
            raise RuntimeError(f"Invalid reduction, got {self.loss.reduction} but expected mean or sum.")
        return ret
        

    def _loss_phys(self, samples, equation):
        """
        TODO
        """
        residual = self.compute_residual(samples, equation)
        return self._compute_loss(residual)

    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the PINN solver based on given
        samples and equation.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor"""
        # train weights
        self.optimizer_weights.zero_grad()
        weighted_loss, _ = self._loss_phys(samples, equation)
        loss_value = - weighted_loss.as_subclass(torch.Tensor)
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

    @property
    def neural_net(self):
        """
        Neural network for the PINN training.
        """
        return self.models[0]
    
    @property
    def weights_dict(self):
        return self.models[1]

    @property
    def scheduler_model(self):
        return self._scheduler[0]
    
    @property
    def scheduler_weights(self):
        return self._scheduler[1]

    @property
    def optimizer_model(self):
        return self.optimizers[0]
    
    @property
    def optimizer_weights(self):
        return self.optimizers[1]
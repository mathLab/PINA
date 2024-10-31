""" Module for PINN """

from abc import ABCMeta, abstractmethod
import torch
from torch.nn.modules.loss import _Loss
from ...condition import InputOutputPointsCondition
from ...solvers.solver import SolverInterface
from ...utils import check_consistency
from ...loss.loss_interface import LossInterface
from ...problem import InverseProblem
from ...condition import DomainEquationCondition
from ...optim import TorchOptimizer, TorchScheduler

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class PINNInterface(SolverInterface, metaclass=ABCMeta):
    """
    Base PINN solver class. This class implements the Solver Interface
    for Physics Informed Neural Network solvers.

    This class can be used to
    define PINNs with multiple ``optimizers``, and/or ``models``.
    By default it takes
    an :class:`~pina.problem.abstract_problem.AbstractProblem`, so it is up
    to the user to choose which problem the implemented solver inheriting from
    this class is suitable for.
    """
    accepted_condition_types = [DomainEquationCondition.condition_type[0],
                                InputOutputPointsCondition.condition_type[0]]
    def __init__(
            self,
            models,
            problem,
            optimizers,
            schedulers,
            extra_features,
            loss,
    ):
        """
        :param models: Multiple torch neural network models instances.
        :type models: list(torch.nn.Module)
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param list(torch.optim.Optimizer) optimizer: A list of neural network
            optimizers to use.
        :param list(dict) optimizer_kwargs: A list of optimizer constructor
            keyword args.
        :param list(torch.nn.Module) extra_features: The additional input
            features to use as augmented input. If ``None`` no extra features
            are passed. If it is a list of :class:`torch.nn.Module`,
            the extra feature list is passed to all models. If it is a list
            of extra features' lists, each single list of extra feature
            is passed to a model.
        :param torch.nn.Module loss: The loss function used as minimizer,
            default :class:`torch.nn.MSELoss`.
        """
        if optimizers is None:
            optimizers = TorchOptimizer(torch.optim.Adam, lr=0.001)

        if schedulers is None:
            schedulers = TorchScheduler(torch.optim.lr_scheduler.ConstantLR)

        if loss is None:
            loss = torch.nn.MSELoss()

        super().__init__(
            models=models,
            problem=problem,
            optimizers=optimizers,
            schedulers=schedulers,
            extra_features=extra_features,
        )

        # check consistency
        check_consistency(loss, (LossInterface, _Loss), subclass=False)

        # assign variables
        self._loss = loss

        # inverse problem handling
        if isinstance(self.problem, InverseProblem):
            self._params = self.problem.unknown_parameters
            self._clamp_params = self._clamp_inverse_problem_params
        else:
            self._params = None
            self._clamp_params = lambda: None

        # variable used internally to store residual losses at each epoch
        # this variable save the residual at each iteration (not weighted)
        self.__logged_res_losses = []

        # variable used internally in pina for logging. This variable points to
        # the current condition during the training step and returns the
        # condition name. Whenever :meth:`store_log` is called the logged
        # variable will be stored with name = self.__logged_metric
        self.__logged_metric = None

        self._model = self._pina_models[0]
        self._optimizer = self._pina_optimizers[0]
        self._scheduler = self._pina_schedulers[0]

    def training_step(self, batch, _):
        """
        The Physics Informed Solver Training Step. This function takes care
        of the physics informed training step, and it must not be override
        if not intentionally. It handles the batching mechanism, the workload
        division for the various conditions, the inverse problem clamping,
        and loggers.

        :param tuple batch: The batch element in the dataloader.
        :param int batch_idx: The batch index.
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """

        condition_losses = []

        physics = batch.physics
        if hasattr(batch, 'supervised'):
            supervised = batch.supervised
            condition_idx = supervised.condition_indices
        else:
            condition_idx = torch.tensor([])

        for condition_id in torch.unique(condition_idx).tolist():
            condition_name = self._dataloader.condition_names[condition_id]
            condition = self.problem.conditions[condition_name]
            self.__logged_metric = condition_name
            pts = batch.supervised.input_points
            out = batch.supervised.output_points
            output_pts = out[condition_idx == condition_id]
            input_pts = pts[condition_idx == condition_id]

            input_pts.labels = pts.labels
            output_pts.labels = out.labels

            loss = self.loss_data(input_points=input_pts, output_points=output_pts)
            loss = loss.as_subclass(torch.Tensor)

        condition_idx = physics.condition_indices
        for condition_id in torch.unique(condition_idx).tolist():

            condition_name = self._dataloader.condition_names[condition_id]
            condition = self.problem.conditions[condition_name]
            self.__logged_metric = condition_name
            pts = batch.physics.input_points
            input_pts = pts[condition_idx == condition_id]

            input_pts.labels = pts.labels
            loss = self.loss_phys(pts, condition.equation)

            # add condition losses for each epoch
            condition_losses.append(loss)

        # clamp unknown parameters in InverseProblem (if needed)
        self._clamp_params()

        # total loss (must be a torch.Tensor)
        total_loss = sum(condition_losses)
        return total_loss.as_subclass(torch.Tensor)

    def loss_data(self, input_points, output_points):
        """
        The data loss for the PINN solver. It computes the loss between
        the network output against the true solution. This function
        should not be override if not intentionally.

        :param LabelTensor input_tensor: The input to the neural networks.
        :param LabelTensor output_tensor: The true solution to compare the
            network solution.
        :return: The residual loss averaged on the input coordinates
        :rtype: torch.Tensor
        """
        loss_value = self.loss(self.forward(input_points), output_points)
        self.store_log(loss_value=float(loss_value))
        return loss_value

    @abstractmethod
    def loss_phys(self, samples, equation):
        """
        Computes the physics loss for the physics informed solver based on given
        samples and equation. This method must be override by all inherited
        classes and it is the core to define a new physics informed solver.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The physics loss calculated based on given
            samples and equation.
        :rtype: LabelTensor
        """
        pass

    def compute_residual(self, samples, equation):
        """
        Compute the residual for Physics Informed learning. This function
        returns the :obj:`~pina.equation.equation.Equation` specified in the
        :obj:`~pina.condition.Condition` evaluated at the ``samples`` points.

        :param LabelTensor samples: The samples to evaluate the physics loss.
        :param EquationInterface equation: The governing equation
            representing the physics.
        :return: The residual of the neural network solution.
        :rtype: LabelTensor
        """
        try:
            residual = equation.residual(samples, self.forward(samples))
        except (
                TypeError
        ):  # this occurs when the function has three inputs, i.e. inverse problem
            residual = equation.residual(
                samples, self.forward(samples), self._params
            )
        return residual

    def store_log(self, loss_value):
        """
        Stores the loss value in the logger. This function should be
        called for all conditions. It automatically handles the storing
        conditions names. It must be used
        anytime a specific variable wants to be stored for a specific condition.
        A simple example is to use the variable to store the residual.

        :param str name: The name of the loss.
        :param torch.Tensor loss_value: The value of the loss.
        """
        self.log(
            self.__logged_metric + "_loss",
            loss_value,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
            batch_size=self._dataloader.batch_size,
        )
        self.__logged_res_losses.append(loss_value)

    def on_train_epoch_end(self):
        """
        At the end of each epoch we free the stored losses. This function
        should not be override if not intentionally.
        """
        if self.__logged_res_losses:
            # storing mean loss
            self.__logged_metric = "mean"
            self.store_log(
                sum(self.__logged_res_losses) / len(self.__logged_res_losses)
            )
            # free the logged losses
            self.__logged_res_losses = []
        return super().on_train_epoch_end()

    def _clamp_inverse_problem_params(self):
        """
        Clamps the parameters of the inverse problem
        solver to the specified ranges.
        """
        for v in self._params:
            self._params[v].data.clamp_(
                self.problem.unknown_parameter_domain.range_[v][0],
                self.problem.unknown_parameter_domain.range_[v][1],
            )

    @property
    def loss(self):
        """
        Loss used for training.
        """
        return self._loss

    @property
    def current_condition_name(self):
        """
        Returns the condition name. This function can be used inside the
        :meth:`loss_phys` to extract the condition at which the loss is
        computed.
        """
        return self.__logged_metric

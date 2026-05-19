"""Module for the BaseSolver class."""

from abc import ABCMeta

import torch
import lightning
from torch._dynamo.eval_frame import OptimizedModule
from pina._src.problem.inverse_problem import InverseProblem
from pina._src.optim.optimizer_interface import OptimizerInterface
from pina._src.optim.scheduler_interface import SchedulerInterface
from pina._src.core.utils import check_consistency
from pina._src.solver.solver_interface import SolverInterface
from pina._src.problem.base_problem import BaseProblem
from pina._src.problem.inverse_problem import InverseProblem
from pina._src.optim.torch_optimizer import TorchOptimizer
from pina._src.optim.torch_scheduler import TorchScheduler
from pina._src.weighting.weighting_interface import WeightingInterface
from pina._src.weighting.no_weighting import _NoWeighting
from pina._src.core.utils import labelize_forward


class BaseSolver(SolverInterface, metaclass=ABCMeta):
    """
    Base class for PINA solvers using a single :class:`torch.nn.Module`.
    """

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        """
        Initialization of the :class:`BaseSolver` class.

        :param BaseProblem problem: The problem to be solved.
        :param torch.nn.Module model: The neural network model to be used.
        :param OptimizerInterface optimizer: The optimizer to be used.
            If ``None``, the :class:`torch.optim.Adam` optimizer is
            used. Default is ``None``.
        :param SchedulerInterface scheduler: The scheduler to be used.
            If ``None``, the :class:`torch.optim.lr_scheduler.ConstantLR`
            scheduler is used. Default is ``None``.
        :param WeightingInterface weighting: The weighting schema to be used.
            If ``None``, no weighting schema is used. Default is ``None``.
        :param bool use_lt: If ``True``, the solver uses LabelTensors as input.
        """
        if optimizer is None:
            optimizer = self.default_torch_optimizer()

        if scheduler is None:
            scheduler = self.default_torch_scheduler()

        if weighting is None:
            weighting = _NoWeighting()

        check_consistency(model, torch.nn.Module)
        check_consistency(scheduler, SchedulerInterface)
        check_consistency(optimizer, OptimizerInterface)
        check_consistency(problem, BaseProblem)
        check_consistency(use_lt, bool)
        check_consistency(weighting, WeightingInterface)

        # initialize the model (needed by Lightining to go to different devices)
        self.reset()
        lightning.pytorch.LightningModule.__init__(self)
        if not isinstance(model, list):
            model = [model]
        self._pina_models = torch.nn.ModuleList(model)
        self._pina_optimizers = [optimizer]
        self._pina_schedulers = [scheduler]
        self._check_solver_consistency(problem)
        self._pina_problem = problem

        self._pina_weighting = weighting
        weighting._solver = self

        # check consistency use_lt
        self._use_lt = use_lt

        # if use_lt is true add extract operation in input
        if use_lt is True:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=problem.input_variables,
                output_variables=problem.output_variables,
            )

        # PINA private attributes (some are overridden by derived classes)

        # inverse problem handling
        if isinstance(self.problem, InverseProblem):
            self._params = self.problem.unknown_parameters
            self._clamp_params = self._clamp_inverse_problem_params
        else:
            self._params = None
            self._clamp_params = lambda: None

    def reset(self):
        self._pina_problem = None
        self._pina_models = None
        self._pina_optimizers = None
        self._pina_schedulers = None

    def forward(self, x):
        """
        Forward pass implementation.

        :param x: Input tensor.
        :type x: torch.Tensor | LabelTensor | Graph | Data
        :return: Solver solution.
        :rtype: torch.Tensor | LabelTensor | Graph | Data
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Optimizer configuration for the solver.

        :return: The optimizer and the scheduler
        :rtype: tuple[list[OptimizerInterface], list[SchedulerInterface]]
        """
        self.optimizer.hook(self.model.parameters())
        if isinstance(self.problem, InverseProblem):
            self.optimizer.instance.add_param_group(
                {
                    "params": [
                        self._params[var]
                        for var in self.problem.unknown_variables
                    ]
                }
            )
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.instance], [self.scheduler.instance])

    @property
    def model(self):
        """
        The model used for training.

        :return: The model used for training.
        :rtype: torch.nn.Module
        """
        return self._pina_models[0]

    @property
    def scheduler(self):
        """
        The scheduler used for training.

        :return: The scheduler used for training.
        :rtype: SchedulerInterface
        """
        return self._pina_schedulers[0]

    @property
    def optimizer(self):
        """
        The optimizer used for training.

        :return: The optimizer used for training.
        :rtype: OptimizerInterface
        """
        return self._pina_optimizers[0]

    def training_step(self, batch, **kwargs):
        """
        Solver training step. It computes the optimization cycle and aggregates
        the losses using the ``weighting`` attribute.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        loss = self._optimization_cycle(batch=batch, **kwargs)
        self.store_log("train_loss", loss, self.get_batch_size(batch))
        return loss

    def validation_step(self, batch, **kwargs):
        """
        Solver validation step. It computes the optimization cycle and
        averages the losses. No aggregation using the ``weighting`` attribute is
        performed.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        losses = self.optimization_cycle(batch=batch, **kwargs)
        loss = (sum(losses.values()) / len(losses)).as_subclass(torch.Tensor)
        self.store_log("val_loss", loss, self.get_batch_size(batch))
        return loss

    def test_step(self, batch, **kwargs):
        """
        Solver test step. It computes the optimization cycle and
        averages the losses. No aggregation using the ``weighting`` attribute is
        performed.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """
        losses = self.optimization_cycle(batch=batch, **kwargs)
        loss = (sum(losses.values()) / len(losses)).as_subclass(torch.Tensor)
        self.store_log("test_loss", loss, self.get_batch_size(batch))
        return loss

    def store_log(self, name, value, batch_size):
        """
        Store the log of the solver.

        :param str name: The name of the log.
        :param torch.Tensor value: The value of the log.
        :param int batch_size: The size of the batch.
        """
        self.log(
            name=name,
            value=value,
            batch_size=batch_size,
            **self.trainer.logging_kwargs,
        )

    def setup(self, stage):
        """
        This method is called at the start of the train and test process to
        compile the model if the :class:`~pina.trainer.Trainer`
        ``compile`` is ``True``.

        :param str stage: The current stage of the training process
            (e.g., ``fit``, ``validate``, ``test``, ``predict``).
        :return: The result of the parent class ``setup`` method.
        :rtype: Any
        """
        if self.trainer.compile and not self._is_compiled():
            self._setup_compile()
        return super().setup(stage)

    def _is_compiled(self):
        """
        Check if the model is compiled.

        :return: ``True`` if the model is compiled, ``False`` otherwise.
        :rtype: bool
        """
        for model in self._pina_models:
            if not isinstance(model, OptimizedModule):
                return False
        return True

    def _setup_compile(self):
        """
        Compile all models in the solver using ``torch.compile``.

        This method iterates through each model stored in the solver
        list and attempts to compile them for optimized execution. It supports
        models of type `torch.nn.Module` and `torch.nn.ModuleDict`. For models
        stored in a `ModuleDict`, each submodule is compiled individually.
        Models on Apple Silicon (MPS) use the 'eager' backend,
        while others use 'inductor'.

        :raises RuntimeError: If a model is neither `torch.nn.Module`
            nor `torch.nn.ModuleDict`.
        """
        for i, model in enumerate(self._pina_models):
            if isinstance(model, torch.nn.ModuleDict):
                for name, module in model.items():
                    self._pina_models[i][name] = self._compile_modules(module)
            elif isinstance(model, torch.nn.Module):
                self._pina_models[i] = self._compile_modules(model)
            else:
                raise RuntimeError(
                    "Compilation available only for "
                    "torch.nn.Module or torch.nn.ModuleDict."
                )

    def _check_solver_consistency(self, problem):
        """
        Check the consistency of the solver with the problem formulation.

        :param BaseProblem problem: The problem to be solved.
        """
        for condition in problem.conditions.values():
            check_consistency(condition, self.accepted_conditions_types)

    def _optimization_cycle(self, batch, **kwargs):
        """
        Aggregate the loss for each condition in the batch.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :param dict kwargs: Additional keyword arguments passed to
            ``optimization_cycle``.
        :return: The losses computed for all conditions in the batch, casted
            to a subclass of :class:`torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict
        """
        # compute losses
        losses = self.optimization_cycle(batch)
        # clamp unknown parameters in InverseProblem (if needed)
        self._clamp_params()
        # store log
        for name, value in losses.items():
            self.store_log(
                f"{name}_loss", value.item(), self.get_batch_size(batch)
            )
        # aggregate
        loss = self.weighting.aggregate(losses).as_subclass(torch.Tensor)
        return loss

    def _clamp_inverse_problem_params(self):
        """
        Clamps the parameters of the inverse problem solver to specified ranges.
        """
        for v in self._params:
            self._params[v].data.clamp_(
                self.problem.unknown_parameter_domain.range[v][0],
                self.problem.unknown_parameter_domain.range[v][1],
            )

    @staticmethod
    def _compile_modules(model):
        """
        Perform the compilation of the model.

        This method attempts to compile the given PyTorch model
        using ``torch.compile`` to improve execution performance. The
        backend is selected based on the device on which the model resides:
        ``eager`` is used for MPS devices (Apple Silicon), and ``inductor``
        is used for all others.

        If compilation fails, the method prints the error and returns the
        original, uncompiled model.

        :param torch.nn.Module model: The model to compile.
        :raises Exception: If the compilation fails.
        :return: The compiled model.
        :rtype: torch.nn.Module
        """
        model_device = next(model.parameters()).device
        try:
            if model_device == torch.device("mps:0"):
                model = torch.compile(model, backend="eager")
            else:
                model = torch.compile(model, backend="inductor")
        except Exception as e:
            print("Compilation failed, running in normal mode.:\n", e)
        return model

    @staticmethod
    def get_batch_size(batch):
        """
        Get the batch size.

        :param list[tuple[str, dict]] batch: A batch of data. Each element is a
            tuple containing a condition name and a dictionary of points.
        :return: The size of the batch.
        :rtype: int
        """
        batch_size = 0
        for data in batch:
            batch_size += len(data[1]["input"])
        return batch_size

    @staticmethod
    def default_torch_optimizer():
        """
        Set the default optimizer to :class:`torch.optim.Adam`.

        :return: The default optimizer.
        :rtype: OptimizerInterface
        """
        return TorchOptimizer(torch.optim.Adam, lr=0.001)

    @staticmethod
    def default_torch_scheduler():
        """
        Set the default scheduler to
        :class:`torch.optim.lr_scheduler.ConstantLR`.

        :return: The default scheduler.
        :rtype: SchedulerInterface
        """
        return TorchScheduler(torch.optim.lr_scheduler.ConstantLR, factor=1.0)

    @property
    def problem(self):
        """
        The problem instance.

        :return: The problem instance.
        :rtype: :class:`~pina.problem.base_problem.BaseProblem`
        """
        return self._pina_problem

    @property
    def use_lt(self):
        """
        Using LabelTensors as input during training.

        :return: The use_lt attribute.
        :rtype: bool
        """
        return self._use_lt

    @property
    def weighting(self):
        """
        The weighting schema.

        :return: The weighting schema.
        :rtype: :class:`~pina.loss.weighting_interface.WeightingInterface`
        """
        return self._pina_weighting

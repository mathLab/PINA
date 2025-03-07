"""Solver module."""

from abc import ABCMeta, abstractmethod
import lightning
import torch

from torch._dynamo.eval_frame import OptimizedModule
from ..problem import AbstractProblem
from ..optim import Optimizer, Scheduler, TorchOptimizer, TorchScheduler
from ..loss import WeightingInterface
from ..loss.scalar_weighting import _NoWeighting
from ..utils import check_consistency, labelize_forward


class SolverInterface(lightning.pytorch.LightningModule, metaclass=ABCMeta):
    """
    SolverInterface base class. This class is a wrapper of LightningModule.
    """

    def __init__(self, problem, weighting, use_lt):
        """
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param weighting: The loss weighting to use.
        :type weighting: WeightingInterface
        :param use_lt: Using LabelTensors as input during training.
        :type use_lt: bool
        """
        super().__init__()

        # check consistency of the problem
        check_consistency(problem, AbstractProblem)
        self._check_solver_consistency(problem)
        self._pina_problem = problem

        # check consistency of the weighting and hook the condition names
        if weighting is None:
            weighting = _NoWeighting()
        check_consistency(weighting, WeightingInterface)
        self._pina_weighting = weighting
        weighting.condition_names = list(self._pina_problem.conditions.keys())

        # check consistency use_lt
        check_consistency(use_lt, bool)
        self._use_lt = use_lt

        # if use_lt is true add extract operation in input
        if use_lt is True:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=problem.input_variables,
                output_variables=problem.output_variables,
            )

        # PINA private attributes (some are overridden by derived classes)
        self._pina_problem = problem
        self._pina_models = None
        self._pina_optimizers = None
        self._pina_schedulers = None

    def _check_solver_consistency(self, problem):
        for condition in problem.conditions.values():
            check_consistency(condition, self.accepted_conditions_types)

    def _optimization_cycle(self, batch):
        """
        Perform a private optimization cycle by computing the loss for each
        condition in the given batch. The loss are later aggregated using the
        specific weighting schema.

        :param batch: A batch of data, where each element is a tuple containing
            a condition name and a dictionary of points.
        :type batch: list of tuples (str, dict)
        :return: The computed loss for the all conditions in the batch,
            cast to a subclass of `torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict(torch.Tensor)
        """
        losses = self.optimization_cycle(batch)
        for name, value in losses.items():
            self.store_log(
                f"{name}_loss", value.item(), self.get_batch_size(batch)
            )
        loss = self.weighting.aggregate(losses).as_subclass(torch.Tensor)
        return loss

    def training_step(self, batch):
        """
        Solver training step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """
        loss = self._optimization_cycle(batch=batch)
        self.store_log("train_loss", loss, self.get_batch_size(batch))
        return loss

    def validation_step(self, batch):
        """
        Solver validation step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        """
        loss = self._optimization_cycle(batch=batch)
        self.store_log("val_loss", loss, self.get_batch_size(batch))

    def test_step(self, batch):
        """
        Solver test step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        """
        loss = self._optimization_cycle(batch=batch)
        self.store_log("test_loss", loss, self.get_batch_size(batch))

    def store_log(self, name, value, batch_size):
        """
        TODO
        """

        self.log(
            name=name,
            value=value,
            batch_size=batch_size,
            **self.trainer.logging_kwargs,
        )

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        TODO
        """

    @abstractmethod
    def optimization_cycle(self, batch):
        """
        Perform an optimization cycle by computing the loss for each condition
        in the given batch.

        :param batch: A batch of data, where each element is a tuple containing
            a condition name and a dictionary of points.
        :type batch: list of tuples (str, dict)
        :return: The computed loss for the all conditions in the batch,
            cast to a subclass of `torch.Tensor`. It should return a dict
            containing the condition name and the associated scalar loss.
        :rtype: dict(torch.Tensor)
        """

    @property
    def problem(self):
        """
        The problem formulation.
        """
        return self._pina_problem

    @property
    def use_lt(self):
        """
        Using LabelTensor in training.
        """
        return self._use_lt

    @property
    def weighting(self):
        """
        The weighting mechanism.
        """
        return self._pina_weighting

    @staticmethod
    def get_batch_size(batch):
        """
        TODO
        """

        batch_size = 0
        for data in batch:
            batch_size += len(data[1]["input"])
        return batch_size

    @staticmethod
    def default_torch_optimizer():
        """
        TODO
        """

        return TorchOptimizer(torch.optim.Adam, lr=0.001)

    @staticmethod
    def default_torch_scheduler():
        """
        TODO
        """

        return TorchScheduler(torch.optim.lr_scheduler.ConstantLR)

    def on_train_start(self):
        """
        Hook that is called before training begins.
        Used to compile the model if the trainer is set to compile.
        """
        super().on_train_start()
        if self.trainer.compile:
            self._compile_model()

    def on_test_start(self):
        """
        Hook that is called before training begins.
        Used to compile the model if the trainer is set to compile.
        """
        super().on_train_start()
        if self.trainer.compile and not self._check_already_compiled():
            self._compile_model()

    def _check_already_compiled(self):
        """
        TODO
        """

        models = self._pina_models
        if len(models) == 1 and isinstance(
            self._pina_models[0], torch.nn.ModuleDict
        ):
            models = list(self._pina_models.values())
        for model in models:
            if not isinstance(model, (OptimizedModule, torch.nn.ModuleDict)):
                return False
        return True

    @staticmethod
    def _perform_compilation(model):
        """
        TODO
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


class SingleSolverInterface(SolverInterface, metaclass=ABCMeta):
    """TODO"""

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
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param model: A torch nn.Module instances.
        :type model: torch.nn.Module
        :param Optimizer optimizers: A neural network optimizers to use.
        :param Scheduler optimizers: A neural network scheduler to use.
        :param WeightingInterface weighting: The loss weighting to use.
        :param bool use_lt: Using LabelTensors as input during training.
        """
        if optimizer is None:
            optimizer = self.default_torch_optimizer()

        if scheduler is None:
            scheduler = self.default_torch_scheduler()

        super().__init__(problem=problem, use_lt=use_lt, weighting=weighting)

        # check consistency of models argument and encapsulate in list
        check_consistency(model, torch.nn.Module)
        # check scheduler consistency and encapsulate in list
        check_consistency(scheduler, Scheduler)
        # check optimizer consistency and encapsulate in list
        check_consistency(optimizer, Optimizer)

        # initialize the model (needed by Lightining to go to different devices)
        self._pina_models = torch.nn.ModuleList([model])
        self._pina_optimizers = [optimizer]
        self._pina_schedulers = [scheduler]

    def forward(self, x):
        """
        Forward pass implementation for the solver.

        :param torch.Tensor x: Input tensor.
        :return: Solver solution.
        :rtype: torch.Tensor
        """
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        Optimizer configuration for the solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        self.optimizer.hook(self.model.parameters())
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.instance], [self.scheduler.instance])

    def _compile_model(self):
        if isinstance(self._pina_models[0], torch.nn.ModuleDict):
            self._compile_module_dict()
        else:
            self._compile_single_model()

    def _compile_module_dict(self):
        for name, model in self._pina_models[0].items():
            self._pina_models[0][name] = self._perform_compilation(model)

    def _compile_single_model(self):
        self._pina_models[0] = self._perform_compilation(self._pina_models[0])

    @property
    def model(self):
        """
        Model for training.
        """
        return self._pina_models[0]

    @property
    def scheduler(self):
        """
        Scheduler for training.
        """
        return self._pina_schedulers[0]

    @property
    def optimizer(self):
        """
        Optimizer for training.
        """
        return self._pina_optimizers[0]


class MultiSolverInterface(SolverInterface, metaclass=ABCMeta):
    """
    Multiple Solver base class. This class inherits is a wrapper of
    SolverInterface class
    """

    def __init__(
        self,
        problem,
        models,
        optimizers=None,
        schedulers=None,
        weighting=None,
        use_lt=True,
    ):
        """
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param models: Multiple torch nn.Module instances.
        :type model: list[torch.nn.Module] | tuple[torch.nn.Module]
        :param list(Optimizer) optimizers: A list of neural network
           optimizers to use.
        :param list(Scheduler) optimizers: A list of neural network
           schedulers to use.
        :param WeightingInterface weighting: The loss weighting to use.
        :param bool use_lt: Using LabelTensors as input during training.
        """
        if not isinstance(models, (list, tuple)) or len(models) < 2:
            raise ValueError(
                "models should be list[torch.nn.Module] or "
                "tuple[torch.nn.Module] with len greater than "
                "one."
            )

        if any(opt is None for opt in optimizers):
            optimizers = [
                self.default_torch_optimizer() if opt is None else opt
                for opt in optimizers
            ]

        if any(sched is None for sched in schedulers):
            schedulers = [
                self.default_torch_scheduler() if sched is None else sched
                for sched in schedulers
            ]

        super().__init__(problem=problem, use_lt=use_lt, weighting=weighting)

        # check consistency of models argument and encapsulate in list
        check_consistency(models, torch.nn.Module)

        # check scheduler consistency and encapsulate in list
        check_consistency(schedulers, Scheduler)

        # check optimizer consistency and encapsulate in list
        check_consistency(optimizers, Optimizer)

        # check length consistency optimizers
        if len(models) != len(optimizers):
            raise ValueError(
                "You must define one optimizer for each model."
                f"Got {len(models)} models, and {len(optimizers)}"
                " optimizers."
            )

        # initialize the model
        self._pina_models = torch.nn.ModuleList(models)
        self._pina_optimizers = optimizers
        self._pina_schedulers = schedulers

    def configure_optimizers(self):
        """Optimizer configuration for the solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        for optimizer, scheduler, model in zip(
            self.optimizers, self.schedulers, self.models
        ):
            optimizer.hook(model.parameters())
            scheduler.hook(optimizer)

        return (
            [optimizer.instance for optimizer in self.optimizers],
            [scheduler.instance for scheduler in self.schedulers],
        )

    def _compile_model(self):
        for i, model in enumerate(self._pina_models):
            if not isinstance(model, torch.nn.ModuleDict):
                self._pina_models[i] = self._perform_compilation(model)

    @property
    def models(self):
        """
        The torch model."""
        return self._pina_models

    @property
    def optimizers(self):
        """
        The torch model."""
        return self._pina_optimizers

    @property
    def schedulers(self):
        """
        The torch model."""
        return self._pina_schedulers

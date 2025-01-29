""" Solver module. """

from abc import ABCMeta, abstractmethod
import lightning
from ..utils import check_consistency, labelize_forward
from ..problem import AbstractProblem
from ..optim import Optimizer, Scheduler, TorchOptimizer, TorchScheduler
import torch
import sys


class SolverInterface(lightning.pytorch.LightningModule, metaclass=ABCMeta):
    """
    SolverInterface base class. This class is a wrapper of the LightningModule
    """

    def __init__(self,
                 models,
                 problem,
                 optimizers,
                 schedulers,
                 use_lt,
                 extra_features=None):
        """
        :param models: Multiple torch nn.Module instances.
        :type models: list[torch.nn.Module] | tuple[torch.nn.Module] | torch.nn.Module
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param optimizers: A list of neural network optimizers to use.
        :type models: list(Optimizer) | tuple(Optimizer) | Optimizer
        :param schedulers: A list of neural network schedulers to use.
        :type models: list(Scheduler) | tuple(Scheduler) | Scheduler
        :param use_lt: Using LabelTensors as input during training.
        :type use_lt: bool
        """
        super().__init__()
        self._problem = problem

        # check consistency of the inputs
        check_consistency(problem, AbstractProblem)
        self._check_solver_consistency(problem)

        # Check consistency of models argument and encapsulate in list
        check_consistency(models, torch.nn.Module)

        # Check scheduler consistency + encapsulation
        check_consistency(schedulers, Scheduler)

        # Check optimizer consistency + encapsulation
        check_consistency(optimizers, Optimizer)

        # Check consistency extra_features
        if extra_features is None:
            extra_features = []
        else:
            check_consistency(extra_features, torch.nn.Module)

        # Check consistency use_lt
        check_consistency(use_lt, bool)
        self._use_lt = use_lt
        # If use_lt is true add extract operation in input
        if use_lt is True:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=problem.input_variables,
                output_variables=problem.output_variables,
                extra_features=extra_features
            )

        self._pina_problem = problem

    def _check_solver_consistency(self, problem):
        for condition in problem.conditions.values():
            check_consistency(condition, self.accepted_conditions_types)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch):
        pass

    @abstractmethod
    def test_step(self, batch):
        pass

    @property
    def problem(self):
        """
        The problem formulation."""
        return self._pina_problem

    @property
    def use_lt(self):
        """
        Using LabelTensor in training."""
        return self._use_lt

class MultipleSolversInterface(SolverInterface,
                               metaclass=ABCMeta):
    """
    Multiple Solver base class. This class inherits is a wrapper of
    SolverInterface class
    """

    def __init__(self,
                 models,
                 problem,
                 optimizers,
                 schedulers,
                 extra_features=None,
                 use_lt=True):
        """
        :param models: Multiple torch nn.Module instances.
        :type model: list[torch.nn.Module] | tuple[torch.nn.Module]
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param list(Optimizer) optimizers: A list of neural network
           optimizers to use.
        :param list(Scheduler) optimizers: A list of neural network
           schedulers to use.
        :param bool use_lt: Using LabelTensors as input during training.
        """
        super().__init__(problem=problem, use_lt=use_lt,
                         extra_features=extra_features, models=models,
                         schedulers=schedulers, optimizers=optimizers)

        # check length consistency optimizers
        len_model = len(models)
        len_optimizer = len(optimizers)
        if len_model != len_optimizer:
            raise ValueError("You must define one optimizer for each model."
                             f"Got {len_model} models, and {len_optimizer}"
                             " optimizers.")

        # extra features handling
        self._pina_models = models
        self._pina_optimizers = optimizers
        self._pina_schedulers = schedulers

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

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


class SingleSolverInterface(SolverInterface, metaclass=ABCMeta):
    def __init__(self,
                 model,
                 problem,
                 optimizer,
                 scheduler,
                 extra_features=None,
                 use_lt=True):
        """
        :param model: A torch nn.Module instances.
        :type model: torch.nn.Module
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param Optimizer optimizers: A neural network optimizers to use.
        :param Scheduler optimizers: A neural network scheduler to use.
        :param extra_features: The additional input features to use as
            augmented input.
        :type extra_features: list[torch.nn.Module] | tuple[torch.nn.Module]
        :param bool use_lt: Using LabelTensors as input during training.
        """
        if optimizer is None:
            optimizer = self.default_torch_optimizer()

        if scheduler is None:
            scheduler = self.default_torch_scheduler()

        super().__init__(models=model,
                         problem=problem,
                         optimizers=optimizer,
                         schedulers=scheduler,
                         extra_features=extra_features,
                         use_lt=use_lt)

        # initialize model (needed for Lightining to go to different devices)
        self._pina_model = model
        self._pina_optimizer = optimizer
        self._pina_scheduler = scheduler

    def forward(self, x):
        """Forward pass implementation for the solver.

        :param torch.Tensor x: Input tensor.
        :return: Solver solution.
        :rtype: torch.Tensor
        """
        return self.model(x)

    def configure_optimizers(self):
        """Optimizer configuration for the solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        self.optimizer.hook(self.model.parameters())
        self.scheduler.hook(self.optimizer)
        return ([self.optimizer.optimizer_instance],
                [self.scheduler.scheduler_instance])

    @staticmethod
    def default_torch_optimizer():
        return TorchOptimizer(torch.optim.Adam, lr=0.001)

    @staticmethod
    def default_torch_scheduler():
        return TorchScheduler(torch.optim.lr_scheduler.ConstantLR)

    @property
    def model(self):
        """
        Model for training.
        """
        return self._pina_model

    @property
    def scheduler(self):
        """
        Scheduler for training.
        """
        return self._pina_scheduler

    @property
    def optimizer(self):
        """
        Optimizer for training.
        """
        return self._pina_optimizer

    def on_train_start(self):
        super().on_train_start()
        if self.trainer.compile:
            model_device = next(self._pina_model.parameters()).device
            try:
                if model_device == torch.device("mps:0"):
                    self._pina_model = torch.compile(self._pina_model,
                                                     backend="eager")
                else:
                    self._pina_model = torch.compile(self._pina_model,
                                                     backend="inductor")
            except Exception as e:
                print("Compilation failed, running in normal mode.:\n", e)
                sys.stdout.flush()

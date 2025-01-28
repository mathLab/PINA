""" Solver module. """

from abc import ABCMeta, abstractmethod
import lightning
from ..utils import check_consistency, labelize_forward
from ..problem import AbstractProblem
from ..optim import Optimizer, Scheduler, TorchOptimizer, TorchScheduler
import torch
import sys


class MultipleSolversInterface(lightning.pytorch.LightningModule, metaclass=ABCMeta):
    """
    Solver base class. This class inherits is a wrapper of
    LightningModule class, inheriting all the
    LightningModule methods.
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
        super().__init__()

        # check consistency of the inputs
        check_consistency(problem, AbstractProblem)
        self._check_solver_consistency(problem)

        # Check consistency of models argument and encapsulate in list
        check_consistency(models, torch.nn.Module)
        len_model = len(models)

        # Check consistency extra_features
        if extra_features is None:
            extra_features = []
        else:
            check_consistency(extra_features, torch.nn.Module)

        # If use_lt is true add extract operation in input
        check_consistency(use_lt, bool)
        self._use_lt = use_lt
        if use_lt is True:
            self.forward = labelize_forward(
                forward=self.forward,
                input_variables=problem.input_variables,
                output_variables=problem.output_variables,
                extra_features=extra_features
                )

        # Check scheduler consistency + encapsulation
        check_consistency(schedulers, Scheduler)

        # Check optimizer consistency + encapsulation
        check_consistency(optimizers, Optimizer)
        len_optimizer = len(optimizers)

        # check length consistency optimizers
        if len_model != len_optimizer:
            raise ValueError("You must define one optimizer for each model."
                             f"Got {len_model} models, and {len_optimizer}"
                             " optimizers.")

        # extra features handling
        self._pina_models = models
        self._pina_optimizers = optimizers
        self._pina_schedulers = schedulers
        self._pina_problem = problem

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

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError

    def on_train_start(self):
        """
        On training epoch start this function is call to do global checks for
        the different solvers.
        """

        # 1. Check the verison for dataloader
        dataloader = self.trainer.train_dataloader
        if sys.version_info < (3, 8):
            dataloader = dataloader.loaders
        self._dataloader = dataloader

        return super().on_train_start()

    @staticmethod
    def get_batch_size(batch):
        # Assuming batch is your custom Batch object
        batch_size = 0
        for data in batch:
            batch_size += len(data[1]['input_points'])
        return batch_size

    def _check_solver_consistency(self, problem):
        for condition in problem.conditions.values():
            check_consistency(condition, self.accepted_conditions_types)

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


class SolverInterface(MultipleSolversInterface):
    def __init__(self, model, problem, optimizer, scheduler, extra_features=None, use_lt=True):
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
            optimizer = SolverInterface.default_torch_optimizer()

        if scheduler is None:
            scheduler = SolverInterface.default_torch_scheduler()

        super().__init__(models = [model],
                         problem = problem,
                         optimizers = [optimizer],
                         schedulers = [scheduler],
                         extra_features = extra_features,
                         use_lt = use_lt)
        # initialize model (needed for Lightining to go to different devices)
        self._pina_model = self.models[0]
    
    def forward(self, x):
        """Forward pass implementation for the solver.

        :param torch.Tensor x: Input tensor.
        :return: Solver solution.
        :rtype: torch.Tensor
        """
        return self.model(x)

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
        return self.schedulers[0]

    @property
    def optimizer(self):
        """
        Optimizer for training.
        """
        return self.optimizers[0]
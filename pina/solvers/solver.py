""" Solver module. """

import lightning
import torch
import sys

from abc import ABCMeta, abstractmethod
from ..problem import AbstractProblem
from ..optim import Optimizer, Scheduler, TorchOptimizer, TorchScheduler
from ..loss import WeightingInterface
from ..loss.scalar_weighting import _NoWeighting
from ..utils import check_consistency, labelize_forward


class SolverInterface(lightning.pytorch.LightningModule, metaclass=ABCMeta):
    """
    SolverInterface base class. This class is a wrapper of the LightningModule
    """

    def __init__(self,
                 problem,
                 weighting,
                 use_lt):
        """
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param use_lt: Using LabelTensors as input during training.
        :type use_lt: bool
        """
        super().__init__()
    
        # check consistency of the problem
        check_consistency(problem, AbstractProblem)
        self._check_solver_consistency(problem)
        self._pina_problem = problem

        # check consistency of the weighting + hook the condition names
        if weighting is None:
            weighting = _NoWeighting()
        check_consistency(weighting, WeightingInterface)
        self._pina_weighting = weighting
        weighting.condition_names = list(self._pina_problem.conditions.keys())

        # Check consistency use_lt
        check_consistency(use_lt, bool)
        self._use_lt = use_lt

        # If use_lt is true add extract operation in input
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
        losses =  self.optimization_cycle(batch)
        loss = self.weighting.aggregate(losses).as_subclass(torch.Tensor)
        return loss

    def training_step(self, batch):
        """Solver training step.

        :param batch: The batch element in the dataloader.
        :type batch: tuple
        :param batch_idx: The batch index.
        :type batch_idx: int
        :return: The sum of the loss functions.
        :rtype: LabelTensor
        """
        loss = self._optimization_cycle(batch=batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True,
                 batch_size=self.get_batch_size(batch), sync_dist=True)
        return loss

    def validation_step(self, batch):
        """
        Solver validation step.
        """
        loss = self._optimization_cycle(batch=batch)
        self.log('val_loss', loss, prog_bar=True, logger=True,
                 batch_size=self.get_batch_size(batch), sync_dist=True)
        
    def test_step(self, batch):
        """
        Solver validation step.
        """
        loss = self._optimization_cycle(batch=batch)
        self.log('test_loss', loss, prog_bar=True, logger=True,
                 batch_size=self.get_batch_size(batch), sync_dist=True)

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

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

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
    
    @property
    def weighting(self):
        """
        The weighting mechanism."""
        return self._pina_weighting

    @staticmethod
    def get_batch_size(batch):
        # Assuming batch is your custom Batch object
        batch_size = 0
        for data in batch:
            batch_size += len(data[1]['input_points'])
        return batch_size

    @staticmethod
    def default_torch_optimizer():
        return TorchOptimizer(torch.optim.Adam, lr=0.001)

    @staticmethod
    def default_torch_scheduler():
        return TorchScheduler(torch.optim.lr_scheduler.ConstantLR)


class SingleSolverInterface(SolverInterface):
    def __init__(self,
                 model,
                 problem,
                 optimizer=None,
                 scheduler=None,
                 weighting=None,
                 use_lt=True):
        """
        :param model: A torch nn.Module instances.
        :type model: torch.nn.Module
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param Optimizer optimizers: A neural network optimizers to use.
        :param Scheduler optimizers: A neural network scheduler to use.
        :param WeightingInterface weighting: The loss weighting to use.
        :param bool use_lt: Using LabelTensors as input during training.
        """
        if optimizer is None:
            optimizer = self.default_torch_optimizer()

        if scheduler is None:
            scheduler = self.default_torch_scheduler()

        super().__init__(problem=problem,
                         use_lt=use_lt,
                         weighting=weighting)

        # Check consistency of models argument and encapsulate in list
        check_consistency(model, torch.nn.Module)
        # Check scheduler consistency + encapsulation
        check_consistency(scheduler, Scheduler)
        # Check optimizer consistency + encapsulation
        check_consistency(optimizer, Optimizer)

        # initialize model (needed for Lightining to go to different devices)
        self._pina_models = [model]
        self._pina_optimizers = [optimizer]
        self._pina_schedulers = [scheduler]

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


class MultiSolverInterface(SolverInterface):
    """
    Multiple Solver base class. This class inherits is a wrapper of
    SolverInterface class
    """

    def __init__(self,
                 models,
                 problem,
                 optimizers=None,
                 schedulers=None,
                 weighting=None,
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
        :param WeightingInterface weighting: The loss weighting to use.
        :param bool use_lt: Using LabelTensors as input during training.
        """
        if not isinstance(models, (list, tuple)) or len(models) < 2:
            raise ValueError(
                'models should be list[torch.nn.Module] or '
                'tuple[torch.nn.Module] with len greater than '
                'one.'
            )

        if optimizers is None:
            optimizers = [self.default_torch_optimizer()] * len(models)

        super().__init__(problem=problem,
                         use_lt=use_lt,
                         weighting=weighting)

        # Check consistency of models argument and encapsulate in list
        check_consistency(models, torch.nn.Module)

        # Check scheduler consistency + encapsulation
        check_consistency(schedulers, Scheduler)

        # Check optimizer consistency + encapsulation
        check_consistency(optimizers, Optimizer)

        # check length consistency optimizers
        if len(models) != len(optimizers):
            raise ValueError(
                "You must define one optimizer for each model."
                f"Got {len(models)} models, and {len(optimizers)}"
                " optimizers."
            )

        # initialize model
        self._pina_models = models
        self._pina_optimizers = optimizers
        self._pina_schedulers = schedulers

    def configure_optimizers(self):
        """Optimizer configuration for the solver.

        :return: The optimizers and the schedulers
        :rtype: tuple(list, list)
        """
        for optimizer, scheduler, model in zip(self.optimizers,
                                               self.schedulers,
                                               self.models):
            optimizer.hook(model.parameters())
            scheduler.hook(optimizer)

        return (
            [optimizer.optimizer_instance for optimizer in self.optimizers],
            [scheduler.scheduler_instance for scheduler in self.schedulers]
        )

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

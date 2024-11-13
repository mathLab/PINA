""" Solver module. """

from abc import ABCMeta, abstractmethod
from ..model.network import Network
import lightning
from ..utils import check_consistency
from ..problem import AbstractProblem
from ..optim import Optimizer, Scheduler
import torch
import sys



class SolverInterface(lightning.pytorch.LightningModule, metaclass=ABCMeta):
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
                 extra_features,
                 use_lt=True):
        """
        :param model: A torch neural network model instance.
        :type model: torch.nn.Module
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param list(torch.optim.Optimizer) optimizer: A list of neural network
           optimizers to use.
        """
        super().__init__()

        # check consistency of the inputs
        check_consistency(problem, AbstractProblem)
        self._check_solver_consistency(problem)

        # Check consistency of models argument and encapsulate in list
        if not isinstance(models, list):
            check_consistency(models, torch.nn.Module)
            # put everything in a list if only one input
            models = [models]
        else:
            for idx in range(len(models)):
                # Check consistency
                check_consistency(models[idx], torch.nn.Module)
        len_model = len(models)

        # If use_lt is true add extract operation in input
        if use_lt is True:
            for idx, model in enumerate(models):
                models[idx] = Network(
                    model=model,
                    input_variables=problem.input_variables,
                    output_variables=problem.output_variables,
                    extra_features=extra_features,
                )

        # Check scheduler consistency + encapsulation
        if not isinstance(schedulers, list):
            check_consistency(schedulers, Scheduler)
            schedulers = [schedulers]
        else:
            for scheduler in schedulers:
                check_consistency(scheduler, Scheduler)

        # Check optimizer consistency + encapsulation
        if not isinstance(optimizers, list):
            check_consistency(optimizers, Optimizer)
            optimizers = [optimizers]
        else:
            for optimizer in optimizers:
                check_consistency(optimizer, Optimizer)
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

        self.validation_condition_losses = {
            k: {'loss': [],
                'count': []} for k in self.problem.conditions.keys()}
        self.train_condition_losses = {
            k: {'loss': [],
                'count': []} for k in self.problem.conditions.keys()}


    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

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
    def problem(self):
        """
        The problem formulation."""
        return self._pina_problem

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

    def _check_solver_consistency(self, problem):
        """
        TODO
        """
        for _, condition in problem.conditions.items():
            if not set(condition.condition_type).issubset(
                    set(self.accepted_condition_types)):
                raise ValueError(
                    f'{self.__name__} dose not support condition '
                    f'{condition.condition_type}')

    def epoch_logger(self, name):
        if name == 'train':
            losses_dict = self.train_condition_losses
        elif name == 'val':
            losses_dict = self.validation_condition_losses
        total_loss = []
        total_count = []
        for k, v in losses_dict.items():
            local_counter = torch.tensor(v['count']).to(self.device)
            n_elements = torch.sum(local_counter)
            loss = torch.sum(
                torch.stack(v['loss']) * local_counter) / n_elements
            loss = loss.as_subclass(torch.Tensor)
            total_loss.append(loss)
            total_count.append(n_elements)
            self.log(
                k + f"_{name}_loss",
                loss,
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=False,
                batch_size=self.trainer.data_module.batch_size,
                )
        total_count = (torch.tensor(total_count, dtype=torch.float32).
                       to(self.device))
        mean_loss = (torch.sum(torch.stack(total_loss) * total_count) /
                     total_count)
        self.log(
            f"_{name}_loss",
            mean_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
            batch_size=self.trainer.data_module.batch_size,
        )
        if name == 'val':
            self.validation_condition_losses = {
                k: {'loss': [],
                    'count': []} for k in self.problem.conditions.keys()}
        elif name == 'train':
            self.train_condition_losses = {
                k: {'loss': [],
                    'count': []} for k in self.problem.conditions.keys()}
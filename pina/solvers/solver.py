""" Solver module. """

from abc import ABCMeta, abstractmethod
from ..model.network import Network
import pytorch_lightning
from ..utils import check_consistency
from ..problem import AbstractProblem
import torch
import sys


class SolverInterface(pytorch_lightning.LightningModule, metaclass=ABCMeta):
    """
    Solver base class. This class inherits is a wrapper of
    LightningModule class, inheriting all the
    LightningModule methods.
    """

    def __init__(
        self,
        models,
        problem,
        optimizers,
        optimizers_kwargs,
        extra_features=None,
    ):
        """
        :param models: A torch neural network model instance.
        :type models: torch.nn.Module
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param list(torch.optim.Optimizer) optimizer: A list of neural network optimizers to
            use.
        :param list(dict) optimizer_kwargs: A list of optimizer constructor keyword args.
        :param list(torch.nn.Module) extra_features: The additional input
            features to use as augmented input. If ``None`` no extra features
            are passed. If it is a list of :class:`torch.nn.Module`, the extra feature
            list is passed to all models. If it is a list of extra features' lists,
            each single list of extra feature is passed to a model.
        """
        super().__init__()

        # check consistency of the inputs
        check_consistency(models, torch.nn.Module)
        check_consistency(problem, AbstractProblem)
        check_consistency(optimizers, torch.optim.Optimizer, subclass=True)
        check_consistency(optimizers_kwargs, dict)

        # put everything in a list if only one input
        if not isinstance(models, list):
            models = [models]
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
            optimizers_kwargs = [optimizers_kwargs]

        # number of models and optimizers
        len_model = len(models)
        len_optimizer = len(optimizers)
        len_optimizer_kwargs = len(optimizers_kwargs)

        # check length consistency optimizers
        if len_model != len_optimizer:
            raise ValueError(
                "You must define one optimizer for each model."
                f"Got {len_model} models, and {len_optimizer}"
                " optimizers."
            )

        # check length consistency optimizers kwargs
        if len_optimizer_kwargs != len_optimizer:
            raise ValueError(
                "You must define one dictionary of keyword"
                " arguments for each optimizers."
                f"Got {len_optimizer} optimizers, and"
                f" {len_optimizer_kwargs} dicitionaries"
            )

        # extra features handling
        if (extra_features is None) or (len(extra_features) == 0):
            extra_features = [None] * len_model
        else:
            # if we only have a list of extra features
            if not isinstance(extra_features[0], (tuple, list)):
                extra_features = [extra_features] * len_model
            else:  # if we have a list of list extra features
                if len(extra_features) != len_model:
                    raise ValueError(
                        "You passed a list of extrafeatures list with len"
                        f"different of models len. Expected {len_model} "
                        f"got {len(extra_features)}. If you want to use "
                        "the same list of extra features for all models, "
                        "just pass a list of extrafeatures and not a list "
                        "of list of extra features."
                    )

        # assigning model and optimizers
        self._pina_models = []
        self._pina_optimizers = []

        for idx in range(len_model):
            model_ = Network(
                model=models[idx],
                input_variables=problem.input_variables,
                output_variables=problem.output_variables,
                extra_features=extra_features[idx],
            )
            optim_ = optimizers[idx](
                model_.parameters(), **optimizers_kwargs[idx]
            )
            self._pina_models.append(model_)
            self._pina_optimizers.append(optim_)

        # assigning problem
        self._pina_problem = problem

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def training_step(self):
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

    # @model.setter
    # def model(self, new_model):
    #     """
    #     Set the torch."""
    #     check_consistency(new_model, nn.Module, 'torch model')
    #     self._model= new_model

    # @problem.setter
    # def problem(self, problem):
    #     """
    #     Set the problem formulation."""
    #     check_consistency(problem, AbstractProblem, 'pina problem')
    #     self._problem = problem

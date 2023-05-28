""" Solver module. """

from abc import ABCMeta, abstractmethod
from typing import Any
import torch.nn as nn
import lightning.pytorch as pl
from .utils import check_consistency
from .problem import AbstractProblem

class SolverInterface(pl.LightningModule, metaclass=ABCMeta):
    """ Solver base class. """
    def __init__(self, model, problem, extra_features=None):
        """
        :param model: A torch neural network model instance.
        :type model: torch.nn.Module
        :param problem: A problem definition instance.
        :type problem: AbstractProblem
        :param list(torch.nn.Module) extra_features: the additional input
        features to use as augmented input.
        """
        super().__init__()
        
        # check inheritance consistency for model and pina problem
        check_consistency(model, nn.Module, 'torch model')
        check_consistency(problem, AbstractProblem, 'pina problem')

        # assigning class variables
        self._model = model
        self._problem = problem

        # check consistency and assign extra fatures 
        if extra_features is None:
            self._extra_features = []
        else:
            for feat in extra_features:
                check_consistency(feat, nn.Module, 'extra features')
            self._extra_features = nn.Sequential(*extra_features)

        # check model works with inputs TODO

    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def training_step(self):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @property
    def model(self):
        """
        The torch model."""
        return self._model

    @property
    def problem(self):
        """
        The problem formulation."""
        return self._problem

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
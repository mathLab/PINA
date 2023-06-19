""" Solver module. """

from abc import ABCMeta, abstractmethod
from .model.network import Network
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
        
        # check inheritance for pina problem
        check_consistency(problem, AbstractProblem)

        # assigning class variables (check consistency inside Network class)
        self._pina_model = Network(model=model, extra_features=extra_features)
        self._problem = problem

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
        return self._pina_model

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
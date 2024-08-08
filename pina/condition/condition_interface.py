
from abc import ABCMeta, abstractmethod


class ConditionInterface(metaclass=ABCMeta):

    def __init__(self) -> None:
        self._problem = None

    @abstractmethod
    def residual(self, model):
        """
        Compute the residual of the condition.

        :param model: The model to evaluate the condition.
        :return: The residual of the condition.
        """
        pass
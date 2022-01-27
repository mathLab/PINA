from abc import ABCMeta, abstractmethod


class AbstractProblem(metaclass=ABCMeta):

    @property
    @abstractmethod
    def input_variables(self):
        pass

    @property
    @abstractmethod
    def output_variables(self):
        pass

    @property
    @abstractmethod
    def conditions(self):
        pass

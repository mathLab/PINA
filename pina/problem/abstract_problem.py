from abc import ABCMeta, abstractmethod


class AbstractProblem(metaclass=ABCMeta):

    @property
    def input_variables(self):
        variables = []

        if hasattr(self, 'spatial_variables'):
            variables += self.spatial_variables
        if hasattr(self, 'temporal_variable'):
            variables += self.temporal_variable
        if hasattr(self, 'parameters'):
            variables += self.parameters
        if hasattr(self, 'custom_variables'):
            variables += self.custom_variables

        return variables

    @input_variables.setter
    def input_variables(self, variables):
        raise NotImplementedError

    @property
    @abstractmethod
    def output_variables(self):
        pass

    @property
    @abstractmethod
    def conditions(self):
        pass

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

    @property
    def domain(self):

        domains = [
            getattr(self, f'{t}_domain')
            for t in ['spatial', 'temporal', 'parameter']
            if hasattr(self, f'{t}_domain')
        ]

        if len(domains) == 1:
            return domains[0]
        elif len(domains) == 0:
            raise RuntimeError

        if len(set(map(type, domains))) == 1:
            domain = domains[0].__class__({})
            [domain.update(d) for d in domains]
            return domain
        else:
            raise RuntimeError('different domains')

    @input_variables.setter
    def input_variables(self, variables):
        raise RuntimeError

    @property
    @abstractmethod
    def output_variables(self):
        pass

    @property
    @abstractmethod
    def conditions(self):
        pass

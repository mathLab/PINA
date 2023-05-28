""" Module for AbstractProblem class """
from abc import ABCMeta, abstractmethod
from ..utils import merge_tensors


class AbstractProblem(metaclass=ABCMeta):
    """
    The abstract `AbstractProblem` class. All the class defining a PINA Problem
    should be inheritied from this class.

    In the definition of a PINA problem, the fundamental elements are:
    the output variables, the condition(s), and the domain(s) where the
    conditions are applied.
    """

    def __init__(self):
        self.have_sampled_points = False
        self.input_pts = {}
        
    @property
    def input_variables(self):
        """
        The input variables of the AbstractProblem, whose type depends on the
        type of domain (spatial, temporal, and parameter).

        :return: the input variables of self
        :rtype: list

        """
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
        """
        The domain(s) where the conditions of the AbstractProblem are valid.

        :return: the domain(s) of self
        :rtype: list (if more than one domain are defined),
            `Span` domain (of only one domain is defined)
        """
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
        """
        The output variables of the problem.
        """
        pass

    @property
    @abstractmethod
    def conditions(self):
        """
        The conditions of the problem.
        """
        pass

    def discretise_domain(self, *args, **kwargs):
        """
        Generate a set of points to span the `Location` of all the conditions of
        the problem.

        >>> pinn.span_pts(n=10, mode='grid')
        >>> pinn.span_pts(n=10, mode='grid', location=['bound1'])
        >>> pinn.span_pts(n=10, mode='grid', variables=['x'])
        """

        if all(key in kwargs for key in ['n', 'mode']):
            argument = {}
            argument['n'] = kwargs['n']
            argument['mode'] = kwargs['mode']
            argument['variables'] = self.input_variables
            arguments = [argument]
        elif any(key in kwargs for key in ['n', 'mode']) and args:
            raise ValueError("Don't mix args and kwargs")
        elif isinstance(args[0], int) and isinstance(args[1], str):
            argument = {}
            argument['n'] = int(args[0])
            argument['mode'] = args[1]
            argument['variables'] = self.input_variables
            arguments = [argument]
        elif all(isinstance(arg, dict) for arg in args):
            arguments = args
        else:
            raise RuntimeError

        locations = kwargs.get('locations', 'all')

        if locations == 'all':
            locations = [condition for condition in self.conditions]
        for location in locations:
            condition = self.conditions[location]

            samples = tuple(condition.location.sample(
                            argument['n'],
                            argument['mode'],
                            variables=argument['variables'])
                            for argument in arguments)
            pts = merge_tensors(samples)
            self.input_pts[location] = pts

            # setting the grad
            self.input_pts[location].requires_grad_(True)
            self.input_pts[location].retain_grad()
        
        self.have_sampled_points = True

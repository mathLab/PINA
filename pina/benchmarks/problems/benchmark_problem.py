from abc import ABCMeta, abstractmethod
from pina.problem import AbstractProblem

class BenchmarkProblemInterface(AbstractProblem, metaclass=ABCMeta):
    """
    This class is used internally in PINA con create benchmark problems. It is
    an extension of :obj:`~pina.problem.AbstractProblem` with additional
    requirements used to create a enriched problem for the user.
    """

    @property
    @abstractmethod
    def description(self):
        """
        A brief description of the benchmark.
        """
        pass
    
    @description.setter
    def description(self, string):
        if not isinstance(string, str):
            raise TypeError('Expected str for description.')

    @property
    @abstractmethod
    def problem_type(self):
        """
        Return a Python str with the problem type. Each problem could be either
        `physics-informed` or `data-driven`.
        """
        pass

    @problem_type.setter
    def problem_type(self, value):
        if value not in ['physics-informed', 'data-driven', 'mixed']:
            raise ValueError('Invalud problem type. Problem type should be of '
                             'in  `physics-informed` or `data-driven`.')

    # @property
    # @abstractmethod
    # def data_directory(self):
    #     """
    #     Returns the data path from where to retrieve the high fiedelity solution
    #     data. Regardless the type of problem, all benchmarks have high fidelity
    #     solution data to compute statistics and perform visualization.
    #     """
    #     pass
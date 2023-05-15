""" Module for EquationInterface class """
from abc import ABCMeta, abstractmethod


class EquationInterface(metaclass=ABCMeta):
    """
    The abstract `AbstractProblem` class. All the class defining a PINA Problem
    should be inheritied from this class.

    In the definition of a PINA problem, the fundamental elements are:
    the output variables, the condition(s), and the domain(s) where the
    conditions are applied.
    """

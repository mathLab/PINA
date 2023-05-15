""" Module """
from .equation_interface import EquationInterface

class Equation(EquationInterface):

    def __init__(self, equation):
        self.__equation = equation

    def residual(self, input_, output_):
        return self.__equation(input_, output_)
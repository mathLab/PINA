""" Module """
import torch
from .equation import Equation

class SystemEquation(Equation):

    def __init__(self, list_equation):
        if not isinstance(list_equation, list):
            raise TypeError('list_equation must be a list of functions')

        self.equations = []
        for i, equation in enumerate(list_equation):
            if not callable(equation):
                raise TypeError('list_equation must be a list of functions')
            
            self.equations.append(Equation(equation))

    def residual(self, input_, output_):
        return torch.mean(
            torch.stack([
                equation.residual(input_, output_)
                for equation in self.equations
            ]),
            dim=0)
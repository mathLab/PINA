""" Module """
from .equation import Equation
from ..operators import grad, div, nabla


class FixedValue(Equation):
    
    def __init__(self, value, components=None):
        def equation(input_, output_):
            if components is None:
                return output_ - value
            return output_.extract(components) - value 
        super().__init__(equation)


class FixedGradient(Equation):

    def __init__(self, value, components=None, d=None):
        def equation(input_, output_):
            return grad(output_, input_, components=components, d=d) - value
        super().__init__(equation)


class FixedFlux(Equation):

    def __init__(self, value, components=None, d=None):
        def equation(input_, output_):
            return div(output_, input_, components=components, d=d) - value
        super().__init__(equation)


class Laplace(Equation):

    def __init__(self, components=None, d=None):
        def equation(input_, output_):
            return nabla(output_, input_, components=components, d=d)
        super().__init__(equation)
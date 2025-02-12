from pina.problem import AbstractProblem
from pina import Condition
from pina import Graph

class SupervisedProblem(AbstractProblem):
    conditions = {}
    output_variables = None

    def __init__(self, input_, output_):
        if isinstance(input_, Graph):
            input_ = input_.data
        self.conditions['data'] = Condition(
            input_points=input_,
            output_points = output_
        )

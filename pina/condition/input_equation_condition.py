
from . import ConditionInterface

class InputOutputCondition(ConditionInterface):
    """
    Condition for input/output data.
    """

    __slots__ = ["input_points", "output_points"]

    def __init__(self, input_points, output_points):
        """
        Constructor for the `InputOutputCondition` class.
        """
        super().__init__()
        self.input_points = input_points
        self.output_points = output_points

    def residual(self, model):
        """
        Compute the residual of the condition.
        """
        return self.output_points - model(self.input_points)
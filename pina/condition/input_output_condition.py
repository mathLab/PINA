
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
        return self.batch_residual(model, self.input_points, self.output_points)

    @staticmethod
    def batch_residual(model, input_points, output_points):
        """
        Compute the residual of the condition for a single batch. Input and
        output points are provided as arguments.

        :param torch.nn.Module model: The model to evaluate the condition.
        :param torch.Tensor input_points: The input points.
        :param torch.Tensor output_points: The output points.
        """
        return output_points - model(input_points)
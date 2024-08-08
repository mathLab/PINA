
from . import ConditionInterface

class DomainOutputCondition(ConditionInterface):
    """
    Condition for input/output data.
    """

    __slots__ = ["domain", "output_points"]

    def __init__(self, domain, output_points):
        """
        Constructor for the `InputOutputCondition` class.
        """
        super().__init__()
        print(self)
        self.domain = domain
        self.output_points = output_points

    @property
    def input_points(self):
        """
        Get the input points of the condition.
        """
        return self._problem.domains[self.domain]

    def residual(self, model):
        """
        Compute the residual of the condition.
        """
        return self.batch_residual(model, self.domain, self.output_points)

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
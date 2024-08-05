from .condition_interface import ConditionInterface

class DomainEquationCondition(ConditionInterface):
    """
    Condition for input/output data.
    """

    __slots__ = ["domain", "equation"]

    def __init__(self, domain, equation):
        """
        Constructor for the `InputOutputCondition` class.
        """
        super().__init__()
        self.domain = domain
        self.equation = equation

    @staticmethod
    def batch_residual(model, input_pts, equation):
        """
        Compute the residual of the condition for a single batch. Input and
        output points are provided as arguments.

        :param torch.nn.Module model: The model to evaluate the condition.
        :param torch.Tensor input_points: The input points.
        :param torch.Tensor output_points: The output points.
        """
        return equation.residual(model(input_pts))
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

    def residual(self, model):
        """
        Compute the residual of the condition.
        """
        self.batch_residual(model, self.domain, self.equation)

    @staticmethod
    def batch_residual(model, input_pts, equation):
        """
        Compute the residual of the condition for a single batch. Input and
        output points are provided as arguments.

        :param torch.nn.Module model: The model to evaluate the condition.
        :param torch.Tensor input_pts: The input points.
        :param torch.Tensor equation: The output points.
        """
        return equation.residual(input_pts, model(input_pts))
import torch
from pina.problem import AbstractProblem
from pina import Condition
from pina import Graph
from pina import LabelTensor


class SupervisedProblem:

    def __new__(cls, *args, **kwargs):

        if sorted(list(kwargs.keys())) == sorted(["input_", "output_"]):
            return SupervisedTensorProblem(**kwargs)
        elif sorted(list(kwargs.keys())) == sorted(["graph_"]):
            return SupervisedGraphProblem(**kwargs)
        raise RuntimeError("Invalid arguments for SupervisedProblem")


class SupervisedTensorProblem(AbstractProblem):
    """
    A problem definition for supervised learning in PINA.

    This class allows an easy and straightforward definition of a Supervised
    problem, based on a single condition of type `InputOutputPointsCondition`

    :Example:
        >>> import torch
        >>> input_data = torch.rand((100, 10))
        >>> output_data = torch.rand((100, 10))
        >>> problem = SupervisedProblem(input_data, output_data)
    """

    conditions = dict()
    output_variables = None

    def __init__(self, input_, output_):
        """
        Initialize the SupervisedProblem class

        :param input_: Input data of the problem
        :type input_: torch.Tensor | Graph
        :param output_: Output data of the problem
        :type output_: torch.Tensor
        """
        if not isinstance(input_, (torch.Tensor, LabelTensor)):
            raise ValueError(
                "The input data must be a torch.Tensor or a LabelTensor"
            )
        if not isinstance(output_, (torch.Tensor, LabelTensor)):
            raise ValueError(
                "The output data must be a torch.Tensor or a LabelTensor"
            )
        if isinstance(output_, LabelTensor):
            self.output_variables = output_.labels

        self.conditions["data"] = Condition(
            input_points=input_, output_points=output_
        )
        super().__init__()


class SupervisedGraphProblem(AbstractProblem):
    """
    A problem definition for supervised learning in PINA.

    This class allows an easy and straightforward definition of a Supervised problem,
    based on a single condition of type `InputOutputPointsCondition`

    :Example:
        >>> import torch
        >>> from pina.graph import RadiusGraph
        >>> x = torch.rand((10, 100, 10))
        >>> pos = torch.rand((10, 100, 2))
        >>> y = torch.rand((10, 100, 2))
        >>> input_data = RadiusGraph(x=x, pos=pos, r=.2, y=y)
        >>> problem = SupervisedProblem(graph_=input_data)
    """

    conditions = dict()
    output_variables = None

    def __init__(self, graph_):
        """
        Initialize the SupervisedProblem class

        :param graph_: Input data of the problem
        :type graph_: Graph
        """
        if not isinstance(graph_, list) or not all(
            isinstance(g, Graph) for g in graph_
        ):
            raise ValueError("The input data must be a Graph")

        self.conditions["data"] = Condition(graph=graph_)
        super().__init__()

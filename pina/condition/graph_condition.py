from .condition_interface import ConditionInterface
from ..graph import Graph
from ..utils import check_consistency
from torch_geometric.data import Data
from ..equation.equation_interface import EquationInterface


class GraphCondition(ConditionInterface):
    """
    TODO
    """

    __slots__ = ["graph"]

    def __new__(cls, graph):
        """
        TODO : add docstring
        """
        check_consistency(graph, (Graph, Data))
        graph = [graph] if isinstance(graph, Data) else graph

        if all(g.y is not None for g in graph):
            return super().__new__(GraphInputOutputCondition)
        else:
            return super().__new__(GraphDataCondition)

    def __init__(self, graph):

        super().__init__()
        self.graph = graph

    def __setattr__(self, key, value):
        if key == "graph":
            check_consistency(value, (Graph, Data))
            GraphCondition.__dict__[key].__set__(self, value)
        elif key in ("_problem", "_condition_type"):
            super().__setattr__(key, value)


class GraphInputEquationCondition(ConditionInterface):

    __slots__ = ["graph", "equation"]

    def __init__(self, graph, equation):
        super().__init__()
        self.graph = graph
        self.equation = equation

    def __setattr__(self, key, value):
        if key == "graph":
            check_consistency(value, (Graph, Data))
            GraphInputEquationCondition.__dict__[key].__set__(self, value)
        elif key == "equation":
            check_consistency(value, (EquationInterface))
            GraphInputEquationCondition.__dict__[key].__set__(self, value)
        elif key in ("_problem", "_condition_type"):
            super().__setattr__(key, value)


# The split between GraphInputOutputCondition and GraphDataCondition
# distinguishes different types of graph conditions passed to problems.
# This separation simplifies consistency checks during problem creation.
class GraphDataCondition(GraphCondition):
    pass


class GraphInputOutputCondition(GraphCondition):
    pass

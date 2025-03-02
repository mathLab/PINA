import torch
from pina.problem import AbstractProblem
from pina.condition import InputOutputPointsCondition
from pina.problem.zoo.supervised_problem import SupervisedProblem
from pina.graph import RadiusGraph


def test_constructor():
    input_ = torch.rand((100, 10))
    output_ = torch.rand((100, 10))
    problem = SupervisedProblem(input_=input_, output_=output_)
    assert isinstance(problem, AbstractProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
    assert list(problem.conditions.keys()) == ["data"]
    assert isinstance(problem.conditions["data"], InputOutputPointsCondition)


def test_constructor_graph():
    x = torch.rand((20, 100, 10))
    pos = torch.rand((20, 100, 2))
    input_ = [
        RadiusGraph(x=x_, pos=pos_, radius=0.2, edge_attr=True)
        for x_, pos_ in zip(x, pos)
    ]
    output_ = torch.rand((100, 10))
    problem = SupervisedProblem(input_=input_, output_=output_)
    assert isinstance(problem, AbstractProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
    assert list(problem.conditions.keys()) == ["data"]
    assert isinstance(problem.conditions["data"], InputOutputPointsCondition)
    assert isinstance(problem.conditions["data"].input_points, list)
    assert isinstance(problem.conditions["data"].output_points, torch.Tensor)

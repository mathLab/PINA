import torch
from pina.problem import AbstractProblem
from pina.condition import InputTargetCondition
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
    assert isinstance(problem.conditions["data"], InputTargetCondition)


def test_constructor_graph():
    x = torch.rand((20, 100, 10))
    pos = torch.rand((20, 100, 2))
    input_ = [
        RadiusGraph(x=x_, pos=pos_, radius=0.2, edge_attr=True)
        for x_, pos_ in zip(x, pos)
    ]
    output_ = torch.rand((20, 100, 10))
    problem = SupervisedProblem(input_=input_, output_=output_)
    assert isinstance(problem, AbstractProblem)
    assert hasattr(problem, "conditions")
    assert isinstance(problem.conditions, dict)
    assert list(problem.conditions.keys()) == ["data"]
    assert isinstance(problem.conditions["data"], InputTargetCondition)
    assert isinstance(problem.conditions["data"].input, list)
    assert isinstance(problem.conditions["data"].target, torch.Tensor)

import torch
import pytest

from pina import LabelTensor, Condition
from pina.condition import (
    TensorInputGraphTargetCondition,
    TensorInputTensorTargetCondition,
    GraphInputGraphTargetCondition,
    GraphInputTensorTargetCondition,
)
from pina.condition import (
    InputTensorEquationCondition,
    InputGraphEquationCondition,
    DomainEquationCondition,
)
from pina.condition import (
    DataCondition,
    TensorDataCondition,
    GraphDataCondition,
)
from pina.domain import CartesianDomain
from pina.equation.equation_factory import FixedValue
from pina.graph import RadiusGraph

example_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

input_tensor = torch.tensor([[0, 0, 0]])
target_tensor = torch.tensor([[1, 2]])
input_lt = LabelTensor(torch.tensor([[0, 0, 0]]), ["x", "y", "z"])
target_lt = LabelTensor(torch.tensor([[1, 2]]), ["a", "b"])

x = torch.rand(10, 20, 2)
pos = torch.rand(10, 20, 2)
radius = 0.1
input_graph = [
    RadiusGraph(
        x=x_,
        pos=pos_,
        radius=radius,
    )
    for x_, pos_ in zip(x, pos)
]
target_graph = [
    RadiusGraph(
        x=x_,
        pos=pos_,
        radius=radius,
    )
    for x_, pos_ in zip(x, pos)
]

input_single_graph = input_graph[0]
target_single_graph = target_graph[0]


def test_init_input_target():
    cond = Condition(input=input_tensor, target=target_tensor)
    assert isinstance(cond, TensorInputTensorTargetCondition)
    cond = Condition(input=input_tensor, target=target_tensor)
    assert isinstance(cond, TensorInputTensorTargetCondition)
    cond = Condition(input=input_tensor, target=target_graph)
    assert isinstance(cond, TensorInputGraphTargetCondition)
    cond = Condition(input=input_graph, target=target_tensor)
    assert isinstance(cond, GraphInputTensorTargetCondition)
    cond = Condition(input=input_graph, target=target_graph)
    assert isinstance(cond, GraphInputGraphTargetCondition)

    cond = Condition(input=input_lt, target=input_single_graph)
    assert isinstance(cond, TensorInputGraphTargetCondition)
    cond = Condition(input=input_single_graph, target=target_lt)
    assert isinstance(cond, GraphInputTensorTargetCondition)
    cond = Condition(input=input_graph, target=target_single_graph)
    assert isinstance(cond, GraphInputGraphTargetCondition)

    with pytest.raises(ValueError):
        Condition(input_tensor, input_tensor)
    with pytest.raises(ValueError):
        Condition(input=3.0, target="example")
    with pytest.raises(ValueError):
        Condition(input=example_domain, target=example_domain)


def test_init_domain_equation():
    cond = Condition(domain=example_domain, equation=FixedValue(0.0))
    assert isinstance(cond, DomainEquationCondition)
    with pytest.raises(ValueError):
        Condition(example_domain, FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(domain=3.0, equation="example")
    with pytest.raises(ValueError):
        Condition(domain=input_tensor, equation=input_graph)


def test_init_input_equation():
    cond = Condition(input=input_tensor, equation=FixedValue(0.0))
    assert isinstance(cond, InputTensorEquationCondition)
    cond = Condition(input=input_lt, equation=FixedValue(0.0))
    assert isinstance(cond, InputTensorEquationCondition)
    cond = Condition(input=input_graph, equation=FixedValue(0.0))
    assert isinstance(cond, InputGraphEquationCondition)
    with pytest.raises(ValueError):
        Condition(example_domain, FixedValue(0.0))
    with pytest.raises(ValueError):
        Condition(input=3.0, equation="example")
    with pytest.raises(ValueError):
        Condition(input=example_domain, equation=input_graph)


def test_init_data_condition():
    cond = Condition(input=input_lt)
    assert isinstance(cond, TensorDataCondition)
    cond = Condition(input=input_tensor)
    assert isinstance(cond, TensorDataCondition)
    cond = Condition(input=input_tensor, conditional_variables=torch.tensor(1))
    assert isinstance(cond, TensorDataCondition)
    cond = Condition(input=input_graph)
    assert isinstance(cond, GraphDataCondition)
    cond = Condition(input=input_graph, conditional_variables=torch.tensor(1))
    assert isinstance(cond, GraphDataCondition)


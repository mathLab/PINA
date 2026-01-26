import torch
import pytest
from pina import Condition
from pina._src.condition.input_equation_condition import InputEquationCondition
from pina.equation import Equation
from pina import LabelTensor
from pina.graph import Graph
from pina._src.condition.data_manager import _DataManager


def _create_pts_and_equation():
    def dummy_equation(pts):
        return pts["x"] ** 2 + pts["y"] ** 2 - 1

    pts = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    equation = Equation(dummy_equation)
    return pts, equation


def _create_graph_and_equation():
    from pina.graph import KNNGraph

    def dummy_equation(pts):
        return pts.x[:, 0] ** 2 + pts.x[:, 1] ** 2 - 1

    x = LabelTensor(torch.randn(100, 2), labels=["u", "v"])
    pos = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    graph = KNNGraph(x=x, pos=pos, neighbours=5, edge_attr=True)
    equation = Equation(dummy_equation)
    return graph, equation


def test_init_tensor_equation_condition():
    pts, equation = _create_pts_and_equation()
    condition = Condition(input=pts, equation=equation)
    assert isinstance(condition, InputEquationCondition)
    assert condition.input.shape == (100, 2)
    assert condition.equation is equation


def test_init_graph_equation_condition():
    graph, equation = _create_graph_and_equation()
    condition = Condition(input=graph, equation=equation)
    assert isinstance(condition, InputEquationCondition)
    assert isinstance(condition.input, Graph)
    assert condition.input.x.shape == (100, 2)
    assert condition.equation is equation


def test_wrong_init_equation_condition():
    pts, equation = _create_pts_and_equation()
    # Wrong input type
    with pytest.raises(ValueError):
        Condition(input=torch.randn(10, 2), equation=equation)
    # Wrong equation type
    with pytest.raises(ValueError):
        Condition(input=pts, equation="not_an_equation")
    # Wrong input type (list with wrong elements)
    with pytest.raises(ValueError):
        Condition(input=[torch.randn(10, 2)], equation=equation)


def test_getitem_tensor_equation_condition():
    pts, equation = _create_pts_and_equation()
    condition = Condition(input=pts, equation=equation)
    item = condition[0]
    assert isinstance(item, _DataManager)
    assert hasattr(item, "input")
    assert item.input.shape == (2,)


def test_getitems_tensor_equation_condition():
    pts, equation = _create_pts_and_equation()
    condition = Condition(input=pts, equation=equation)
    idxs = [0, 1, 3]
    item = condition[idxs]
    assert isinstance(item, _DataManager)
    assert hasattr(item, "input")
    assert item.input.shape == (3, 2)

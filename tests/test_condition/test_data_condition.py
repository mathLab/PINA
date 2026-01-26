import pytest
import torch
from pina import Condition, LabelTensor
from pina.condition import DataCondition
from pina.graph import RadiusGraph
from torch_geometric.data import Data
from pina._src.condition.data_manager import _DataManager


def _create_tensor_data(use_lt=False, conditional_variables=False):
    input_tensor = torch.rand((10, 3))
    if use_lt:
        input_tensor = LabelTensor(input_tensor, ["x", "y", "z"])
    if conditional_variables:
        cond_vars = torch.rand((10, 2))
        if use_lt:
            cond_vars = LabelTensor(cond_vars, ["a", "b"])
    else:
        cond_vars = None
    return input_tensor, cond_vars


def _create_graph_data(use_lt=False, conditional_variables=False):
    if use_lt:
        x = LabelTensor(torch.rand(10, 20, 2), ["u", "v"])
        pos = LabelTensor(torch.rand(10, 20, 2), ["x", "y"])
    else:
        x = torch.rand(10, 20, 2)
        pos = torch.rand(10, 20, 2)
    radius = 0.1
    input_graph = [
        RadiusGraph(pos=pos[i], radius=radius, x=x[i]) for i in range(len(x))
    ]
    if conditional_variables:
        if use_lt:
            cond_vars = LabelTensor(torch.rand(10, 20, 1), ["f"])
        else:
            cond_vars = torch.rand(10, 20, 1)
    else:
        cond_vars = None
    return input_graph, cond_vars


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_init_tensor_data_condition_tensor(conditional_variables):
    # Setup for standard torch.Tensor
    input_tensor, cond_vars = _create_tensor_data(
        use_lt=False, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_tensor, conditional_variables=cond_vars)

    assert isinstance(condition, DataCondition)

    # Input assertions
    assert isinstance(condition.input, torch.Tensor)
    assert not isinstance(condition.input, LabelTensor)

    # Conditional variables assertions
    if conditional_variables:
        assert condition.conditional_variables is not None
        assert isinstance(condition.conditional_variables, torch.Tensor)
        assert not isinstance(condition.conditional_variables, LabelTensor)
    else:
        assert condition.conditional_variables is None


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_init_tensor_data_condition_label_tensor(conditional_variables):
    # Setup for LabelTensor
    input_tensor, cond_vars = _create_tensor_data(
        use_lt=True, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_tensor, conditional_variables=cond_vars)

    assert isinstance(condition, DataCondition)

    # Input assertions with label validation
    assert isinstance(condition.input, LabelTensor)
    assert condition.input.labels == ["x", "y", "z"]

    # Conditional variables assertions with label validation
    if conditional_variables:
        assert isinstance(condition.conditional_variables, LabelTensor)
        assert condition.conditional_variables.labels == ["a", "b"]
    else:
        assert condition.conditional_variables is None


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_init_graph_data_condition_tensor(conditional_variables):
    # Setup for standard torch.Tensor
    input_graph, cond_vars = _create_graph_data(
        use_lt=False, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)

    assert isinstance(condition, DataCondition)

    # Validate Input list
    assert isinstance(condition.input, list)
    for graph in condition.input:
        assert isinstance(graph, Data)
        assert isinstance(graph.x, torch.Tensor)
        assert not isinstance(graph.x, LabelTensor)
        assert isinstance(graph.pos, torch.Tensor)

    # Validate Conditional Variables
    if conditional_variables:
        assert isinstance(condition.conditional_variables, torch.Tensor)
        assert not isinstance(condition.conditional_variables, LabelTensor)
    else:
        assert condition.conditional_variables is None


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_init_graph_data_condition_label_tensor(conditional_variables):
    # Setup for LabelTensor
    input_graph, cond_vars = _create_graph_data(
        use_lt=True, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)

    assert isinstance(condition, DataCondition)

    # Validate Input list and Labels
    for graph in condition.input:
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == ["u", "v"]

        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == ["x", "y"]

    # Validate Conditional Variables and Labels
    if conditional_variables:
        assert isinstance(condition.conditional_variables, LabelTensor)
        assert condition.conditional_variables.labels == ["f"]
    else:
        assert condition.conditional_variables is None


def test_wrong_init_data_condition():
    input_tensor, cond_vars = _create_tensor_data()
    # Wrong input type
    with pytest.raises(ValueError):
        Condition(input="invalid_input", conditional_variables=cond_vars)
    # Wrong conditional_variables type
    with pytest.raises(ValueError):
        Condition(input=input_tensor, conditional_variables="invalid_cond_vars")
    # Wrong input type (list with wrong elements)
    with pytest.raises(ValueError):
        Condition(input=[input_tensor], conditional_variables=cond_vars)
    # Wrong conditional_variables type (list)
    with pytest.raises(ValueError):
        Condition(input=input_tensor, conditional_variables=[cond_vars])


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitem_tensor_data_condition_tensor(conditional_variables):
    # Setup for standard torch.Tensor
    input_tensor, cond_vars = _create_tensor_data(
        use_lt=False, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_tensor, conditional_variables=cond_vars)

    item = condition[0]

    # Input assertions
    assert isinstance(item.input, torch.Tensor)
    assert not isinstance(item.input, LabelTensor)
    assert item.input.shape == (3,)

    # Conditional variables assertions
    if conditional_variables:
        assert isinstance(item.conditional_variables, torch.Tensor)
        assert item.conditional_variables.shape == (2,)
    else:
        assert not hasattr(item, "conditional_variables")


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitem_tensor_data_condition_label_tensor(conditional_variables):
    # Setup for LabelTensor
    input_tensor, cond_vars = _create_tensor_data(
        use_lt=True, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_tensor, conditional_variables=cond_vars)

    item = condition[0]

    # Input assertions with label validation
    assert isinstance(item.input, LabelTensor)
    assert item.input.shape == (3,)
    assert item.input.labels == ["x", "y", "z"]

    # Conditional variables assertions with label validation
    if conditional_variables:
        assert isinstance(item.conditional_variables, LabelTensor)
        assert item.conditional_variables.shape == (2,)
        assert item.conditional_variables.labels == ["a", "b"]
    else:
        assert not hasattr(item, "conditional_variables")


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitem_graph_data_condition_tensor(conditional_variables):
    # Setup specifically for standard torch.Tensor
    input_graph, cond_vars = _create_graph_data(
        use_lt=False, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)

    item = condition[0]

    # Assertions for the graph data
    assert isinstance(item.input, Data)
    assert isinstance(item.input.x, torch.Tensor)
    assert not isinstance(item.input.x, LabelTensor)
    assert item.input.x.shape == (20, 2)

    # Assertions for conditional variables
    if conditional_variables:
        assert isinstance(item.conditional_variables, torch.Tensor)
        assert item.conditional_variables.shape == (1, 20, 1)


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitem_graph_data_condition_label_tensor(conditional_variables):
    # Setup specifically for LabelTensor
    input_graph, cond_vars = _create_graph_data(
        use_lt=True, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)

    item = condition[0]
    graph = item.input

    # Assertions for LabelTensor attributes
    assert isinstance(graph.x, LabelTensor)
    assert graph.x.labels == ["u", "v"]
    assert graph.x.shape == (20, 2)

    assert isinstance(graph.pos, LabelTensor)
    assert graph.pos.labels == ["x", "y"]

    # Assertions for labeled conditional variables
    if conditional_variables:
        cond_var = item.conditional_variables
        assert isinstance(cond_var, LabelTensor)
        assert cond_var.labels == ["f"]
        assert cond_var.shape == (1, 20, 1)


@pytest.mark.parametrize("use_lt", [False, True])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitems_tensor_data_condition(use_lt, conditional_variables):
    input_tensor, cond_vars = _create_tensor_data(
        use_lt=use_lt, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_tensor, conditional_variables=cond_vars)
    idxs = [0, 1, 3]
    items = condition[idxs]
    assert isinstance(items, _DataManager)
    assert hasattr(items, "input")
    type_ = LabelTensor if use_lt else torch.Tensor
    inputs = items.input
    assert isinstance(inputs, type_)
    assert inputs.shape == (3, 3)
    if use_lt:
        assert inputs.labels == ["x", "y", "z"]
    if conditional_variables:
        assert hasattr(items, "conditional_variables")
        cond_vars_items = items.conditional_variables
        assert isinstance(cond_vars_items, type_)
        assert cond_vars_items.shape == (3, 2)
        if use_lt:
            assert cond_vars_items.labels == ["a", "b"]
    else:
        assert not hasattr(items, "conditional_variables")


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitems_graph_data_condition_tensor(conditional_variables):
    # Setup with use_lt=False
    input_graph, cond_vars = _create_graph_data(
        use_lt=False, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)

    idxs = [0, 1, 3]
    items = condition[idxs]

    # Assertions for DataManager and Graphs
    assert isinstance(items, _DataManager)
    graphs = items.input
    assert len(graphs) == 3

    for graph in graphs:
        assert isinstance(graph.x, torch.Tensor)
        assert not isinstance(graph.x, LabelTensor)
        assert graph.x.shape == (20, 2)

    # Assertions for Conditional Variables
    if conditional_variables:
        assert isinstance(items.conditional_variables, torch.Tensor)
        assert items.conditional_variables.shape == (3, 20, 1)


@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitems_graph_data_condition_label_tensor(conditional_variables):
    # Setup with use_lt=True
    input_graph, cond_vars = _create_graph_data(
        use_lt=True, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)

    idxs = [0, 1, 3]
    items = condition[idxs]

    # Assertions for LabelTensor specific attributes in Graphs
    for graph in items.input:
        assert isinstance(graph.x, LabelTensor)
        assert graph.x.labels == ["u", "v"]

        assert isinstance(graph.pos, LabelTensor)
        assert graph.pos.labels == ["x", "y"]

    # Assertions for LabelTensor in Conditional Variables
    if conditional_variables:
        cv = items.conditional_variables
        assert isinstance(cv, LabelTensor)
        assert cv.labels == ["f"]
        assert cv.shape == (3, 20, 1)

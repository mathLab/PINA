import pytest
import torch
from pina import Condition, LabelTensor
from pina.condition import DataCondition
from pina.graph import RadiusGraph
from torch_geometric.data import Data, Batch
from pina.graph import Graph, LabelBatch
from pina.condition.data_manager import _DataManager


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


@pytest.mark.parametrize("use_lt", [False, True])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_init_tensor_data_condition(use_lt, conditional_variables):
    input_tensor, cond_vars = _create_tensor_data(
        use_lt=use_lt, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_tensor, conditional_variables=cond_vars)
    print(condition)
    assert isinstance(condition, DataCondition)

    type_ = LabelTensor if use_lt else torch.Tensor
    if conditional_variables:
        assert condition.conditional_variables is not None
        assert isinstance(condition.conditional_variables, type_)
        if use_lt:
            assert condition.conditional_variables.labels == ["a", "b"]
    else:
        assert condition.conditional_variables is None
    assert isinstance(condition.input, type_)
    if use_lt:
        assert condition.input.labels == ["x", "y", "z"]


@pytest.mark.parametrize("use_lt", [False, True])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_init_graph_data_condition(use_lt, conditional_variables):
    input_graph, cond_vars = _create_graph_data(
        use_lt=use_lt, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)
    assert isinstance(condition, DataCondition)
    type_ = LabelTensor if use_lt else torch.Tensor
    if conditional_variables:
        assert condition.conditional_variables is not None
        assert isinstance(condition.conditional_variables, type_)
        if use_lt:
            assert condition.conditional_variables.labels == ["f"]
    else:
        assert condition.conditional_variables is None
        # assert "conditional_variables" not in condition.data.keys()
    assert isinstance(condition.input, list)
    for graph in condition.input:
        assert isinstance(graph, Data)
        assert isinstance(graph.x, type_)
        if use_lt:
            assert graph.x.labels == ["u", "v"]
        assert isinstance(graph.pos, type_)
        if use_lt:
            assert graph.pos.labels == ["x", "y"]


@pytest.mark.parametrize("use_lt", [False, True])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitem_tensor_data_condition(use_lt, conditional_variables):
    input_tensor, cond_vars = _create_tensor_data(
        use_lt=use_lt, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_tensor, conditional_variables=cond_vars)
    item = condition[0]
    assert isinstance(item, _DataManager)
    assert hasattr(item, "input")
    type_ = LabelTensor if use_lt else torch.Tensor
    assert isinstance(item.input, type_)
    assert item.input.shape == (3,)
    if type_ is LabelTensor:
        assert item.input.labels == ["x", "y", "z"]
    if conditional_variables:
        assert hasattr(item, "conditional_variables")
        assert isinstance(item.conditional_variables, type_)
        assert item.conditional_variables.shape == (2,)
        if type_ is LabelTensor:
            assert item.conditional_variables.labels == ["a", "b"]
    else:
        assert not hasattr(item, "conditional_variables")


@pytest.mark.parametrize("use_lt", [False, True])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitem_graph_data_condition(use_lt, conditional_variables):
    input_graph, cond_vars = _create_graph_data(
        use_lt=use_lt, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)
    item = condition[0]
    assert isinstance(item, _DataManager)
    assert hasattr(item, "input")
    graph = item.input
    assert isinstance(graph, Data)
    type_ = LabelTensor if use_lt else torch.Tensor
    assert isinstance(graph.x, type_)
    assert graph.x.shape == (20, 2)
    if use_lt:
        assert graph.x.labels == ["u", "v"]
    assert isinstance(graph.pos, type_)
    assert graph.pos.shape == (20, 2)
    if use_lt:
        assert graph.pos.labels == ["x", "y"]
    if conditional_variables:
        assert hasattr(item, "conditional_variables")
        cond_var = item.conditional_variables
        assert isinstance(cond_var, type_)
        assert cond_var.shape == (1, 20, 1)
        if use_lt:
            assert cond_var.labels == ["f"]


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


@pytest.mark.parametrize("use_lt", [False, True])
@pytest.mark.parametrize("conditional_variables", [False, True])
def test_getitems_graph_data_condition(use_lt, conditional_variables):
    input_graph, cond_vars = _create_graph_data(
        use_lt=use_lt, conditional_variables=conditional_variables
    )
    condition = Condition(input=input_graph, conditional_variables=cond_vars)
    idxs = [0, 1, 3]
    items = condition[idxs]
    assert isinstance(items, _DataManager)
    assert hasattr(items, "input")
    graphs = items.input
    assert isinstance(graphs, list)
    assert len(graphs) == 3
    for graph in graphs:
        assert isinstance(graph, Data)
        type_ = LabelTensor if use_lt else torch.Tensor
        assert isinstance(graph.x, type_)
        assert graph.x.shape == (20, 2)
        if use_lt:
            assert graph.x.labels == ["u", "v"]
        assert isinstance(graph.pos, type_)
        assert graph.pos.shape == (20, 2)
        if use_lt:
            assert graph.pos.labels == ["x", "y"]
    if conditional_variables:
        type_ = LabelTensor if use_lt else torch.Tensor
        assert hasattr(items, "conditional_variables")
        cond_vars_batch = items.conditional_variables
        assert isinstance(cond_vars_batch, type_)
        assert cond_vars_batch.shape == (3, 20, 1)
        if use_lt:
            assert cond_vars_batch.labels == ["f"]


if __name__ == "__main__":
    test_init_tensor_data_condition(use_lt=False, conditional_variables=False)
    print("Passed tensor data condition init test without LT and cond vars.")
    test_init_tensor_data_condition(use_lt=True, conditional_variables=False)
    print(
        "Passed tensor data condition init test with LT and without cond vars."
    )
    test_init_tensor_data_condition(use_lt=False, conditional_variables=True)
    print(
        "Passed tensor data condition init test without LT and with cond vars."
    )
    test_init_tensor_data_condition(use_lt=True, conditional_variables=True)
    print("Passed tensor data condition init test with LT and cond vars.")
    test_init_graph_data_condition(use_lt=False, conditional_variables=False)
    print("Passed graph data condition init test without LT and cond vars.")
    test_init_graph_data_condition(use_lt=True, conditional_variables=False)
    print(
        "Passed graph data condition init test with LT and without cond vars."
    )
    test_init_graph_data_condition(use_lt=False, conditional_variables=True)
    print(
        "Passed graph data condition init test without LT and with cond vars."
    )
    test_init_graph_data_condition(use_lt=True, conditional_variables=True)
    print("Passed graph data condition init test with LT and cond vars.")

    test_getitem_tensor_data_condition(
        use_lt=False, conditional_variables=False
    )
    print("Passed tensor data condition getitem test without LT and cond vars.")
    test_getitem_tensor_data_condition(use_lt=True, conditional_variables=False)
    print(
        "Passed tensor data condition getitem test with LT and without cond vars."
    )
    test_getitem_tensor_data_condition(use_lt=False, conditional_variables=True)
    print(
        "Passed tensor data condition getitem test without LT and with cond vars."
    )
    test_getitem_tensor_data_condition(use_lt=True, conditional_variables=True)
    print("Passed tensor data condition getitem test with LT and cond vars.")

    test_getitem_graph_data_condition(use_lt=False, conditional_variables=False)
    print("Passed graph data condition getitem test without LT and cond vars.")
    test_getitem_graph_data_condition(use_lt=True, conditional_variables=False)
    print(
        "Passed graph data condition getitem test with LT and without cond vars."
    )
    test_getitem_graph_data_condition(use_lt=False, conditional_variables=True)
    print(
        "Passed graph data condition getitem test without LT and with cond vars."
    )
    test_getitem_graph_data_condition(use_lt=True, conditional_variables=True)
    print("Passed graph data condition getitem test with LT and cond vars.")

    test_getitems_tensor_data_condition(
        use_lt=False, conditional_variables=False
    )
    print(
        "Passed tensor data condition getitems test without LT and cond vars."
    )
    test_getitems_tensor_data_condition(
        use_lt=True, conditional_variables=False
    )
    print(
        "Passed tensor data condition getitems test with LT and without cond vars."
    )
    test_getitems_tensor_data_condition(
        use_lt=False, conditional_variables=True
    )
    print(
        "Passed tensor data condition getitems test without LT and with cond vars."
    )
    test_getitems_tensor_data_condition(use_lt=True, conditional_variables=True)
    print("Passed tensor data condition getitems test with LT and cond vars.")

import torch
import pytest
from pina.graph import RadiusGraph, Graph
from pina import LabelTensor, Condition
from pina.condition import InputTargetCondition
from pina.data.manager import (
    _TensorDataManager,
    _GraphDataManager,
    _BatchManager,
)


# Helper function to create tensor data
def _create_tensor_data(use_lt):

    # If LabelTensor is used, create tensors with labels
    if use_lt:
        input_tensor = LabelTensor(torch.rand((10, 3)), ["x", "y", "z"])
        target_tensor = LabelTensor(torch.rand((10, 2)), ["a", "b"])
        return input_tensor, target_tensor

    # Standard torch.Tensor without labels
    input_tensor = torch.rand((10, 3))
    target_tensor = torch.rand((10, 2))

    return input_tensor, target_tensor


# Helper function to create graph data
def _create_graph_data(is_input, use_lt):

    # If LabelTensor is used, create graph data with LabelTensors
    if use_lt:
        x = LabelTensor(torch.rand(10, 20, 2), ["u", "v"])
        pos = LabelTensor(torch.rand(10, 20, 2), ["x", "y"])
        tensor = LabelTensor(torch.rand(10, 20, 1), ["f"])

    # Standard torch.Tensor without labels
    else:
        x = torch.rand(10, 20, 2)
        pos = torch.rand(10, 20, 2)
        tensor = torch.rand(10, 20, 1)

    # Create a list of Graphs
    graph = [
        RadiusGraph(
            pos=pos[i],
            radius=0.1,
            x=x[i] if is_input else None,
            y=x[i] if not is_input else None,
        )
        for i in range(len(x))
    ]

    return graph, tensor


# Helper function to check tensor types
def _assert_tensor_type(t, use_lt):
    if use_lt:
        assert isinstance(t, LabelTensor)
    else:
        assert isinstance(t, torch.Tensor) and not isinstance(t, LabelTensor)


# Helper function to check input graph
def _assert_graph_type(graph_list, use_lt, is_input):

    assert isinstance(graph_list, list)
    for graph in graph_list:
        value = graph.x if is_input else graph.y
        _assert_tensor_type(value, use_lt)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize(
    "case", [["tensor", "tensor"], ["tensor", "graph"], ["graph", "tensor"]]
)
def test_constructor(use_lt, case):

    # Tensor - tensor
    if case == ["tensor", "tensor"]:

        # Define the condition
        input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
        condition = Condition(input=input_tensor, target=target_tensor)

        # Assert correct types
        assert isinstance(condition, InputTargetCondition)
        _assert_tensor_type(condition.input, use_lt)
        _assert_tensor_type(condition.target, use_lt)

        # Assert numerical parity
        assert torch.allclose(condition.input, input_tensor)
        assert torch.allclose(condition.target, target_tensor)

        # Assert labels if LabelTensor is used
        if use_lt:
            assert condition.input.labels == ["x", "y", "z"]
            assert condition.target.labels == ["a", "b"]

    # Tensor - graph
    elif case == ["tensor", "graph"]:

        # Define the condition
        target_graph, input_tensor = _create_graph_data(
            is_input=False, use_lt=use_lt
        )
        condition = Condition(input=input_tensor, target=target_graph)

        # Assert correct types
        assert isinstance(condition, InputTargetCondition)
        _assert_tensor_type(condition.input, use_lt)
        _assert_graph_type(condition.target, use_lt, is_input=False)

        # Assert numerical parity
        assert torch.allclose(condition.input, input_tensor)
        for i, graph in enumerate(target_graph):
            assert torch.allclose(condition.target[i].y, graph.y)

        # Assert labels if LabelTensor is used
        if use_lt:
            assert condition.input.labels == ["f"]
            for i in range(len(target_graph)):
                assert condition.target[i].y.labels == ["u", "v"]
                assert condition.target[i].pos.labels == ["x", "y"]

    # Graph - tensor
    elif case == ["graph", "tensor"]:

        # Define the condition
        input_graph, target_tensor = _create_graph_data(
            is_input=True, use_lt=use_lt
        )
        condition = Condition(input=input_graph, target=target_tensor)

        # Assert correct types
        assert isinstance(condition, InputTargetCondition)
        _assert_graph_type(condition.input, use_lt, is_input=True)
        _assert_tensor_type(condition.target, use_lt)

        # Assert numerical parity
        assert torch.allclose(condition.target, target_tensor)
        for i, graph in enumerate(input_graph):
            assert torch.allclose(condition.input[i].x, graph.x)

        # Assert labels if LabelTensor is used
        if use_lt:
            assert condition.target.labels == ["f"]
            for i in range(len(input_graph)):
                assert condition.input[i].x.labels == ["u", "v"]
                assert condition.input[i].pos.labels == ["x", "y"]

    # Prepare for invalid input tests
    input_ = input_tensor if case[0] == "tensor" else input_graph
    target_ = target_tensor if case[1] == "tensor" else target_graph

    # Should fail if the input is neither a tensor nor a graph
    with pytest.raises(ValueError):
        Condition(input="invalid_input", target=target_)

    # Should fail if the target is neither a tensor nor a graph
    with pytest.raises(ValueError):
        Condition(input=input_, target="invalid_target")

    # Should fail if the input is a list of tensors
    if case[0] == "tensor":
        with pytest.raises(ValueError):
            Condition(input=[input_], target=target_)

    # Should fail if the target is a list of tensors
    if case[1] == "tensor":
        with pytest.raises(ValueError):
            Condition(input=input_, target=[target_])


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize(
    "case", [["tensor", "tensor"], ["tensor", "graph"], ["graph", "tensor"]]
)
def test_get_item(use_lt, case):

    # Tensor - tensor
    if case == ["tensor", "tensor"]:

        # Define the condition
        input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
        condition = Condition(input=input_tensor, target=target_tensor)

        # Extract item using __getitem__
        index = 0
        item = condition[index]

        # Assert correct types
        assert isinstance(item, _TensorDataManager)
        _assert_tensor_type(item.input, use_lt)
        _assert_tensor_type(item.target, use_lt)

        # Assert numerical parity
        assert torch.allclose(item.input, input_tensor[index])
        assert torch.allclose(item.target, target_tensor[index])

    # Tensor - graph
    elif case == ["tensor", "graph"]:

        # Define the condition
        target_graph, input_tensor = _create_graph_data(
            is_input=False, use_lt=use_lt
        )
        condition = Condition(input=input_tensor, target=target_graph)

        # Extract item using __getitem__
        index = 0
        item = condition[index]

        # Assert correct types
        assert isinstance(item, _GraphDataManager)
        _assert_tensor_type(item.input, use_lt)
        assert isinstance(item.target, Graph)
        _assert_tensor_type(item.target.y, use_lt)

        # Assert numerical parity
        assert torch.allclose(item.input, input_tensor[index])
        assert torch.allclose(item.target.y, target_graph[index].y)

    # Graph - tensor
    elif case == ["graph", "tensor"]:

        # Define the condition
        input_graph, target_tensor = _create_graph_data(
            is_input=True, use_lt=use_lt
        )
        condition = Condition(input=input_graph, target=target_tensor)

        # Extract item using __getitem__
        index = 0
        item = condition[index]

        # Assert correct types
        assert isinstance(item, _GraphDataManager)
        assert isinstance(item.input, Graph)
        _assert_tensor_type(item.input.x, use_lt)
        _assert_tensor_type(item.target, use_lt)

        # Assert numerical parity
        assert torch.allclose(item.target, target_tensor[index])
        assert torch.allclose(item.input.x, input_graph[index].x)


@pytest.mark.parametrize("use_lt", [True, False])
@pytest.mark.parametrize(
    "case", [["tensor", "tensor"], ["tensor", "graph"], ["graph", "tensor"]]
)
def test_create_batch(use_lt, case):

    # Tensor - tensor
    if case == ["tensor", "tensor"]:

        # Define the condition
        input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
        condition = Condition(input=input_tensor, target=target_tensor)

        # Create batches using automatic batching or condition's collate_fn
        idx = [0, 2]
        data_to_collate = [condition.data[i] for i in idx]
        batch_auto = condition.automatic_batching_collate_fn(data_to_collate)
        batch_collate = condition.collate_fn(idx, condition)

        # Check that the automatic batch has been properly created
        assert isinstance(batch_auto, _BatchManager)
        assert hasattr(batch_auto, "input")
        assert hasattr(batch_auto, "target")

        # Check that the collate_fn batch has been properly created
        assert isinstance(batch_collate, dict)
        assert hasattr(batch_collate, "input")
        assert hasattr(batch_collate, "target")

        # Create expected input and target batches
        expected_input = torch.stack([input_tensor[i] for i in idx])
        expected_target = torch.stack([target_tensor[i] for i in idx])

        # Assert that the automatic batch input and target are correct
        assert torch.allclose(batch_auto.input, expected_input)
        assert torch.allclose(batch_auto.target, expected_target)
        assert batch_auto.input.shape == expected_input.shape
        assert batch_auto.target.shape == expected_target.shape

        # Assert that the collate_fn batch input and target are correct
        assert torch.allclose(batch_collate.input, expected_input)
        assert torch.allclose(batch_collate.target, expected_target)
        assert batch_collate.input.shape == expected_input.shape
        assert batch_collate.target.shape == expected_target.shape

    # Tensor - graph
    elif case == ["tensor", "graph"]:

        # Define the condition
        target_graph, input_tensor = _create_graph_data(
            is_input=False, use_lt=use_lt
        )
        condition = Condition(input=input_tensor, target=target_graph)

        # Create batches using automatic batching or condition's collate_fn
        idx = [0, 2]
        data_to_collate = [condition.data[i] for i in idx]
        batch_auto = condition.automatic_batching_collate_fn(data_to_collate)
        batch_collate = condition.collate_fn(idx, condition)

        # Check that the automatic batch has been properly created
        assert isinstance(batch_auto, _BatchManager)
        assert hasattr(batch_auto, "input")
        assert hasattr(batch_auto, "target")

        # Check that the collate_fn batch has been properly created
        assert isinstance(batch_collate, dict)
        assert hasattr(batch_collate, "input")
        assert hasattr(batch_collate, "target")

        # Create expected input and target batches
        expected_input = torch.cat([input_tensor[i] for i in idx])
        expected_target = [target_graph[i] for i in idx]

        # Assert that the automatic batch input and target are correct
        assert torch.allclose(batch_auto.input, expected_input)
        for i, graph in enumerate(expected_target):
            assert torch.allclose(batch_auto.target[i].y, graph.y)
        assert batch_auto.input.shape == expected_input.shape
        assert batch_auto.target.num_graphs == len(idx)

        # Assert that the collate_fn batch input and target are correct
        assert torch.allclose(batch_collate.input, expected_input)
        for i, graph in enumerate(expected_target):
            assert torch.allclose(batch_collate.target[i].y, graph.y)
        assert batch_collate.input.shape == expected_input.shape
        assert batch_collate.target.num_graphs == len(idx)

    # Graph - tensor
    elif case == ["graph", "tensor"]:

        # Define the condition
        input_graph, target_tensor = _create_graph_data(
            is_input=True, use_lt=use_lt
        )
        condition = Condition(input=input_graph, target=target_tensor)

        # Create batches using automatic batching or condition's collate_fn
        idx = [0, 2]
        data_to_collate = [condition.data[i] for i in idx]
        batch_auto = condition.automatic_batching_collate_fn(data_to_collate)
        batch_collate = condition.collate_fn(idx, condition)

        # Check that the automatic batch has been properly created
        assert isinstance(batch_auto, _BatchManager)
        assert hasattr(batch_auto, "input")
        assert hasattr(batch_auto, "target")

        # Check that the collate_fn batch has been properly created
        assert isinstance(batch_collate, dict)
        assert hasattr(batch_collate, "input")
        assert hasattr(batch_collate, "target")

        # Create expected input and target batches
        expected_input = [input_graph[i] for i in idx]
        expected_target = torch.cat([target_tensor[i] for i in idx])

        # Assert that the automatic batch input and target are correct
        for i, graph in enumerate(expected_input):
            assert torch.allclose(batch_auto.input[i].x, graph.x)
        assert torch.allclose(batch_auto.target, expected_target)
        assert batch_auto.input.num_graphs == len(idx)
        assert batch_auto.target.shape == expected_target.shape

        # Assert that the collate_fn batch input and target are correct
        for i, graph in enumerate(expected_input):
            assert torch.allclose(batch_collate.input[i].x, graph.x)
        assert torch.allclose(batch_collate.target, expected_target)
        assert batch_collate.input.num_graphs == len(idx)
        assert batch_collate.target.shape == expected_target.shape

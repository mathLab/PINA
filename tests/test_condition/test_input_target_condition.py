import torch
import pytest
from pina import LabelTensor, Condition
from pina.graph import RadiusGraph
from pina._src.condition.batch_manager import _BatchManager


def _create_tensor_data(use_lt=False):
    if use_lt:
        input_tensor = LabelTensor(torch.rand((10, 3)), ["x", "y", "z"])
        target_tensor = LabelTensor(torch.rand((10, 2)), ["a", "b"])
        return input_tensor, target_tensor
    input_tensor = torch.rand((10, 3))
    target_tensor = torch.rand((10, 2))
    return input_tensor, target_tensor


def _create_graph_data(tensor_input=True, use_lt=False):
    if use_lt:
        x = LabelTensor(torch.rand(10, 20, 2), ["u", "v"])
        pos = LabelTensor(torch.rand(10, 20, 2), ["x", "y"])
    else:
        x = torch.rand(10, 20, 2)
        pos = torch.rand(10, 20, 2)
    radius = 0.1
    graph = [
        RadiusGraph(
            pos=pos[i],
            radius=radius,
            x=x[i] if not tensor_input else None,
            y=x[i] if tensor_input else None,
        )
        for i in range(len(x))
    ]
    if use_lt:
        tensor = LabelTensor(torch.rand(10, 20, 1), ["f"])
    else:
        tensor = torch.rand(10, 20, 1)
    return graph, tensor


def test_init_tensor_input_tensor_target_condition_tensor():
    # Setup for standard torch.Tensor
    input_tensor, target_tensor = _create_tensor_data(use_lt=False)
    condition = Condition(input=input_tensor, target=target_tensor)

    # Numerical assertions
    assert torch.allclose(
        condition.input, input_tensor
    ), "Standard input tensor equality failed"
    assert torch.allclose(
        condition.target, target_tensor
    ), "Standard target tensor equality failed"

    # Type assertions
    assert isinstance(condition.input, torch.Tensor)
    assert not isinstance(condition.input, LabelTensor)
    assert isinstance(condition.target, torch.Tensor)
    assert not isinstance(condition.target, LabelTensor)


def test_init_tensor_input_tensor_target_condition_label_tensor():
    # Setup for LabelTensor
    input_tensor, target_tensor = _create_tensor_data(use_lt=True)
    condition = Condition(input=input_tensor, target=target_tensor)

    # Type and Label assertions for Input
    assert isinstance(
        condition.input, LabelTensor
    ), "Input did not preserve LabelTensor type"
    assert condition.input.labels == [
        "x",
        "y",
        "z",
    ], "Input labels were lost or corrupted"

    # Type and Label assertions for Target
    assert isinstance(
        condition.target, LabelTensor
    ), "Target did not preserve LabelTensor type"
    assert condition.target.labels == [
        "a",
        "b",
    ], "Target labels were lost or corrupted"

    # Numerical parity check still applies
    assert torch.allclose(condition.input, input_tensor)
    assert torch.allclose(condition.target, target_tensor)


def test_init_tensor_input_graph_target_condition_tensor():
    # Setup for standard torch.Tensor
    target_graph, input_tensor = _create_graph_data(use_lt=False)
    condition = Condition(input=input_tensor, target=target_graph)

    # Input assertions (Tensor)
    assert isinstance(condition.input, torch.Tensor)
    assert not isinstance(condition.input, LabelTensor)
    assert torch.allclose(condition.input, input_tensor)

    # Target assertions (Graph List)
    assert isinstance(condition.target, list)
    for i, graph in enumerate(target_graph):
        assert isinstance(condition.target[i].y, torch.Tensor)
        assert not isinstance(condition.target[i].y, LabelTensor)
        assert torch.allclose(condition.target[i].y, graph.y)


def test_init_tensor_input_graph_target_condition_label_tensor():
    # Setup for LabelTensor
    target_graph, input_tensor = _create_graph_data(use_lt=True)
    condition = Condition(input=input_tensor, target=target_graph)

    # Input assertions with label validation
    assert isinstance(condition.input, LabelTensor)
    assert condition.input.labels == ["f"]
    assert torch.allclose(condition.input, input_tensor)

    # Target assertions with nested label validation
    for i, graph in enumerate(target_graph):
        target_y = condition.target[i].y
        assert isinstance(target_y, LabelTensor)
        assert target_y.labels == ["u", "v"]
        assert torch.allclose(target_y, graph.y)


def test_init_graph_input_tensor_target_condition_tensor():
    # Setup for standard torch.Tensor (use_lt=False)
    input_graph, target_tensor = _create_graph_data(False, use_lt=False)
    condition = Condition(input=input_graph, target=target_tensor)

    # Input assertions: Check graph list integrity
    assert isinstance(condition.input, list)
    for i, original_graph in enumerate(input_graph):
        assert torch.allclose(condition.input[i].x, original_graph.x)
        assert isinstance(condition.input[i].x, torch.Tensor)
        assert not isinstance(condition.input[i].x, LabelTensor)

    # Target assertions: Check raw tensor integrity
    assert torch.allclose(condition.target, target_tensor)
    assert isinstance(condition.target, torch.Tensor)
    assert not isinstance(condition.target, LabelTensor)


def test_init_graph_input_tensor_target_condition_label_tensor():
    # Setup for LabelTensor (use_lt=True)
    input_graph, target_tensor = _create_graph_data(False, use_lt=True)
    condition = Condition(input=input_graph, target=target_tensor)

    # Input assertions: Check LabelTensor preservation in Graphs
    for i, original_graph in enumerate(input_graph):
        input_x = condition.input[i].x
        assert isinstance(input_x, LabelTensor)
        assert input_x.labels == original_graph.x.labels
        assert torch.allclose(input_x, original_graph.x)

    # Target assertions: Check LabelTensor preservation in Target
    assert isinstance(condition.target, LabelTensor)
    assert condition.target.labels == ["f"]
    assert torch.allclose(condition.target, target_tensor)


def test_wrong_init():
    input_tensor, target_tensor = _create_tensor_data()
    with pytest.raises(ValueError):
        Condition(input="invalid_input", target=target_tensor)
    with pytest.raises(ValueError):
        Condition(input=input_tensor, target="invalid_target")
    with pytest.raises(ValueError):
        Condition(input=[input_tensor], target=target_tensor)
    with pytest.raises(ValueError):
        Condition(input=input_tensor, target=[target_tensor])


def test_getitem_tensor_input_tensor_target_condition_tensor():
    # Setup for standard torch.Tensor
    input_tensor, target_tensor = _create_tensor_data(use_lt=False)
    condition = Condition(input=input_tensor, target=target_tensor)

    # We test a single index to verify __getitem__ logic
    index = 0
    item = condition[index]

    # Numerical and Type Assertions
    assert torch.allclose(item.input, input_tensor[index])
    assert isinstance(item.input, torch.Tensor)
    assert not isinstance(item.input, LabelTensor)

    assert torch.allclose(item.target, target_tensor[index])
    assert isinstance(item.target, torch.Tensor)
    assert not isinstance(item.target, LabelTensor)


def test_getitem_tensor_input_tensor_target_condition_label_tensor():
    # Setup for LabelTensor
    input_tensor, target_tensor = _create_tensor_data(use_lt=True)
    condition = Condition(input=input_tensor, target=target_tensor)

    index = 0
    item = condition[index]

    # Verify Input LabelTensor preservation
    assert isinstance(item.input, LabelTensor)
    assert item.input.labels == input_tensor.labels
    assert torch.allclose(item.input, input_tensor[index])

    # Verify Target LabelTensor preservation
    assert isinstance(item.target, LabelTensor)
    assert item.target.labels == target_tensor.labels
    assert torch.allclose(item.target, target_tensor[index])


@pytest.mark.parametrize("use_lt", [True, False])
def test_getitem_graph_input_tensor_target_condition(use_lt):
    input_graph, target_tensor = _create_graph_data(False, use_lt=use_lt)
    condition = Condition(input=input_graph, target=target_tensor)
    assert len(condition) == len(input_graph)
    for i in range(len(input_graph)):
        item = condition[i]
        assert torch.allclose(
            item.input.x, input_graph[i].x
        ), "GraphInputTensorTargetCondition __getitem__ input failed"
        assert torch.allclose(
            item.target, target_tensor[i]
        ), "GraphInputTensorTargetCondition __getitem__ target failed"
        if use_lt:
            assert isinstance(
                item.input.x, LabelTensor
            ), "GraphInputTensorTargetCondition __getitem__ input type failed"
            assert (
                item.input.x.labels == input_graph[i].x.labels
            ), "GraphInputTensorTargetCondition __getitem__ input labels failed"
            assert isinstance(
                item.target, LabelTensor
            ), "GraphInputTensorTargetCondition __getitem__ target type failed"
            assert item.target.labels == [
                "f"
            ], "GraphInputTensorTargetCondition __getitem__ target labels failed"


def test_getitem_tensor_input_graph_target_condition_tensor():
    # Setup for standard torch.Tensor
    target_graph, input_tensor = _create_graph_data(use_lt=False)
    condition = Condition(input=input_tensor, target=target_graph)

    # Check first item indexing
    idx = 0
    item = condition[idx]

    # Input assertions (Tensor)
    assert torch.allclose(item.input, input_tensor[idx])
    assert isinstance(item.input, torch.Tensor)
    assert not isinstance(item.input, LabelTensor)

    # Target assertions (Graph Data)
    assert torch.allclose(item.target.y, target_graph[idx].y)
    assert isinstance(item.target.y, torch.Tensor)
    assert not isinstance(item.target.y, LabelTensor)


def test_getitem_tensor_input_graph_target_condition_label_tensor():
    # Setup for LabelTensor
    target_graph, input_tensor = _create_graph_data(use_lt=True)
    condition = Condition(input=input_tensor, target=target_graph)

    idx = 0
    item = condition[idx]

    # Input LabelTensor validation
    assert isinstance(item.input, LabelTensor)
    assert item.input.labels == input_tensor.labels
    assert torch.allclose(item.input, input_tensor[idx])

    # Target Graph LabelTensor validation
    target_y = item.target.y
    assert isinstance(target_y, LabelTensor)
    assert target_y.labels == ["u", "v"]
    assert torch.allclose(target_y, target_graph[idx].y)


def test_getitems_tensor_input_tensor_target_condition_tensor():
    # Setup for standard torch.Tensor
    input_tensor, target_tensor = _create_tensor_data(use_lt=False)
    condition = Condition(input=input_tensor, target=target_tensor)

    indices = [1, 3, 5, 7]
    items = condition[indices]

    # Verify values by comparing against manually stacked slices
    expected_input = torch.stack([input_tensor[i] for i in indices])
    expected_target = torch.stack([target_tensor[i] for i in indices])

    assert torch.allclose(items.input, expected_input)
    assert torch.allclose(items.target, expected_target)

    # Ensure types remain standard torch.Tensor
    assert isinstance(items.input, torch.Tensor)
    assert not isinstance(items.input, LabelTensor)
    assert isinstance(items.target, torch.Tensor)


def test_getitems_tensor_input_tensor_target_condition_label_tensor():
    # Setup for LabelTensor
    input_tensor, target_tensor = _create_tensor_data(use_lt=True)
    condition = Condition(input=input_tensor, target=target_tensor)

    indices = [1, 3, 5, 7]
    items = condition[indices]

    # Assertions for Input LabelTensor
    assert isinstance(items.input, LabelTensor)
    assert items.input.labels == ["x", "y", "z"]
    assert torch.allclose(items.input, input_tensor[indices])

    # Assertions for Target LabelTensor
    assert isinstance(items.target, LabelTensor)
    assert items.target.labels == ["a", "b"]
    assert torch.allclose(items.target, target_tensor[indices])


def test_getitems_tensor_input_graph_target_condition_tensor():
    # Setup for standard torch.Tensor
    target_graph, input_tensor = _create_graph_data(True, use_lt=False)
    condition = Condition(input=input_tensor, target=target_graph)

    indices = [0, 2, 4]
    items = condition[indices]

    # 1. Verify Input Batch (Tensor)
    expected_input = torch.stack([input_tensor[i] for i in indices])
    assert torch.allclose(items.input, expected_input)
    assert isinstance(items.input, torch.Tensor)
    assert not isinstance(items.input, LabelTensor)

    # 2. Verify Target Batch (Graph List)
    assert len(items.target) == len(indices)
    for i, original_idx in enumerate(indices):
        assert torch.allclose(items.target[i].y, target_graph[original_idx].y)
        assert isinstance(items.target[i].y, torch.Tensor)


def test_getitems_tensor_input_graph_target_condition_label_tensor():
    # Setup for LabelTensor
    target_graph, input_tensor = _create_graph_data(True, use_lt=True)
    condition = Condition(input=input_tensor, target=target_graph)

    indices = [0, 2, 4]
    items = condition[indices]

    # 1. Verify Input LabelTensor preservation
    assert isinstance(items.input, LabelTensor)
    assert items.input.labels == ["f"]
    # Verify values still match
    assert torch.allclose(items.input, input_tensor[indices])

    # 2. Verify Target Graphs LabelTensor preservation
    assert len(items.target) == len(indices)
    for i, original_idx in enumerate(indices):
        target_y = items.target[i].y
        assert isinstance(target_y, LabelTensor)
        assert target_y.labels == ["u", "v"]
        # Verify numerical parity
        assert torch.allclose(target_y, target_graph[original_idx].y)


def test_create_batch_tensor():
    input_tensor, target_tensor = _create_tensor_data()
    condition = Condition(input=input_tensor, target=target_tensor)
    idx = [0, 2, 4, 6]
    data_to_collate = [condition.data[i] for i in idx]
    batch = condition.automatic_batching_collate_fn(data_to_collate)
    assert isinstance(batch, _BatchManager)
    assert hasattr(batch, "input")
    assert hasattr(batch, "target")
    expected_input = torch.stack([input_tensor[i] for i in idx])
    expected_target = torch.stack([target_tensor[i] for i in idx])
    assert torch.allclose(batch.input, expected_input)
    assert torch.allclose(batch.target, expected_target)

    batch = condition.collate_fn(idx, condition)
    # assert isinstance(batch, _BatchManager)
    assert hasattr(batch, "input")
    assert hasattr(batch, "target")
    expected_input = torch.stack([input_tensor[i] for i in idx])
    expected_target = torch.stack([target_tensor[i] for i in idx])
    assert torch.allclose(batch.input, expected_input)
    assert torch.allclose(batch.target, expected_target)


def test_create_batch_graph():
    input_graph, target_tensor = _create_graph_data(False)
    condition = Condition(input=input_graph, target=target_tensor)
    idx = [1, 3, 5]
    data_to_collate = [condition.data[i] for i in idx]
    batch = condition.automatic_batching_collate_fn(data_to_collate)
    assert isinstance(batch, _BatchManager)
    assert hasattr(batch, "input")
    assert hasattr(batch, "target")
    expected_target = torch.cat([target_tensor[i] for i in idx])
    print(expected_target.shape, batch.target.shape)
    assert torch.allclose(batch.target, expected_target)
    assert batch.input.num_graphs == len(idx)

    batch = condition.collate_fn(idx, condition)
    assert isinstance(batch, _BatchManager)
    assert hasattr(batch, "input")
    assert hasattr(batch, "target")
    assert torch.allclose(batch.target, expected_target)
    assert batch.input.num_graphs == len(idx)

import torch
import pytest
from pina import LabelTensor, Condition
from pina.graph import RadiusGraph
from pina.condition.batch_manager import _BatchManager


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


@pytest.mark.parametrize("use_lt", [True, False])
def test_init_tensor_input_tensor_target_condition(use_lt):
    input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
    condition = Condition(input=input_tensor, target=target_tensor)
    # assert isinstance(condition, TensorInputTensorTargetCondition)
    assert torch.allclose(
        condition.input, input_tensor
    ), "TensorInputTensorTargetCondition input failed"
    assert torch.allclose(
        condition.target, target_tensor
    ), "TensorInputTensorTargetCondition target failed"
    if use_lt:
        assert isinstance(
            condition.input, LabelTensor
        ), "TensorInputTensorTargetCondition input type failed"
        assert condition.input.labels == [
            "x",
            "y",
            "z",
        ], "TensorInputTensorTargetCondition input labels failed"
        assert isinstance(
            condition.target, LabelTensor
        ), "TensorInputTensorTargetCondition target type failed"
        assert condition.target.labels == [
            "a",
            "b",
        ], "TensorInputTensorTargetCondition target labels failed"


@pytest.mark.parametrize("use_lt", [True, False])
def test_init_tensor_input_graph_target_condition(use_lt):
    target_graph, input_tensor = _create_graph_data(use_lt=use_lt)
    condition = Condition(input=input_tensor, target=target_graph)
    # assert isinstance(condition, TensorInputGraphTargetCondition)
    assert torch.allclose(
        condition.input, input_tensor
    ), "TensorInputGraphTargetCondition input failed"
    for i in range(len(target_graph)):
        assert torch.allclose(
            condition.target[i].y, target_graph[i].y
        ), "TensorInputGraphTargetCondition target failed"
        if use_lt:
            assert isinstance(
                condition.target[i].y, LabelTensor
            ), "TensorInputGraphTargetCondition target type failed"
            assert condition.target[i].y.labels == [
                "u",
                "v",
            ], "TensorInputGraphTargetCondition target labels failed"
    if use_lt:
        assert isinstance(
            condition.input, LabelTensor
        ), "TensorInputGraphTargetCondition target type failed"
        assert condition.input.labels == [
            "f"
        ], "TensorInputGraphTargetCondition target labels failed"


@pytest.mark.parametrize("use_lt", [True, False])
def test_init_graph_input_tensor_target_condition(use_lt):
    input_graph, target_tensor = _create_graph_data(False, use_lt=use_lt)
    condition = Condition(input=input_graph, target=target_tensor)
    # assert isinstance(condition, GraphInputTensorTargetCondition)
    for i in range(len(input_graph)):
        assert torch.allclose(
            condition.input[i].x, input_graph[i].x
        ), "GraphInputTensorTargetCondition input failed"
        if use_lt:
            assert isinstance(
                condition.input[i].x, LabelTensor
            ), "GraphInputTensorTargetCondition input type failed"
            assert (
                condition.input[i].x.labels == input_graph[i].x.labels
            ), "GraphInputTensorTargetCondition labels failed"

    assert torch.allclose(
        condition.target[i], target_tensor[i]
    ), "GraphInputTensorTargetCondition target failed"
    if use_lt:
        assert isinstance(
            condition.target, LabelTensor
        ), "GraphInputTensorTargetCondition target type failed"
        assert condition.target.labels == [
            "f"
        ], "GraphInputTensorTargetCondition target labels failed"


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


@pytest.mark.parametrize("use_lt", [True, False])
def test_getitem_tensor_input_tensor_target_condition(use_lt):
    input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
    condition = Condition(input=input_tensor, target=target_tensor)
    for i in range(len(input_tensor)):
        item = condition[i]
        assert torch.allclose(
            item.input, input_tensor[i]
        ), "TensorInputTensorTargetCondition __getitem__ input failed"
        assert torch.allclose(
            item.target, target_tensor[i]
        ), "TensorInputTensorTargetCondition __getitem__ target failed"


@pytest.mark.parametrize("use_lt", [True, False])
def test_getitem_tensor_input_graph_target_condition(use_lt):
    target_graph, input_tensor = _create_graph_data(use_lt=use_lt)
    condition = Condition(input=input_tensor, target=target_graph)
    for i in range(len(input_tensor)):
        item = condition[i]
        assert torch.allclose(
            item.input, input_tensor[i]
        ), "TensorInputGraphTargetCondition __getitem__ input failed"
        assert torch.allclose(
            item.target.y, target_graph[i].y
        ), "TensorInputGraphTargetCondition __getitem__ target failed"
        if use_lt:
            assert isinstance(
                item.target.y, LabelTensor
            ), "TensorInputGraphTargetCondition __getitem__ target type failed"
            assert item.target.y.labels == [
                "u",
                "v",
            ], "TensorInputGraphTargetCondition __getitem__ target labels failed"


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


@pytest.mark.parametrize("use_lt", [True, False])
def test_getitems_tensor_input_tensor_target_condition(use_lt):
    input_tensor, target_tensor = _create_tensor_data(use_lt=use_lt)
    condition = Condition(input=input_tensor, target=target_tensor)
    indices = [1, 3, 5, 7]
    items = condition[indices]
    candidate_input = items.input
    candidate_target = items.target

    if use_lt:
        input_ = LabelTensor.stack([input_tensor[i] for i in indices])
        target_ = LabelTensor.stack([target_tensor[i] for i in indices])
    else:
        input_ = torch.stack([input_tensor[i] for i in indices])
        target_ = torch.stack([target_tensor[i] for i in indices])
    assert torch.allclose(
        candidate_input, input_
    ), "TensorInputTensorTargetCondition __getitems__ input failed"
    assert torch.allclose(
        candidate_target, target_
    ), "TensorInputTensorTargetCondition __getitems__ target failed"
    if use_lt:
        assert isinstance(
            candidate_input, LabelTensor
        ), "TensorInputTensorTargetCondition __getitems__ input type failed"
        assert candidate_input.labels == [
            "x",
            "y",
            "z",
        ], "TensorInputTensorTargetCondition __getitems__ input labels failed"
        assert isinstance(
            candidate_target, LabelTensor
        ), "TensorInputTensorTargetCondition __getitems__ target type failed"
        assert candidate_target.labels == [
            "a",
            "b",
        ], "TensorInputTensorTargetCondition __getitems__ target labels failed"


@pytest.mark.parametrize("use_lt", [True, False])
def test_getitems_tensor_input_graph_target_condition(use_lt):
    target_graph, input_tensor = _create_graph_data(True, use_lt=use_lt)
    condition = Condition(input=input_tensor, target=target_graph)
    indices = [0, 2, 4]
    items = condition[indices]
    candidate_input = items.input
    candidate_target = items.target
    if use_lt:
        input_ = LabelTensor.stack([input_tensor[i] for i in indices])
        # target_ = LabelBatch.from_data_list([target_graph[i] for i in indices])
    else:
        input_ = torch.stack([input_tensor[i] for i in indices])
        # target_ = Batch.from_data_list([target_graph[i] for i in indices])
    assert torch.allclose(
        candidate_input, input_
    ), "TensorInputGraphTargetCondition __getitems__ input failed"

    assert len(candidate_target) == len(
        indices
    ), "TensorInputGraphTargetCondition __getitems__ target length failed"
    for idx, graph_idx in enumerate(indices):
        assert torch.allclose(
            candidate_target[idx].y, target_graph[graph_idx].y
        ), "TensorInputGraphTargetCondition __getitems__ target failed"

    if use_lt:
        assert isinstance(
            candidate_input, LabelTensor
        ), "TensorInputGraphTargetCondition __getitems__ input type failed"
        assert candidate_input.labels == [
            "f"
        ], "TensorInputGraphTargetCondition __getitems__ input labels failed"
        for g in candidate_target:
            assert isinstance(
                g.y, LabelTensor
            ), "TensorInputGraphTargetCondition __getitems__ target type failed"
            assert g.y.labels == [
                "u",
                "v",
            ], "TensorInputGraphTargetCondition __getitems__ target labels failed"


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

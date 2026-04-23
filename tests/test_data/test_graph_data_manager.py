import torch
import pytest
from pina import LabelTensor
from pina.graph import Graph
from pina.data.manager import _DataManager, _GraphDataManager, _BatchManager


# Define data for testing
standard_graph = [
    Graph(
        x=torch.rand((10, 3)),
        pos=torch.rand((10, 2)),
        edge_index=torch.randint(0, 10, (2, 20)),
    )
    for _ in range(3)
]
label_graph = [
    Graph(
        x=LabelTensor(torch.rand((10, 3)), labels=["a", "b", "c"]),
        pos=LabelTensor(torch.rand((10, 2)), labels=["x", "y"]),
        edge_index=torch.randint(0, 10, (2, 20)),
    )
    for _ in range(3)
]
target_ = torch.rand((3, 10, 1))
label_target = LabelTensor(target_, labels=["target"])


@pytest.mark.parametrize("case", ["standard", "labeled"])
def test_constructor(case):

    # Define data for testing
    if case == "standard":
        graph = standard_graph
        target = target_
        exp_type = torch.Tensor
    else:
        graph = label_graph
        target = label_target
        exp_type = LabelTensor

    # Create data manager
    data_manager = _DataManager(graph=graph, target=target)

    # Check that the data manager is an instance of _GraphDataManager
    assert isinstance(data_manager, _GraphDataManager)

    # Check that the attributes are set correctly
    assert hasattr(data_manager, "graph_key")
    assert hasattr(data_manager, "graph")
    assert hasattr(data_manager, "target")
    assert data_manager.graph_key == "graph"

    # Check that the graph length is correct
    assert len(data_manager.graph) == len(graph)

    # Check that the attributes have the correct types
    assert isinstance(data_manager.target, exp_type)
    assert isinstance(data_manager.graph, list)
    for g in data_manager.graph:
        assert isinstance(g, Graph)

    # Check that the values of the attributes are correct
    assert torch.equal(data_manager.target, target)
    for i in range(len(graph)):
        assert torch.equal(data_manager.graph[i].x, graph[i].x)
        assert torch.equal(data_manager.graph[i].pos, graph[i].pos)
        assert torch.equal(
            data_manager.graph[i].edge_index, graph[i].edge_index
        )
        assert torch.equal(data_manager.graph[i].target, graph[i].target)


@pytest.mark.parametrize("case", ["standard", "labeled"])
def test_create_batch(case):

    # Define data for testing
    if case == "standard":
        graph = standard_graph
        target = target_
        exp_type = torch.Tensor
    else:
        graph = label_graph
        target = label_target
        exp_type = LabelTensor

    # Create data manager
    data_manager = _DataManager(graph=graph, target=target)

    # Batch over indices
    idx = [0, 2]
    batch = _GraphDataManager.create_batch([data_manager[idx] for idx in idx])

    # Check that the batch is an instance of _BatchManager
    assert isinstance(batch, _BatchManager)

    # Check that the attributes are set correctly
    assert hasattr(batch, "graph")
    assert hasattr(batch, "target")

    # Check that the graph length is correct
    assert batch.graph.num_graphs == len(idx)

    # Check that the attributes have the correct types
    assert isinstance(batch.target, exp_type)
    assert isinstance(batch.graph, Graph)

    # Check that the values of the attributes are correct
    assert torch.equal(batch.target, torch.cat([target[i] for i in idx], dim=0))
    assert torch.equal(
        batch.graph.x, torch.cat([graph[i].x for i in idx], dim=0)
    )
    assert torch.equal(
        batch.graph.pos, torch.cat([graph[i].pos for i in idx], dim=0)
    )

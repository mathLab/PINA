import torch
from pina.condition.data_manager import (
    _DataManager,
    _TensorDataManager,
    _GraphDataManager,
)
from pina.graph import Graph
from pina.equation import Equation


def test_tensor_data_manager_init():
    pippo = torch.rand((10, 5))
    pluto = torch.rand((10, 7))
    paperino = torch.rand((10, 11))
    data_manager = _DataManager(pippo=pippo, pluto=pluto, paperino=paperino)
    assert isinstance(data_manager, _TensorDataManager)
    assert hasattr(data_manager, "pippo")
    assert hasattr(data_manager, "pluto")
    assert hasattr(data_manager, "paperino")
    assert torch.equal(data_manager.pippo, pippo)
    assert torch.equal(data_manager.pluto, pluto)
    assert torch.equal(data_manager.paperino, paperino)

    paperino = Equation(lambda x: x**2)
    data_manager3 = _DataManager(pippo=pippo, pluto=pluto, paperino=paperino)
    assert isinstance(data_manager3, _TensorDataManager)
    assert hasattr(data_manager3, "pippo")
    assert hasattr(data_manager3, "pluto")
    assert hasattr(data_manager3, "paperino")
    assert torch.equal(data_manager3.pippo, pippo)
    assert torch.equal(data_manager3.pluto, pluto)
    assert isinstance(data_manager3.paperino, Equation)


def test_graph_data_manager_init():
    x = [torch.rand((10, 5)) for _ in range(3)]
    pos = [torch.rand((10, 3)) for _ in range(3)]
    edge_index = [torch.randint(0, 10, (2, 20)) for _ in range(3)]
    graph = [
        Graph(x=x_, pos=pos_, edge_index=edge_index_)
        for x_, pos_, edge_index_ in zip(x, pos, edge_index)
    ]
    target = torch.rand((3, 10, 1))
    data_manager = _DataManager(graph=graph, target=target)
    assert hasattr(data_manager, "graph_key")
    assert data_manager.graph_key == "graph"
    assert hasattr(data_manager, "graph")
    assert len(data_manager.data) == 3
    for i in range(3):
        g = data_manager.graph[i]
        assert torch.equal(g.x, x[i])
        assert torch.equal(g.pos, pos[i])
        assert torch.equal(g.edge_index, edge_index[i])
        assert torch.equal(g.target, target[i])


def test_graph_data_manager_getattribute():
    x = [torch.rand((10, 5)) for _ in range(3)]
    pos = [torch.rand((10, 3)) for _ in range(3)]
    edge_index = [torch.randint(0, 10, (2, 20)) for _ in range(3)]
    graph = [
        Graph(x=x_, pos=pos_, edge_index=edge_index_)
        for x_, pos_, edge_index_ in zip(x, pos, edge_index)
    ]
    target = torch.rand((3, 10, 1))
    data_manager = _DataManager(graph=graph, target=target)
    target_retrieved = data_manager.target
    assert torch.equal(target_retrieved, target)


def test_graph_data_manager_getitem():
    x = [torch.rand((10, 5)) for _ in range(3)]
    pos = [torch.rand((10, 3)) for _ in range(3)]
    edge_index = [torch.randint(0, 10, (2, 20)) for _ in range(3)]
    graph = [
        Graph(x=x_, pos=pos_, edge_index=edge_index_)
        for x_, pos_, edge_index_ in zip(x, pos, edge_index)
    ]
    target = torch.rand((3, 10, 1))
    data_manager = _DataManager(graph=graph, target=target)
    item = data_manager[1]
    assert isinstance(item, _DataManager)
    assert hasattr(item, "graph_key")
    assert item.graph_key == "graph"
    assert hasattr(item, "graph")
    assert torch.equal(item.graph.x, x[1])
    assert torch.equal(item.graph.pos, pos[1])
    assert torch.equal(item.graph.edge_index, edge_index[1])
    assert torch.equal(item.target, target[1].unsqueeze(0))


def test_graph_data_create_batch():
    x = [torch.rand((10, 5)) for _ in range(3)]
    pos = [torch.rand((10, 3)) for _ in range(3)]
    edge_index = [torch.randint(0, 10, (2, 20)) for _ in range(3)]
    graph = [
        Graph(x=x_, pos=pos_, edge_index=edge_index_)
        for x_, pos_, edge_index_ in zip(x, pos, edge_index)
    ]
    target = torch.rand((3, 10, 1))
    data_manager = _DataManager(graph=graph, target=target)
    item1 = data_manager[0]
    item2 = data_manager[1]
    batch_data = _GraphDataManager._create_batch([item1, item2])
    assert hasattr(batch_data, "graph")
    assert hasattr(batch_data, "target")
    batched_graphs = batch_data.graph
    batched_target = batch_data.target
    assert batched_graphs.num_graphs == 2
    assert batched_target.shape == (20, 1)
    assert torch.equal(batched_target, torch.cat([target[0], target[1]], dim=0))
    mps_data = batch_data.to("mps")
    assert mps_data.graph.num_graphs == 2
    assert torch.equal(mps_data.target, batched_target.to("mps"))
    assert torch.equal(mps_data.graph.x, batched_graphs.x.to("mps"))


def test_tensor_data_create_batch():
    pippo = torch.rand((10, 5))
    pluto = torch.rand((10, 7))
    paperino = torch.rand((10, 11))
    data_manager = _DataManager(pippo=pippo, pluto=pluto, paperino=paperino)
    item1 = data_manager[0]
    item2 = data_manager[1]
    batch_data = _TensorDataManager._create_batch([item1, item2])
    assert hasattr(batch_data, "pippo")
    assert hasattr(batch_data, "pluto")
    assert hasattr(batch_data, "paperino")
    assert torch.equal(
        batch_data.pippo, torch.stack([pippo[0], pippo[1]], dim=0)
    )
    assert torch.equal(
        batch_data.pluto, torch.stack([pluto[0], pluto[1]], dim=0)
    )
    assert torch.equal(
        batch_data.paperino, torch.stack([paperino[0], paperino[1]], dim=0)
    )

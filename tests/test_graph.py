import pytest
import torch
from pina import Graph
from pina.graph import RadiusGraph, KNNGraph, TemporalGraph


@pytest.mark.parametrize(
    "x, pos",
    [
        ([torch.rand(10, 2) for _ in range(3)], [torch.rand(10, 3) for _ in range(3)]),
        ([torch.rand(10, 2) for _ in range(3)], [torch.rand(10, 3) for _ in range(3)]),
        (torch.rand(3,10,2), torch.rand(3,10,3)),
        (torch.rand(3,10,2), torch.rand(3,10,3)),
    ]
)
def test_build_multiple_graph_multiple_val(x, pos):
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=.3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=.3)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k = 3)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


def test_build_single_graph_multiple_val():
    x = torch.rand(10, 2)
    pos = torch.rand(10, 3)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=.3)
    assert len(graph.data) == 1
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=.3)
    data = graph.data
    assert len(graph.data) == 1
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    x = torch.rand(10, 2)
    pos = torch.rand(10, 3)
    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    assert len(graph.data) == 1
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    data = graph.data
    assert len(graph.data) == 1
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


@pytest.mark.parametrize(
    "pos",
    [
        ([torch.rand(10, 3) for _ in range(3)]),
        ([torch.rand(10, 3) for _ in range(3)]),
        (torch.rand(3, 10, 3)),
        (torch.rand(3, 10, 3))
    ]
)
def test_build_single_graph_single_val(pos):
    x = torch.rand(10, 2)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=.3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=.3)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)
    x = torch.rand(10, 2)
    graph = KNNGraph(x=x, pos=pos, build_edge_attr=False, k=3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

def test_additional_parameters_1():
    x = torch.rand(3, 10, 2)
    pos = torch.rand(3, 10, 2)
    additional_parameters = {'y': torch.ones(3)}
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=.3,
                  additional_params=additional_parameters)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(hasattr(d, 'y') for d in data)
    assert all(d_.y == 1 for d_ in data)

@pytest.mark.parametrize(
    "additional_parameters",
    [
        ({'y': torch.rand(3,10,1)}),
        ({'y': [torch.rand(10,1) for _ in range(3)]}),
    ]
)
def test_additional_parameters_2(additional_parameters):
    x = torch.rand(3, 10, 2)
    pos = torch.rand(3, 10, 2)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=.3,
                  additional_params=additional_parameters)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(hasattr(d, 'y') for d in data)
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))

def test_temporal_graph():
    x = torch.rand(3, 10, 2)
    pos = torch.rand(3, 10, 2)
    t = torch.rand(3)
    graph = TemporalGraph(x=x, pos=pos, build_edge_attr=True, r=.3, t=t)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(hasattr(d, 't') for d in data)
    assert all(d_.t == t_ for (d_, t_) in zip(data, t))

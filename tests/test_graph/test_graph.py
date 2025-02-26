import pytest
import torch
from pina.graph import RadiusGraph, KNNGraph, Graph
from torch_geometric.data import Data


@pytest.mark.parametrize(
    "x, pos, edge_index",
    [
        (
            [torch.rand(10, 2) for _ in range(3)],
            [torch.rand(10, 3) for _ in range(3)],
            [
                torch.tensor(
                    [
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    ]
                )
            ]
            * 3,
        ),
        (
            torch.rand(3, 10, 2),
            torch.rand(3, 10, 3),
            torch.stack(
                [
                    torch.tensor(
                        [
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                        ]
                    )
                ]
                * 3
            ),
        ),
    ],
)
def test_build_multiple_graph_multiple_val(x, pos, edge_index):
    data = [
        Graph(x=x_, pos=pos_, edge_index=edge_index_)
        for x_, pos_, edge_index_ in zip(x, pos, edge_index)
    ]
    assert all(isinstance(d, Data) for d in data)
    assert all(torch.isclose(d.x, x_).all() for d, x_ in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is None for d in data)

    data = [
        Graph(x=x_, pos=pos_, edge_index=edge_index_, build_edge_attr=True)
        for x_, pos_, edge_index_ in zip(x, pos, edge_index)
    ]
    assert all(isinstance(d, Data) for d in data)
    assert all(torch.isclose(d.x, x_).all() for d, x_ in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    data = [RadiusGraph(x=x_, pos=pos_, r=0.2) for x_, pos_ in zip(x, pos)]
    assert all(isinstance(d, Data) for d in data)
    assert all(torch.isclose(d.x, x_).all() for d, x_ in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is None for d in data)

    data = [
        RadiusGraph(x=x_, pos=pos_, r=0.2, build_edge_attr=True)
        for x_, pos_, in zip(x, pos)
    ]
    assert all(isinstance(d, Data) for d in data)
    assert all(torch.isclose(d.x, x_).all() for d, x_ in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    data = [KNNGraph(x=x_, pos=pos_, k=3) for x_, pos_ in zip(x, pos)]
    assert all(isinstance(d, Data) for d in data)
    assert all(torch.isclose(d.x, x_).all() for d, x_ in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is None for d in data)

    data = [
        KNNGraph(x=x_, pos=pos_, k=3, build_edge_attr=True)
        for x_, pos_, in zip(x, pos)
    ]
    assert all(isinstance(d, Data) for d in data)
    assert all(torch.isclose(d.x, x_).all() for d, x_ in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


@pytest.mark.parametrize(
    "x", [([torch.rand(10, 2) for _ in range(3)]), (torch.rand(3, 10, 2))]
)
def test_build_single_graph_multi_val(x):
    pos = torch.rand(10, 3)
    graph = RadiusGraph(pos=pos, build_edge_attr=False, r=0.3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )

    graph = Graph(pos=pos, edge_index=edge_index)
    data = [graph(x=x_) for x_ in x]
    print(type(data[0].pos))
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = Graph(pos=pos, edge_index=edge_index, build_edge_attr=True)
    data = [graph(x=x_) for x_ in x]
    assert len(data) == 3
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    graph = RadiusGraph(pos=pos, r=0.3)
    data = [graph(x=x_) for x_ in x]
    print(type(data[0].pos))
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = RadiusGraph(pos=pos, build_edge_attr=True, r=0.3)
    data = [graph(x=x_) for x_ in x]
    assert len(data) == 3
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    graph = KNNGraph(pos=pos, k=3)
    data = [graph(x=x_) for x_ in x]
    print(type(data[0].pos))
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = KNNGraph(pos=pos, build_edge_attr=True, k=3)
    data = [graph(x=x_) for x_ in x]
    assert len(data) == 3
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


def test_build_none_x():
    x = None
    pos = torch.rand(10, 3)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        ]
    )
    data = Graph(x=x, pos=pos, edge_index=edge_index)
    assert isinstance(data, Data)
    assert data.x is None
    assert torch.isclose(data.pos, pos).all()
    assert torch.isclose(data.edge_index, edge_index).all()

    data = RadiusGraph(x=x, pos=pos, r=0.3)
    assert isinstance(data, Data)
    assert data.x is None
    assert torch.isclose(data.pos, pos).all()

    data = KNNGraph(x=x, pos=pos, k=3)
    assert isinstance(data, Data)
    assert data.x is None
    assert torch.isclose(data.pos, pos).all()


def test_additional_parameters_1():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    x = torch.rand(10, 2)
    pos = torch.rand(10, 2)
    y = torch.ones(10, 3)

    data = Graph(x=x, pos=pos, edge_index=edge_index, y=y)
    assert isinstance(data, Data)
    assert torch.isclose(data.x, x).all()
    assert hasattr(data, "y")
    assert torch.isclose(data.y, y).all()

    data = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3, y=y)
    assert isinstance(data, Data)
    assert torch.isclose(data.x, x).all()
    assert hasattr(data, "y")
    assert torch.isclose(data.y, y).all()

    data = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3, y=y)
    assert isinstance(data, Data)
    assert torch.isclose(data.x, x).all()
    assert hasattr(data, "y")
    assert torch.isclose(data.y, y).all()


def test_custom_build_edge_attr_func():
    x = torch.rand(3, 10, 2)
    pos = torch.rand(3, 10, 2)

    def build_edge_attr(x, pos, edge_index):
        return torch.cat([pos[edge_index[0]], pos[edge_index[1]]], dim=-1)

    data = [
        RadiusGraph(
            x=x_,
            pos=pos_,
            build_edge_attr=True,
            r=0.3,
            custom_build_edge_attr=build_edge_attr,
        )
        for x_, pos_ in zip(x, pos)
    ]

    assert len(data) == 3
    assert all(hasattr(d, "edge_attr") for d in data)
    assert all(d.edge_attr.shape[1] == 4 for d in data)
    assert all(
        torch.isclose(
            d.edge_attr, build_edge_attr(d.x, d.pos, d.edge_index)
        ).all()
        for d in data
    )

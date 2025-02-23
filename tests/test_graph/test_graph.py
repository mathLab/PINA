import pytest
import torch
from pina.graph import RadiusGraph, KNNGraph, Graph


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
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=0.3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=False, k=3)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index, build_edge_attr=True)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


def test_build_single_graph_single_val():
    x = torch.rand(10, 2)
    pos = torch.rand(10, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=0.3)
    assert len(graph.data) == 1
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3)
    data = graph.data
    assert len(graph.data) == 1
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=False, k=3)
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

    graph = Graph(x=x, pos=pos, edge_index=edge_index)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index, build_edge_attr=True)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


@pytest.mark.parametrize(
    "pos, edge_index",
    [
        (
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
def test_build_multi_graph_single_val(pos, edge_index):
    x = torch.rand(10, 2)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=0.3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

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

    graph = Graph(x=x, pos=pos, edge_index=edge_index)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index, build_edge_attr=True)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


@pytest.mark.parametrize(
    "x", [([torch.rand(10, 2) for _ in range(3)]), (torch.rand(3, 10, 2))]
)
def test_build_single_graph_multi_val(x):
    pos = torch.rand(10, 3)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=0.3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3)
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=False, k=3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index)
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index, build_edge_attr=True)
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


def test_additional_parameters_1():
    x = torch.rand(3, 10, 2)
    pos = torch.rand(3, 10, 2)
    additional_parameters = {"y": torch.ones(3)}
    graph = RadiusGraph(
        x=x,
        pos=pos,
        build_edge_attr=True,
        r=0.3,
        additional_params=additional_parameters,
    )
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(hasattr(d, "y") for d in data)
    assert all(d_.y == 1 for d_ in data)


@pytest.mark.parametrize(
    "additional_parameters",
    [
        ({"y": torch.rand(3, 10, 1)}),
        ({"y": [torch.rand(10, 1) for _ in range(3)]}),
    ],
)
def test_additional_parameters_2(additional_parameters):
    x = torch.rand(3, 10, 2)
    pos = torch.rand(3, 10, 2)
    graph = RadiusGraph(
        x=x,
        pos=pos,
        build_edge_attr=True,
        r=0.3,
        additional_params=additional_parameters,
    )
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(hasattr(d, "y") for d in data)
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))


def test_custom_build_edge_attr_func():
    x = torch.rand(3, 10, 2)
    pos = torch.rand(3, 10, 2)

    def build_edge_attr(x, pos, edge_index):
        return torch.cat([pos[edge_index[0]], pos[edge_index[1]]], dim=-1)

    graph = RadiusGraph(
        x=x,
        pos=pos,
        build_edge_attr=True,
        r=0.3,
        custom_build_edge_attr=build_edge_attr,
    )
    assert len(graph.data) == 3
    data = graph.data
    assert all(hasattr(d, "edge_attr") for d in data)
    assert all(d.edge_attr.shape[1] == 4 for d in data)
    assert all(
        torch.isclose(
            d.edge_attr, build_edge_attr(d.x, d.pos, d.edge_index)
        ).all()
        for d in data
    )

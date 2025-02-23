import pytest
import torch
from pina.graph import RadiusGraph, KNNGraph, Graph
from pina import LabelTensor


@pytest.mark.parametrize(
    "x, pos, edge_index",
    [
        (
            [LabelTensor(torch.rand(10, 2), ["u", "v"]) for _ in range(3)],
            [LabelTensor(torch.rand(10, 3), ["x", "y", "z"]) for _ in range(3)],
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
            LabelTensor(torch.rand(3, 10, 2), ["u", "v"]),
            LabelTensor(torch.rand(3, 10, 3), ["x", "y", "z"]),
            torch.stack(
                [
                    torch.tensor(
                        [
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                        ]
                    )
                ]
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
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index, build_edge_attr=True)
    data = graph.data
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


def test_build_single_graph_single_val():
    x = LabelTensor(torch.rand(10, 2), ["u", "v"])
    pos = LabelTensor(torch.rand(10, 3), ["x", "y", "z"])
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=0.3)
    assert len(graph.data) == 1
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3)
    data = graph.data
    assert len(graph.data) == 1
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    assert len(graph.data) == 1
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index)
    assert len(graph.data) == 1
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = Graph(x=x, pos=pos, edge_index=edge_index, build_edge_attr=True)
    assert len(graph.data) == 1
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos).all() for d_ in data)
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


@pytest.mark.parametrize(
    "pos, edge_index",
    [
        (
            [LabelTensor(torch.rand(10, 3), ["x", "y", "z"]) for _ in range(3)],
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
            LabelTensor(torch.rand(3, 10, 3), ["x", "y", "z"]),
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
    x = LabelTensor(torch.rand(10, 2), ["u", "v"])
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=0.3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=False, k=3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = Graph(x=x, pos=pos, build_edge_attr=False, edge_index=edge_index)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = Graph(x=x, pos=pos, build_edge_attr=True, edge_index=edge_index)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.x, x).all() for d in data)
    assert all(torch.isclose(d_.pos, pos_).all() for d_, pos_ in zip(data, pos))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


@pytest.mark.parametrize(
    "x",
    [
        ([LabelTensor(torch.rand(10, 2), ["u", "v"]) for _ in range(3)]),
        (LabelTensor(torch.rand(3, 10, 2), ["u", "v"])),
    ],
)
def test_build_single_graph_multi_val(x):
    pos = LabelTensor(torch.rand(10, 3), ["x", "y", "z"])
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
    )
    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=False, r=0.3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = RadiusGraph(x=x, pos=pos, build_edge_attr=True, r=0.3)
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=False, k=3)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=3)
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = Graph(x=x, pos=pos, build_edge_attr=False, edge_index=edge_index)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)

    graph = Graph(x=x, pos=pos, build_edge_attr=True, edge_index=edge_index)
    assert len(graph.data) == 3
    data = graph.data
    assert all(torch.isclose(d.pos, pos).all() for d in data)
    assert all(torch.isclose(d_.x, x_).all() for d_, x_ in zip(data, x))
    assert all(len(d.edge_index) == 2 for d in data)
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(isinstance(d.pos, LabelTensor) for d in data)
    assert all(d.pos.labels == ["x", "y", "z"] for d in data)
    assert all(d.edge_attr is not None for d in data)
    assert all([d.edge_index.shape[1] == d.edge_attr.shape[0]] for d in data)


def test_additional_parameters_1():
    x = LabelTensor(torch.rand(3, 10, 2), ["u", "v"])
    pos = LabelTensor(torch.rand(3, 10, 3), ["x", "y", "z"])
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
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(hasattr(d, "y") for d in data)
    assert all(d_.y == 1 for d_ in data)


@pytest.mark.parametrize(
    "additional_parameters",
    [
        ({"y": LabelTensor(torch.rand(3, 10, 1), ["y"])}),
        ({"y": [LabelTensor(torch.rand(10, 1), ["y"]) for _ in range(3)]}),
    ],
)
def test_additional_parameters_2(additional_parameters):
    x = LabelTensor(torch.rand(3, 10, 2), ["u", "v"])
    pos = LabelTensor(torch.rand(3, 10, 3), ["x", "y", "z"])
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
    assert all(isinstance(d.x, LabelTensor) for d in data)
    assert all(d.x.labels == ["u", "v"] for d in data)
    assert all(hasattr(d, "y") for d in data)
    assert all(torch.isclose(d_.x, x_).all() for (d_, x_) in zip(data, x))
    assert all(isinstance(d.y, LabelTensor) for d in data)
    assert all(d.y.labels == ["y"] for d in data)


def test_custom_build_edge_attr_func():
    x = LabelTensor(torch.rand(3, 10, 2), ["u", "v"])
    pos = LabelTensor(torch.rand(3, 10, 2), ["x", "y"])

    def build_edge_attr(x, pos, edge_index):
        return LabelTensor(
            torch.cat(
                [pos.tensor[edge_index[0]], pos.tensor[edge_index[1]]], dim=1
            ),
            ["x1", "y1", "x2", "y2"],
        )

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
    assert all(isinstance(d.edge_attr, LabelTensor) for d in data)
    assert all(d.edge_attr.labels == ["x1", "y1", "x2", "y2"] for d in data)

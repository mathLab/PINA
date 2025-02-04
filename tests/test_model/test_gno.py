import pytest
import torch
from pina.graph import KNNGraph
from pina.model import GNO
from torch_geometric.data import Batch

x = [torch.rand(100, 6) for _ in range(10)]
pos = [torch.rand(100, 3) for _ in range(10)]
graph = KNNGraph(x=x, pos=pos, build_edge_attr=True, k=6)
input_ = Batch.from_data_list(graph.data)



@pytest.mark.parametrize(
    "shared_weights",
    [
        True,
        False
    ]
)
def test_constructor(shared_weights):
    lifting_operator = torch.nn.Linear(6, 16)
    projection_operator = torch.nn.Linear(16, 3)
    GNO(lifting_operator=lifting_operator,
        projection_operator=projection_operator,
        edge_features=3,
        internal_layers=[16, 16],
        shared_weights=shared_weights)

    GNO(lifting_operator=lifting_operator,
        projection_operator=projection_operator,
        edge_features=3,
        inner_size=16,
        internal_n_layers=10,
        shared_weights=shared_weights)

    int_func = torch.nn.Softplus
    ext_func = torch.nn.ReLU

    GNO(lifting_operator=lifting_operator,
        projection_operator=projection_operator,
        edge_features=3,
        internal_n_layers=10,
        shared_weights=shared_weights,
        internal_func=int_func,
        external_func=ext_func)


@pytest.mark.parametrize(
    "shared_weights",
    [
        True,
        False
    ]
)
def test_forward_1(shared_weights):
    lifting_operator = torch.nn.Linear(6, 16)
    projection_operator = torch.nn.Linear(16, 3)
    model = GNO(lifting_operator=lifting_operator,
                projection_operator=projection_operator,
                edge_features=3,
                internal_layers=[16, 16],
                shared_weights=shared_weights)
    output_ = model(input_)
    assert output_.shape == torch.Size([1000, 3])

@pytest.mark.parametrize(
    "shared_weights",
    [
        True,
        False
    ]
)
def test_forward_2(shared_weights):
    lifting_operator = torch.nn.Linear(6, 16)
    projection_operator = torch.nn.Linear(16, 3)
    model = GNO(lifting_operator=lifting_operator,
                projection_operator=projection_operator,
                edge_features=3,
                inner_size=32,
                internal_n_layers=2,
                shared_weights=shared_weights)
    output_ = model(input_)
    assert output_.shape == torch.Size([1000, 3])

@pytest.mark.parametrize(
    "shared_weights",
    [
        True,
        False
    ]
)
def test_backward(shared_weights):
    lifting_operator = torch.nn.Linear(6, 16)
    projection_operator = torch.nn.Linear(16, 3)
    model = GNO(lifting_operator=lifting_operator,
                projection_operator=projection_operator,
                edge_features=3,
                internal_layers=[16, 16],
                shared_weights=shared_weights)
    input_.x.requires_grad = True
    output_ = model(input_)
    l = torch.mean(output_)
    l.backward()
    assert input_.x.grad.shape == torch.Size([1000, 6])

@pytest.mark.parametrize(
    "shared_weights",
    [
        True,
        False
    ]
)
def test_backward_2(shared_weights):
    lifting_operator = torch.nn.Linear(6, 16)
    projection_operator = torch.nn.Linear(16, 3)
    model = GNO(lifting_operator=lifting_operator,
                projection_operator=projection_operator,
                edge_features=3,
                inner_size=32,
                internal_n_layers=2,
                shared_weights=shared_weights)
    input_.x.requires_grad = True
    output_ = model(input_)
    l = torch.mean(output_)
    l.backward()
    assert input_.x.grad.shape == torch.Size([1000, 6])
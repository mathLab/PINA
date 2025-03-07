import torch
from pina.model import LowRankNeuralOperator
from pina import LabelTensor
import pytest


batch_size = 15
n_layers = 4
embedding_dim = 24
func = torch.nn.Tanh
rank = 4
n_kernel_layers = 3
field_indices = ["u"]
coordinates_indices = ["x", "y"]


def test_constructor():
    # working constructor
    lifting_net = torch.nn.Linear(
        len(coordinates_indices) + len(field_indices), embedding_dim
    )
    projecting_net = torch.nn.Linear(
        embedding_dim + len(coordinates_indices), len(field_indices)
    )
    LowRankNeuralOperator(
        lifting_net=lifting_net,
        projecting_net=projecting_net,
        coordinates_indices=coordinates_indices,
        field_indices=field_indices,
        n_kernel_layers=n_kernel_layers,
        rank=rank,
    )

    # not working constructor
    with pytest.raises(ValueError):
        LowRankNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_kernel_layers=3.2,  # wrong
            rank=rank,
        )

        LowRankNeuralOperator(
            lifting_net=[0],  # wrong
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_kernel_layers=n_kernel_layers,
            rank=rank,
        )

        LowRankNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=[0],  # wront
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_kernel_layers=n_kernel_layers,
            rank=rank,
        )

        LowRankNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=[0],  # wrong
            field_indices=field_indices,
            n_kernel_layers=n_kernel_layers,
            rank=rank,
        )

        LowRankNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=[0],  # wrong
            n_kernel_layers=n_kernel_layers,
            rank=rank,
        )

        lifting_net = torch.nn.Linear(len(coordinates_indices), embedding_dim)
        LowRankNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_kernel_layers=n_kernel_layers,
            rank=rank,
        )

        lifting_net = torch.nn.Linear(
            len(coordinates_indices) + len(field_indices), embedding_dim
        )
        projecting_net = torch.nn.Linear(embedding_dim, len(field_indices))
        LowRankNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_kernel_layers=n_kernel_layers,
            rank=rank,
        )


def test_forward():
    lifting_net = torch.nn.Linear(
        len(coordinates_indices) + len(field_indices), embedding_dim
    )
    projecting_net = torch.nn.Linear(
        embedding_dim + len(coordinates_indices), len(field_indices)
    )
    lno = LowRankNeuralOperator(
        lifting_net=lifting_net,
        projecting_net=projecting_net,
        coordinates_indices=coordinates_indices,
        field_indices=field_indices,
        n_kernel_layers=n_kernel_layers,
        rank=rank,
    )

    input_ = LabelTensor(
        torch.rand(
            batch_size, 100, len(coordinates_indices) + len(field_indices)
        ),
        coordinates_indices + field_indices,
    )

    out = lno(input_)
    assert out.shape == torch.Size(
        [batch_size, input_.shape[1], len(field_indices)]
    )


def test_backward():
    lifting_net = torch.nn.Linear(
        len(coordinates_indices) + len(field_indices), embedding_dim
    )
    projecting_net = torch.nn.Linear(
        embedding_dim + len(coordinates_indices), len(field_indices)
    )
    lno = LowRankNeuralOperator(
        lifting_net=lifting_net,
        projecting_net=projecting_net,
        coordinates_indices=coordinates_indices,
        field_indices=field_indices,
        n_kernel_layers=n_kernel_layers,
        rank=rank,
    )
    input_ = LabelTensor(
        torch.rand(
            batch_size, 100, len(coordinates_indices) + len(field_indices)
        ),
        coordinates_indices + field_indices,
    )
    input_ = input_.requires_grad_()
    out = lno(input_)
    tmp = torch.linalg.norm(out)
    tmp.backward()
    grad = input_.grad
    assert grad.shape == torch.Size(
        [
            batch_size,
            input_.shape[1],
            len(coordinates_indices) + len(field_indices),
        ]
    )

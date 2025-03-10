import torch
from pina.model import AveragingNeuralOperator
from pina import LabelTensor
import pytest


batch_size = 15
n_layers = 4
embedding_dim = 24
func = torch.nn.Tanh
coordinates_indices = ["p"]
field_indices = ["v"]


def test_constructor():
    # working constructor
    lifting_net = torch.nn.Linear(
        len(coordinates_indices) + len(field_indices), embedding_dim
    )
    projecting_net = torch.nn.Linear(
        embedding_dim + len(field_indices), len(field_indices)
    )
    AveragingNeuralOperator(
        lifting_net=lifting_net,
        projecting_net=projecting_net,
        coordinates_indices=coordinates_indices,
        field_indices=field_indices,
        n_layers=n_layers,
        func=func,
    )

    # not working constructor
    with pytest.raises(ValueError):
        AveragingNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_layers=3.2,  # wrong
            func=func,
        )

        AveragingNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_layers=n_layers,
            func=1,
        )  # wrong

        AveragingNeuralOperator(
            lifting_net=[0],  # wrong
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_layers=n_layers,
            func=func,
        )

        AveragingNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=[0],  # wront
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_layers=n_layers,
            func=func,
        )

        AveragingNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=[0],  # wrong
            field_indices=field_indices,
            n_layers=n_layers,
            func=func,
        )

        AveragingNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=[0],  # wrong
            n_layers=n_layers,
            func=func,
        )

        lifting_net = torch.nn.Linear(len(coordinates_indices), embedding_dim)
        AveragingNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_layers=n_layers,
            func=func,
        )

        lifting_net = torch.nn.Linear(
            len(coordinates_indices) + len(field_indices), embedding_dim
        )
        projecting_net = torch.nn.Linear(embedding_dim, len(field_indices))
        AveragingNeuralOperator(
            lifting_net=lifting_net,
            projecting_net=projecting_net,
            coordinates_indices=coordinates_indices,
            field_indices=field_indices,
            n_layers=n_layers,
            func=func,
        )


def test_forward():
    lifting_net = torch.nn.Linear(
        len(coordinates_indices) + len(field_indices), embedding_dim
    )
    projecting_net = torch.nn.Linear(
        embedding_dim + len(field_indices), len(field_indices)
    )
    avno = AveragingNeuralOperator(
        lifting_net=lifting_net,
        projecting_net=projecting_net,
        coordinates_indices=coordinates_indices,
        field_indices=field_indices,
        n_layers=n_layers,
        func=func,
    )

    input_ = LabelTensor(
        torch.rand(
            batch_size, 100, len(coordinates_indices) + len(field_indices)
        ),
        ["p", "v"],
    )

    out = avno(input_)
    assert out.shape == torch.Size(
        [batch_size, input_.shape[1], len(field_indices)]
    )


def test_backward():
    lifting_net = torch.nn.Linear(
        len(coordinates_indices) + len(field_indices), embedding_dim
    )
    projecting_net = torch.nn.Linear(
        embedding_dim + len(field_indices), len(field_indices)
    )
    avno = AveragingNeuralOperator(
        lifting_net=lifting_net,
        projecting_net=projecting_net,
        coordinates_indices=coordinates_indices,
        field_indices=field_indices,
        n_layers=n_layers,
        func=func,
    )
    input_ = LabelTensor(
        torch.rand(
            batch_size, 100, len(coordinates_indices) + len(field_indices)
        ),
        ["p", "v"],
    )
    input_ = input_.requires_grad_()
    out = avno(input_)
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

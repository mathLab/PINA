import torch
import pytest
from pina.model.block import PirateNetBlock

data = torch.rand((20, 3))


@pytest.mark.parametrize("inner_size", [10, 20])
def test_constructor(inner_size):

    PirateNetBlock(inner_size=inner_size, activation=torch.nn.Tanh)

    # Should fail if inner_size is negative
    with pytest.raises(AssertionError):
        PirateNetBlock(inner_size=-1, activation=torch.nn.Tanh)


@pytest.mark.parametrize("inner_size", [10, 20])
def test_forward(inner_size):

    model = PirateNetBlock(inner_size=inner_size, activation=torch.nn.Tanh)

    # Create dummy embedding
    dummy_embedding = torch.nn.Linear(data.shape[1], inner_size)
    x = dummy_embedding(data)

    # Create dummy U and V tensors
    U = torch.rand((data.shape[0], inner_size))
    V = torch.rand((data.shape[0], inner_size))

    output_ = model(x, U, V)
    assert output_.shape == (data.shape[0], inner_size)


@pytest.mark.parametrize("inner_size", [10, 20])
def test_backward(inner_size):

    model = PirateNetBlock(inner_size=inner_size, activation=torch.nn.Tanh)
    data.requires_grad_()

    # Create dummy embedding
    dummy_embedding = torch.nn.Linear(data.shape[1], inner_size)
    x = dummy_embedding(data)

    # Create dummy U and V tensors
    U = torch.rand((data.shape[0], inner_size))
    V = torch.rand((data.shape[0], inner_size))

    output_ = model(x, U, V)

    loss = torch.mean(output_)
    loss.backward()
    assert data.grad.shape == data.shape

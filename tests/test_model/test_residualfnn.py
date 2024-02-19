import torch
import pytest
from pina.model import ResidualFeedForward


def test_constructor():
    # simple constructor
    ResidualFeedForward(input_dimensions=2, output_dimensions=1)

    # wrong transformer nets (not 2)
    with pytest.raises(ValueError):
        ResidualFeedForward(input_dimensions=2,
                            output_dimensions=1,
                            transformer_nets=[torch.nn.Linear(2, 20)])

    # wrong transformer nets (not nn.Module)
    with pytest.raises(ValueError):
        ResidualFeedForward(input_dimensions=2,
                            output_dimensions=1,
                            transformer_nets=[2, 2])


def test_forward():
    x = torch.rand(10, 2)
    model = ResidualFeedForward(input_dimensions=2, output_dimensions=1)
    model(x)


def test_backward():
    x = torch.rand(10, 2)
    x.requires_grad = True
    model = ResidualFeedForward(input_dimensions=2, output_dimensions=1)
    model(x)
    l = torch.mean(model(x))
    l.backward()
    assert x.grad.shape == torch.Size([10, 2])
    
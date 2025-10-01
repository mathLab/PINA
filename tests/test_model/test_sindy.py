import torch
import pytest
from pina.model import SINDy

# Define a simple library of candidate functions and some test data
library = [lambda x: torch.pow(x, 2), lambda x: torch.sin(x)]


@pytest.mark.parametrize("data", [torch.rand((20, 1)), torch.rand((5, 20, 1))])
def test_constructor(data):
    SINDy(library, data.shape[-1])

    # Should fail if output_dimension is not a positive integer
    with pytest.raises(AssertionError):
        SINDy(library, "not_int")
    with pytest.raises(AssertionError):
        SINDy(library, -1)

    # Should fail if library is not a list
    with pytest.raises(ValueError):
        SINDy(lambda x: torch.pow(x, 2), 3)

    # Should fail if library is not a list of callables
    with pytest.raises(ValueError):
        SINDy([1, 2, 3], 3)


@pytest.mark.parametrize("data", [torch.rand((20, 1)), torch.rand((5, 20, 1))])
def test_forward(data):

    # Define model
    model = SINDy(library, data.shape[-1])
    with torch.no_grad():
        model.coefficients.data.fill_(1.0)

    # Evaluate model
    output_ = model(data)
    vals = data.pow(2) + torch.sin(data)

    print(data.shape, output_.shape, vals.shape)

    assert output_.shape == data.shape
    assert torch.allclose(output_, vals, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("data", [torch.rand((20, 1)), torch.rand((5, 20, 1))])
def test_backward(data):

    # Define and evaluate model
    model = SINDy(library, data.shape[-1])
    output_ = model(data.requires_grad_())

    loss = output_.mean()
    loss.backward()
    assert data.grad.shape == data.shape

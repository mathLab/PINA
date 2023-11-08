import torch
import pytest

from pina.model import FeedForward

data = torch.rand((20, 3))
input_vars = 3
output_vars = 4


def test_constructor():
    FeedForward(input_vars, output_vars)
    FeedForward(input_vars, output_vars, inner_size=10, n_layers=20)
    FeedForward(input_vars, output_vars, layers=[10, 20, 5, 2])
    FeedForward(input_vars,
                output_vars,
                layers=[10, 20, 5, 2],
                func=torch.nn.ReLU)
    FeedForward(input_vars,
                output_vars,
                layers=[10, 20, 5, 2],
                func=[torch.nn.ReLU, torch.nn.ReLU, None, torch.nn.Tanh])


def test_constructor_wrong():
    with pytest.raises(RuntimeError):
        FeedForward(input_vars,
                    output_vars,
                    layers=[10, 20, 5, 2],
                    func=[torch.nn.ReLU, torch.nn.ReLU])


def test_forward():
    dim_in, dim_out = 3, 2
    fnn = FeedForward(dim_in, dim_out)
    output_ = fnn(data)
    assert output_.shape == (data.shape[0], dim_out)

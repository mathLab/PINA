import torch
import pytest

from pina import LabelTensor
from pina.model import FeedForward

class myFeature(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """
    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        return LabelTensor(torch.sin(torch.pi * x.extract('a')), 'sin(a)')


data = torch.rand((20, 3))
input_vars = ['a', 'b', 'c']
output_vars = ['d', 'e']
input_ = LabelTensor(data, input_vars)


def test_constructor():
    FeedForward(input_vars, output_vars)
    FeedForward(3, 4)
    FeedForward(input_vars, output_vars, extra_features=[myFeature()])
    FeedForward(input_vars, output_vars, inner_size=10, n_layers=20)
    FeedForward(input_vars, output_vars, layers=[10, 20, 5, 2])
    FeedForward(input_vars, output_vars, layers=[10, 20, 5, 2],
                func=torch.nn.ReLU)
    FeedForward(input_vars, output_vars, layers=[10, 20, 5, 2],
                func=[torch.nn.ReLU, torch.nn.ReLU, None, torch.nn.Tanh])


def test_constructor_wrong():
    with pytest.raises(RuntimeError):
        FeedForward(input_vars, output_vars, layers=[10, 20, 5, 2],
                    func=[torch.nn.ReLU, torch.nn.ReLU])


def test_forward():
    fnn = FeedForward(input_vars, output_vars)
    output_ = fnn(input_)
    assert output_.labels == output_vars


def test_forward2():
    dim_in, dim_out = 3, 2
    fnn = FeedForward(dim_in, dim_out)
    output_ = fnn(input_)
    assert output_.shape == (input_.shape[0], dim_out)


def test_forward_features():
    fnn = FeedForward(input_vars, output_vars, extra_features=[myFeature()])
    output_ = fnn(input_)
    assert output_.labels == output_vars

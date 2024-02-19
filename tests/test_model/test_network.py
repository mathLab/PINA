import torch
import pytest

from pina.model.network import Network
from pina.model import FeedForward
from pina import LabelTensor

data = torch.rand((20, 3))
data_lt = LabelTensor(data, ['x', 'y', 'z'])
input_dim = 3
output_dim = 4
torchmodel = FeedForward(input_dim, output_dim)
extra_feat = []


def test_constructor():
    Network(model=torchmodel,
            input_variables=['x', 'y', 'z'],
            output_variables=['a', 'b', 'c', 'd'],
            extra_features=None)

def test_forward():
    net = Network(model=torchmodel,
                input_variables=['x', 'y', 'z'],
                output_variables=['a', 'b', 'c', 'd'],
                extra_features=None)
    out = net.torchmodel(data)
    out_lt = net(data_lt)
    assert isinstance(out, torch.Tensor)
    assert isinstance(out_lt, LabelTensor)
    assert out.shape == (20, 4)
    assert out_lt.shape == (20, 4)
    assert torch.allclose(out_lt, out)
    assert out_lt.labels == ['a', 'b', 'c', 'd']

    with pytest.raises(AssertionError):
        net(data)

def test_backward():
    net = Network(model=torchmodel,
                input_variables=['x', 'y', 'z'],
                output_variables=['a', 'b', 'c', 'd'],
                extra_features=None)
    data = torch.rand((20, 3))
    data.requires_grad = True
    out = net.torchmodel(data)
    l = torch.mean(out)
    l.backward()
    assert data._grad.shape == torch.Size([20, 3])
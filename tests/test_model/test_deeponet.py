import pytest
import torch
from torch.nn import Linear

from pina import LabelTensor
from pina.model import DeepONet
from pina.model import FeedForward

data = torch.rand((20, 3))
input_vars = ['a', 'b', 'c']
input_ = LabelTensor(data, input_vars)
symbol_funcs_red = DeepONet._symbol_functions(dim=-1)
output_dims = [1, 5, 10, 20]

def test_constructor():
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    DeepONet(branch_net=branch_net,
             trunk_net=trunk_net,
             input_indeces_branch_net=['a'],
             input_indeces_trunk_net=['b', 'c'],
             reduction='+',
             aggregator='*')


def test_constructor_fails_when_invalid_inner_layer_size():
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=8)
    with pytest.raises(ValueError):
        DeepONet(branch_net=branch_net,
                 trunk_net=trunk_net,
                 input_indeces_branch_net=['a'],
                 input_indeces_trunk_net=['b', 'c'],
                 reduction='+',
                 aggregator='*')

def test_forward_extract_str():
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    model = DeepONet(branch_net=branch_net,
                     trunk_net=trunk_net,
                     input_indeces_branch_net=['a'],
                     input_indeces_trunk_net=['b', 'c'],
                     reduction='+',
                     aggregator='*')
    model(input_)
    assert model(input_).shape[-1] == 1


def test_forward_extract_int():
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    model = DeepONet(branch_net=branch_net,
                     trunk_net=trunk_net,
                     input_indeces_branch_net=[0],
                     input_indeces_trunk_net=[1, 2],
                     reduction='+',
                     aggregator='*')
    model(data)

def test_backward_extract_int():
    data = torch.rand((20, 3))
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    model = DeepONet(branch_net=branch_net,
                     trunk_net=trunk_net,
                     input_indeces_branch_net=[0],
                     input_indeces_trunk_net=[1, 2],
                     reduction='+',
                     aggregator='*')
    data.requires_grad = True
    model(data)
    l=torch.mean(model(data))
    l.backward()
    assert data._grad.shape == torch.Size([20,3])

def test_forward_extract_str_wrong():
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    model = DeepONet(branch_net=branch_net,
                     trunk_net=trunk_net,
                     input_indeces_branch_net=['a'],
                     input_indeces_trunk_net=['b', 'c'],
                     reduction='+',
                     aggregator='*')
    with pytest.raises(RuntimeError):
        model(data)

def test_backward_extract_str_wrong():
    data = torch.rand((20, 3))
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    model = DeepONet(branch_net=branch_net,
                     trunk_net=trunk_net,
                     input_indeces_branch_net=['a'],
                     input_indeces_trunk_net=['b', 'c'],
                     reduction='+',
                     aggregator='*')
    data.requires_grad = True
    with pytest.raises(RuntimeError):
        model(data)
        l=torch.mean(model(data))
        l.backward()
        assert data._grad.shape == torch.Size([20,3])

@pytest.mark.parametrize('red', symbol_funcs_red)
def test_forward_symbol_funcs(red):
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    model = DeepONet(branch_net=branch_net,
                     trunk_net=trunk_net,
                     input_indeces_branch_net=['a'],
                     input_indeces_trunk_net=['b', 'c'],
                     reduction=red,
                     aggregator='*')
    model(input_)
    assert model(input_).shape[-1] == 1

@pytest.mark.parametrize('out_dim', output_dims)
def test_forward_callable_reduction(out_dim):
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    reduction_layer = Linear(10, out_dim)
    model = DeepONet(branch_net=branch_net,
                     trunk_net=trunk_net,
                     input_indeces_branch_net=['a'],
                     input_indeces_trunk_net=['b', 'c'],
                     reduction=reduction_layer,
                     aggregator='*')
    model(input_)
    assert model(input_).shape[-1] == out_dim

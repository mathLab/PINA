import pytest
import torch

from pina import LabelTensor
from pina.model import DeepONet
from pina.model import FeedForward

data = torch.rand((20, 3))
input_vars = ['a', 'b', 'c']
input_ = LabelTensor(data, input_vars)


def test_constructor():
    branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=2, output_dimensions=10)
    DeepONet(branch_net=branch_net,
             trunk_net=trunk_net,
             input_indeces_branch_net=['a'],
             input_indeces_trunk_net=['b', 'c'],
             reduction='+',
             aggregator='*')


# This test is wrong! The user could define a custom network and do a 
# reshape at the end! A more general way to check input and output is
# needed
# def test_constructor_fails_when_invalid_inner_layer_size():
#     branch_net = FeedForward(input_dimensions=1, output_dimensions=10)
#     trunk_net = FeedForward(input_dimensions=2, output_dimensions=8)
#     with pytest.raises(ValueError):
#         DeepONet(branch_net=branch_net,
#                  trunk_net=trunk_net,
#                  input_indeces_branch_net=['a'],
#                  input_indeces_trunk_net=['b', 'c'],
#                  reduction='+',
#                  aggregator='*')


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

import pytest
import torch

from pina import LabelTensor
from pina.model import MIONet
from pina.model import FeedForward

data = torch.rand((20, 3))
input_vars = ['a', 'b', 'c']
input_ = LabelTensor(data, input_vars)


def test_constructor():
    branch_net1 = FeedForward(input_dimensons=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensons=2, output_dimensions=10)
    trunk_net = FeedForward(input_dimensons=1, output_dimensions=10)
    networks = {branch_net1 : ['x'],
                branch_net2 : ['x', 'y'],
                trunk_net : ['z']}
    MIONet(networks=networks,
           reduction='+',
           aggregator='*')


def test_constructor_fails_when_invalid_inner_layer_size():
    branch_net1 = FeedForward(input_dimensons=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensons=2, output_dimensions=10)
    trunk_net = FeedForward(input_dimensons=1, output_dimensions=12)
    networks = {branch_net1 : ['x'],
                branch_net2 : ['x', 'y'],
                trunk_net : ['z']}
    with pytest.raises(ValueError):
        MIONet(networks=networks,
               reduction='+',
               aggregator='*')

def test_forward_extract_str():
    branch_net1 = FeedForward(input_dimensons=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensons=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensons=1, output_dimensions=10)
    networks = {branch_net1 : ['a'],
                branch_net2 : ['b'],
                trunk_net : ['c']}
    model = MIONet(networks=networks,
                   reduction='+',
                   aggregator='*')
    model(input_)

def test_forward_extract_int():
    branch_net1 = FeedForward(input_dimensons=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensons=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensons=1, output_dimensions=10)
    networks = {branch_net1 : [0],
                branch_net2 : [1],
                trunk_net : [2]}
    model = MIONet(networks=networks,
                   reduction='+',
                   aggregator='*')
    model(data)

def test_forward_extract_str_wrong():
    branch_net1 = FeedForward(input_dimensons=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensons=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensons=1, output_dimensions=10)
    networks = {branch_net1 : ['a'],
                branch_net2 : ['b'],
                trunk_net : ['c']}
    model = MIONet(networks=networks,
                   reduction='+',
                   aggregator='*')
    with pytest.raises(RuntimeError):
        model(data)

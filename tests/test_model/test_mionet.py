import pytest
import torch

from pina import LabelTensor
from pina.model import MIONet
from pina.model import FeedForward

data = torch.rand((20, 3))
input_vars = ["a", "b", "c"]
input_ = LabelTensor(data, input_vars)


def test_constructor():
    branch_net1 = FeedForward(input_dimensions=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensions=2, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=1, output_dimensions=10)
    networks = {branch_net1: ["x"], branch_net2: ["x", "y"], trunk_net: ["z"]}
    MIONet(networks=networks, reduction="+", aggregator="*")


def test_constructor_fails_when_invalid_inner_layer_size():
    branch_net1 = FeedForward(input_dimensions=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensions=2, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=1, output_dimensions=12)
    networks = {branch_net1: ["x"], branch_net2: ["x", "y"], trunk_net: ["z"]}
    with pytest.raises(ValueError):
        MIONet(networks=networks, reduction="+", aggregator="*")


def test_forward_extract_str():
    branch_net1 = FeedForward(input_dimensions=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=1, output_dimensions=10)
    networks = {branch_net1: ["a"], branch_net2: ["b"], trunk_net: ["c"]}
    model = MIONet(networks=networks, reduction="+", aggregator="*")
    model(input_)


def test_backward_extract_str():
    data = torch.rand((20, 3))
    data.requires_grad = True
    input_vars = ["a", "b", "c"]
    input_ = LabelTensor(data, input_vars)
    branch_net1 = FeedForward(input_dimensions=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=1, output_dimensions=10)
    networks = {branch_net1: ["a"], branch_net2: ["b"], trunk_net: ["c"]}
    model = MIONet(networks=networks, reduction="+", aggregator="*")
    model(input_)
    l = torch.mean(model(input_))
    l.backward()
    assert data._grad.shape == torch.Size([20, 3])


def test_forward_extract_int():
    branch_net1 = FeedForward(input_dimensions=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=1, output_dimensions=10)
    networks = {branch_net1: [0], branch_net2: [1], trunk_net: [2]}
    model = MIONet(networks=networks, reduction="+", aggregator="*")
    model(data)


def test_backward_extract_int():
    data = torch.rand((20, 3))
    data.requires_grad = True
    branch_net1 = FeedForward(input_dimensions=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=1, output_dimensions=10)
    networks = {branch_net1: [0], branch_net2: [1], trunk_net: [2]}
    model = MIONet(networks=networks, reduction="+", aggregator="*")
    model(data)
    l = torch.mean(model(data))
    l.backward()
    assert data._grad.shape == torch.Size([20, 3])


def test_forward_extract_str_wrong():
    branch_net1 = FeedForward(input_dimensions=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=1, output_dimensions=10)
    networks = {branch_net1: ["a"], branch_net2: ["b"], trunk_net: ["c"]}
    model = MIONet(networks=networks, reduction="+", aggregator="*")
    with pytest.raises(RuntimeError):
        model(data)


def test_backward_extract_str_wrong():
    data = torch.rand((20, 3))
    data.requires_grad = True
    branch_net1 = FeedForward(input_dimensions=1, output_dimensions=10)
    branch_net2 = FeedForward(input_dimensions=1, output_dimensions=10)
    trunk_net = FeedForward(input_dimensions=1, output_dimensions=10)
    networks = {branch_net1: ["a"], branch_net2: ["b"], trunk_net: ["c"]}
    model = MIONet(networks=networks, reduction="+", aggregator="*")
    with pytest.raises(RuntimeError):
        model(data)
        l = torch.mean(model(data))
        l.backward()
        assert data._grad.shape == torch.Size([20, 3])

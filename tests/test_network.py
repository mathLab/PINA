import torch
import torch.nn as nn
import pytest
from pina.model import Network
from pina import LabelTensor


class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.layers(x)


class SimpleNetExtraFeat(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.layers(x)


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (torch.sin(x.extract(['x'])*torch.pi) *
             torch.sin(x.extract(['y'])*torch.pi))
        return LabelTensor(t, ['sin(x)sin(y)'])


input_variables = ['x', 'y']
output_variables = ['u']
data = torch.rand((20, 2))
input_ = LabelTensor(data, input_variables)


def test_constructor():
    net = SimpleNet()
    pina_net = Network(model=net, input_variables=input_variables,
                       output_variables=output_variables)


def test_forward():
    net = SimpleNet()
    pina_net = Network(model=net, input_variables=input_variables,
                       output_variables=output_variables)
    output_ = pina_net(input_)
    assert output_.labels == output_variables


def test_constructor_extrafeat():
    net = SimpleNetExtraFeat()
    feat = [myFeature()]
    pina_net = Network(model=net, input_variables=input_variables,
                       output_variables=output_variables, extra_features=feat)


def test_forward_extrafeat():
    net = SimpleNetExtraFeat()
    feat = [myFeature()]
    pina_net = Network(model=net, input_variables=input_variables,
                       output_variables=output_variables, extra_features=feat)
    output_ = pina_net(input_)
    assert output_.labels == output_variables

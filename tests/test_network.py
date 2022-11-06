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

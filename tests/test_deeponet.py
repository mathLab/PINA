import torch
import pytest

from pina import LabelTensor
from pina.model import DeepONet, FeedForward as FFN


data = torch.rand((20, 3))
input_vars = ['a', 'b', 'c']
output_vars = ['d']
input_ = LabelTensor(data, input_vars)


def test_constructor():
    branch = FFN(input_variables=['a', 'c'], output_variables=20)
    trunk = FFN(input_variables=['b'], output_variables=20)
    onet = DeepONet(trunk_net=trunk, branch_net=branch,
                    output_variables=output_vars)

def test_forward():
    branch = FFN(input_variables=['a', 'c'], output_variables=10)
    trunk = FFN(input_variables=['b'], output_variables=10)
    onet = DeepONet(trunk_net=trunk, branch_net=branch,
                    output_variables=output_vars)
    output_ = onet(input_)
    assert output_.labels == output_vars

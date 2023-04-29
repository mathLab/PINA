import pytest
import torch

from pina import LabelTensor
from pina.model import DeepONet
from pina.model import FeedForward as FFN

data = torch.rand((20, 3))
input_vars = ['a', 'b', 'c']
output_vars = ['d']
input_ = LabelTensor(data, input_vars)

# TODO

# def test_constructor():
#     branch = FFN(input_variables=['a', 'c'], output_variables=20)
#     trunk = FFN(input_variables=['b'], output_variables=20)
#     onet = DeepONet(nets=[trunk, branch], output_variables=output_vars)

# def test_constructor_fails_when_invalid_inner_layer_size():
#     branch = FFN(input_variables=['a', 'c'], output_variables=20)
#     trunk = FFN(input_variables=['b'], output_variables=19)
#     with pytest.raises(ValueError):
#         DeepONet(nets=[trunk, branch], output_variables=output_vars)

# def test_forward():
#     branch = FFN(input_variables=['a', 'c'], output_variables=10)
#     trunk = FFN(input_variables=['b'], output_variables=10)
#     onet = DeepONet(nets=[trunk, branch], output_variables=output_vars)
#     output_ = onet(input_)
#     assert output_.labels == output_vars

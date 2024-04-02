import torch
import pytest

from pina.model.layers import LowRankBlock
from pina import LabelTensor


input_dimensions=2
embedding_dimenion=1
rank=4
inner_size=20
n_layers=2
func=torch.nn.Tanh
bias=True

def test_constructor():
    LowRankBlock(input_dimensions=input_dimensions,
                 embedding_dimenion=embedding_dimenion,
                 rank=rank,
                 inner_size=inner_size,
                 n_layers=n_layers,
                 func=func,
                 bias=bias)
    
def test_constructor_wrong():
    with pytest.raises(ValueError):
        LowRankBlock(input_dimensions=input_dimensions,
                 embedding_dimenion=embedding_dimenion,
                 rank=0.5,
                 inner_size=inner_size,
                 n_layers=n_layers,
                 func=func,
                 bias=bias)
           
def test_forward():
    block = LowRankBlock(input_dimensions=input_dimensions,
                 embedding_dimenion=embedding_dimenion,
                 rank=rank,
                 inner_size=inner_size,
                 n_layers=n_layers,
                 func=func,
                 bias=bias)
    data = LabelTensor(torch.rand(10, 30, 3), labels=['x', 'y', 'u'])
    block(data.extract('u'), data.extract(['x', 'y']))

def test_backward():
    block = LowRankBlock(input_dimensions=input_dimensions,
                 embedding_dimenion=embedding_dimenion,
                 rank=rank,
                 inner_size=inner_size,
                 n_layers=n_layers,
                 func=func,
                 bias=bias)
    data = LabelTensor(torch.rand(10, 30, 3), labels=['x', 'y', 'u'])
    data.requires_grad_(True)
    out = block(data.extract('u'), data.extract(['x', 'y']))
    loss = out.mean()
    loss.backward()
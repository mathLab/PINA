import torch
from pina.model import AVNO

output_channels = 5
batch_size = 15


def test_constructor():
    input_channels = 1
    output_channels = 1
    #minimuum constructor
    AVNO(input_channels, output_channels, torch.rand(10000, 2))

    #all constructor
    AVNO(input_channels, output_channels, torch.rand(100, 2), inner_size=5,n_layers=5,func=torch.nn.ReLU)



def test_forward():
    input_channels = 1
    output_channels = 1
    input_ = torch.rand(batch_size, 1000, input_channels)
    points=torch.rand(1000,2)
    ano = AVNO(input_channels, output_channels, points)
    out = ano(input_)
    assert out.shape == torch.Size([batch_size, points.shape[0], output_channels])



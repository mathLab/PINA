from pina.model.layers import ResidualBlock
import torch


# INPUTS
input_dim = 10
hidden_dim = 5
output_dim = 7
N = 37
x = torch.rand(size=(N, input_dim))

def test_constructor():
    ResidualBlock(input_dim=input_dim, 
                  output_dim=output_dim, 
                  hidden_dim=hidden_dim)



def test_forward():
    model = ResidualBlock(input_dim=input_dim, 
                          output_dim=output_dim, 
                          hidden_dim=hidden_dim)
    output_ = model(x)
    assert output_.shape == (N, output_dim)



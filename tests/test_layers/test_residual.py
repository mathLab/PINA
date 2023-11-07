from pina.model.layers import ResidualBlock
import torch


def test_constructor():

    res_block = ResidualBlock(input_dim=10, output_dim=3, hidden_dim=4)

    res_block = ResidualBlock(input_dim=10,
                              output_dim=3,
                              hidden_dim=4,
                              spectral_norm=True)


def test_forward():

    res_block = ResidualBlock(input_dim=10, output_dim=3, hidden_dim=4)

    x = torch.rand(size=(80, 10))
    y = res_block(x)
    assert y.shape[1] == 3
    assert y.shape[0] == x.shape[0]
